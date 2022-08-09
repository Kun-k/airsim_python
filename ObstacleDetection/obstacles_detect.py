import airsim
import cv2
from .calculate_pose import *


def get_images(client, img_request, vehicle_name):
    responses = client.simGetImages(requests=img_request, vehicle_name=vehicle_name)
    return responses


def get_min_and_max_with_mask(pos, mask, threshold):
    pos_for_min = pos.copy()
    pos_for_min[mask == 0] = threshold + 1
    pos_min = np.min(np.min(pos_for_min))

    pos_for_max = pos.copy()
    pos_for_max[mask == 0] = -(threshold + 1)
    pos_max = np.max(np.max(pos_for_max))

    return pos_min, pos_max


def obstacles_detect(client, dis_threshold, vehicle_name=''):
    """
    detect obstacles
    :param client: multi-rotor client
    :param dis_threshold: distance threshold
    :return: obstacles(list[Box3D])
    """
    obstacles = []

    # get response
    image_requests = [
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False),
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)]
    image_responses = get_images(client, image_requests, vehicle_name)

    # get:(1)img_DPer(depth_perspective); (2)img_DPla(depth_planar); (3)img_Seg(segment)
    img_response_DPer = image_responses[0]
    img_DPer = airsim.get_pfm_array(img_response_DPer)
    img_response_DPla = image_responses[1]
    img_DPla = airsim.get_pfm_array(img_response_DPla)
    img_response_Seg = image_responses[2]
    img_Seg = np.frombuffer(img_response_Seg.image_data_uint8, dtype=np.uint8)
    img_Seg = img_Seg.reshape(img_response_Seg.height, img_response_Seg.width, 3)
    img_Seg = cv2.cvtColor(img_Seg, cv2.COLOR_RGB2GRAY)

    # get relative pose
    x_pos, y_pos, z_pos = calculate_pose(depth_planar=img_DPla, depth_perspective=img_DPer)

    # seg and detect
    seg_labels = np.unique(img_Seg)
    for seg_label in seg_labels:
        # get connected components
        img_Seg_binary = (img_Seg == seg_label).astype(np.uint8) * 255
        num_objects, labels = cv2.connectedComponents(img_Seg_binary)

        # detect by threshold
        for i in range(num_objects - 1):
            label = i + 1
            mask = (labels == label).astype(int)  # get mask
            mask_dist = mask * img_DPer  # get mask distance
            mask_dist[mask_dist == 0] = dis_threshold + 1  # turn 0 to (threshold + 1) to get min

            # judge by threshold
            if np.min(np.min(mask_dist)) < dis_threshold:
                x_min, x_max = get_min_and_max_with_mask(pos=x_pos, mask=mask, threshold=dis_threshold)
                y_min, y_max = get_min_and_max_with_mask(pos=y_pos, mask=mask, threshold=dis_threshold)
                z_min, z_max = get_min_and_max_with_mask(pos=z_pos, mask=mask, threshold=dis_threshold)
                obstacle = airsim.Box3D()
                obstacle.min = airsim.Vector3r(x_val=x_min, y_val=y_min, z_val=z_min)
                obstacle.max = airsim.Vector3r(x_val=x_max, y_val=y_max, z_val=z_max)
                obstacles.append(obstacle)

    return obstacles
