from json import detect_encoding
from . import setup_path
# import setup_path
import airsim
import cv2
import numpy as np 
import pprint
import time
import threading


imageInfo_show = []

# connect to the AirSim simulator
# client = airsim.VehicleClient()
def Detection_keep(client, camera_name="0", radius_m=10):
    
    # client.confirmConnection()

    # set camera name and image type to request images and detections
    image_type = airsim.ImageType.Scene

    # set detection radius in [cm]
    client.simSetDetectionFilterRadius(camera_name, image_type, radius_m * 100)
    # add desired object name to detect in wild card/regex format
    '''
    垃圾箱:SM_Bin*
    消防栓:SM_FireHydrant*
    路灯:BP_LightPole*
    邮箱:MailBox*
    红绿灯:BP_TraficLightCorner*
        BP_TraficLightSingle* // 
    树:InstancedFoliageActor*
    黑色柱子:Goal*

    '''
    client.simAddDetectionFilterMeshName(camera_name, image_type, "InstancedFoliageActor*") # InstancedFoliageActor* SM*
    client.simAddDetectionFilterMeshName(camera_name, image_type, "SM_Bin*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "SM_FireHydrant*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_LightPole*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "MailBox*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_TraficLightCorner*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_TraficLightSingle*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "Goal*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder_*")


    while True:
        rawImage = client.simGetImage(camera_name, image_type)
        if not rawImage:
            continue
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        cylinders = client.simGetDetections(camera_name, image_type)
        if cylinders:
            for cylinder in cylinders:
                s = pprint.pformat(cylinder)
                print("SM: %s" % s)

                cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                cv2.putText(png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))

    
        cv2.imshow("Camera", png)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            client.simClearDetectionMeshNames(camera_name, image_type)
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            client.simAddDetectionFilterMeshName(camera_name, image_type, "InstancedFoliageActor*")

        time.sleep(2)
    cv2.destroyAllWindows()


def Detection(client, camera_name="0", radius_m=200):
    image_type = airsim.ImageType.Scene

    client.simSetDetectionFilterRadius(camera_name, image_type, radius_m * 100)
    '''
    垃圾箱:SM_Bin*
    消防栓:SM_FireHydrant*
    路灯:BP_LightPole*
    邮箱:MailBox*
    红绿灯:BP_TraficLightCorner*
        BP_TraficLightSingle* // 
    树:InstancedFoliageActor*
    黑色柱子:Goal*
    '''
    client.simAddDetectionFilterMeshName(camera_name, image_type,
                                         "InstancedFoliageActor*")  # InstancedFoliageActor*
    client.simAddDetectionFilterMeshName(camera_name, image_type,
                                         "SM*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "SM_Bin*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "SM_FireHydrant*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_LightPole*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "MailBox*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_TraficLightCorner*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_TraficLightSingle*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "Goal*")
    client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*")

    rawImage = client.simGetImage(camera_name, image_type)
    global  imageInfo_show
    if not rawImage:
        imageInfo_show = [None, []]
        return []
    cylinders = client.simGetDetections(camera_name, image_type)

    imageInfo_show = [rawImage, cylinders]

    return cylinders


def show(delay=0.2, camera_name="0", image_type=airsim.ImageType.Scene):
    print('camera show start')
    time.sleep(0.2)
    count = 0
    global imageInfo_show
    while count < 5:
        [rawImage, cylinders] = imageInfo_show
        if rawImage is not None and cylinders is not None:
            count = 0
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            if cylinders:
                for cylinder in cylinders:
                    cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                    cv2.putText(png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
            # cv2.namedWindow("Camera", cv2.WINDOW_FREERATIO)
            cv2.imshow("Camera", png)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                client.simClearDetectionMeshNames(camera_name, image_type)
            elif cv2.waitKey(1) & 0xFF == ord('a'):
                client.simAddDetectionFilterMeshName(camera_name, image_type, "InstancedFoliageActor*")
            imageInfo_show = [None, None]
        elif rawImage is None and cylinders is None:
            count += 1
        time.sleep(delay)
    cv2.destroyAllWindows()
    print('camera show end')


if __name__ == '__main__':
    client = airsim.MultirotorClient(port=41451)
    # starttime = time.time()
    Detection(client)
    # print(time.time() - starttime)
