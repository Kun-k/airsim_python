import numpy as np


def calculate_pose(depth_planar, depth_perspective):
    # get the width and height of img
    height = depth_planar.shape[0]
    width = depth_planar.shape[1]

    # get orientation
    y_orientation = np.arange(0, width).reshape(1, -1)
    y_orientation = np.repeat(y_orientation, height, axis=0)
    y_orientation = y_orientation - (width - 1) / 2

    z_orientation = np.arange(0, height).reshape(-1, 1)
    z_orientation = np.repeat(z_orientation, width, axis=1)
    z_orientation = z_orientation - (height - 1) / 2

    scale_orientation = np.sqrt(np.square(y_orientation) + np.square(z_orientation))

    # calculate position
    x_pos = depth_planar.copy()
    dis_from_center = np.sqrt(np.square(depth_perspective) - np.square(depth_planar))
    y_pos = dis_from_center * y_orientation / scale_orientation
    z_pos = dis_from_center * z_orientation / scale_orientation

    return x_pos, y_pos, z_pos
