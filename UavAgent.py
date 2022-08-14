import math
import airsim
import numpy as np
from scipy import linalg


def get_state(client, vehicle_name=''):
    """
    获取无人机状态
    :param client: AirSim连接客户端
    :param vehicle_name: 无人机名称
    :return: 无人机状态
    """
    # 获取无人机状态
    DIG = 6
    State = client.getMultirotorState(vehicle_name=vehicle_name)
    kinematics = State.kinematics_estimated
    state = {
        "timestamp": str(State.timestamp),
        "position": [round(ele, DIG) for i, ele in
                     enumerate(kinematics.position.to_numpy_array().tolist())],
        "orientation": [round(i, DIG) for i in airsim.to_eularian_angles(kinematics.orientation)],
        "linear_velocity": [round(i, DIG) for i in kinematics.linear_velocity.to_numpy_array().tolist()],
        "linear_acceleration": [round(i, DIG) for i in kinematics.linear_acceleration.to_numpy_array().tolist()],
        "angular_velocity": [round(i, DIG) for i in kinematics.angular_velocity.to_numpy_array().tolist()],
        "angular_acceleration": [round(i, DIG) for i in kinematics.angular_acceleration.to_numpy_array().tolist()]
    }
    return state


def move_by_acceleration(client, ax_cmd, ay_cmd, az_cmd, duration=1, waited=False, vehicle_name=''):
    """
    四旋翼加速度控制，不改变偏航角
    :param client: AirSim连接客户端
    :param ax_cmd: x轴方向加速度
    :param ay_cmd: y轴方向加速度
    :param az_cmd: z轴方向加速度
    :param duration: 命令持续时间
    :param waited: 命令是否阻塞
    :param vehicle_name: 无人机名称
    :return: 无
    """
    # 读取自身yaw角度
    state = get_state(client)
    angles = state['orientation']
    z = state['position'][2]
    yaw_my = angles[2]
    g = 9.8  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1 / (g + az_cmd) * np.dot(A_psi_inverse, np.array([[ax_cmd], [ay_cmd]]))
    theta = math.atan(angle_h_cmd[0, 0])
    phi = math.atan(angle_h_cmd[1, 0] * math.cos(theta))
    # client.moveToZAsync(z_cmd, vz).join()
    # client.moveByRollPitchYawZAsync(phi, theta, 0, z_cmd, duration)
    throttle = -0.3 * az_cmd + 0.6
    if throttle < 0:
        throttle = 0
    elif throttle > 1:
        throttle = 1
    if waited:
        client.moveByRollPitchYawThrottleAsync(phi, theta, 0, throttle, duration, vehicle_name=vehicle_name).join()
    else:
        client.moveByRollPitchYawThrottleAsync(phi, theta, 0, throttle, duration, vehicle_name=vehicle_name)


def move_by_acceleration_horizontal_yaw(client, ax_cmd, ay_cmd, z_cmd, yaw, duration=1, waited=False, vehicle_name=''):
    """
    四旋翼x-y平面加速度控制，可设置偏航角
    :param client: AirSim连接客户端
    :param ax_cmd: x轴方向加速度
    :param ay_cmd: y轴方向加速度
    :param z_cmd: 无人机飞行高度
    :param yaw: 无人机偏航角
    :param duration: 命令持续时间
    :param waited: 命令是否阻塞
    :param vehicle_name: 无人机名称
    :return: 无
    """
    # 读取自身yaw角度
    # state = client.simGetGroundTruthKinematics()
    # angles = airsim.to_eularian_angles(state.orientation)
    # yaw_my = angles[2]
    yaw_my = -yaw
    g = 9.8  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1 / g * np.dot(A_psi_inverse, np.array([[ax_cmd], [ay_cmd]]))
    theta = math.atan(angle_h_cmd[0, 0])
    phi = math.atan(angle_h_cmd[1, 0] * math.cos(theta))

    if waited:
        client.moveByRollPitchYawZAsync(phi, theta, yaw_my, z_cmd, duration, vehicle_name=vehicle_name).join()
    else:
        client.moveByRollPitchYawZAsync(phi, theta, yaw_my, z_cmd, duration, vehicle_name=vehicle_name)


def move_tracking_lqr(client, p_aim, v_aim, height, a_aim, duration, waited=False):
    """
    x-y平面内，LQR方法控制无人机跟踪目标状态
    :param client: AirSim连接客户端
    :param p_aim: 目标位置
    :param v_aim: 目标速度
    :param height: 飞行高度
    :param a_aim: 目标加速度
    :param duration: 命令持续时间
    :param waited: 命令是否阻塞
    :return: 无
    """
    A = np.array([[1, 0, duration, 0],
                  [0, 1, 0, duration],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    B = np.array([[0, 0],
                  [0, 0],
                  [duration, 0],
                  [0, duration]])
    Q = np.diag([1, 1, 1, 1])
    R = np.diag([.4, .4])
    S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(linalg.inv(B.T * S * B + R) * (B.T * S * A))
    # 读取当前的位置和速度
    UAV_state = client.simGetGroundTruthKinematics()
    pos_now = np.array([[UAV_state.position.x_val],
                        [UAV_state.position.y_val],
                        [UAV_state.position.z_val]])
    vel_now = np.array([[UAV_state.linear_velocity.x_val],
                        [UAV_state.linear_velocity.y_val],
                        [UAV_state.linear_velocity.z_val]])
    state_now = np.vstack((pos_now[0:2], vel_now[0:2]))
    # 目标状态
    state_des = np.array([[p_aim[0]], [p_aim[1]], [v_aim[0]], [v_aim[1]]])
    # LQR轨迹跟踪
    a = np.dot(K, state_des - state_now) + np.array([[a_aim[0]], [a_aim[1]]])
    # 四旋翼加速度控制
    yaw = math.atan2(state_des[3], state_des[2])
    move_by_acceleration_horizontal_yaw(client, a[0, 0], a[1, 0], height, yaw, duration, waited=waited)

