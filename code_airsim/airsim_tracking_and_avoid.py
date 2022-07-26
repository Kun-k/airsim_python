'''
python_tracking_and_avoid.py
CarrotChasing轨迹跟踪算法与APF避障算法融合
'''


import math
import numpy as np
from ObstacleDetection.Detection import Detection, Detection_keep, show
import airsim
import threading
from UAV import demo_keyboard_uav
import keyboard
from scipy import linalg
import cv2
import time


def get_state(client):
    # 获取无人机状态
    DIG = 6
    State = client.getMultirotorState()
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


def move_by_acceleration_horizontal(client, ax_cmd, ay_cmd, z_cmd, duration=1):
    # 读取自身yaw角度
    state = get_state(client)
    angles = state['orientation']
    yaw_my = -angles[2]
    g = 9.8  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1 / (g) * np.dot(A_psi_inverse, np.array([[ax_cmd], [ay_cmd]]))
    theta = math.atan(angle_h_cmd[0, 0])
    phi = math.atan(angle_h_cmd[1, 0] * math.cos(theta))
    client.moveByRollPitchYawZAsync(phi, theta, yaw_my, z_cmd, duration)


def move_by_acceleration_horizontal_yaw(client, ax_cmd, ay_cmd, z_cmd, yaw, duration=1):
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
    client.moveByRollPitchYawZAsync(phi, theta, yaw_my, z_cmd, duration)


'''
P           初始位置
V           初始速度
Path        航路点集合
mymap       储存有障碍物信息的地图
K_track     轨迹跟踪控制器参数
delta       向前搜索下一个航路点的距离
K_avoid     避障控制器参数（引力，斥力）
Q_search    搜索障碍物距离
epsilon     误差上限
Vl          速率上限
Ul          控制器输出上限
dt          迭代时间
'''
def move_by_path_and_avoid(client, Path, K_track=None, delta=1, K_avoid=None,
                           Q_search=10, epsilon=2, Vl=1, Ul=None, dt=0.6):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def myatan(a, b):
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return math.pi / 2 if b > 0 else math.pi * 3 / 2
        if b == 0:
            return 0 if a > 0 else -math.pi
        if b > 0:
            return math.atan(b / a) if a > 0 else (math.atan(b / a) + math.pi)
        return math.atan(b / a + 2 * math.pi) if a > 0 else math.atan(b / a) + math.pi

    def isClockwise(a, b):
        da = b - a
        if 0 < da < math.pi or -math.pi * 2 < da < -math.pi:
            return False
        return True

    # 读取初始状态
    if K_track == None:
        K_track = [2, 2, 2]
    [K0, K1, K2] = K_track
    if K_avoid == None:
        K_avoid = [3, 30]
    [Kg, Kr] = K_avoid
    if Ul == None:
        Ul = [0.5, 0.5]
    [Ul_avoid, Ul_track] = Ul
    # 读取初始状态
    state = get_state(client)
    P_start = np.array(state['position'])              # 初始位置
    V_start = np.array(state['linear_velocity'])[0:2]  # 初始速度
    pos_record = [P_start]  # 记录位置
    I2 = np.matrix([[1, 0],            # 用于计算轨迹跟踪控制器参数
                    [0, 1]])
    # 开始飞行
    pos_num = 0  # 已经记录的位置的总数
    P_curr = P_start  # 当前位置
    V_curr = V_start
    pos_z = -3  # TODO
    ob_flag = False  # 用于判断局部极小值
    Wb = P_curr[0:2]
    for path_num in range(len(Path)):
        Wa = Wb                                                             # 出发航路点
        Wb = np.array([Path[path_num].x_val,
                       Path[path_num].y_val])                                         # 目标航路点
        Wa_sub_Wb_matrix = np.matrix((Wa - Wb)).T                           # 计算差矩阵
        A = I2 - Wa_sub_Wb_matrix.dot(Wa_sub_Wb_matrix.T) / (distance(Wa, Wb) ** 2)
        Pt_matrix = np.matrix(P_curr[0:2] - Wb).T                                # 用于计算的中间矩阵，目标点到当前点的向量
        # e = np.linalg.norm(A.dot(Pt_matrix))                              # 误差，当前到目标轨迹的垂直距离
        # e_record.append(e)
        d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))              # 沿着轨迹方向，当前点距离目标点的距离
        print('跟踪航路点:', path_num, 'Start:', Wa, ',Aim:', Wb)
        # 首先偏航，转向目标点
        theta_to_Wb = myatan(Wb[0] - Wa[0], Wb[1] - Wa[1])
        client.moveByRollPitchYawZAsync(0, 0, -theta_to_Wb, pos_z, 2).join()
        # 开始跟踪当前航路Wa-Wb，
        # 如Wb为最后一个航点，则P距离Wb小于epsilon时终止
        # 否则，在delta距离内发现航点Wb时终止
        while (path_num != len(Path) - 1 or distance(Wb, P_curr[0:2]) > epsilon) and \
                (path_num == len(Path) - 1 or d_to_Wb > delta):
            Pt_matrix = np.matrix(P_curr[0:2] - Wb).T
            Frep = np.array([0, 0])  # 斥力
            info_obstacles = []
            for cam in ['0', '1', '2']:
                info_obstacles.extend(Detection(client=client, radius_m=Q_search, camera_name=cam))
            for obs in info_obstacles:
                pos_obstacle = obs.relative_pose.position
                P_search = np.array([pos_obstacle.x_val, pos_obstacle.z_val])
                d_search = distance(P_search, P_curr[0:2])
                Frep = Frep + \
                       Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                       (P_curr[0:2] - P_search) * (distance(P_search, Wb) ** 2)
            # 未检测到障碍物，执行航路点跟踪
            if Frep[0] == 0 and Frep[1] == 0:
                U1 = np.array((K0 * Pt_matrix + K1 * A.dot(Pt_matrix)).T)[0]
                if np.linalg.norm(U1, ord=np.inf) > Ul_track:
                    U1 = U1 * Ul_track / np.linalg.norm(U1, ord=np.inf)
                U = -(U1 + V_curr) / K2                             # 计算控制器输出并转换为array
            # 检测到障碍物，执行避障
            else:
                Frep = Frep
                Fatt = -Kg * (P_curr[0:2] - Wb)                          # 计算引力
                if pos_num >= 1:
                    # 计算上两个时刻物体相对终点的位移，以判断是否陷入局部极小值
                    p0 = pos_record[pos_num - 1]
                    p1 = pos_record[pos_num - 2]
                    Vra = (distance(p0, Wb) - distance(p1, Wb)) / dt
                    if abs(Vra) < 0.6 * Vl and Frep[0] != 0 and Frep[1] != 0:  # 陷入局部极小值
                        if ob_flag == False:
                            # 之前不是局部极小状态时，根据当前位置计算斥力偏向角theta
                            angle_g = myatan(Fatt[0], Fatt[1])
                            angle_r = myatan(Frep[0], Frep[1])
                            if isClockwise(angle_g, angle_r):
                                theta = 15 * math.pi / 180
                            else:
                                theta = -15 * math.pi / 180
                            ob_flag = True
                        # 之前为局部极小，则继续使用上一时刻的斥力偏向角theta
                        Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                                math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]
                    else:                                           # 离开局部极小值
                        ob_flag = False
                    l = Vl
                    Kv = 3 * l / (2 * l + abs(Vra))
                    Kd = 15 * math.exp(-(distance(P_curr[0:2], Wb) - 3) ** 2 / 2) + 1
                    Ke = 3
                    Fatt = Kv * Kd * Ke * Fatt                      # 改进引力
                U = Fatt + Frep                                     # 计算合力
                if np.linalg.norm(U, ord=np.inf) > Ul_avoid:        # 控制器输出限幅
                    U = Ul_avoid * U / np.linalg.norm(U, ord=np.inf)
            # 速度预测并限幅
            # V_pred = V_curr + U * dt
            # if abs(V_pred[0]) > Vl or abs(V_pred[1]) > Vl:
            #     U_lim = abs(Vl - max(abs(V_curr))) / dt
            #     U = U_lim * U / np.linalg.norm(U, ord=np.inf)
            # yaw_cmd = -myatan(V_pred[0], V_pred[1]) * 0
            # 执行
            move_by_acceleration_horizontal(client, U[0], U[1], pos_z, dt)
            # V_curr = V_curr + U * dt                                # 计算速度
            # if np.linalg.norm(V_curr) > Vl:                         # 速度限幅
            #     V_curr = Vl * V_curr / np.linalg.norm(V_curr)
            # P_curr = P_curr + V_curr * dt                           # 计算位置

            # 画图和记录
            plot_p1 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
            state = get_state(client)
            P_curr = np.array(state['position'])
            plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
            client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

            V_curr = np.array(state['linear_velocity'])[0:2]
            d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))
            pos_record.append(P_curr)
            # print(P_curr, V_curr, U, distance(P_curr[0:2], Wb))
            pos_num += 1


'''
P           初始位置
V           初始速度
Path        航路点集合
mymap       储存有障碍物信息的地图
K_track     轨迹跟踪控制器参数
delta       向前搜索下一个航路点的距离
K_avoid     避障控制器参数（引力，斥力）
Q_search    搜索障碍物距离
epsilon     误差上限
Vl          速率上限
Ul          控制器输出上限
dt          迭代时间
'''
def move_by_path_and_avoid_lqr(client, Path, K_track=None, delta=1, K_avoid=None,
                           Q_search=10, epsilon=2, Vl=1, Ul=None, dt=0.6):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def myatan(a, b):
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return math.pi / 2 if b > 0 else math.pi * 3 / 2
        if b == 0:
            return 0 if a > 0 else -math.pi
        if b > 0:
            return math.atan(b / a) if a > 0 else (math.atan(b / a) + math.pi)
        return math.atan(b / a + 2 * math.pi) if a > 0 else math.atan(b / a) + math.pi

    def isClockwise(a, b):
        da = b - a
        if 0 < da < math.pi or -math.pi * 2 < da < -math.pi:
            return False
        return True

    # def DLQR(A, B, Q, R):
    #     S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
    #     K = np.matrix(linalg.inv(B.T * S * B + R) * (B.T * S * A))
    #     return K

    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    B = np.array([[0, 0],
                  [0, 0],
                  [dt, 0],
                  [0, dt]])
    Q = np.diag([2, 2, 2, 2])
    R = np.diag([.1, .1])
    S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
    K = np.matrix(linalg.inv(B.T * S * B + R) * (B.T * S * A))

    # 读取初始状态
    if K_track == None:
        K_track = [2, 2, 2]
    [K0, K1, K2] = K_track
    if K_avoid == None:
        K_avoid = [3, 30]
    [Kg, Kr] = K_avoid
    if Ul == None:
        Ul = [0.5, 0.5]
    [Ul_avoid, Ul_track] = Ul
    # 读取初始状态
    state = get_state(client)
    P_start = np.array(state['position'])              # 初始位置
    V_start = np.array(state['linear_velocity'])[0:2]  # 初始速度
    pos_record = [P_start]  # 记录位置
    I2 = np.matrix([[1, 0],            # 用于计算轨迹跟踪控制器参数
                    [0, 1]])
    # 开启摄像头图像显示线程
    thread_cam = threading.Thread(target=show)
    thread_cam.start()
    # 开始飞行
    pos_num = 0  # 已经记录的位置的总数
    P_curr = P_start  # 当前位置
    V_curr = V_start
    pos_z = -3
    ob_flag = False  # 用于判断局部极小值
    Wb = P_curr[0:2]
    for path_num in range(len(Path)):
        Wa = Wb                                                             # 出发航路点
        Wb = np.array([Path[path_num].x_val,
                       Path[path_num].y_val])                                         # 目标航路点
        Wa_sub_Wb_matrix = np.matrix((Wa - Wb)).T                           # 计算差矩阵
        A = I2 - Wa_sub_Wb_matrix.dot(Wa_sub_Wb_matrix.T) / (distance(Wa, Wb) ** 2)
        Pt_matrix = np.matrix(P_curr[0:2] - Wb).T                                # 用于计算的中间矩阵，目标点到当前点的向量
        # e = np.linalg.norm(A.dot(Pt_matrix))                              # 误差，当前到目标轨迹的垂直距离
        # e_record.append(e)
        d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))              # 沿着轨迹方向，当前点距离目标点的距离
        print('跟踪航路点:', path_num, 'Start:', Wa, ',Aim:', Wb)
        # 开始跟踪当前航路Wa-Wb，
        # 如Wb为最后一个航点，则P距离Wb小于epsilon时终止
        # 否则，在delta距离内发现航点Wb时终止
        while (path_num != len(Path) - 1 or distance(Wb, P_curr[0:2]) > epsilon) and \
                (path_num == len(Path) - 1 or d_to_Wb > delta):
            Pt_matrix = np.matrix(P_curr[0:2] - Wb).T
            Frep = np.array([0, 0])  # 斥力
            info_obstacles = []
            for cam in ['0']:
                info_obstacles.extend(Detection(client=client, radius_m=Q_search, camera_name=cam))
            for cylinder in info_obstacles:
                pos_obstacle = cylinder.relative_pose.position
                P_search = np.array([pos_obstacle.x_val, pos_obstacle.z_val])
                d_search = distance(P_search, P_curr[0:2])
                Frep = Frep + \
                       Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                       (P_curr[0:2] - P_search) * (distance(P_search, Wb) ** 2)
            # 未检测到障碍物，执行航路点跟踪
            if Frep[0] == 0 and Frep[1] == 0:
                U1 = np.array((K0 * Pt_matrix + K1 * A.dot(Pt_matrix)).T)[0]
                if np.linalg.norm(U1, ord=np.inf) > Ul_track:
                    U1 = U1 * Ul_track / np.linalg.norm(U1, ord=np.inf)
                U = -(U1 + V_curr) / K2                             # 计算控制器输出并转换为array
            # 检测到障碍物，执行避障
            else:
                Fatt = -Kg * (P_curr[0:2] - Wb)                          # 计算引力
                if pos_num >= 1:
                    # 计算上两个时刻物体相对终点的位移，以判断是否陷入局部极小值
                    p0 = pos_record[pos_num - 1]
                    p1 = pos_record[pos_num - 2]
                    Vra = (distance(p0, Wb) - distance(p1, Wb)) / dt
                    if abs(Vra) < 0.6 * Vl and Frep[0] != 0 and Frep[1] != 0:  # 陷入局部极小值
                        if ob_flag == False:
                            # 之前不是局部极小状态时，根据当前位置计算斥力偏向角theta
                            angle_g = myatan(Fatt[0], Fatt[1])
                            angle_r = myatan(Frep[0], Frep[1])
                            if isClockwise(angle_g, angle_r):
                                theta = 15 * math.pi / 180
                            else:
                                theta = -15 * math.pi / 180
                            ob_flag = True
                        # 之前为局部极小，则继续使用上一时刻的斥力偏向角theta
                        Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                                math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]
                    else:                                           # 离开局部极小值
                        ob_flag = False
                    l = Vl
                    Kv = 3 * l / (2 * l + abs(Vra))
                    Kd = 15 * math.exp(-(distance(P_curr[0:2], Wb) - 3) ** 2 / 2) + 1
                    Ke = 3
                    Fatt = Kv * Kd * Ke * Fatt                      # 改进引力
                U = Fatt + Frep                                     # 计算合力
                if np.linalg.norm(U, ord=np.inf) > Ul_avoid:        # 控制器输出限幅
                    U = Ul_avoid * U / np.linalg.norm(U, ord=np.inf)
            # 执行
            V_next = V_curr + U * dt                                # 计算速度
            if np.linalg.norm(V_next) > Vl:                         # 速度限幅
                V_next = Vl * V_next / np.linalg.norm(V_next)
                # U = (np.linalg.norm(V_next - V_curr)) * U / np.linalg.norm(U) / dt
            P_next = P_curr[0:2] + V_next * dt                           # 计算位置
            state_now = np.array([[P_curr[0], P_curr[1], V_curr[0], V_curr[1]]]).T  # 当前状态
            state_des = np.array([[P_next[0], P_next[1], V_next[0], V_next[1]]]).T  # 目标状态
            # LQR轨迹跟踪
            a = -np.dot(K, state_now - state_des) + np.array([U[0], U[1]]).T
            # 四旋翼加速度控制
            yaw = math.atan2(state_des[3], state_des[2])
            move_by_acceleration_horizontal_yaw(client, a[0, 0], a[1, 0], pos_z, yaw, duration=dt)
            # 画图和记录
            plot_p1 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
            state = get_state(client)
            P_curr = np.array(state['position'])
            plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
            client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

            V_curr = np.array(state['linear_velocity'])[0:2]
            d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))
            pos_record.append(P_curr)
            # print(P_curr, V_curr, U, distance(P_curr[0:2], Wb))
            pos_num += 1


if __name__ == "__main__":
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.simFlushPersistentMarkers()
    points = [airsim.Vector3r(20, 0, -3),
              airsim.Vector3r(20, -20, -3),
              airsim.Vector3r(0, 0, -3)]
    # points = [airsim.Vector3r(20, 0, -3)]
    client.simPlotPoints(points, color_rgba=[0, 1, 0, 1], size=5, is_persistent=True)

    client.moveToZAsync(-3, 1).join()
    move_by_path_and_avoid_lqr(client, points, epsilon=1, delta=2, dt=0.12, Ul=[1.5, 2], Q_search=7, Vl=2, K_avoid=[3, 40])
    #
    # client.moveToZAsync(-3, 2)
    # thread_detect = threading.Thread(target=Detection_keep, args=(client,))
    # thread_detect.start()
    # demo_keyboard_uav.keyboard_uav(client)
