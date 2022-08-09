"""
python_tracking_and_avoid.py
CarrotChasing轨迹跟踪算法与APF避障算法融合
"""

import math
import numpy as np
import airsim
import time
from ObstacleDetection.obstacles_detect import obstacles_detect
from UavAgent import move_by_acceleration_horizontal_yaw, get_state
from mymath import distance, myatan, isClockwise


def move_by_path_and_avoid_APF(client, Path, K_track=None, delta=1, K_avoid=None,
                               Q_search=10, epsilon=2, Ul=None, dt=0.6, vehicle_name='UAV0'):
    """
    :param client: AirSim连接客户端
    :param Path: 被跟踪航路点
    :param K_track: 轨迹跟踪控制器参数
    :param delta: 向前搜索下一个航路点的距离
    :param K_avoid: 轨迹跟踪控制器参数
    :param Q_search: 搜索障碍物距离
    :param epsilon: 误差上限
    :param Ul: 控制器输出上限
    :param dt: 迭代时间
    :param vehicle_name: 被控无人机名称
    :return: 无
    """

    # 读取初始参数
    [K0, K1, K2] = K_track
    [Kg, Kr] = K_avoid
    [Ul_avoid, Ul_track] = Ul
    # 读取无人机初始状态
    state = get_state(client, vehicle_name=vehicle_name)
    P_start = np.array(state['position'])  # 初始位置
    V_start = np.array(state['linear_velocity'])[0:2]  # 初始速度
    pos_record = [P_start]  # 记录位置
    I2 = np.matrix([[1, 0],  # 用于计算轨迹跟踪控制器参数
                    [0, 1]])
    # 开始飞行
    count = 0  # 已经记录的位置的总数
    P_curr = P_start  # 当前位置
    V_curr = V_start
    V_last = np.array([0, 0])
    plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
    pos_z = -3
    Wb = P_curr[0:2]
    nowtime, lasttime = 0, 0
    for path_num in range(len(Path)):
        Wa = Wb  # 出发航路点
        Wb = np.array([Path[path_num].x_val,
                       Path[path_num].y_val])  # 目标航路点
        Wa_sub_Wb_matrix = np.matrix((Wa - Wb)).T  # 计算差矩阵
        A = I2 - Wa_sub_Wb_matrix.dot(Wa_sub_Wb_matrix.T) / (distance(Wa, Wb) ** 2)
        Pt_matrix = np.matrix(P_curr[0:2] - Wb).T  # 用于计算的中间矩阵，目标点到当前点的向量
        d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))  # 沿着轨迹方向，当前点距离目标点的距离
        # 开始跟踪当前航路Wa-Wb，
        # 如Wb为最后一个航点，则P距离Wb小于epsilon时终止
        # 否则，在delta距离内发现航点Wb时终止
        while (path_num != len(Path) - 1 or distance(Wb, P_curr[0:2]) > epsilon) and \
                (path_num == len(Path) - 1 or d_to_Wb > delta):
            Pt_matrix = np.matrix(P_curr[0:2] - Wb).T
            Frep = np.array([0, 0])  # 斥力
            info_obstacles = obstacles_detect(client, Q_search, vehicle_name=vehicle_name)
            num_obstacles = 0
            for obstacle in info_obstacles:
                state = get_state(client, vehicle_name=vehicle_name)
                yaw = -state['orientation'][2]
                Rz = np.array([[math.cos(yaw), math.sin(yaw)],
                               [-math.sin(yaw), math.cos(yaw)]])
                pos_obstacle_min = obstacle.min
                pos_obstacle_max = obstacle.max
                P_search_list = np.array([np.array([pos_obstacle_min.x_val, pos_obstacle_min.y_val]),
                                          np.array([pos_obstacle_min.x_val, pos_obstacle_max.y_val]),
                                          np.array([pos_obstacle_max.x_val, pos_obstacle_min.y_val]),
                                          np.array([pos_obstacle_max.x_val, pos_obstacle_max.y_val])]).dot(Rz.T)
                for P_search in P_search_list:
                    d_search = np.linalg.norm(P_search)
                    if d_search <= Q_search:
                        num_obstacles += 1
                        Frep = Frep + \
                               Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                               (-P_search) * (distance(P_search + P_curr[0:2], Wb) ** 2)
            # 未检测到障碍物，执行航路点跟踪
            if num_obstacles == 0:
                U1 = np.array((K0 * Pt_matrix + K1 * A.dot(Pt_matrix)).T)[0]
                if np.linalg.norm(U1, ord=np.inf) > Ul_track:
                    U1 = U1 * Ul_track / np.linalg.norm(U1, ord=np.inf)
                U = -(U1 + V_curr) / K2  # 计算控制器输出并转换为array
            # 检测到障碍物，执行避障
            else:
                Frep = Frep / num_obstacles
                Fatt = -Kg * (P_curr[0:2] - Wb)  # 计算引力
                if count >= 1:
                    # 计算上两个时刻物体相对终点的位移，以判断是否陷入局部极小值
                    p0 = pos_record[-1]
                    p1 = pos_record[-2]
                    nowtime = time.time()
                    Vra = (distance(p0, Wb) - distance(p1, Wb)) / (nowtime - lasttime)
                    lasttime = nowtime
                    if abs(Vra) < 0.95 * Ul_avoid and len(info_obstacles) != 0 \
                            and np.linalg.norm(V_curr, ord=np.inf) < np.linalg.norm(V_last, ord=np.inf):  # 陷入局部极小值
                        # 之前不是局部极小状态时，根据当前位置计算斥力偏向角theta
                        angle_g = myatan([0, 0], [Fatt[0], Fatt[1]])
                        angle_g = 0 if angle_g is None else angle_g
                        angle_r = myatan([0, 0], [Frep[0], Frep[1]])
                        angle_r = 0 if angle_r is None else angle_r
                        if isClockwise(angle_g, angle_r):
                            theta = 60 * math.pi / 180
                        else:
                            theta = -60 * math.pi / 180
                        # 之前为局部极小，则继续使用上一时刻的斥力偏向角theta
                        Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                                math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]
                    l = Ul_avoid
                    Kv = 3 * l / (2 * l + abs(Vra))
                    Kd = 15 * math.exp(-(distance(P_curr[0:2], Wb) - 1.5) ** 2 / 2) + 1
                    Ke = 5
                    Fatt = Kv * Kd * Ke * Fatt  # 改进引力
                U = Fatt + Frep  # 计算合力
                if np.linalg.norm(U, ord=np.inf) > Ul_avoid:  # 控制器输出限幅
                    U = Ul_avoid * U / np.linalg.norm(U, ord=np.inf)
                U = (U - V_curr) / K2
            # 执行
            V_next = V_curr + U * dt  # 计算速度
            yaw = math.atan2(V_next[1], V_next[0])
            move_by_acceleration_horizontal_yaw(client, U[0], U[1], pos_z, yaw, duration=dt, vehicle_name=vehicle_name)
            # 画图和记录
            V_last = V_curr
            plot_p1 = plot_p2
            state = get_state(client, vehicle_name=vehicle_name)
            P_curr = np.array(state['position'])
            V_curr = np.array(state['linear_velocity'])[0:2]
            plot_p2 = [airsim.Vector3r(P_curr[0], P_curr[1], P_curr[2])]
            client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

            d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))
            pos_record.append(P_curr)
            count += 1


if __name__ == "__main__":
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True, vehicle_name='UAV0')
    client.armDisarm(True, vehicle_name='UAV0')
    client.simFlushPersistentMarkers()
    client.takeoffAsync(vehicle_name='UAV0')
    client.moveToZAsync(-3, 1, vehicle_name='UAV0').join()

    points = [airsim.Vector3r(60, 0, -3),
              airsim.Vector3r(70, -80, -3),
              airsim.Vector3r(55, -120, -3),
              airsim.Vector3r(0, 0, -3)]

    move_by_path_and_avoid_APF(client, points, K_track=[1.5, 6, 1], delta=5, K_avoid=[3, 60], Q_search=8, epsilon=1,
                               Ul=[2, 3], dt=0.3, vehicle_name='UAV0')

    client.landAsync(vehicle_name='UAV0').join()
    client.armDisarm(False, vehicle_name='UAV0')
    client.enableApiControl(False, vehicle_name='UAV0')
