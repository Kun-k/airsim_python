'''
tracking_carrot.py
airsim 四旋翼轨迹跟踪 Carrot控制算法
'''

import airsim
import math
import numpy as np
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


def move_by_acceleration_horizontal(client, ax_cmd, ay_cmd, az_cmd, z_cmd, duration=1):
    # 读取自身yaw角度
    state = get_state(client)
    angles = state['orientation']
    yaw_my = angles[2]
    g = 9.8  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1 / (g) * np.dot(A_psi_inverse, np.array([[ax_cmd], [ay_cmd]]))
    theta = math.atan(angle_h_cmd[0, 0])
    phi = math.atan(angle_h_cmd[1, 0] * math.cos(theta))
    # client.moveToZAsync(z_cmd, vz).join()
    client.moveByRollPitchYawZAsync(phi, theta, 0, z_cmd, duration).join()


def move_by_path(client, Va, Path, Pz, delta=0.8, K=0.8, K2=0, dt=0.02):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def myatan(A, B):
        x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
        if x1 != x2:
            if x1 > x2:
                return math.atan((y1 - y2) / (x1 - x2)) + math.pi
            else:
                return math.atan((y1 - y2) / (x1 - x2))
        if x1 == x2 and y1 == y2:
            return None
        if x1 == x2 and y1 != y2:
            if y1 > y2:
                return -math.pi / 2
            else:
                return math.pi / 2

    state = client.simGetGroundTruthKinematics()
    psi = airsim.to_eularian_angles(state.orientation)[2]
    Px = state.position.x_val
    Py = state.position.y_val
    Wb = [Px, Py]

    for i in range(len(Path)):
        Wa = Wb
        Wb = [Path[i].x_val, Path[i].y_val]
        theta = myatan(Wa, Wb)
        while True:
            theta_u = myatan(Wa, [Px, Py])
            if theta_u == None:
                theta_u = theta
            beta = theta - theta_u
            Ru = distance(Wa, [Px, Py])
            R = Ru * math.cos(beta)
            e = Ru * math.sin(beta)
            print(Px, Py, e)
            xt = Wa[0] + (R + delta) * math.cos(theta)
            yt = Wa[1] + (R + delta) * math.sin(theta)
            if i == len(Path) - 1:
                if (Px - Wb[0]) * (Wb[0] - Wa[0]) > 0 \
                    or (Py - Wb[1]) * (Wb[1] - Wa[1]) > 0:
                    break
            elif (xt - Wb[0]) * (Wb[0] - Wa[0]) > 0 \
                    or (yt - Wb[1]) * (Wb[1] - Wa[1]) > 0:
                break
            psi_d = myatan([Px, Py], [xt, yt])
            u = K * (psi_d - psi) * Va + K2 * e
            if u > 1:  # 限制u的范围
                u = 1
            psi = psi_d
            Vy = Va * math.sin(psi) + u * dt
            if abs(Vy) >= Va:
                Vy = Va
                Vx = 0
            else:
                Vx = np.sign(math.cos(psi)) * math.sqrt(Va ** 2 - Vy ** 2)
            client.moveByVelocityZAsync(Vx, Vy, Pz, dt).join()
            # 画图
            plot_p1 = [airsim.Vector3r(Px, Py, Pz)]
            state = client.simGetGroundTruthKinematics()
            Px = state.position.x_val
            Py = state.position.y_val
            plot_p2 = [airsim.Vector3r(Px, Py, Pz)]
            client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)


def move_by_path_3d(client, Path, K0=1.5, K1=4, K2=0.6, dt=0.5, a0=1, delta=0.7):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 +
                         (A[1] - B[1]) ** 2 +
                         (A[2] - B[2]) ** 2)

    state = get_state(client)
    P = state['position']
    V = state['linear_velocity']
    Wb = P
    Wb_m = np.matrix(Wb).T

    P_m = np.matrix(P).T
    V_m = np.matrix(V).T
    # pos_record = [P]
    # e_record = []
    I3 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    for i in range(len(Path)):
        Wa = Wb
        Wb = [Path[i].x_val, Path[i].y_val, Path[i].z_val]
        Wa_m = Wb_m
        Wb_m = np.matrix(Wb).T
        A = I3 - (Wa_m - Wb_m).dot((Wa_m - Wb_m).T) / (distance(Wa_m, Wb_m) ** 2)
        Pt = P_m - Wb_m
        e = np.linalg.norm(A.dot(Pt))
        # e_record.append(e)
        # pos_record.append(P)
        d = np.linalg.norm(Pt - A.dot(Pt))
        print('i,', i, 'Start:', Wa, ',Aim:', Wb)
        print('\tP:', P, 'V:', V, 'e:', e)
        while d >= delta or \
                (i == len(Path) - 1
                 and ((P[0] - Wb[0]) * (Wb[0] - Wa[0]) < 0
                      or (P[1] - Wb[1]) * (Wb[1] - Wa[1]) < 0
                      or (P[2] - Wb[2]) * (Wb[2] - Wa[2]) < 0)):
            Pt = P_m - Wb_m
            U1 = K0 * Pt + K1 * A.dot(Pt)
            if np.linalg.norm(U1, ord=np.inf) > a0:
                U1 = U1 * a0 / np.linalg.norm(U1, ord=np.inf)
            U = -(U1 + V_m) / K2
            U_cmd = np.array(U)[:, 0]
            z_cmd = P[2] + (V[2] + U_cmd[2] * dt) * dt
            move_by_acceleration_horizontal(client, U_cmd[0], U_cmd[1], U_cmd[2], z_cmd, dt)
            e = np.linalg.norm(A.dot(Pt))
            d = np.linalg.norm(Pt - A.dot(Pt))
            print('\tP:', P, 'V:', V, 'e:', e, 'U:', U_cmd)
            # e_record.append(e)
            # pos_record.append(P)
            # 画图
            plot_p1 = [airsim.Vector3r(P[0], P[1], P[2])]
            state = get_state(client)
            P = state['position']
            V = state['linear_velocity']
            P_m = np.matrix(P).T
            V_m = np.matrix(V).T
            plot_p2 = [airsim.Vector3r(P[0], P[1], P[2])]
            client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)
    # print((P[0] - Wb[0]) * (Wb[0] - Wa[0]))
    # print((P[1] - Wb[1]) * (Wb[1] - Wa[1]))
    # print((P[2] - Wb[2]) * (Wb[2] - Wa[2]))


client = airsim.MultirotorClient()  # 创建连接
client.confirmConnection()          # 检查连接
client.reset()
client.enableApiControl(True)       # 获取控制权
client.armDisarm(True)              # 电机启转
client.takeoffAsync().join()        # 起飞
client.moveToZAsync(-3, 1).join()   # 上升到3米高度
client.simSetTraceLine([1, 0, 0, 1], thickness=5)
client.simFlushPersistentMarkers()  # 清空画图

# state = get_state(client)
# print(state['position'])
# print(state['orientation'])
# print(state['linear_velocity'])

# 二维航路点跟踪
# 初始化4个点的坐标，并在视口中标识出来
# points = [airsim.Vector3r(5, 0, -3),
#           airsim.Vector3r(5, 8, -3),
#           airsim.Vector3r(8, 12, -3),
#           airsim.Vector3r(4, 9, -3)]
# client.simPlotPoints(points, color_rgba=[0, 1, 0, 1], size=30, is_persistent=True)

# 方法1：按照逐个点飞，形成正方形
# client.moveToPositionAsync(5, 0, -3, 1).join()
# client.moveToPositionAsync(5, 5, -3, 1).join()
# client.moveToPositionAsync(0, 5, -3, 1).join()
# client.moveToPositionAsync(0, 0, -3, 1).join()

# 方法2：直接按照航路点飞正方形轨迹
# client.moveOnPathAsync(points, 1).join()

# 方法3：自己实现的航路点方法(二维)
# move_by_path(client, 3, points, -3)

# 三维航路点跟踪
# 初始化4个点的坐标，并在视口中标识出来
# points = [airsim.Vector3r(5, 0, -3)]
path_for_airsim = np.load('path_for_airsim.npy')
points = []
for p in path_for_airsim:
    points.append(airsim.Vector3r(p[0], p[1], p[2]))

# client.simPlotPoints(points, color_rgba=[0, 1, 0, 1], size=30, is_persistent=True)
# client.simPlotLineStrip(points, color_rgba=[0, 1, 0, 1], thickness=5, is_persistent=True)
move_by_path_3d(client, points, delta=1, a0=1)
# client.moveOnPathAsync(points, 1).join()


# 仿真结束
client.landAsync().join()           # 降落
client.armDisarm(False)             # 电机上锁
client.enableApiControl(False)      # 释放控制权