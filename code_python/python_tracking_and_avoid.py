'''
python_tracking_and_avoid.py
CarrotChasing轨迹跟踪算法与APF避障算法融合
'''


import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
def move_by_path_and_avoid(P_start, V_start, Path, mymap, K_track=None, delta=1,
                           K_avoid=None, Q_search=5, epsilon=2, Vl=5, Ul=2, dt=0.2, ontime=False):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def myatan(a, b):
        if a == 0 and b == 0:
            return None
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
        K_avoid = [0.5, 20]
    [Kg, Kr] = K_avoid
    P_start = np.array(P_start)        # 初始位置
    pos_record = [P_start]             # 记录位置
    # e_record = []                    # 记录误差
    V_start = np.array(V_start).T      # 初始速度
    I2 = np.matrix([[1, 0],            # 用于计算轨迹跟踪控制器参数
                    [0, 1]])
    # 地图尺寸
    size_x = mymap.shape[0]
    size_y = mymap.shape[1]
    # 设置绘图参数
    plt.axis('equal')
    plt.xlim(0, size_x)
    plt.ylim(0, size_y)
    # plt.ion()  # 开启实时绘图
    # 绘制地图（障碍物和航路点）
    plt.imshow(mymap.T)
    path_plot = np.array(Path).T
    plt.plot(path_plot[0], path_plot[1], 'o')
    # plt.savefig('tmp.png')
    # 计算周边障碍物搜素位置
    direction_search = np.array([-2, -1, 0, 1, 2]) * math.pi / 4
    # direction_search = []
    # for t in range(- Q_search, Q_search + 1):
    #     for s in range(-Q_search, Q_search + 1):
    #         if (t != 0 or s != 0) and distance([s, t], [0, 0]) <= Q_search:
    #             direction_search.append(np.array([t, s]))
    # 开始飞行
    pos_num = 0         # 已经记录的位置的总数
    P_curr = P_start    # 当前位置
    V_curr = V_start
    Wb = P_curr
    ob_flag = False     # 用于判断局部极小值
    for path_num in range(len(Path)):
        Wa = Wb                                                             # 出发航路点
        Wb = np.array(Path[path_num])                                       # 目标航路点
        Wa_sub_Wb_matrix = np.matrix((Wa - Wb)).T                           # 计算差矩阵
        A = I2 - Wa_sub_Wb_matrix.dot(Wa_sub_Wb_matrix.T) / (distance(Wa, Wb) ** 2)
        Pt_matrix = np.matrix(P_curr - Wb).T                                # 用于计算的中间矩阵，目标点到当前点的向量
        # e = np.linalg.norm(A.dot(Pt_matrix))                              # 误差，当前到目标轨迹的垂直距离
        # e_record.append(e)
        d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))              # 沿着轨迹方向，当前点距离目标点的距离
        print('跟踪航路点:', path_num, 'Start:', Wa, ',Aim:', Wb)
        # 开始跟踪当前航路Wa-Wb，
        # 如Wb为最后一个航点，则P距离Wb小于epsilon时终止
        # 否则，在delta距离内发现航点Wb时终止
        while (path_num != len(Path) - 2 or distance(Wb, P_curr) > epsilon) and \
                (path_num == len(Path) - 2 or d_to_Wb > delta):
            Pt_matrix = np.matrix(P_curr - Wb).T
            Frep = np.array([0, 0])                                 # 斥力
            # # 搜索障碍物
            # for dir in direction_search:
            #     P_search = P_curr + dir                             # 被搜索点的位置
            #     if not (0 <= P_search[0] < size_x and                # 超出地图范围，地图内障碍，均视作障碍物
            #             0 <= P_search[1] < size_y and
            #             mymap[int(P_search[0])][int(P_search[1])] == 1):
            #         d_search = distance(P_curr, P_search)           # 被搜索点与当前点的距离
            #         Frep = Frep + \
            #                Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
            #                (P_curr - P_search) * (distance(P_search, Wb) ** 2)
            angle_curr = myatan(V_curr[0], V_curr[1])
            for dir in direction_search:
                angle_search = angle_curr + dir
                for dis_search in range(Q_search):
                    P_search = [int(P_curr[0] + dis_search * math.sin(angle_search)),
                                int(P_curr[1] + dis_search * math.cos(angle_search))]
                    if not (0 <= P_search[0] < size_x and  # 超出地图范围，地图内障碍，均视作障碍物
                            0 <= P_search[1] < size_y and
                            mymap[int(P_search[0])][int(P_search[1])] == 1):
                        d_search = distance(P_curr, P_search)  # 被搜索点与当前点的距离
                        Frep = Frep + \
                               Kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                               (P_curr - P_search) * (distance(P_search, Wb) ** 2)
                        break
            # 未检测到障碍物，执行航路点跟踪
            if Frep[0] == 0 and Frep[1] == 0:
                U1 = np.array((K0 * Pt_matrix + K1 * A.dot(Pt_matrix)).T)[0]
                if np.linalg.norm(U1, ord=np.inf) > Ul:
                    U1 = U1 * Ul / np.linalg.norm(U1, ord=np.inf)
                U = -(U1 + V_curr) / K2                             # 计算控制器输出并转换为array
            # 检测到障碍物，执行避障
            else:
                Fatt = -Kg * (P_curr - Wb)                          # 计算引力
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
                    Kd = 15 * math.exp(-(distance(P_curr, Wb) - 3) ** 2 / 2) + 1
                    Ke = 3
                    Fatt = Kv * Kd * Ke * Fatt                      # 改进引力
                U = Fatt + Frep                                     # 计算合力
                if np.linalg.norm(U, ord=np.inf) > Ul:              # 控制器输出限幅
                    U = Ul * U / np.linalg.norm(U, ord=np.inf)
            # 执行
            V_curr = V_curr + U * dt                                # 计算速度
            if np.linalg.norm(V_curr) > Vl:                         # 速度限幅
                V_curr = Vl * V_curr / np.linalg.norm(V_curr)
            P_curr = P_curr + V_curr * dt                           # 计算位置
            pos_num += 1
            # 记录和绘图
            print(P_curr, V_curr, distance(P_curr, Wb))
            # e = np.linalg.norm(A.dot(Pt_matrix))
            # e_record.append(e)
            d_to_Wb = np.linalg.norm(Pt_matrix - A.dot(Pt_matrix))
            pos_record.append(P_curr)

            if ontime:
                plt.plot([float(pos_record[pos_num - 1][0]), float(pos_record[pos_num][0])],
                         [float(pos_record[pos_num - 1][1]), float(pos_record[pos_num][1])], 'r')
                plt.pause(0.001)
                plt.ioff()

    pos_record = np.array(pos_record).T
    plt.plot(pos_record[0], pos_record[1], '--')
    plt.show()


if __name__ == "__main__":
    # 读取地图图像并二值化
    img = cv2.imread('1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=1)
    dst = cv2.erode(dst, None, iterations=4) / 255

    K_track = [0.5, 0, 2]
    K_avoid = [0.5, 20]
    K = [1, 0, 2, 20]
    Path = [[65, 65]]
    move_by_path_and_avoid([0, 0], [1, 1], Path, dst, Q_search=15, K_avoid=K_avoid, ontime=False)
