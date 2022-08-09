"""
python_avoid_APF.py
人工势场法避障的python实现
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import cv2


def avoid_APF(P_start, V_start, P_aim, mymap, Kg=0.5, kr=20,
              Q_search=20, epsilon=2, Vl=2, Ul=2, dt=0.2, draw_ontime=False):
    """
    :param P_start: 初始位置
    :param V_start: 初始速度
    :param P_aim: 目标点
    :param mymap: 储存有障碍物信息的0-1二值化地图，0表示障碍物
    :param Kg: 避障控制器参数（引力）
    :param kr: 避障控制器参数（斥力）
    :param Q_search: 搜索障碍物距离阈值
    :param epsilon: 误差上限
    :param Vl: 速率上限
    :param Ul: 控制器输出上限
    :param dt: 迭代时间
    :param draw_ontime:
    :return: 无
    """

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
    P_start = np.array(P_start)        # 初始位置
    V_start = np.array(V_start).T      # 初始速度
    pos_record = [P_start]  # 记录位置
    # 地图尺寸
    size_x = mymap.shape[0]
    size_y = mymap.shape[1]
    # 设置绘图参数
    plt.axis('equal')
    plt.xlim(0, size_x)
    plt.ylim(0, size_y)
    # 绘制地图（障碍物和航路点）
    plt.imshow(mymap.T)
    plt.plot([P_start[0], P_aim[0]], [P_start[1], P_aim[1]], 'o')
    # 计算周边障碍物搜素范围（在-90到90度范围）
    direction_search = np.array([-2, -1, 0, 1, 2]) * math.pi / 4
    # 开始飞行
    pos_num = 0         # 已经记录的位置的总数
    P_curr = P_start    # 当前位置
    V_curr = V_start    # 当前速度
    ob_flag = False     # 是否处于局部极小值
    while distance(P_curr, P_aim) > epsilon:
        angle_curr = myatan(V_curr[0], V_curr[1])               # 当前无人机朝向
        Frep = np.array([0, 0])                                 # 斥力初始化为0
        for dir in direction_search:                            # 搜索周围障碍物
            angle_search = angle_curr + dir
            for dis_search in range(Q_search):
                P_search = [int(P_curr[0] + dis_search * math.sin(angle_search)),
                            int(P_curr[1] + dis_search * math.cos(angle_search))]
                if not (0 <= P_search[0] < size_x and           # 超出地图范围和地图内障碍，均视作障碍物
                        0 <= P_search[1] < size_y and
                        mymap[int(P_search[0])][int(P_search[1])] == 1):
                    d_search = distance(P_curr, P_search)       # 被搜索点与当前点的距离
                    Frep = Frep + \
                           kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                           (P_curr - P_search) * (distance(P_search, P_aim) ** 2)
                    break
        Fatt = -Kg * (P_curr - P_aim)                           # 计算引力
        if pos_num >= 1:                                        # 从第二个时刻开始，判断是否陷入局部极小值
            p0 = pos_record[pos_num - 1]                        # 取上两个时刻物体相对终点的位移
            p1 = pos_record[pos_num - 2]
            Vra = (distance(p0, P_aim) - distance(p1, P_aim)) / dt
            if abs(Vra) < 0.6 * Vl:                              # 陷入局部极小值
                if not ob_flag:                                  # 之前不是局部极小状态时，根据当前状态计算斥力偏向角theta
                    angle_g = myatan(Fatt[0], Fatt[1])
                    angle_r = myatan(Frep[0], Frep[1])
                    if angle_r is None or angle_g is None:
                        print('111')
                    if isClockwise(angle_g, angle_r):
                        theta = 15 * math.pi / 180
                    else:
                        theta = -15 * math.pi / 180
                    ob_flag = True
                # 之前为局部极小，则继续使用上一时刻的斥力偏向角theta
                Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                        math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]
            else:                                               # 离开局部极小值
                ob_flag = False
            l = Vl
            Kv = 3 * l / (2 * l + abs(Vra))
            Kd = 15 * math.exp(-(distance(P_curr, P_aim) - 3) ** 2 / 2) + 1
            Ke = 3
            Fatt = Kv * Kd * Ke * Fatt                          # 改进引力
        U = Fatt + Frep                                         # 计算合力
        if np.linalg.norm(U, ord=np.inf) > Ul:                  # 控制器输出限幅
            U = Ul * U / np.linalg.norm(U, ord=np.inf)
        V_curr = V_curr + U * dt                                # 计算速度
        if np.linalg.norm(V_curr) > Vl:                         # 速度限幅
            V_curr = Vl * V_curr / np.linalg.norm(V_curr)
        P_curr = P_curr + V_curr * dt                           # 计算位置
        print(P_curr, V_curr, distance(P_curr, P_aim))
        pos_record.append(P_curr)
        pos_num += 1
        if draw_ontime:
            plt.plot([float(pos_record[pos_num - 1][0]), float(pos_record[pos_num][0])],
                     [float(pos_record[pos_num - 1][1]), float(pos_record[pos_num][1])], 'r')
            plt.pause(0.0001)
            plt.ioff()
    pos_record = np.array(pos_record).T
    plt.plot(pos_record[0], pos_record[1], '--')
    plt.show()


if __name__ == "__main__":
    # 读取地图图像并二值化
    img = cv2.imread('map_avoid/6.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=1)
    dst = cv2.erode(dst, None, iterations=4) / 255
    avoid_APF(P_start=[15, 15], V_start=[0, 2], P_aim=[400, 400], Q_search=15, mymap=dst)

