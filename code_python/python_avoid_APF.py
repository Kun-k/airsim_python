"""
python_avoid_APF.py
人工势场法避障的python实现
"""


import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import threading
import time


class avoid():
    def __init__(self):
        self.p_curr = np.array([0, 0])
        self.v_curr = np.array([0, 0])
        self.p_aim = np.array([0, 0])
        self.mymap = None
        self.size_x = 0
        self.size_y = 0
        self.obs = None

    def distance(self, A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def myatan(self, a, b):
        if a == 0 and b == 0:
            return 0
        if a == 0:
            return math.pi / 2 if b > 0 else math.pi * 3 / 2
        if b == 0:
            return 0 if a > 0 else -math.pi
        if b > 0:
            return math.atan(b / a) if a > 0 else (math.atan(b / a) + math.pi)
        return math.atan(b / a + 2 * math.pi) if a > 0 else math.atan(b / a) + math.pi

    def isclockwise(self, a, b):
        da = b - a
        if 0 < da < math.pi or -math.pi * 2 < da < -math.pi:
            return False
        return True

    def setmap(self, mapsize):
        # 地图尺寸
        self.size_x = mapsize[0]
        self.size_y = mapsize[1]
        self.mymap = np.ones([self.size_x, self.size_y])
        # 设置绘图参数
        plt.axis('equal')
        plt.xlim(0, self.size_x)
        plt.ylim(0, self.size_y)

    def setobstacle(self, obposition):
        tmp_obs = []
        tmp_mymap = np.ones([self.size_x, self.size_y])
        for i in range(len(obposition)):
            [x, y] = obposition[i].tolist()
            tmp_mymap[x][y] = 0
            tmp_obs.append([x, y])
        self.obs = tmp_obs
        self.mymap = tmp_mymap

    def avoid_apf(self, P_start, V_start, P_aim, Kg=0.5, kr=20,
                  Q_search=20, epsilon=2, Vl=2, Ul=2, dt=0.2, ontime=False):
        # 读取初始状态
        self.p_curr = np.array(P_start)  # 初始位置
        pos_record = [self.p_curr]  # 记录位置
        self.v_curr = np.array(V_start).T  # 初始速度
        # 计算周边障碍物搜素位置
        direction_search = np.array([-2, -1, 0, 1, 2]) * math.pi / 4
        # 开始飞行
        pos_num = 0  # 已经记录的位置的总数
        ob_flag = False  # 用于判断局部极小值
        while self.distance(self.p_curr, P_aim) > epsilon:
            Frep = np.array([0, 0])  # 斥力
            angle_curr = self.myatan(self.v_curr[0], self.v_curr[1])
            for dir in direction_search:
                angle_search = angle_curr + dir
                for dis_search in range(Q_search):
                    P_search = [int(self.p_curr[0] + dis_search * math.sin(angle_search)),
                                int(self.p_curr[1] + dis_search * math.cos(angle_search))]
                    if not (0 <= P_search[0] < self.size_x and  # 超出地图范围，地图内障碍，均视作障碍物
                            0 <= P_search[1] < self.size_y and
                            self.mymap[int(P_search[0])][int(P_search[1])] == 1):
                        d_search = self.distance(self.p_curr, P_search)  # 被搜索点与当前点的距离
                        Frep = Frep + \
                               kr * (1 / d_search - 1 / Q_search) / (d_search ** 3) * \
                               (self.p_curr - P_search) * (self.distance(P_search, P_aim) ** 2)
                        break
            Fatt = -Kg * (self.p_curr - P_aim)  # 计算引力
            if pos_num >= 1:
                # 计算上两个时刻物体相对终点的位移，以判断是否陷入局部极小值
                p0 = pos_record[pos_num - 1]
                p1 = pos_record[pos_num - 2]
                Vra = (self.distance(p0, P_aim) - self.distance(p1, P_aim)) / dt
                if abs(Vra) < 0.6 * Vl and Frep[0] != 0 and Frep[1] != 0:  # 陷入局部极小值
                    if ob_flag == False:
                        # 之前不是局部极小状态时，根据当前位置计算斥力偏向角theta
                        angle_g = self.myatan(Fatt[0], Fatt[1])
                        angle_r = self.myatan(Frep[0], Frep[1])
                        # if angle_r == None or angle_g == None:
                        #     print('111')
                        if self.isclockwise(angle_g, angle_r):
                            theta = 15 * math.pi / 180
                        else:
                            theta = -15 * math.pi / 180
                        ob_flag = True
                    # 之前为局部极小，则继续使用上一时刻的斥力偏向角theta
                    Frep = [math.cos(theta) * Frep[0] - math.sin(theta) * Frep[1],
                            math.sin(theta) * Frep[0] + math.cos(theta) * Frep[1]]
                else:  # 离开局部极小值
                    ob_flag = False
                l = Vl
                Kv = 3 * l / (2 * l + abs(Vra))
                Kd = 15 * math.exp(-(self.distance(self.p_curr, P_aim) - 3) ** 2 / 2) + 1
                Ke = 3
                Fatt = Kv * Kd * Ke * Fatt  # 改进引力
            U = Fatt + Frep  # 计算合力
            if np.linalg.norm(U, ord=np.inf) > Ul:  # 控制器输出限幅
                U = Ul * U / np.linalg.norm(U, ord=np.inf)
            self.v_curr = self.v_curr + U * dt  # 计算速度
            if np.linalg.norm(self.v_curr) > Vl:  # 速度限幅
                self.v_curr = Vl * self.v_curr / np.linalg.norm(self.v_curr)
            self.p_curr = self.p_curr + self.v_curr * dt  # 计算位置
            print(self.p_curr, self.v_curr, self.distance(self.p_curr, P_aim))
            pos_record.append(self.p_curr)
            pos_num += 1
            if ontime:
                plt.clf()
                plt.axis('equal')
                plt.xlim(0, self.size_x)
                plt.ylim(0, self.size_y)
                plt.plot([P_start[0], P_aim[0]], [P_start[1], P_aim[1]], 'x')
                for i in range(len(self.obs)):
                    [x, y] = self.obs[i]
                    self.mymap[x][y] = 0
                    plt.plot(x, y, 'o', c='black')
                pos_plot = np.array(pos_record).T
                plt.plot(pos_plot[0], pos_plot[1], '--')
                plt.pause(0.001)
                plt.ioff()
        plt.clf()
        plt.axis('equal')
        plt.xlim(0, self.size_x)
        plt.ylim(0, self.size_y)
        plt.plot([P_start[0], P_aim[0]], [P_start[1], P_aim[1]], 'x')
        for i in range(len(self.obs)):
            [x, y] = self.obs[i]
            self.mymap[x][y] = 0
            plt.plot(x, y, 'o', c='black')
        pos_plot = np.array(pos_record).T
        plt.plot(pos_plot[0], pos_plot[1], '--')
        plt.show()

    def obs_dynamic(self):
        dt = 0.1
        tmp = True
        obs1 = []
        obs2 = []
        obs3 = []
        for x in range(30, 40):
            y = 100 - x
            obs1.append([x, y])
        for x in range(40, 50):
            y = 50 - x
            obs2.append([x, y])
        for x in range(90, 100):
            y = 150 - x
            obs3.append([x, y])
        obs1 = np.array(obs1)
        obs2 = np.array(obs2)
        obs3 = np.array(obs3)
        for i in range(1000):
            if obs1[0][0] == 30:
                tmp = True
            elif obs1[0][0] == 59:
                tmp = False
            if tmp:
                obs1 = obs1 + [1, -1]
                obs2 = obs2 + [-1, 1]
                obs3 = obs3 + [-1, 1]
            else:
                obs1 = obs1 + [-1, 1]
                obs2 = obs2 + [1, -1]
                obs3 = obs3 + [1, -1]
            obs = np.append(obs1, obs2, axis=0)
            obs = np.append(obs, obs3, axis=0)
            self.setobstacle(obs)
            time.sleep(dt)


'''
二维空间内避障，输入地图信息为二值化的图像
P_start     初始位置
V_start     初始速度
P_aim       目标点
mymap       储存有障碍物信息的地图
Kg,Kr       避障控制器参数（引力，斥力）
Q_search    搜索障碍物距离阈值
epsilon     误差上限
Vl          速率上限
Ul          控制器输出上限
dt          迭代时间
draw_ontime 是否开启实时绘图
'''
def avoid_APF(P_start, V_start, P_aim, mymap, Kg=0.5, kr=20,
              Q_search=20, epsilon=2, Vl=2, Ul=2, dt=0.2, draw_ontime=False):
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
    img = cv2.imread('4.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=1)
    dst = cv2.erode(dst, None, iterations=4) / 255
    avoid_APF(P_start=[15, 15], V_start=[0, 2], P_aim=[95, 95], Q_search=15, mymap=dst)

    # mapsize = [100, 100]
    # # obs = []
    # # for x in range(46, 56):
    # #     y = 100 - x
    # #     obs.append([x, y])
    # # obs = np.array(obs)
    # avo = avoid()
    # avo.setmap(mapsize)
    # # avo.setobstacle(obs)
    #
    # thread_obs = threading.Thread(target=avo.obs_dynamic)
    # thread_obs.start()
    # avo.avoid_apf(P_start=[15, 15], V_start=[0, 2], P_aim=[95, 95], ontime=True, Q_search=15)
