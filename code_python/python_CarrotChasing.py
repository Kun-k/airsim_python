'''
CarrotChasing.py
CarrotChasing轨迹跟踪算法的python实现
'''


import math
import matplotlib.pyplot as plt
import numpy as np


def move_to_point(P, Va, psi, Wa, Wb, delta=1, K=0.5, K2=0, epsilon=0.02, dt=0.01):
    print(Wa, Wb, P)
    [Px, Py] = P
    theta = math.atan((Wb[1] - Wa[1]) / (Wb[0] - Wa[0]))
    if Wb[0] < Wa[0]:
        theta += math.pi
    pos_record = [[Px], [Py]]
    e = epsilon
    while abs(e) > epsilon:
        beta = theta - math.atan((Py - Wa[1]) / (Px - Wa[0]))
        if Px < Wa[0]:
            beta -= math.pi
        Ru = math.sqrt((Px - Wa[0]) ** 2 + (Py - Wa[1]) ** 2)
        R = Ru * math.cos(beta)
        e = Ru * math.sin(beta)
        print(Px, Py, e)
        xt = Wa[0] + (R + delta) * math.cos(theta)
        yt = Wa[1] + (R + delta) * math.sin(theta)
        psi_d = math.atan((yt - Py) / (xt - Px))
        if xt < Px:
            psi_d += math.pi
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
        Px = Px + Vx * dt
        Py = Py + Vy * dt
        pos_record[0].append(Px)
        pos_record[1].append(Py)
    Vx = Va * math.cos(theta)
    Vy = Va * math.sin(theta)
    epsilon = math.sqrt(e ** 2 + (Va * dt / 2) ** 2)
    while True:
        e = math.sqrt((Px - Wb[0]) ** 2 + (Py - Wb[1]) ** 2)
        if e < epsilon:
            break
        Px = Px + Vx * dt
        Py = Py + Vy * dt
        pos_record[0].append(Px)
        pos_record[1].append(Py)
        print(Px, Py, e)
    plt.plot([Wa[0], Wb[0]], [Wa[1], Wb[1]])
    plt.plot(pos_record[0], pos_record[1])
    plt.axis('equal')
    plt.grid()
    plt.show()


def move_by_path(P, Va, psi, Path, delta=1, K=0.5, K2=0, dt=0.01):
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

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    [Px, Py] = P
    pos_record = [[Px, Py]]
    aim_record = []
    for i in range(1, len(Path)):
        Wa = Path[i - 1]
        Wb = Path[i]
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
            if (xt - Wb[0]) * (Wb[0] - Wa[0]) > 0 \
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
            Px = Px + Vx * dt
            Py = Py + Vy * dt
            pos_record.append([Px, Py])
            aim_record.append([xt, yt])
    # 点采样和绘图
    pos_plot = []
    aim_plot = []
    num = 25
    gap = int(len(pos_record) / num)
    for i in range(num):
        pos_plot.append(pos_record[i * gap])
        aim_plot.append(aim_record[i * gap])
    pos_plot = np.array(pos_plot).T
    aim_plot = np.array(aim_plot).T
    path = np.array(Path).T
    plt.plot(path[0], path[1])
    plt.plot(pos_plot[0], pos_plot[1], '--')
    plt.plot(aim_plot[0], aim_plot[1], '*')
    plt.legend(['目标轨迹', '实际轨迹', '目标航点'])
    plt.axis('equal')
    plt.grid()
    plt.show()


def move_to_point_3d(P, V, Wa, Wb, K0=2, K1=2, K2=1, eposilon=0.02, dt=0.1, a0=2):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 +
                         (A[1] - B[1]) ** 2 +
                         (A[2] - B[2]) ** 2)

    P = np.matrix(P).T
    Wa = np.matrix(Wa).T
    Wb = np.matrix(Wb).T
    V = np.matrix(V).T
    I3 = np.matrix([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
    A = I3 - (Wa - Wb).dot((Wa - Wb).T) / (distance(Wa, Wb) ** 2)
    e = np.linalg.norm(A.dot(P - Wb))
    pos_record = [P]
    print('P:', P, 'V:', V, 'e:', e)
    while e >= eposilon:
        Pt = P - Wb
        U1 = K0 * Pt + K1 * A.dot(Pt)
        if np.linalg.norm(U1, ord=np.inf) > a0:
            U1 = U1 * a0 / np.linalg.norm(U1, ord=np.inf)
        U = -(U1 + V) / K2
        V = V + U * dt
        P = P + V * dt
        e = np.linalg.norm(A.dot(Pt))
        print('P:', P, 'V:', V, 'e:', e)
        pos_record.append(P)
    pos_record = np.array(pos_record).T[0]
    path = np.array([Wa, Wb]).T[0]
    ax = plt.subplot(projection='3d')
    ax.plot(path[0], path[1], path[2])
    ax.plot(pos_record[0], pos_record[1], pos_record[2], '--')
    # ax.scatter(pos_record[0],
    #            pos_record[1],
    #            pos_record[2], c='#00DDAA')
    plt.show()


def move_by_path_3d(P, V, Path, K0=2, K1=2, K2=1, dt=0.1, a0=2, delta=0.5):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 +
                         (A[1] - B[1]) ** 2 +
                         (A[2] - B[2]) ** 2)

    P = np.matrix(P).T
    pos_record = [P]
    e_record = []
    V = np.matrix(V).T
    I3 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    for i in range(len(Path) - 1):
        Wa = np.matrix(Path[i]).T
        Wb = np.matrix(Path[i + 1]).T
        A = I3 - (Wa - Wb).dot((Wa - Wb).T) / (distance(Wa, Wb) ** 2)
        Pt = P - Wb
        e = np.linalg.norm(A.dot(Pt))
        e_record.append(e)
        d = np.linalg.norm(Pt - A.dot(Pt))
        print('i,', 'Start:', Path[i], ',Aim:', Path[i + 1])
        print('\tP:', P, 'V:', V, 'e:', e)
        while d >= delta:
            Pt = P - Wb
            U1 = K0 * Pt + K1 * A.dot(Pt)
            if np.linalg.norm(U1, ord=np.inf) > a0:
                U1 = U1 * a0 / np.linalg.norm(U1, ord=np.inf)
            U = -(U1 + V) / K2
            V = V + U * dt
            P = P + V * dt
            e = np.linalg.norm(A.dot(Pt))
            e_record.append(e)
            d = np.linalg.norm(Pt - A.dot(Pt))
            print('\tP:', P, 'V:', V, 'e:', e)
            pos_record.append(P)
    pos_record = np.array(pos_record).T[0]
    path_plot = np.array(Path).T
    ax = plt.subplot(projection='3d')
    ax.plot(path_plot[0], path_plot[1], path_plot[2])
    ax.plot(pos_record[0], pos_record[1], pos_record[2], '--')
    plt.show()
    plt.plot(e_record)
    plt.show()


if __name__ == "__main__":
    # Path = [[0, 0], [10, 15], [15, 20], [20, 5]]
    # move_by_path([6, 3], 3, 0.5, Path, delta=0.1)
    # move_to_point([3, 3], 3, 0, [0, 0], [10, 15])
    # move_to_point_3d([1, 1, 1], [1, 1, 1], [0, 0, 0], [10, 15, 15], K0=1, K1=2, K2=0.8)
    Path = [[0, 0, 0], [10, 15, 15], [15, 20, 20], [20, 5, 5]]
    move_by_path_3d([1, 1, 1], [1, 1, 1], Path, K0=1, K1=2, K2=0.8)

