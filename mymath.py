import math


def distance(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def distance_3d(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 +
                     (A[1] - B[1]) ** 2 +
                     (A[2] - B[2]) ** 2)


# 计算向量AB的角度
def myatan(A, B):
    x1, y1, x2, y2 = A[0], A[1], B[0], B[1]
    if x1 != x2:
        if x1 > x2:
            return math.atan((y1 - y2) / (x1 - x2)) + math.pi
        else:
            return math.atan((y1 - y2) / (x1 - x2))
    if x1 == x2 and y1 == y2:
        return None  # 两点重合时返回None，可自行处理
    if x1 == x2 and y1 != y2:
        if y1 > y2:
            return -math.pi / 2
        else:
            return math.pi / 2


def isClockwise(a, b):
    da = b - a
    if 0 < da < math.pi or -math.pi * 2 < da < -math.pi:
        return False
    return True

