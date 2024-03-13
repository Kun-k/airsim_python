import numpy as np
import airsim

# 输入：期望的位置、速度、加速度
# 输出：力和力矩
def Geometric_control(p_des, v_des, a_des, yaw_des, p_now, v_now, R_now, omega_now, m):
    g = 9.81        # 重力加速度
    e3 = np.array([[0], [0], [1]])
    # 4个控制系数
    kp = 2
    kv = 2
    kR = 0.4
    komega = 0.08
    # 位置和速度误差
    e_p = p_now - p_des
    e_v = v_now - v_des
    # 求合力 f
    acc = -kp*e_p -kv*e_v - m*g*e3 + m*a_des
    f = -np.dot((acc).T, np.dot(R_now, e3))
    # 求期望的旋转矩阵 R_des
    proj_xb = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
    z_b = - acc.reshape(3) / np.linalg.norm(acc)
    y_b = np.cross(z_b, proj_xb)
    y_b = y_b / np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / np.linalg.norm(x_b)
    R_des = np.hstack([np.hstack([x_b.reshape([3, 1]), y_b.reshape([3, 1])]), z_b.reshape([3, 1])])
    # 求合力矩 M
    e_R_tem = np.dot(R_des.T, R_now) - np.dot(R_now.T, R_des)/2
    e_R = np.array([[e_R_tem[2, 1]], [e_R_tem[0, 2]], [e_R_tem[1, 0]]])
    M = -kR * e_R - komega * omega_now
    return f[0, 0], M

# 力和力矩到电机控制的转换
def fM2u(f, M):
    mat = np.array([[4.179446268,       4.179446268,        4.179446268,        4.179446268],
                    [-0.6723341164784,  0.6723341164784,    0.6723341164784,    -0.6723341164784],
                    [0.6723341164784,   -0.6723341164784,   0.6723341164784,    -0.6723341164784],
                    [0.055562,          0.055562,           -0.055562,          -0.055562]])
    fM = np.vstack([f, M])
    u = np.dot(np.linalg.inv(mat), fM)
    u1 = u[0, 0]
    u2 = u[1, 0]
    u3 = u[2, 0]
    u4 = u[3, 0]
    return u1, u2, u3, u4

# 欧拉角到旋转矩阵的转换
def angle2R(roll, pitch, yaw):
    sphi = np.sin(roll)
    cphi = np.cos(roll)
    stheta = np.sin(pitch)
    ctheta = np.cos(pitch)
    spsi = np.sin(yaw)
    cpsi = np.cos(yaw)
    R = np.array([[ctheta*cpsi, sphi*stheta*cpsi-cphi*spsi, cphi*stheta*cpsi+sphi*spsi],
                  [ctheta*spsi, sphi*stheta*spsi+cphi*cpsi, cphi*stheta*spsi-sphi*cpsi],
                  [-stheta,     sphi*ctheta,                cphi*ctheta]])
    return R

def move_by_geometricControl(client, p_des, v_des, a_des, yaw, pos_now, vel_now, dt):
    """
    几何控制无人机飞行
    client: AirSim client
    p_des: desired position
    v_des: desired velocity
    a_des: desired acceleration
    yaw: yaw angle
    pos_now: current position
    vel_now: current velocity
    dt: duration
    """
    m = 1
    state = client.getMultirotorState()
    pos_now = np.array([[state.kinematics_estimated.position.x_val],
                        [state.kinematics_estimated.position.y_val],
                        [state.kinematics_estimated.position.z_val]])
    vel_now = np.array([[state.kinematics_estimated.linear_velocity.x_val],
                        [state.kinematics_estimated.linear_velocity.y_val],
                        [state.kinematics_estimated.linear_velocity.z_val]])
    omega_now = np.array([[state.kinematics_estimated.angular_velocity.x_val],
                          [state.kinematics_estimated.angular_velocity.y_val],
                          [state.kinematics_estimated.angular_velocity.z_val]])
    pitch_now, roll_now, yaw_now = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
    R_now = angle2R(roll_now, pitch_now, yaw_now)

    f, M = Geometric_control(p_des, v_des, a_des, yaw, pos_now, vel_now, R_now, omega_now, m)
    u1, u2, u3, u4 = fM2u(f, M)
    client.moveByMotorPWMsAsync(u1, u2, u3, u4, dt)