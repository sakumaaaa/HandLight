from sense_hat import SenseHat
import math
import time
from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
import sympy

# SenseHATを初期化する
sense = SenseHat()
# センサの初期化
sense.set_imu_config(True, True, True)  # ジャイロスコープ、加速度、磁気センサを有効にする

dt = 0.1 #時間間隔
x = np.zeros(3) #指定対象の状態
sigma = np.eye(3) #指定対象の分散
x_pred = np.zeros(3) #状態予測モデル
x_obs = np.zeros(3) #状態観測モデル
Q = np.eye(3) * 0.01 #状態遷移ノイズ
R = np.eye(3) * 0.1 #状態観測ノイズ

# 初期姿勢を推定する
init_accel = sense.get_accelerometer_raw()
init_mag = sense.get_compass_raw()
#init_roll = math.atan2(init_accel['y'], math.sqrt(init_accel['x'] ** 2 + init_accel['z'] ** 2)) * 180 / math.pi
#init_pitch = math.atan2(-init_accel['x'], math.sqrt(init_accel['y'] ** 2 + init_accel['z'] ** 2)) * 180 / math.pi

init_roll = math.atan2(init_accel['y'], init_accel['z'])
init_pitch = math.atan2(-init_accel['x'], math.sqrt(init_accel['y']**2 + init_accel['z']**2))
init_heading = math.atan2(init_mag['y'], init_mag['x'])
x[0] = init_roll
x[1] = init_pitch
x[2] = init_heading

def read_sensors():
    accel = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()
    compass = sense.get_compass_raw()
    return accel, gyro, compass

while True:
    accel, gyro, compass = read_sensors()
    x_pred = np.array[x[0]+(gyro['x']+gyro['y']*math.tan(x[1])*math.sin(x[0])+gyro['z']*math.tan(x[1])*math.cos(x[0]))*dt,
              x[1]+(gyro['y']*math.cos(x[0])-gyro['z']*math.sin(x[0]))*dt,
              x[2]+(gyro['y']*math.sin(x[0])/math.cos(x[1])+gyro['z']*math.cos(x[0])/math.sin(x[1]))]
    
    x_obs = np.array[math.atan2(-accel['y'], -accel['z']),
             math.atan2(accel['x'], math.sqrt(accel['y'] ** 2 + accel['z'] ** 2)),
             math.atan2(compass['x']*math.cos(x_obs[1])+compass['y']*math.sin(x_obs[1])*math.sin(x_obs[0])+compass['z']*math.sin(x_obs[1])*math.cos(x_obs[0]), 
                        compass['y']*math.cos(x_obs[0]-compass['z']*math.sin(x_obs[0])))]
    
    A = sympy.diff(x_pred, x) #予測状態のヤコビ行列
    C = np.eye(3) #観測状態のヤコビ行列

    #sigma_Q = np.dot(np.dot(A, sigma), A.T) + Q
    #sigma_R = np.dot(np.dot(C, sigma_Q), C.T) + R
    #K = np.dot(np.dot(sigma_Q, C.T), np.linalg.inv(sigma_R))
    #x = x_pred + np.dot(K, (x_obs - np.dot(C, x_obs)))
    #sigma = np.dot((np.eye(3) - np.dot(K, C)), sigma_Q)

    sigma_Q = A @ sigma @ A.T+ Q
    sigma_R = C @ sigma_Q @ C.T+ R
    K = sigma_Q @ C.T @ np.linalg.inv(sigma_R)
    x = x_pred + K @ (x_obs - (C @ x_obs))
    sigma = (np.eye(3) - (K @ C)) @ sigma_Q

    # 補正された姿勢角
    roll = x[0]
    pitch = x[1]
    heading = x[2]

    print("Roll: {:.2f}   Pitch: {:.2f}   Heading: {:.2f}".format(roll, pitch, heading))

    # 補正された姿勢角
    roll = math.degrees(x[0])
    pitch = math.degrees(x[1])
    heading = math.degrees(x[2])

    print("Roll: {:.2f} degree   Pitch: {:.2f} degree  Heading: {:.2f} degree".format(roll, pitch, heading))

    time.sleep(0.1)
