import yaml
import os
import cv2
import numpy as np

curPath = os.path.dirname(os.path.realpath(__file__))
# yaml_stereo = os.path.join(curPath, "j7-00006_20191018_front_left_right_camera.yml")

def load_stereo_conf(yaml_file):
    f = open(yaml_file)  
    d = yaml.load(f) 
    size = (d['width'], d['height'])
    P1 = np.asarray(d['P1']['data']).reshape((d['P1']['rows'], d['P1']['cols']))
    R1 = np.asarray(d['R1']['data']).reshape((d['R1']['rows'], d['R1']['cols']))
    M1 = np.asarray(d['M1']['data']).reshape((d['M1']['rows'], d['M1']['cols']))
    D1 = np.asarray(d['D1']['data']).reshape((d['D1']['rows'], d['D1']['cols']))
    P2 = np.asarray(d['P2']['data']).reshape((d['P2']['rows'], d['P2']['cols']))
    R2 = np.asarray(d['R2']['data']).reshape((d['R2']['rows'], d['R2']['cols']))
    M2 = np.asarray(d['M2']['data']).reshape((d['M2']['rows'], d['M2']['cols']))
    D2 = np.asarray(d['D2']['data']).reshape((d['D2']['rows'], d['D2']['cols']))
    Q = np.asarray(d['Q']['data']).reshape((d['Q']['rows'], d['Q']['cols']))
    Tr_cam_to_imu = np.asarray(d['Tr_cam_to_imu']['data']).reshape((d['Tr_cam_to_imu']['rows'], d['Tr_cam_to_imu']['cols']))
    return size, P1, R1, M1, D1, P2, R2, M2, D2, Q, Tr_cam_to_imu

def load_lidar_conf(yaml_file):
    f = open(yaml_file)
    d = yaml.load(f)
    Tr_lidar_to_imu = np.asarray(d['Tr_lidar_to_imu']['data']).reshape((d['Tr_lidar_to_imu']['rows'], d['Tr_lidar_to_imu']['cols']))
    return Tr_lidar_to_imu


def search_calib_file(car_name, date):
    yaml_stereo = ""
    yaml_lidar = ""
    dirs = os.listdir(curPath)
    latest = [0, 0]
    for file in dirs:
        if file.find('.yml') != -1:
            calib_name = file.split('_')[0]
            calib_date = int(file.split('_')[1])
            calib_sensor = file.split('_')[-1].split('.')[0]
            if calib_name.find(car_name) != -1 and calib_date < date:
                if calib_sensor.find('camera') != -1:
                    if latest[0] < calib_date:
                        latest[0] = calib_date
                        yaml_stereo = os.path.join(curPath, file)
                elif calib_sensor.find('lidar') != -1:
                    if latest[1] < calib_date:
                        latest[1] = calib_date
                        yaml_lidar = os.path.join(curPath, file)
    return yaml_stereo, yaml_lidar

def unwarp(image_left, image_right, car_name, date):
    yaml_stereo, yaml_lidar = search_calib_file(car_name, date)
    size, P1, R1, M1, D1, P2, R2, M2, D2, Q, Tr_cam_to_imu = load_stereo_conf(yaml_stereo)
    Tr_lidar_to_imu = load_lidar_conf(yaml_lidar)
    h, w, c = image_left.shape
    origin_size = (w, h)

    if size != origin_size:
        image_left = cv2.resize(image_left, size, interpolation=cv2.INTER_NEAREST)
        image_right = cv2.resize(image_right, size, interpolation=cv2.INTER_NEAREST)
    map1_left, map2_left = cv2.initUndistortRectifyMap(M1, D1, R1, P1, size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(M2, D2, R2, P2, size, cv2.CV_16SC2)
    image_left_rectified = cv2.remap(image_left, map1_left, map2_left, cv2.INTER_LINEAR)
    image_right_rectified = cv2.remap(image_right,map1_right, map2_right, cv2.INTER_LINEAR)
    return image_left_rectified, image_right_rectified, P1, P2, Tr_cam_to_imu, Tr_lidar_to_imu

