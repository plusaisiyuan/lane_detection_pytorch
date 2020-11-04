import yaml
import os
import cv2
import numpy as np

curPath = os.path.dirname(os.path.realpath(__file__))

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

class loadCalibration(object):
    def __init__(self, bag_name):
        super(loadCalibration, self).__init__()
        def searchCalibFile(car_name, date):
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

        def loadStereoConf(yaml_file):
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
            Tr_cam_to_imu = np.asarray(d['Tr_cam_to_imu']['data']).reshape(
                (d['Tr_cam_to_imu']['rows'], d['Tr_cam_to_imu']['cols']))
            return size, P1, R1, M1, D1, P2, R2, M2, D2, Q, Tr_cam_to_imu

        car_name = bag_name.split('_')[1]
        date = int(bag_name.split('_')[0].split('T')[0])
        self.yaml_stereo, self.yaml_lidar = searchCalibFile(car_name, date)
        if self.yaml_stereo == "":
            return false
        self.size, self.P1, self.R1, self.M1, self.D1, self.P2, self.R2, self.M2, self.D2, self.Q, self.Tr_cam_to_imu = \
            loadStereoConf(self.yaml_stereo)

    def loadLidarConf(self):
        f = open(self.yaml_lidar)
        d = yaml.load(f)
        Tr_lidar_to_imu = np.asarray(d['Tr_lidar_to_imu']['data']).reshape(
            (d['Tr_lidar_to_imu']['rows'], d['Tr_lidar_to_imu']['cols']))
        return Tr_lidar_to_imu

    def getCalibParams(self):
        return self.P1, self.P2, self.Tr_cam_to_imu

    def unwarp(self, image, use_left):
        h, w, c = image.shape
        origin_size = (w, h)
        if self.size != origin_size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
        if use_left:
            M, D, R, P = self.M1, self.D1, self.R1, self.P1
        else:
            M, D, R, P = self.M2, self.D2, self.R2, self.P2
        map1, map2 = cv2.initUndistortRectifyMap(M, D, R, P, self.size, cv2.CV_16SC2)
        image_rectified = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return image_rectified