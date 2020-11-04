#!/usr/bin/env python

import sys
import rosbag
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2

import cv2
from cv_bridge import CvBridge
from tf.transformations import *
import os
from conf.sensor_config import loadCalibration, unwarp
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.optimize import curve_fit

bridge = CvBridge()

INPUT_MEAN = [123.675, 116.28, 103.53]
INPUT_STD = [58.395, 57.12, 57.375]
up_crop = 80
bottom_crop = 0
EG0_POINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),  (255, 255, 255),  (255, 255, 0)]

# def gen_bev_frame(frame_points, pose, Tr_imu_to_world, height, width):
#     h = height
#     w = width
#     c = 100
#     h_offset = 0
#     w_offset = 0
#     # bev_ids = np.zeros((h, w, 3)).astype(np.uint8)
#     # bev_count = np.zeros((h, w, 3)).astype(np.uint8)
#     # bev_image = np.zeros((h, w, 3)).astype(np.uint8)
#     # bev_image_rotated = np.zeros((h, w, 3)).astype(np.uint8)
#     rotation_vector = np.array([[1., 0., 0.]])
#     rotation = Tr_imu_to_world[0:3, 0:3]
#     rotation_vector = np.matmul(rotation, rotation_vector.T)
#     rotation_vector = rotation_vector.squeeze()
#
#     # bev_count = np.zeros((h, w, c, 2)).astype(np.int32)
#     bev_count = dict([(x, []) for x in range(h)])
#     bev_image = np.zeros((h, w, 3)).astype(np.uint8)
#     count = 0
#     for points4d in frame_points:
#         # count += 1
#         # if count % 2 == 0 or count < 20:
#         #     continue
#         for point4d in points4d:
#             x = point4d[0] - pose[0][0]
#             y = point4d[1] - pose[0][1]
#             z = point4d[2] - pose[0][2]
#             point_utm = np.array([x, y, z])
#             x_utm = int(-y * 10) + (h - h_offset) / 2 + h_offset
#             y_utm = int(x * 10) + (w - w_offset) / 2 + w_offset
#             if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
#                 bev_image[x_utm, y_utm, :] = POINT_COLORS[point4d[3]]
#             point_imu = np.matmul(np.array(np.matrix(rotation).I), point_utm.T)
#             point_imu = point_imu.squeeze()
#             x = int(point_imu[0] * 10) + h/2
#             y = int(point_imu[1] * 10) + w/2
#             z = int(point_imu[2] * 10) + c/2
#             if x >= 0 and x < h and y >= 0 and y < w:
#                 # bev_count[x, y, z, 0] += 1
#                 # bev_count[x, y, z, 1] = int(point4d[3]+1)
#                 flag = 0
#                 for i in range(len(bev_count[x])):
#                     info = bev_count[x][i]
#                     if [y, z, point4d[3]] == [info[0], info[1], info[2]]:
#                         flag = 1
#                         bev_count[x][i][3] += 1
#                 if flag == 0:
#                     bev_count[x].append([y, z, point4d[3], 1])
#     filter_points4d = []
#     for x in range(h):
#         ids_max = [0,0,0,0]
#         ids_point = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
#         for info in bev_count[x]:
#             if ids_max[info[2]] < info[3]:
#                 ids_max[info[2]] = info[3]
#                 ids_point[info[2]] = (info[0], info[1])
#         for i in range(len(ids_point)):
#             filter_points4d.append([pose[0][0], pose[0][1], pose[0][2], i])
#             if ids_point[i] != (-1,-1) and abs(ids_point[2][0]-ids_point[i][0]) >= abs(2-i)* 28 and abs(ids_point[2][0]-ids_point[i][0]) <= abs(2-i)* 48:
#                 filter_imu_point = np.array([[float(x-h/2)/10, float(ids_point[i][0]-w/2)/10, float(ids_point[i][1]-c/2)/10]])
#                 filter_utm_point = np.matmul(rotation, filter_imu_point.T)
#                 filter_utm_point = filter_utm_point.squeeze()
#                 x_utm = int(-filter_utm_point[1] * 10) + (h-h_offset) /2 + h_offset
#                 y_utm = int(filter_utm_point[0] * 10) + (w-w_offset) / 2 + w_offset
#                 # if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
#                     # bev_image[x_utm, y_utm, :] = POINT_COLORS[i]
#                 filter_points4d.append([filter_utm_point[0] + pose[0][0], filter_utm_point[1] + pose[0][1], filter_utm_point[2] + pose[0][2], i])
#         if ids_point[1] != (-1,-1) or ids_point[2] != (-1,-1):
#             if ids_point[1] == (-1,-1):
#                 ids_point[1] = (ids_point[2][0]+40, ids_point[2][1])
#             if ids_point[2] == (-1, -1):
#                 ids_point[2] = (ids_point[1][0] - 40, ids_point[1][1])
#             ego_lane_point =  np.array([[float(x-h/2)/10, float(ids_point[1][0]+ids_point[2][0]-w)/2/10, float(ids_point[1][1]+ids_point[2][1]-c)/2/10]])
#             ego_lane_utm_point = np.matmul(rotation, ego_lane_point.T)
#             ego_lane_utm_point = ego_lane_utm_point.squeeze()
#             x_utm = int(-ego_lane_utm_point[1] * 10) + (h - h_offset) / 2 + h_offset
#             y_utm = int(ego_lane_utm_point[0] * 10) + (w - w_offset) / 2 + w_offset
#             # if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
#                 # bev_image[x_utm, y_utm, :] = POINT_COLORS[5]
#             filter_points4d.append([ego_lane_utm_point[0] + pose[0][0], ego_lane_utm_point[1] + pose[0][1], ego_lane_utm_point[2] + pose[0][2], 5])
#                 # if x >= 0 and x < h and y >= 0 and y < w:
#                 #     bev_count[x, y, 0] += 1
#                 #     bev_count[x, y, 1] = int(point4d[3]+1)
#
#     # for points4d in frame_points:
#     #     for point4d in points4d:
#             # x = int((pose[0][1] - point4d[1]) * 10) + (h-h_offset) /2 + h_offset
#             # y = -int((pose[0][0] - point4d[0]) * 10) + (w-w_offset) / 2 + w_offset
#             # if x >= 0 and x < h and y >= 0 and y < w:
#             #     bev_count[x, y, 0] += 1
#             #     bev_count[x, y, 1] = int(point4d[3]+1)
#             #     bev_ids[x, y, 0] = int(point4d[3]+1)*40
#             #     bev_ids[x, y, 1] += 1
#
#
#     # P = math.degrees(math.tanh(rotation_vector[0]/rotation_vector[1]))
#     # M = cv2.getRotationMatrix2D(((w-w_offset) / 2 + w_offset, (h-h_offset) / 2 + h_offset), P, 1)
#     # bev_count_rotated = cv2.warpAffine(bev_count, M, (w, h))
#     # bev_ids_rotated = cv2.warpAffine(bev_ids, M, (w, h))
#     # bev_merge = np.concatenate([bev_ids, bev_ids_rotated], axis=1)
#     # cv2.imshow("bev_ids", bev_merge)
#     # cv2.waitKey(50)
#     #
#     # for x in range(h):
#     #     ids_max = [0,0,0,0]
#     #     ids_point = [-1, -1, -1, -1]
#     #     for y in range(w):
#     #         if ids_max[bev_count_rotated[x, y, 1]-1] < bev_count_rotated[x, y, 0]:
#     #             ids_max[bev_count_rotated[x, y, 1]-1] = bev_count_rotated[x, y, 0]
#     #             ids_point[bev_count_rotated[x, y, 1]-1] = y
#     #     for i in range(len(ids_point)):
#     #         if ids_point[i] != -1:
#     #             bev_image_rotated[x, ids_point[i], :] = POINT_COLORS[i]
#     # M = cv2.getRotationMatrix2D(((w - w_offset) / 2 + w_offset, (h - h_offset) / 2 + h_offset), -P, 1)
#     # bev_image = cv2.warpAffine(bev_image_rotated, M, (w, h))
#     bev_image = cv2.arrowedLine(bev_image, ((w - w_offset) / 2 + w_offset, (h - h_offset) / 2 + h_offset), (
#     int(rotation_vector[0] * 20) + (w - w_offset) / 2 + w_offset,
#     -int(rotation_vector[1] * 20) + (h - h_offset) / 2 + h_offset), (255, 0, 255), 2, 8, 0, 0.3)
#     return bev_image, filter_points4d

# def imageToGround(image_points, P, Tr_cam_to_imu, imu_height=0):
#     imu_p0 = np.array([0, 0, -imu_height, 1])
#     imu_p1 = np.array([0, 0, -imu_height + 1, 1])
#     Tr_imu_to_cam = np.mat(Tr_cam_to_imu).I
#
#     cam_p0 = np.matmul(Tr_imu_to_cam, imu_p0)
#     cam_p0 /= cam_p0[0, 3]
#
#     cam_p1 = np.matmul(Tr_imu_to_cam, imu_p1)
#     cam_p1 /= cam_p1[0, 3]
#
#     A = cam_p1[0, 0] - cam_p0[0, 0]
#     B = cam_p1[0, 1] - cam_p0[0, 1]
#     C = cam_p1[0, 2] - cam_p0[0, 2]
#     D = -1 * (A * cam_p0[0, 0] + B * cam_p0[0, 1] + C * cam_p0[0, 2])
#     fx = P[0, 0]
#     fy = P[1, 1]
#     px0 = P[0, 2]
#     py0 = P[1, 2]
#     ground_points = []
#     for image_point in image_points:
#         px = image_point[0]
#         py = image_point[1]
#         if px == -1:
#             ground_points.append([-1, -1, -1])
#         else:
#             z = - D / (C + A * (px - px0) / fx + B * (py - py0) / fy)
#             x = (px - px0) * z / fx
#             y = (py - py0) * z / fy
#             ground_pt = np.array([x, y, z, 1])
#             ground_pt = np.matmul(Tr_cam_to_imu, ground_pt)
#             ground_pt /= ground_pt[3]
#             ground_points.append([ground_pt[0], ground_pt[1], ground_pt[2]])
#     return ground_points
#
# def msg_loop(output_dir, rate, frame_limit, topics, velo_topics, cam_topics,
#               odom_topics, msg_it, offset, model, car_name, date, sim_path, use_stereo):
#     global frame_points, EG0_POINT_COLORS
#     start_time = None
#     last_frame = None
#     frame_number = None
#     index = 0
#     for m in msg_it:
#         if start_time is None:
#             try:
#                 start_time = msg_time(m[topics[0]][0])
#             except:
#                 continue
#         frame_number = int(((msg_time(m[topics[0]][0]) - start_time).to_sec() + (rate / 2.0)) / rate) + offset
#         if last_frame == frame_number:
#             continue
#         sys.stdout.flush()
#         left_orig_image, right_orig_image, left_timestamp, right_timestamp = msg_to_png(m, cam_topics)
#         left_image, right_image, P1, P2, Tr_cam_to_imu, Tr_lidar_to_imu = unwarp(left_orig_image, right_orig_image, car_name, date)
#         mono_image = left_image
#         input = model.preprocess(mono_image)
#         start = time.time()
#         h_output_cls, h_output_ego = model.forward(input)
#         end = time.time()
#         result_ego, lines = model.postprocess(h_output_ego)
#
#         result_ego_color = np.copy(mono_image)
#         result_points_color = np.copy(mono_image)
#         for id in range(len(lines)):
#             result_ego_color[result_ego == id+1] = EG0_POINT_COLORS[id]
#             for point in lines[id]:
#                 if point[0] >= 0:
#                     cv2.circle(result_points_color, point, 1, EG0_POINT_COLORS[id], -1)
#         cv2.imwrite(os.path.join(output_dir, "mono_%.3f_ego.png" % (left_timestamp)), result_ego_color)
#         cv2.imwrite(os.path.join(output_dir, "mono_%.3f_points.png" % (left_timestamp)), result_points_color)
#
#         if use_stereo:
#             input = model.preprocess(right_image)
#             start = time.time()
#             h_output_cls, h_output_ego = model.forward(input)
#             end = time.time()
#             result_ego_right, lines_right = model.postprocess(h_output_ego)
#
#             result_ego_color_right = np.copy(right_image)
#             result_points_color_right = np.copy(right_image)
#             for id in range(len(lines_right)):
#                 result_ego_color_right[result_ego_right == id + 1] = EG0_POINT_COLORS[id]
#                 for point in lines_right[id]:
#                     if point[0] >= 0:
#                         cv2.circle(result_points_color_right, point, 1, EG0_POINT_COLORS[id], -1)
#             cv2.imwrite(os.path.join(output_dir, "right_%.3f_ego.png" % (left_timestamp)), result_ego_color_right)
#             cv2.imwrite(os.path.join(output_dir, "right_%.3f_points.png" % (left_timestamp)), result_points_color_right)
#
#         Tr_imu_to_world, pose = msg_to_odom(m, odom_topics, left_timestamp)
#         Tr_cam_to_world = np.matmul(Tr_imu_to_world, Tr_cam_to_imu)
#         Tr_velo_to_world = np.matmul(Tr_imu_to_world, Tr_lidar_to_imu)
#
#         points3d_utm = []
#         points3d_imu = []
#         max_depth = 0
#         lines_imu = []
#
#         def func(x, a, b, c, d, e, f):
#             return a*pow(x,5) + b*pow(x,4) + c*pow(x,3) + d*pow(x,2) + e*pow(x,1)+f
#
#         left_params, right_params = [], []
#         left_min, right_min = 0, 0
#         fit_map = np.copy(mono_image)
#         for id in range(len(lines)):
#             x, y = [], []
#             v_min = len(lines[id])-1
#             v_max = 0
#             for dot in lines[id]:
#                 if dot[0] < 0 or dot[1] > 240:
#                     continue
#                 if v_max < dot[1]:
#                     v_max = dot[1]
#                 if v_min > dot[1]:
#                     v_min = dot[1]
#                 x.append(dot[1])
#                 y.append(dot[0])
#             if id == 0:
#                 left_params = curve_fit(func, x, y)[0]
#                 left_min = v_min
#             else:
#                 right_params = curve_fit(func, x, y)[0]
#                 right_min = v_min
#         #     x3 = np.arange(v_min, v_max, 1)
#         #     y3 = A3 * x3 * x3 * x3 * x3* x3 + B3 * x3 * x3 * x3* x3 + C3 * x3 * x3 * x3+ D3*x3* x3+E3* x3+F3
#         #     for i in range(len(x3)):
#         #         cv2.circle(fit_map, (int(y3[i]+0.5), int(x3[i]+0.5)), 1, EG0_POINT_COLORS[id+1], -1)
#         #     # plt.grid(True)
#         #     # plt.axis("equal")
#         #     # plt.show()
#         # cv2.imshow('fit_map', fit_map)
#         # cv2.waitKey()
#         # continue
#
#         for lid in range(len(lines)):
#             # points3d_utm.append([0, 0, 0, lid])
#             lines_imu.append(imageToGround(lines[lid], P1, Tr_cam_to_imu))
#         lens = len(lines[0])
#         lane_width_list = []
#         for i in range(lens):
#             if lines_imu[0][lens-1-i][0] < 20 and lines_imu[0][lens-1-i][0] >= 10 \
#                 and lines_imu[1][lens-1-i][0] < 20 and lines_imu[1][lens-1-i][0] >= 10:
#                 lane_width_list.append(lines_imu[0][lens-1-i][1] - lines_imu[1][lens-1-i][1])
#
#         lane_width = np.mean(lane_width_list)
#         lane_width = 3.75
#         fx = P1[0, 0]
#         P1_inv = np.mat(P1[0:3,0:3]).I
#
#         for i in range(lens):
#             # if i <= 240:
#             #     continue
#             if lines[0][lens-1-i][0] < 0 or lines[1][lens-1-i][0] < 0:
#                 continue
#             pixel_width = lines[1][lens - 1 - i][0] - lines[0][lens - 1 - i][0]
#             cam_depth = lane_width / pixel_width * fx
#             if max_depth < cam_depth:
#                 max_depth = cam_depth
#             else:
#                 continue
#             # if cam_depth > 150:
#             #     continue
#             cam_ego_left = cam_depth * np.matmul(P1_inv, [lines[0][lens - 1 - i][0], lines[0][lens - 1 - i][1], 1])
#             cam_ego_right = cam_depth * np.matmul(P1_inv, [lines[1][lens - 1 - i][0], lines[1][lens - 1 - i][1], 1])
#             imu_ego_left = np.matmul(Tr_cam_to_imu, np.asarray(
#                 [[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
#             imu_ego_left /= imu_ego_left[3]
#             imu_ego_right = np.matmul(Tr_cam_to_imu, np.asarray(
#                 [[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
#             imu_ego_right /= imu_ego_right[3]
#             points3d_imu.append([imu_ego_left[0, 0], imu_ego_left[1, 0], imu_ego_left[2, 0], 0])
#             points3d_imu.append([imu_ego_right[0, 0], imu_ego_right[1, 0], imu_ego_right[2, 0], 1])
#             utm_ego_left = np.matmul(Tr_cam_to_world, np.asarray([[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
#             utm_ego_left /= utm_ego_left[3]
#             utm_ego_right = np.matmul(Tr_cam_to_world, np.asarray([[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
#             utm_ego_right /= utm_ego_right[3]
#             points3d_utm.append([utm_ego_left[0, 0], utm_ego_left[1, 0], utm_ego_left[2, 0], 0])
#             points3d_utm.append([utm_ego_right[0, 0], utm_ego_right[1, 0], utm_ego_right[2, 0], 1])
#
#         points3d_utm_right = []
#         points3d_imu_right = []
#
#         if use_stereo:
#             fx = P2[0, 0]
#             P2_inv = np.mat(P2[0:3, 0:3]).I
#             max_depth_right = 0
#             for i in range(lens):
#                 # if i <= 240:
#                 #     continue
#                 if lines_right[0][lens-1-i][0] < 0 or lines_right[1][lens-1-i][0] < 0:
#                     continue
#                 pixel_width = lines_right[1][lens - 1 - i][0] - lines_right[0][lens - 1 - i][0]
#                 cam_depth = lane_width / pixel_width * fx
#                 if max_depth_right < cam_depth:
#                     max_depth_right = cam_depth
#                 else:
#                     continue
#                 # if cam_depth > 150:
#                 #     continue
#                 cam_ego_left = cam_depth * np.matmul(P2_inv, [lines_right[0][lens - 1 - i][0], lines_right[0][lens - 1 - i][1], 1])
#                 cam_ego_right = cam_depth * np.matmul(P2_inv, [lines_right[1][lens - 1 - i][0], lines_right[1][lens - 1 - i][1], 1])
#                 imu_ego_left = np.matmul(Tr_cam_to_imu, np.asarray(
#                     [[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
#                 imu_ego_left /= imu_ego_left[3]
#                 imu_ego_right = np.matmul(Tr_cam_to_imu, np.asarray(
#                     [[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
#                 imu_ego_right /= imu_ego_right[3]
#                 points3d_imu_right.append([imu_ego_left[0, 0], imu_ego_left[1, 0], imu_ego_left[2, 0], 2])
#                 points3d_imu_right.append([imu_ego_right[0, 0], imu_ego_right[1, 0], imu_ego_right[2, 0], 3])
#                 utm_ego_left = np.matmul(Tr_cam_to_world, np.asarray([[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
#                 utm_ego_left /= utm_ego_left[3]
#                 utm_ego_right = np.matmul(Tr_cam_to_world, np.asarray([[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
#                 utm_ego_right /= utm_ego_right[3]
#                 points3d_utm_right.append([utm_ego_left[0, 0], utm_ego_left[1, 0], utm_ego_left[2, 0], 2])
#                 points3d_utm_right.append([utm_ego_right[0, 0], utm_ego_right[1, 0], utm_ego_right[2, 0], 3])
#
#         # min_h = max(left_min, right_min)
#         # fit_H = np.arange(min_h, 240, 0.05)
#         # for i in range(len(fit_H)):
#         #     x_left = func(fit_H[len(fit_H)-1-i], *left_params)
#         #     x_right = func(fit_H[len(fit_H)-1-i], *right_params)
#         #     pixel_width = x_right - x_left
#         #     cam_depth = lane_width / pixel_width * fx
#         #     if max_depth < cam_depth:
#         #         max_depth = cam_depth
#         #     # else:
#         #     #     continue
#         #     cam_ego_left = cam_depth * np.matmul(P1_inv, [x_left, fit_H[len(fit_H)-1-i], 1])
#         #     cam_ego_right = cam_depth * np.matmul(P1_inv, [x_right, fit_H[len(fit_H)-1-i], 1])
#         #     imu_ego_left = np.matmul(Tr_cam_to_imu, np.asarray(
#         #         [[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
#         #     imu_ego_left /= imu_ego_left[3]
#         #     imu_ego_right = np.matmul(Tr_cam_to_imu, np.asarray(
#         #         [[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
#         #     imu_ego_right /= imu_ego_right[3]
#         #     points3d_imu.append([imu_ego_left[0, 0], imu_ego_left[1, 0], imu_ego_left[2, 0], 0])
#         #     points3d_imu.append([imu_ego_right[0, 0], imu_ego_right[1, 0], imu_ego_right[2, 0], 1])
#         #     utm_ego_left = np.matmul(Tr_cam_to_world,
#         #                              np.asarray([[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]],
#         #                                         dtype=np.float).T)
#         #     utm_ego_left /= utm_ego_left[3]
#         #     utm_ego_right = np.matmul(Tr_cam_to_world,
#         #                               np.asarray([[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]],
#         #                                          dtype=np.float).T)
#         #     utm_ego_right /= utm_ego_right[3]
#         #     points3d_utm.append([utm_ego_left[0, 0], utm_ego_left[1, 0], utm_ego_left[2, 0], 0])
#         #     points3d_utm.append([utm_ego_right[0, 0], utm_ego_right[1, 0], utm_ego_right[2, 0], 1])
#
#         header = """VERSION 0.7
#             FIELDS x y z intensity
#             SIZE 4 4 4 4
#             TYPE F F F F
#             COUNT 1 1 1 1
#             WIDTH %d
#             HEIGHT 1
#             VIEWPOINT 0 0 0 1 0 0 0
#             POINTS %d
#             DATA ascii
#             """ % (len(points3d_utm)+len(pose), len(points3d_utm)+len(pose))
#         o = open(os.path.join(output_dir, "mono_%.3f_utm.pcd" % (left_timestamp)), "w")
#         o.writelines(header)
#         for j in range(len(points3d_utm)):
#             o.write("%f %f %f %f\n" % (points3d_utm[j][0], points3d_utm[j][1], points3d_utm[j][2], points3d_utm[j][3]))
#         for j in range(len(pose)):
#             o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], pose[j][3]))
#         o.close()
#
#         header = """VERSION 0.7
#                     FIELDS x y z intensity
#                     SIZE 4 4 4 4
#                     TYPE F F F F
#                     COUNT 1 1 1 1
#                     WIDTH %d
#                     HEIGHT 1
#                     VIEWPOINT 0 0 0 1 0 0 0
#                     POINTS %d
#                     DATA ascii
#                     """ % (len(points3d_imu), len(points3d_imu))
#         o = open(os.path.join(output_dir, "mono_%.3f_imu.pcd" % (left_timestamp)), "w")
#         o.writelines(header)
#         for j in range(len(points3d_imu)):
#             o.write("%f %f %f %f\n" % (-points3d_imu[j][1], points3d_imu[j][0], points3d_imu[j][2], points3d_imu[j][3]))
#         o.close()
#
#         header = """VERSION 0.7
#                     FIELDS x y z intensity
#                     SIZE 4 4 4 4
#                     TYPE F F F F
#                     COUNT 1 1 1 1
#                     WIDTH %d
#                     HEIGHT 1
#                     VIEWPOINT 0 0 0 1 0 0 0
#                     POINTS %d
#                     DATA ascii
#                     """ % (len(points3d_utm_right) + 3*len(pose), len(points3d_utm_right) + 3*len(pose))
#         o = open(os.path.join(output_dir, "right_%.3f_utm.pcd" % (left_timestamp)), "w")
#         o.writelines(header)
#         for j in range(len(points3d_utm_right)):
#             o.write("%f %f %f %f\n" % (points3d_utm_right[j][0], points3d_utm_right[j][1], points3d_utm_right[j][2], points3d_utm_right[j][3]))
#         for j in range(len(pose)):
#             o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], pose[j][3]))
#             o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], 0))
#             o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], 1))
#             # o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], 2))
#         o.close()
#
#         header = """VERSION 0.7
#                             FIELDS x y z intensity
#                             SIZE 4 4 4 4
#                             TYPE F F F F
#                             COUNT 1 1 1 1
#                             WIDTH %d
#                             HEIGHT 1
#                             VIEWPOINT 0 0 0 1 0 0 0
#                             POINTS %d
#                             DATA ascii
#                             """ % (len(points3d_imu_right), len(points3d_imu_right))
#         o = open(os.path.join(output_dir, "right_%.3f_imu.pcd" % (left_timestamp)), "w")
#         o.writelines(header)
#         for j in range(len(points3d_imu_right)):
#             o.write("%f %f %f %f\n" % (-points3d_imu_right[j][1], points3d_imu_right[j][0], points3d_imu_right[j][2], points3d_imu_right[j][3]))
#         o.close()
#
#         # velo = msg_to_velo(m, velo_topics)
#         # header = """VERSION 0.7
#         #                     FIELDS x y z intensity
#         #                     SIZE 4 4 4 4
#         #                     TYPE F F F F
#         #                     COUNT 1 1 1 1
#         #                     WIDTH %d
#         #                     HEIGHT 1
#         #                     VIEWPOINT 0 0 0 1 0 0 0
#         #                     POINTS %d
#         #                     DATA ascii
#         #                     """ % (len(velo), len(velo))
#         # o = open(os.path.join(output_dir, "%.3f_velo.pcd" % (left_timestamp)), "w")
#         # o.writelines(header)
#         # for j in range(len(velo)):
#         #     point4d = np.asarray([[velo[j][0], velo[j][1], velo[j][2], 1.0]], dtype=np.float)
#         #     world_point4d = np.matmul(Tr_velo_to_world, point4d.T)
#         #     world_point4d = world_point4d.squeeze()
#         #     o.write("%f %f %f %f\n" % (world_point4d[0], world_point4d[1], world_point4d[2], velo[j][3]))
#         # o.close()
#
#         bev_image = draw_bev(points3d_imu)
#         bev_image = cv2.putText(bev_image, "max_depth: %.3f m" % (max_depth), (bev_image.shape[1]-300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                                 (0, 255, 0), 1)
#         sim_img = cv2.imread(os.path.join(sim_path, "%04d.jpg" % (index)))
#         sim_img = sim_img[sim_img.shape[0] - bev_image.shape[0]:, 0:result_ego.shape[1]]
#         result_points_color_zoom = np.zeros((bev_image.shape[0], result_ego.shape[1], 3)).astype(np.uint8)
#         result_points_color_zoom[result_points_color_zoom.shape[0]-result_ego.shape[0]:, :] = result_points_color
#         merged = np.concatenate([result_points_color_zoom, sim_img], axis=1)
#         merged = np.concatenate([merged, bev_image], axis=1)
#         cv2.imwrite(os.path.join(output_dir, "mono_%.3f_bev.png" % (left_timestamp)), merged)
#         latency = end - start
#         print("timestamp: %.3f, latency: %.3f s, distance: %.3f m" % (float(left_timestamp), latency, max_depth))
#         last_frame = frame_number
#         index += 1
#         if frame_limit > 0 and frame_number >= frame_limit:
#             sys.stdout.flush()
#             exit(0)
#
#     return frame_number

class Spline:
    u"""
    Cubic Spline class
    usage:
        spline=Spline(x,y)
        rx=np.arange(0,4,0.1)
        ry=[spline.calc(i) for i in rx]
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc__A(h)
        B = self.__calc__B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        u"""
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """

        for i in range(self.nx):
            if self.x[i] - x > 0:
                return i - 1

    def __calc__A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != self.nx - 2:
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc__B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B

    def calcd(self, t):
        u"""
        Calc first derivative
        """

        j = int(math.floor(t))
        if(j < 0):
            j = 0
        elif(j >= len(self.a)):
            j = len(self.a) - 1

        dt = t - j
        result = self.b[j] + 2.0 * self.c[j] * dt + 3.0 * self.d[j] * dt * dt
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """
        j = int(math.floor(t))
        if(j < 0):
            j = 0
        elif(j >= len(self.a)):
            j = len(self.a) - 1

        dt = t - j
        result = 2.0 * self.c[j] + 6.0 * self.d[j] * dt
        return result

class cal3DLane(object):
    def __init__(self, model, P1, P2, Tr_cam_to_imu):
        super(cal3DLane, self).__init__()
        self.model = model
        self.lane_width = 3.75
        self.P1 = P1
        self.P2 = P2
        self.Tr_cam_to_imu = Tr_cam_to_imu

    def prediction(self, mono_image):
        input = self.model.preprocess(mono_image)
        start = time.time()
        h_output_cls, h_output_ego = self.model.forward(input)
        end = time.time()
        latency = end - start
        result_ego, lines = self.model.postprocess(h_output_ego)
        return latency, result_ego, lines

    def drawPrediction(self, mono_image, result_ego, lines):
        global EG0_POINT_COLORS
        result_ego_color = np.copy(mono_image)
        result_points_color = np.copy(mono_image)
        for id in range(len(lines)):
            result_ego_color[result_ego == id + 1] = EG0_POINT_COLORS[id]
            for point in lines[id]:
                if point[0] >= 0:
                    cv2.circle(result_points_color, point, 1, EG0_POINT_COLORS[id], -1)
        return result_ego_color, result_points_color

    def imageToGround(self, lines, Tr_imu_to_world, imu_height=0):
        Tr_cam_to_world = np.matmul(Tr_imu_to_world, self.Tr_cam_to_imu)
        imu_p0 = np.array([0, 0, -imu_height, 1])
        imu_p1 = np.array([0, 0, -imu_height + 1, 1])
        Tr_imu_to_cam = np.mat(self.Tr_cam_to_imu).I

        cam_p0 = np.matmul(Tr_imu_to_cam, imu_p0)
        cam_p0 /= cam_p0[0, 3]

        cam_p1 = np.matmul(Tr_imu_to_cam, imu_p1)
        cam_p1 /= cam_p1[0, 3]

        A = cam_p1[0, 0] - cam_p0[0, 0]
        B = cam_p1[0, 1] - cam_p0[0, 1]
        C = cam_p1[0, 2] - cam_p0[0, 2]
        D = -1 * (A * cam_p0[0, 0] + B * cam_p0[0, 1] + C * cam_p0[0, 2])
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]
        px0 = self.P1[0, 2]
        py0 = self.P1[1, 2]
        ground_points = []
        points3d_imu = []
        points3d_utm = []
        max_depth = 0
        for i in range(len(lines)):
            for image_point in lines[i]:
                px = image_point[0]
                py = image_point[1]
                if px == -1:
                    ground_points.append([-1, -1, -1])
                else:
                    z = - D / (C + A * (px - px0) / fx + B * (py - py0) / fy)
                    x = (px - px0) * z / fx
                    y = (py - py0) * z / fy
                    cam_pt = np.array([x, y, z, 1])
                    imu_pt = np.matmul(self.Tr_cam_to_imu, cam_pt)
                    imu_pt /= imu_pt[3]
                    if imu_pt[0] > 300 or imu_pt[0] < 0:
                        continue
                    if max_depth < imu_pt[0]:
                        max_depth = imu_pt[0]
                    points3d_imu.append([imu_pt[0], imu_pt[1], imu_pt[2], i])
                    utm_pt = np.matmul(Tr_cam_to_world, cam_pt)
                    utm_pt /= utm_pt[3]
                    points3d_utm.append([utm_pt[0], utm_pt[1], utm_pt[2], i])
        return points3d_imu, points3d_utm, max_depth

    def mono3DLane(self, lines, Tr_imu_to_world, use_left):
        Tr_cam_to_world = np.matmul(Tr_imu_to_world, self.Tr_cam_to_imu)
        max_depth_cam = 0
        max_depth = 0
        points3d_imu = []
        points3d_utm = []
        fx = self.P1[0, 0]
        P_inv = np.mat(self.P1[0:3, 0:3]).I
        P_offset = self.P1[:, 3:4] / 1000
        if not use_left:
            fx = self.P2[0, 0]
            P_inv = np.mat(self.P2[0:3, 0:3]).I
            P_offset = self.P2[:, 3:4] / 1000
        lens = len(lines[0])
        for i in range(lens):
            if lines[0][lens-1-i][0] < 0 or lines[1][lens-1-i][0] < 0:
                continue
            pixel_width = lines[1][lens - 1 - i][0] - lines[0][lens - 1 - i][0]
            cam_depth = self.lane_width / pixel_width * fx
            if max_depth_cam < cam_depth:
                max_depth_cam = cam_depth
            else:
                continue
            cam_ego_left = cam_depth * np.matmul(P_inv, [lines[0][lens - 1 - i][0], lines[0][lens - 1 - i][1], 1]) - P_offset.T
            cam_ego_right = cam_depth * np.matmul(P_inv, [lines[1][lens - 1 - i][0], lines[1][lens - 1 - i][1], 1]) - P_offset.T
            imu_ego_left = np.matmul(self.Tr_cam_to_imu, np.asarray(
                [[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
            imu_ego_left /= imu_ego_left[3]
            imu_ego_right = np.matmul(self.Tr_cam_to_imu, np.asarray(
                [[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
            imu_ego_right /= imu_ego_right[3]
            if max_depth < imu_ego_left[0]:
                max_depth = imu_ego_left[0]
            if max_depth < imu_ego_right[0]:
                max_depth = imu_ego_right[0]
            points3d_imu.append([imu_ego_left[0, 0], imu_ego_left[1, 0], imu_ego_left[2, 0], 0])
            points3d_imu.append([imu_ego_right[0, 0], imu_ego_right[1, 0], imu_ego_right[2, 0], 1])
            utm_ego_left = np.matmul(Tr_cam_to_world, np.asarray(
                [[cam_ego_left[0, 0], cam_ego_left[0, 1], cam_ego_left[0, 2], 1.0]], dtype=np.float).T)
            utm_ego_left /= utm_ego_left[3]
            utm_ego_right = np.matmul(Tr_cam_to_world, np.asarray(
                [[cam_ego_right[0, 0], cam_ego_right[0, 1], cam_ego_right[0, 2], 1.0]], dtype=np.float).T)
            utm_ego_right /= utm_ego_right[3]
            points3d_utm.append([utm_ego_left[0, 0], utm_ego_left[1, 0], utm_ego_left[2, 0], 0])
            points3d_utm.append([utm_ego_right[0, 0], utm_ego_right[1, 0], utm_ego_right[2, 0], 1])
        return points3d_imu, points3d_utm, max_depth

    def stereo3DLane(self, lines_left, lines_right, Tr_imu_to_world):
        Tr_cam_to_world = np.matmul(Tr_imu_to_world, self.Tr_cam_to_imu)
        max_depth = 0
        points3d_imu = []
        points3d_utm = []
        lid_left = lid_right = 0
        while lid_left < len(lines_left) and lid_right < len(lines_right):
            if lid_left < 4 and lid_right < 4:
                line_left = lines_left[lid_left]
                line_right = lines_right[lid_right]
                xl, yl, xr, yr = [], [], [], []
                if len(line_left) == len(line_right):
                    for j in range(len(line_left)):
                        point_left = line_left[j]
                        point_right = line_right[j]
                        if point_left[0] >= 0 and point_right[0] >= 0:
                            xl.append(point_left[0])
                            yl.append(point_left[1])
                            xr.append(point_right[0])
                            yr.append(point_right[1])
                    if len(xl) == 0 or len(xr) == 0:
                        lid_left += 1
                        lid_right += 1
                        continue
                    ptl = np.concatenate(
                        (np.asarray(xl, dtype=np.float).reshape(1, -1), np.asarray(yl, dtype=np.float).reshape(1, -1)),
                        axis=0)
                    ptr = np.concatenate(
                        (np.asarray(xr, dtype=np.float).reshape(1, -1), np.asarray(yr, dtype=np.float).reshape(1, -1)),
                        axis=0)
                    points4d = cv2.triangulatePoints(self.P1, self.P2, ptl, ptr)
                    for j in range(points4d.shape[1]):
                        point4d = np.asarray([[points4d[0][j] / points4d[3][j], points4d[1][j] / points4d[3][j],
                                               points4d[2][j] / points4d[3][j], 1.0]], dtype=np.float)
                        imu_point4d = np.matmul(self.Tr_cam_to_imu, point4d.T)
                        imu_point4d /= imu_point4d[3]
                        if abs(imu_point4d[2]) > 10 or abs(imu_point4d[1]) > 5 or imu_point4d[0] > 400 or imu_point4d[0] < 0:
                            continue
                        if max_depth < imu_point4d[0]:
                            max_depth = imu_point4d[0]
                        points3d_imu.append([imu_point4d[0], imu_point4d[1], imu_point4d[2], lid_left])
                        world_point4d = np.matmul(Tr_cam_to_world, point4d.T)
                        world_point4d /= world_point4d[3]
                        points3d_utm.append([world_point4d[0], world_point4d[1], world_point4d[2], lid_left])
            lid_left += 1
            lid_right += 1
        return points3d_imu, points3d_utm, max_depth

class EngineModel(object):
    def __init__(self, engine_file, up_crop, bottom_crop, input_mean, input_std):
        super(EngineModel, self).__init__()
        self.up_crop = up_crop
        self.bottom_crop = bottom_crop
        self.input_mean = input_mean
        self.input_std = input_std

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.input_size = self.engine.get_binding_shape(0)
            self.output_cls_size = self.engine.get_binding_shape(1)
            self.output_ego_size = self.engine.get_binding_shape(2)
            self.h_input = np.empty(trt.volume(self.engine.get_binding_shape(0)), np.float32)
            self.h_output_cls = np.empty(trt.volume(self.engine.get_binding_shape(1)), np.float32)
            self.h_output_ego = np.empty(trt.volume(self.engine.get_binding_shape(2)), np.float32)
            self.d_input = cuda.mem_alloc(1 * self.h_input.nbytes)
            self.d_output_cls = cuda.mem_alloc(1 * self.h_output_cls.nbytes)
            self.d_output_ego = cuda.mem_alloc(1 * self.h_output_ego.nbytes)

            self.bindings = [int(self.d_input), int(self.d_output_cls), int(self.d_output_ego)]
            self.stream = cuda.Stream()

    def forward(self, input):
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod_async(self.d_input, input.ravel(), self.stream)
            context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output_cls, self.d_output_cls, self.stream)
            cuda.memcpy_dtoh_async(self.h_output_ego, self.d_output_ego, self.stream)
            self.stream.synchronize()
            self.h_output_cls = self.h_output_cls.reshape(self.output_cls_size)
            self.h_output_ego = self.h_output_ego.reshape(self.output_ego_size)
            return self.h_output_cls, self.h_output_ego

    def preprocess(self, image):
        self.image_height, self.image_width, _ = image.shape
        input_width = self.input_size[2]
        input_height = self.input_size[1]
        image_bgr = image[self.up_crop: self.image_height - self.bottom_crop, :]
        image_bgr = cv2.resize(image_bgr, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = (image_rgb - self.input_mean) / self.input_std
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.float32)
        image_rgb = np.expand_dims(image_rgb, 0)
        return image_rgb

    def postprocess(self, output, threshold=0.8):
        def get_centroid(ary):
            ret = []
            seg = []
            for i in ary:
                if len(seg) == 0 or seg[-1] + 1 == i:
                    seg.append(i)
                else:
                    ret.append(seg[len(seg) / 2])
                    seg = [i]
            if len(seg) != 0:
                ret.append(seg[len(seg) / 2])
            return ret

        output = np.transpose(output, (2, 0, 1))
        result = np.zeros((self.input_size[1], self.input_size[2])).astype(np.uint8)
        result_orig = np.zeros((self.image_height, self.image_width)).astype(np.uint8)
        for num in range(output.shape[0]-1):
            result[output[num + 1] >= threshold] = num + 1
        result_orig[self.up_crop:self.image_height-self.bottom_crop, :] = cv2.resize(result,
            (self.image_width, self.image_height-self.up_crop-self.bottom_crop), interpolation=cv2.INTER_NEAREST)

        coordinates = []
        YS = np.arange(self.image_height)
        XS = np.arange(self.image_height)
        for lid in range(output.shape[0]-1):
            ys, xs = np.where(result_orig == lid + 1)
            ytox = {}
            for x, y in zip(xs, ys):
                ytox.setdefault(y, []).append(x)
            for y in range(self.image_height):
                xs = ytox.get(y, [])
                # only use the center of consecutive pixels
                xs = get_centroid(xs)
                if len(xs) > 0:
                    XS[y] = xs[0]
                else:
                    XS[y] = -1
            coordinates.append(list(zip(XS, YS)))
        return result_orig, coordinates

class runRosbag(object):
    def __init__(self, args, model):
        super(runRosbag, self).__init__()
        self.topics = []
        self.velo_topics = []
        self.odom_topics = []
        self.cam_topics = []
        self.pose_init = []
        self.args = args
        self.model = model

        if self.args.odom_topics != None:
            for t in self.args.odom_topics.split(","):
                self.odom_topics.append(t)
                self.topics.append(t)

        if self.args.cam_topics != None:
            for t in self.args.cam_topics.split(","):
                self.cam_topics.append(t)
                self.topics.append(t)

        if self.args.velo_topics != None:
            for t in self.args.velo_topics.split(","):
                self.velo_topics.append(t)
                self.topics.append(t)

    def msgTime(self, msg):
        # return msg.timestamp
        return msg.header.stamp

    def bufferedMessageGenerator(self, bag):
        buffers = dict([(t, []) for t in self.topics])
        skipcounts = dict([(t, 0) for t in self.topics])
        for msg in bag.read_messages(topics=self.topics):
            if msg.topic in self.topics:
                buffers[msg.topic].append(msg)
            else:
                continue
            while all(buffers.values()):
                cam_timestamp = 0
                for topic, buf in buffers.iteritems():
                    if 'left' in topic:
                        cam_timestamp = self.msgTime(buf[0].message).to_sec()
                        break
                flag = 0
                for topic, buf in buffers.iteritems():
                    if topic in self.odom_topics:
                        for i in buf:
                            if self.msgTime(i.message).to_sec() - cam_timestamp > 2 * self.args.tolerance:
                                flag = 1
                if flag == 0:
                    break
                msg_set = dict([(t, []) for t in self.topics])

                for topic, buf in buffers.iteritems():
                    pre_timestamp = cam_timestamp - 0.1
                    next_timestamp = cam_timestamp + 0.1
                    pre_odom = 0
                    next_odom = 0
                    if topic in self.odom_topics:
                        for i in buf:
                            m = i.message
                            if self.msgTime(m).to_sec() - cam_timestamp < 2 * self.args.tolerance and \
                                    self.msgTime(m).to_sec() - cam_timestamp > 0:
                                if next_timestamp > self.msgTime(m).to_sec():
                                    next_timestamp = self.msgTime(m).to_sec()
                                    next_odom = m
                            elif cam_timestamp - self.msgTime(m).to_sec() < 2 * self.args.tolerance and \
                                    cam_timestamp - self.msgTime(m).to_sec() > 0:
                                if pre_timestamp < self.msgTime(m).to_sec():
                                    pre_timestamp = self.msgTime(m).to_sec()
                                    pre_odom = m
                        msg_set[topic].append(next_odom)
                        msg_set[topic].insert(0, pre_odom)
                        # print(topic, len(buf), 'pre_odom', pre_timestamp, 'next_odom', next_timestamp)
                    else:
                        for i in buf:
                            m = i.message
                            msg_set[topic].append(m)
                            # print(topic, self.msgTime(m).to_sec())
                        buf.pop(0).message
                yield msg_set
        for t, c in skipcounts.iteritems():
            print("skipped %d %s messages" % (c, t))
        sys.stdout.flush()

    def msgToPng(self, msgs):
        for topic in self.cam_topics:
            if 'left' in topic:
                left_img = bridge.compressed_imgmsg_to_cv2(msgs[topic][0], desired_encoding="bgr8")
                left_timestamp = self.msgTime(msgs[topic][0]).to_sec()
            elif 'right' in topic:
                right_img = bridge.compressed_imgmsg_to_cv2(msgs[topic][0], desired_encoding="bgr8")
                right_timestamp = self.msgTime(msgs[topic][0]).to_sec()
            else:
                exit(0)
        return left_img, right_img, left_timestamp, right_timestamp

    def msgToOdom(self, msgs, timestamp):
        def linear_interpolation(before, after, proportion):
            return before + proportion * (after - before)

        def wrapToPi(a):
            # if a < 0:
            #     return a + 2 * math.pi
            # else:
            return a

        def slerp(before, after, proportion):
            return wrapToPi(before + proportion * wrapToPi(after - before))

        def getPoseFromOdom(odom0, odom1, proportion):
            x = linear_interpolation(odom0.position.x, odom1.position.x, proportion)
            y = linear_interpolation(odom0.position.y, odom1.position.y, proportion)
            z = linear_interpolation(odom0.position.z, odom1.position.z, proportion)
            if len(self.pose_init) == 0:
                self.pose_init = [x, y, z]
                # self.pose_init = [0, 0, 0]
                print("pose_init:", self.pose_init)
            x -= self.pose_init[0]
            y -= self.pose_init[1]
            z -= self.pose_init[2]
            (roll0, pitch0, yaw0) = euler_from_quaternion(
                [odom0.orientation.x, odom0.orientation.y, odom0.orientation.z, odom0.orientation.w])
            (roll1, pitch1, yaw1) = euler_from_quaternion(
                [odom1.orientation.x, odom1.orientation.y, odom1.orientation.z, odom1.orientation.w])
            roll = slerp(roll0, roll1, proportion)
            pitch = slerp(pitch0, pitch1, proportion)
            yaw = slerp(yaw0, yaw1, proportion)
            q = quaternion_from_euler(roll, pitch, yaw)
            rotation = [[1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2], 2.0 * q[0] * q[1] - 2.0 * q[2] * q[3],
                         2.0 * q[0] * q[2] + 2.0 * q[1] * q[3]],
                        [2.0 * q[0] * q[1] + 2.0 * q[2] * q[3], 1.0 - 2.0 * q[0] * q[0] - 2.0 * q[2] * q[2],
                         2.0 * q[1] * q[2] - 2.0 * q[0] * q[3]],
                        [2.0 * q[0] * q[2] - 2.0 * q[1] * q[3], 2.0 * q[1] * q[2] + 2.0 * q[0] * q[3],
                         1.0 - 2.0 * q[0] * q[0] - 2.0 * q[1] * q[1]]]
            # rotation_quaternion = np.asarray([q[3], q[0], q[1], q[2]])
            translation = np.asarray([x, y, z])
            # rigidtrans = RigidTransform(rotation_quaternion, translation)
            Tr_imu_to_world = np.concatenate((rotation, translation.reshape(1, -1).T), axis=1)
            Tr_imu_to_world = np.concatenate((Tr_imu_to_world, [[0., 0., 0., 1.0]]), axis=0)
            pose = []
            pose.append((x, y, z, 0, roll, pitch, yaw))
            return Tr_imu_to_world, pose

        for topic in self.odom_topics:
            odom_msgs = msgs[topic]
            for odom in odom_msgs:
                odom_timestamp = self.msgTime(odom).to_sec()
                if odom_timestamp <= timestamp:
                    odom0 = odom
                else:
                    odom1 = odom

        t0 = self.msgTime(odom0).to_sec()
        t1 = self.msgTime(odom1).to_sec()
        proportion = (timestamp - t0) / (t1 - t0)
        Tr_imu_to_world, pose = getPoseFromOdom(odom0.pose.pose, odom1.pose.pose, proportion)
        return Tr_imu_to_world, pose

    def writePcd(self, points3d, pose, method, writepath):
        ego_id = 2 * method
        if not len(pose):
            ego_id = 0
        header = """VERSION 0.7
                        FIELDS x y z intensity
                        SIZE 4 4 4 4
                        TYPE F F F F
                        COUNT 1 1 1 1
                        WIDTH %d
                        HEIGHT 1
                        VIEWPOINT 0 0 0 1 0 0 0
                        POINTS %d
                        DATA ascii
                        """ % (len(points3d) + len(pose) + ego_id, len(points3d) + len(pose) + ego_id)

        o = open(writepath, "w")
        o.writelines(header)
        for j in range(len(pose)):
            o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], pose[j][3]))
            for m in range(8):
                o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], m+1))
        for j in range(len(points3d)):
            o.write("%f %f %f %f\n" % (points3d[j][0], points3d[j][1], points3d[j][2], points3d[j][3] + ego_id + 1))
        o.close()

    def drawBevImage(self, points_imu):
        global EG0_POINT_COLORS
        bev_image = np.zeros((800, 700, 3)).astype(np.uint8)
        h, w, c = bev_image.shape
        for v in range(h / 40 - 1):
            cv2.line(bev_image, (0, (v + 1) * 40), (w - 1, (v + 1) * 40), color=(255, 255, 255), thickness=1)
            bev_image = cv2.putText(bev_image, str(h / 2 - 20 * (v + 2)), (5, (v + 1) * 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 1)
        for point_imu in points_imu:
            v = int(h - 20 - point_imu[0] * 2 + 0.5)
            u = int(-point_imu[1] * 10 + 0.5 + w / 2)
            if v >= 0 and v < h and u >= 0 and u < w:
                cv2.circle(bev_image, (u, v), 2, EG0_POINT_COLORS[int(point_imu[3])], -1)
        return bev_image

    def msgLoop(self, msg_it, offset, calib, cal3dlane, sim_path):
        global frame_points, EG0_POINT_COLORS
        start_time = None
        last_frame = None
        frame_number = None
        index = 0
        for m in msg_it:
            if start_time is None:
                try:
                    start_time = self.msgTime(m[self.topics[0]][0])
                except:
                    continue
            frame_number = int(((self.msgTime(m[self.topics[0]][0]) - start_time).to_sec() +
                                (self.args.rate / 2.0)) / self.args.rate) + offset
            if last_frame == frame_number:
                continue
            sys.stdout.flush()
            left_orig_image, right_orig_image, left_timestamp, right_timestamp = self.msgToPng(m)
            Tr_imu_to_world, pose = self.msgToOdom(m, left_timestamp)
            max_depth_left, max_depth_right, max_depth_stereo = -1, -1, -1
            if self.args.use_stereo:
                left_image = calib.unwarp(left_orig_image, 1)
                latency_left, result_ego_left, lines_left = cal3dlane.prediction(left_image)
                result_ego_color_left, result_points_color_left = \
                    cal3dlane.drawPrediction(left_image, result_ego_left, lines_left)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_ego_left.png" % (left_timestamp)),
                            result_ego_color_left)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_points_left.png" % (left_timestamp)),
                            result_points_color_left)
                points3d_imu_left, points3d_utm_left, max_depth_left = \
                    cal3dlane.mono3DLane(lines_left, Tr_imu_to_world, 1)
                imupath_left = os.path.join(self.args.out, "%.3f_imu_left.pcd" % (left_timestamp))
                utmpath_left = os.path.join(self.args.out, "%.3f_utm_left.pcd" % (left_timestamp))
                self.writePcd(points3d_imu_left, [], 0, imupath_left)
                self.writePcd(points3d_utm_left, pose, 0, utmpath_left)
                # bev_image_left = self.drawBevImage(points3d_imu_left)

                right_image = calib.unwarp(right_orig_image, 0)
                latency_right, result_ego_right, lines_right = cal3dlane.prediction(right_image)
                result_ego_color_right, result_points_color_right = \
                    cal3dlane.drawPrediction(right_image, result_ego_right, lines_right)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_ego_right.png" % (left_timestamp)),
                            result_ego_color_right)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_points_right.png" % (left_timestamp)),
                            result_points_color_right)
                points3d_imu_right, points3d_utm_right, max_depth_right = \
                    cal3dlane.mono3DLane(lines_right, Tr_imu_to_world, 0)
                imupath_right = os.path.join(self.args.out, "%.3f_imu_right.pcd" % (left_timestamp))
                utmpath_right = os.path.join(self.args.out, "%.3f_utm_right.pcd" % (left_timestamp))
                self.writePcd(points3d_imu_right, [], 1, imupath_right)
                self.writePcd(points3d_utm_right, pose, 1, utmpath_right)
                # bev_image_right = self.drawBevImage(points3d_imu_right)

                points3d_imu_stereo, points3d_utm_stereo, max_depth_stereo = \
                    cal3dlane.stereo3DLane(lines_left, lines_right, Tr_imu_to_world)
                imupath_stereo = os.path.join(self.args.out, "%.3f_imu_stereo.pcd" % (left_timestamp))
                utmpath_stereo = os.path.join(self.args.out, "%.3f_utm_stereo.pcd" % (left_timestamp))
                self.writePcd(points3d_imu_stereo, [], 2, imupath_stereo)
                self.writePcd(points3d_utm_stereo, pose, 2, utmpath_stereo)
                # bev_image_stereo = self.drawBevImage(points3d_imu_stereo)
                #
                # bev_image = np.concatenate([bev_image_left, bev_image_right], axis=1)
                # bev_image = np.concatenate([bev_image, bev_image_stereo], axis=1)

            if self.args.use_mono and not self.args.use_stereo:
                if self.args.use_left:
                    left_image = calib.unwarp(left_orig_image, self.args.use_left)
                    latency_left, result_ego_left, lines_left = cal3dlane.prediction(left_image)
                    result_ego_color_left, result_points_color_left = \
                        cal3dlane.drawPrediction(left_image, result_ego_left, lines_left)
                    cv2.imwrite(os.path.join(self.args.out, "%.3f_ego_left.png" % (left_timestamp)),
                                result_ego_color_left)
                    cv2.imwrite(os.path.join(self.args.out, "%.3f_points_left.png" % (left_timestamp)),
                                result_points_color_left)
                    points3d_imu_left, points3d_utm_left, max_depth_left = \
                        cal3dlane.mono3DLane(lines_left, Tr_imu_to_world, self.args.use_left)
                    imupath_left = os.path.join(self.args.out, "%.3f_imu_left.pcd" % (left_timestamp))
                    utmpath_left = os.path.join(self.args.out, "%.3f_utm_left.pcd" % (left_timestamp))
                    self.writePcd(points3d_imu_left, [], 0, imupath_left)
                    self.writePcd(points3d_utm_left, pose, 0, utmpath_left)
                    # bev_image_left = self.drawBevImage(points3d_imu_left)
                    # bev_image = bev_image_left
                else:
                    right_image = calib.unwarp(right_orig_image, self.args.use_left)
                    latency_right, result_ego_right, lines_right = cal3dlane.prediction(right_image)
                    result_ego_color_right, result_points_color_right = \
                        cal3dlane.drawPrediction(right_image, result_ego_right, lines_right)
                    cv2.imwrite(os.path.join(output_dir, "%.3f_ego_right.png" % (left_timestamp)),
                                result_ego_color_right)
                    cv2.imwrite(os.path.join(output_dir, "%.3f_points_right.png" % (left_timestamp)),
                                result_points_color_right)
                    points3d_imu_right, points3d_utm_right, max_depth_right = \
                        cal3dlane.mono3DLane(lines_right, Tr_imu_to_world, self.args.use_left)
                    imupath_right = os.path.join(self.args.out, "%.3f_imu_right.pcd" % (left_timestamp))
                    utmpath_right = os.path.join(self.args.out, "%.3f_utm_right.pcd" % (left_timestamp))
                    self.writePcd(points3d_imu_right, [], 1, imupath_right)
                    self.writePcd(points3d_utm_right, pose, 1, utmpath_right)
                    # bev_image_right = self.drawBevImage(points3d_imu_right)
                    # bev_image = bev_image_right

            if not self.args.use_mono and not self.args.use_stereo:
                left_image = calib.unwarp(left_orig_image, self.args.use_left)
                latency_left, result_ego_left, lines_left = cal3dlane.prediction(left_image)
                result_ego_color_left, result_points_color_left = \
                    cal3dlane.drawPrediction(left_image, result_ego_left, lines_left)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_ego_left.png" % (left_timestamp)),
                            result_ego_color_left)
                cv2.imwrite(os.path.join(self.args.out, "%.3f_points_left.png" % (left_timestamp)),
                            result_points_color_left)
                points3d_imu_left, points3d_utm_left, max_depth_left = \
                    cal3dlane.imageToGround(lines_left, Tr_imu_to_world, imu_height=0.52)
                imupath_left = os.path.join(self.args.out, "%.3f_imu_flat.pcd" % (left_timestamp))
                utmpath_left = os.path.join(self.args.out, "%.3f_utm_flat.pcd" % (left_timestamp))
                self.writePcd(points3d_imu_left, [], 3, imupath_left)
                self.writePcd(points3d_utm_left, pose, 3, utmpath_left)
                # bev_image_left = self.drawBevImage(points3d_imu_left)
                # bev_image = bev_image_left


            # sim_img = cv2.imread(os.path.join(sim_path, "%04d.jpg" % (index)))
            # sim_img = sim_img[sim_img.shape[0] - bev_image.shape[0]:, 0:left_orig_image.shape[1]]
            # result_points_color_zoom = np.zeros((bev_image.shape[0], left_orig_image.shape[1], 3)).astype(np.uint8)
            # result_points_color_zoom[result_points_color_zoom.shape[0] - left_orig_image.shape[0]:, :] = result_points_color
            # merged = np.concatenate([result_points_color_zoom, sim_img], axis=1)
            # merged = np.concatenate([merged, bev_image], axis=1)
            # cv2.imwrite(os.path.join(output_dir, "mono_%.3f_bev.png" % (left_timestamp)), merged)
            # latency = end - start
            # print("timestamp: %.3f, latency: %.3f s, distance: %.3f m" % (float(left_timestamp), latency, max_depth))
            print("timestamp: %.3f, max_depth_left: %.3f m, max_depth_right: %.3f m, max_depth_stereo: %.3f m"
                  % (float(left_timestamp), max_depth_left, max_depth_right, max_depth_stereo))
            last_frame = frame_number
            index += 1
            if self.args.frame_limit > 0 and frame_number >= self.args.frame_limit:
                sys.stdout.flush()
                exit(0)

        return frame_number

    def forward(self):
        if not os.path.isdir(self.args.out):
            os.mkdir(self.args.out)
        print("start bag", self.args.bag)
        bag_name = self.args.bag.split('/')[-1]
        calib = loadCalibration(bag_name)
        P1, P2, Tr_cam_to_imu = calib.getCalibParams()
        cal3dlane = cal3DLane(self.model, P1, P2, Tr_cam_to_imu)
        sim_path = self.args.sim_path + '/' + bag_name.split('.')[0]
        sys.stdout.flush()
        bag = rosbag.Bag(self.args.bag)
        offset = 0
        self.pose_init = []
        msg_it = iter(self.bufferedMessageGenerator(bag))
        offset = self.msgLoop(msg_it, offset, calib, cal3dlane, sim_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # ========================= Rosbag Configs ===========================
    parser.add_argument("--bag", type=str, help="path to bags (comma separated)")
    parser.add_argument("--out", type=str, help="output path (a directory)")
    parser.add_argument("--rate", type=float, help="desired sample rate in seconds", default=0.1)
    parser.add_argument("--tolerance", type=float, help="tolerance", default=0.005)
    parser.add_argument("--velo_topics", type=str,
                        help="velodyne topic (comma separated, don't add space between topics)")
    parser.add_argument("--cam_topics", type=str,
                        help="camera topics (comma separated, don't add space between topics)")
    parser.add_argument("--odom_topics", type=str,
                        help="odometry topic (comma separated, don't add space between topics)")
    parser.add_argument("--frame_limit", type=int, help="frame limit if > 0", default=0)
    parser.add_argument("--model_file", type=str, help="the path of model file")
    parser.add_argument("--sim_path", type=str, help="the path of simulator file")
    parser.add_argument("--use_mono", type=bool, help="use mono", default=False)
    parser.add_argument("--use_left", type=bool, help="use left", default=True)
    parser.add_argument("--use_stereo", type=bool, help="use stereo", default=True)
    parser.add_argument("--imu_height", type=int, help="imu height", default=0.52)
    args = parser.parse_args()

    global up_crop, bottom_crop, INPUT_MEAN, INPUT_STD
    model = EngineModel(args.model_file, up_crop, bottom_crop, INPUT_MEAN, INPUT_STD)
    runbag = runRosbag(args, model)
    runbag.forward()


if __name__ == '__main__':
    main()

