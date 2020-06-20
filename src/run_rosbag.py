#!/usr/bin/env python

import sys
import rosbag
import math
import numpy as np
import sensor_msgs.point_cloud2
import cv2
from cv_bridge import CvBridge
# from tf.transformations import *
import os
from options.config import cfg
from options.options import parser
# from demo import load_model, infer_model
from conf.sensor_config import unwarp

bridge = CvBridge()
pose_init = []
frame_points = []

def msg_time(msg):
    # return msg.timestamp
    return msg.header.stamp


def msg_to_velo(msgs, topics):
    for topic in topics:
        velo = np.array([p for p in sensor_msgs.point_cloud2.read_points(msgs[topic][0])]).astype(np.float64)
        velo = velo
    return velo


def msg_to_png(msgs, topics):
    for topic in topics:
        if 'left' in topic:
            left_img = bridge.compressed_imgmsg_to_cv2(msgs[topic][0], desired_encoding="bgr8")
            left_timestamp = msg_time(msgs[topic][0]).to_sec()
        elif 'right' in topic:
            right_img = bridge.compressed_imgmsg_to_cv2(msgs[topic][0], desired_encoding="bgr8")
            right_timestamp = msg_time(msgs[topic][0]).to_sec()
        else:
            exit(0)
    return left_img, right_img, left_timestamp, right_timestamp


def msg_to_odom(msgs, topics, timestamp):
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
        global pose_init
        x = linear_interpolation(odom0.position.x, odom1.position.x, proportion)
        y = linear_interpolation(odom0.position.y, odom1.position.y, proportion)
        z = linear_interpolation(odom0.position.z, odom1.position.z, proportion)
        if len(pose_init) == 0:
            pose_init = [x, y, z]
            print("pose_init:", pose_init)
        x -= pose_init[0]
        y -= pose_init[1]
        z -= pose_init[2]
        (roll0, pitch0, yaw0) = euler_from_quaternion(
            [odom0.orientation.x, odom0.orientation.y, odom0.orientation.z, odom0.orientation.w])
        (roll1, pitch1, yaw1) = euler_from_quaternion(
            [odom1.orientation.x, odom1.orientation.y, odom1.orientation.z, odom1.orientation.w])
        roll = slerp(roll0, roll1, proportion)
        pitch = slerp(pitch0, pitch1, proportion)
        yaw = slerp(yaw0, yaw1, proportion)
        q = quaternion_from_euler(roll, pitch, yaw)
        rotation = [[1.0 - 2.0 * q[1] * q[1] - 2.0 * q[2] * q[2], 2.0 * q[0] * q[1] - 2.0 * q[2] * q[3], 2.0 * q[0] * q[2] + 2.0 * q[1] * q[3]],
                    [2.0 * q[0] * q[1] + 2.0 * q[2] * q[3], 1.0 - 2.0 * q[0] * q[0] - 2.0 * q[2] * q[2], 2.0 * q[1] * q[2] - 2.0 * q[0] * q[3]],
                    [2.0 * q[0] * q[2] - 2.0 * q[1] * q[3], 2.0 * q[1] * q[2] + 2.0 * q[0] * q[3], 1.0 - 2.0 * q[0] * q[0] - 2.0 * q[1] * q[1]]]
        # rotation_quaternion = np.asarray([q[3], q[0], q[1], q[2]])
        translation = np.asarray([x, y, z])
        # rigidtrans = RigidTransform(rotation_quaternion, translation)
        Tr_imu_to_world = np.concatenate((rotation, translation.reshape(1,-1).T), axis=1)
        Tr_imu_to_world = np.concatenate((Tr_imu_to_world, [[0., 0., 0., 1.0]]), axis=0)
        pose = []
        # for i in range(10):
        #     x = linear_interpolation(odom0.position.x, odom1.position.x, float(i) / 10) - pose_init[0]
        #     y = linear_interpolation(odom0.position.y, odom1.position.y, float(i) / 10) - pose_init[1]
        #     z = linear_interpolation(odom0.position.z, odom1.position.z, float(i) / 10) - pose_init[2]
        #     pose.append((x,y,z, 4))
        pose.append((x,y,z,4, roll, pitch, yaw))
        return Tr_imu_to_world, pose

    for topic in topics:
        odom_msgs = msgs[topic]
        for odom in odom_msgs:
            odom_timestamp = msg_time(odom).to_sec()
            if odom_timestamp <= timestamp:
                odom0 = odom
            else:
                odom1 = odom

    t0 = msg_time(odom0).to_sec()
    t1 = msg_time(odom1).to_sec()
    proportion = (timestamp - t0) / (t1 - t0)
    Tr_imu_to_world, pose = getPoseFromOdom(odom0.pose.pose, odom1.pose.pose, proportion)
    return Tr_imu_to_world, pose

def buffered_message_generator(bag, tolerance, topics, odom_topics):
    buffers = dict([(t, []) for t in topics])
    skipcounts = dict([(t, 0) for t in topics])
    for msg in bag.read_messages(topics=topics):
        if msg.topic in topics:
            buffers[msg.topic].append(msg)
        else:
            continue
        while all(buffers.values()):
            cam_timestamp = 0
            for topic, buf in buffers.iteritems():
                if 'left' in topic:
                    cam_timestamp = msg_time(buf[0].message).to_sec()
                    break
            count = 0
            for topic, buf in buffers.iteritems():
                if topic in odom_topics:
                    for i in buf:
                        if abs(msg_time(i.message).to_sec() - cam_timestamp) < 2 * tolerance:
                            count += 1
            if count < 2:
                break
            msg_set = dict([(t, []) for t in topics])
            for topic, buf in buffers.iteritems():
                if topic in odom_topics:
                    for i in buf:
                        m = i.message
                        if msg_time(m).to_sec() - cam_timestamp < 2 * tolerance and msg_time(m).to_sec() - cam_timestamp > 0:
                            msg_set[topic].append(m)
                        elif cam_timestamp - msg_time(m).to_sec() < 2 * tolerance and cam_timestamp - msg_time(m).to_sec() > 0:
                            msg_set[topic].insert(0, m)
                else:
                    for i in buf:
                        m = i.message
                        msg_set[topic].append(m)
                buf.pop(0).message
            yield msg_set
    for t, c in skipcounts.iteritems():
        print("skipped %d %s messages" % (c, t))
    sys.stdout.flush()

def distance(odom1, odom2):
    x1 = odom1.pose.pose.position.x
    y1 = odom1.pose.pose.position.y
    x2 = odom2.pose.pose.position.x
    y2 = odom2.pose.pose.position.y
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def sort_items(path):
    frames = []
    items = os.listdir(path)
    for item in items:
        sec = int(item.split('.')[0])
        nsec = int(item.split('.')[1])
        frame = sec * 1000 + nsec
        frames.append(frame)
    frames.sort()
    return frames

# def gen_bev_frame(frame_points, pose, Tr_imu_to_world, height, width):
#     h = height
#     w = width
#     h_offset = 500
#     w_offset = 0
#     rotation_vector = np.array([[1., 0., 0.]])
#     rotation = Tr_imu_to_world[0:3, 0:3]
#     # rotation_vector = np.matmul(rotation, rotation_vector.T)
#     rotation_vector = rotation_vector.squeeze()
#     filter_points4d = []
#     # bev_count = np.zeros((h, w, c, 2)).astype(np.int32)
#     bev_count = dict([(id, []) for id in range(4)])
#     bev_image = np.zeros((h, w, 3)).astype(np.uint8)
#     ids_max = [0, 0, 0, 0]
#     count = 0
#     for points4d in frame_points:
#         count += 1
#         if count % 5 != 1 and count > 5:
#             continue
#         for point4d in points4d:
#             x = point4d[0] - pose[0][0]
#             y = point4d[1] - pose[0][1]
#             z = point4d[2] - pose[0][2]
#             point_utm = np.array([x, y, z])
#             point_imu = np.matmul(np.array(np.matrix(rotation).I), point_utm.T)
#             point_imu = point_imu.squeeze()
#             x = -int(point_imu[0] * 5) + (h - h_offset) / 2 + h_offset
#             y = -int(point_imu[1] * 5) + (w - w_offset) / 2 + w_offset
#             if x >= 0 and x < h and y >= 0 and y < w:
#                 if point4d[3] > 0 and point4d[3] < 3:
#                     bev_image[x, y, :] = POINT_COLORS[point4d[3]]
#
#     #         x_utm = -(point4d[1] - pose[0][1])* 10 + (h - h_offset) / 2 + h_offset
#     #         y_utm = (point4d[0] - pose[0][0]) * 10 + (w - w_offset) / 2 + w_offset
#     #         if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
#     #             # flag = 0
#     #             # for x in range(len(bev_count[point4d[3]])):
#     #             #     if bev_count[point4d[3]][x][0] > x_utm:
#     #             #         bev_count[point4d[3]].insert(x, [x_utm, y_utm])
#     #             #         flag = 1
#     #             #         break
#     #             # if flag == 0:
#                 if x > ids_max[point4d[3]]:
#                     ids_max[point4d[3]] = x
#                 bev_count[point4d[3]].append([x, y])
#                 # bev_image[int(x), int(y), :] = POINT_COLORS[point4d[3]]
#     for i in range(len(bev_count)):
#         if len(bev_count[i]) != 0 and i > 0 and i < 3:
#             fitline = Ransac()
#             fitline.ransac(bev_count[i])
#             for x in range(ids_max[point4d[3]]):
#                 y = 0
#                 for n in range(len(fitline.params)):
#                     y += fitline.params[n] * math.pow(x, n)
#                 # y = (y - (w - w_offset) / 2 - w_offset)*4 + (w - w_offset) / 2 + w_offset
#                 bev_image[x, int(y), :] = POINT_COLORS[i]
#             # p1 = (int(data_y[0]), int(data_x[0]))
#             # p2 = (int(data_y[-1]), int(data_x[-1]))
#             # cv2.line(bev_image, p1, p2, color=POINT_COLORS[i], thickness=1)
#     bev_image = cv2.arrowedLine(bev_image, ((w - w_offset) / 2 + w_offset, (h - h_offset) / 2 + h_offset), (
#         int(rotation_vector[1] * 20) + (w - w_offset) / 2 + w_offset,
#         -int(rotation_vector[0] * 20) + (h - h_offset) / 2 + h_offset), (255, 0, 255), 2, 8, 0, 0.3)
#     return bev_image, filter_points4d

def gen_bev_frame(frame_points, pose, Tr_imu_to_world, height, width):
    h = height
    w = width
    c = 100
    h_offset = 0
    w_offset = 0
    # bev_ids = np.zeros((h, w, 3)).astype(np.uint8)
    # bev_count = np.zeros((h, w, 3)).astype(np.uint8)
    # bev_image = np.zeros((h, w, 3)).astype(np.uint8)
    # bev_image_rotated = np.zeros((h, w, 3)).astype(np.uint8)
    rotation_vector = np.array([[1., 0., 0.]])
    rotation = Tr_imu_to_world[0:3, 0:3]
    rotation_vector = np.matmul(rotation, rotation_vector.T)
    rotation_vector = rotation_vector.squeeze()

    # bev_count = np.zeros((h, w, c, 2)).astype(np.int32)
    bev_count = dict([(x, []) for x in range(h)])
    bev_image = np.zeros((h, w, 3)).astype(np.uint8)
    count = 0
    for points4d in frame_points:
        # count += 1
        # if count % 2 == 0 or count < 20:
        #     continue
        for point4d in points4d:
            x = point4d[0] - pose[0][0]
            y = point4d[1] - pose[0][1]
            z = point4d[2] - pose[0][2]
            point_utm = np.array([x, y, z])
            x_utm = int(-y * 10) + (h - h_offset) / 2 + h_offset
            y_utm = int(x * 10) + (w - w_offset) / 2 + w_offset
            if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
                bev_image[x_utm, y_utm, :] = POINT_COLORS[point4d[3]]
            point_imu = np.matmul(np.array(np.matrix(rotation).I), point_utm.T)
            point_imu = point_imu.squeeze()
            x = int(point_imu[0] * 10) + h/2
            y = int(point_imu[1] * 10) + w/2
            z = int(point_imu[2] * 10) + c/2
            if x >= 0 and x < h and y >= 0 and y < w:
                # bev_count[x, y, z, 0] += 1
                # bev_count[x, y, z, 1] = int(point4d[3]+1)
                flag = 0
                for i in range(len(bev_count[x])):
                    info = bev_count[x][i]
                    if [y, z, point4d[3]] == [info[0], info[1], info[2]]:
                        flag = 1
                        bev_count[x][i][3] += 1
                if flag == 0:
                    bev_count[x].append([y, z, point4d[3], 1])
    filter_points4d = []
    for x in range(h):
        ids_max = [0,0,0,0]
        ids_point = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
        for info in bev_count[x]:
            if ids_max[info[2]] < info[3]:
                ids_max[info[2]] = info[3]
                ids_point[info[2]] = (info[0], info[1])
        for i in range(len(ids_point)):
            filter_points4d.append([pose[0][0], pose[0][1], pose[0][2], i])
            if ids_point[i] != (-1,-1) and abs(ids_point[2][0]-ids_point[i][0]) >= abs(2-i)* 28 and abs(ids_point[2][0]-ids_point[i][0]) <= abs(2-i)* 48:
                filter_imu_point = np.array([[float(x-h/2)/10, float(ids_point[i][0]-w/2)/10, float(ids_point[i][1]-c/2)/10]])
                filter_utm_point = np.matmul(rotation, filter_imu_point.T)
                filter_utm_point = filter_utm_point.squeeze()
                x_utm = int(-filter_utm_point[1] * 10) + (h-h_offset) /2 + h_offset
                y_utm = int(filter_utm_point[0] * 10) + (w-w_offset) / 2 + w_offset
                # if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
                    # bev_image[x_utm, y_utm, :] = POINT_COLORS[i]
                filter_points4d.append([filter_utm_point[0] + pose[0][0], filter_utm_point[1] + pose[0][1], filter_utm_point[2] + pose[0][2], i])
        if ids_point[1] != (-1,-1) or ids_point[2] != (-1,-1):
            if ids_point[1] == (-1,-1):
                ids_point[1] = (ids_point[2][0]+40, ids_point[2][1])
            if ids_point[2] == (-1, -1):
                ids_point[2] = (ids_point[1][0] - 40, ids_point[1][1])
            ego_lane_point =  np.array([[float(x-h/2)/10, float(ids_point[1][0]+ids_point[2][0]-w)/2/10, float(ids_point[1][1]+ids_point[2][1]-c)/2/10]])
            ego_lane_utm_point = np.matmul(rotation, ego_lane_point.T)
            ego_lane_utm_point = ego_lane_utm_point.squeeze()
            x_utm = int(-ego_lane_utm_point[1] * 10) + (h - h_offset) / 2 + h_offset
            y_utm = int(ego_lane_utm_point[0] * 10) + (w - w_offset) / 2 + w_offset
            # if x_utm >= 0 and x_utm < h and y_utm >= 0 and y_utm < w:
                # bev_image[x_utm, y_utm, :] = POINT_COLORS[5]
            filter_points4d.append([ego_lane_utm_point[0] + pose[0][0], ego_lane_utm_point[1] + pose[0][1], ego_lane_utm_point[2] + pose[0][2], 5])
                # if x >= 0 and x < h and y >= 0 and y < w:
                #     bev_count[x, y, 0] += 1
                #     bev_count[x, y, 1] = int(point4d[3]+1)

    # for points4d in frame_points:
    #     for point4d in points4d:
            # x = int((pose[0][1] - point4d[1]) * 10) + (h-h_offset) /2 + h_offset
            # y = -int((pose[0][0] - point4d[0]) * 10) + (w-w_offset) / 2 + w_offset
            # if x >= 0 and x < h and y >= 0 and y < w:
            #     bev_count[x, y, 0] += 1
            #     bev_count[x, y, 1] = int(point4d[3]+1)
            #     bev_ids[x, y, 0] = int(point4d[3]+1)*40
            #     bev_ids[x, y, 1] += 1


    # P = math.degrees(math.tanh(rotation_vector[0]/rotation_vector[1]))
    # M = cv2.getRotationMatrix2D(((w-w_offset) / 2 + w_offset, (h-h_offset) / 2 + h_offset), P, 1)
    # bev_count_rotated = cv2.warpAffine(bev_count, M, (w, h))
    # bev_ids_rotated = cv2.warpAffine(bev_ids, M, (w, h))
    # bev_merge = np.concatenate([bev_ids, bev_ids_rotated], axis=1)
    # cv2.imshow("bev_ids", bev_merge)
    # cv2.waitKey(50)
    #
    # for x in range(h):
    #     ids_max = [0,0,0,0]
    #     ids_point = [-1, -1, -1, -1]
    #     for y in range(w):
    #         if ids_max[bev_count_rotated[x, y, 1]-1] < bev_count_rotated[x, y, 0]:
    #             ids_max[bev_count_rotated[x, y, 1]-1] = bev_count_rotated[x, y, 0]
    #             ids_point[bev_count_rotated[x, y, 1]-1] = y
    #     for i in range(len(ids_point)):
    #         if ids_point[i] != -1:
    #             bev_image_rotated[x, ids_point[i], :] = POINT_COLORS[i]
    # M = cv2.getRotationMatrix2D(((w - w_offset) / 2 + w_offset, (h - h_offset) / 2 + h_offset), -P, 1)
    # bev_image = cv2.warpAffine(bev_image_rotated, M, (w, h))
    bev_image = cv2.arrowedLine(bev_image, ((w - w_offset) / 2 + w_offset, (h - h_offset) / 2 + h_offset), (
    int(rotation_vector[0] * 20) + (w - w_offset) / 2 + w_offset,
    -int(rotation_vector[1] * 20) + (h - h_offset) / 2 + h_offset), (255, 0, 255), 2, 8, 0, 0.3)
    return bev_image, filter_points4d

def msg_loop(output_dir, rate, frame_limit, topics, velo_topics, cam_topics,
              odom_topics, msg_it, offset, model, car_name, date):
    global frame_points
    start_time = None
    last_frame = None
    frame_number = None
    index = 0
    for m in msg_it:
        if start_time is None:
            start_time = msg_time(m[topics[0]][0])
        frame_number = int(((msg_time(m[topics[0]][0]) - start_time).to_sec() + (rate / 2.0)) / rate) + offset
        if last_frame == frame_number:
            continue
        sys.stdout.flush()

        left_orig_image, right_orig_image, left_timestamp, right_timestamp = msg_to_png(m, cam_topics)
        cv2.imwrite(os.path.join(output_dir, "#usb_cam_left#image_raw#compressed_%.3f.png" % (left_timestamp)), left_orig_image)
        # velo = msg_to_velo(m, velo_topics)
        # left_image, right_image, P1, P2, Tr_cam_to_imu, Tr_lidar_to_imu = unwarp(left_orig_image, right_orig_image, car_name, date)
        # merged = np.concatenate([left_image, right_image], axis=1)
        # cv2.imwrite(os.path.join(output_dir, "%.3f_unwarp.png" % (left_timestamp)), merged)
        #
        # result_cls_left, result_ego_left, disp_left, lines_left, latency_left = infer_model(model, left_image)
        # result_cls_right, result_ego_right, disp_right, lines_right, latency_right = infer_model(model, right_image)
        #
        # h, w = result_ego_left.shape
        # result_cls_left_color = np.zeros((h, w, 3)).astype(np.uint8)
        # result_cls_right_color = np.zeros((h, w, 3)).astype(np.uint8)
        # result_ego_left_color = np.zeros((h, w, 3)).astype(np.uint8)
        # result_ego_right_color = np.zeros((h, w, 3)).astype(np.uint8)
        # for i in range(h):
        #     for j in range(w):
        #         if result_cls_left[i, j] != 0:
        #             result_cls_left_color[i, j, :] = cfg.EG0_POINT_COLORS[result_cls_left[i, j]-1]
        #         if result_cls_right[i, j] != 0:
        #             result_cls_right_color[i, j, :] = cfg.EG0_POINT_COLORS[result_cls_right[i, j]]
        #         if result_ego_left[i, j] != 0:
        #             result_ego_left_color[i, j, :] = cfg.EG0_POINT_COLORS[result_ego_left[i, j]]
        #         if result_ego_right[i, j] != 0:
        #             result_ego_right_color[i, j, :] = cfg.EG0_POINT_COLORS[result_ego_right[i, j]]
        #
        # result_cls_merged = np.concatenate([result_cls_left, result_cls_right], axis=1)
        # result_ego_merged = np.concatenate([result_ego_left, result_ego_right], axis=1)
        # result_cls_color_merged = np.concatenate([result_cls_left_color, result_cls_right_color], axis=1)
        # result_ego_color_merged = np.concatenate([result_ego_left_color, result_ego_right_color], axis=1)
        # disp_merged = np.concatenate([disp_left, disp_right], axis=1)
        #
        # cv2.imwrite(os.path.join(output_dir, "%.3f_cls.png" % (left_timestamp)), result_cls_color_merged * 0.5 + merged * 0.5)
        # cv2.imwrite(os.path.join(output_dir, "%.3f_ego.png" % (left_timestamp)), result_ego_color_merged * 0.5 + merged * 0.5)
        # cv2.imwrite(os.path.join(output_dir, "%.3f_points.png" % (left_timestamp)), disp_merged)
        # Tr_imu_to_world, pose = msg_to_odom(m, odom_topics, left_timestamp)
        # Tr_cam_to_world = np.matmul(Tr_imu_to_world, Tr_cam_to_imu)
        # Tr_velo_to_world = np.matmul(Tr_imu_to_world, Tr_lidar_to_imu)
        #
        # points3d = []
        # colors = []
        # max_depth = 0
        # o = open(os.path.join(output_dir, "%.3f_det.pcd" % (left_timestamp)), "w")
        # for lid in range(len(lines_left)):
        #     points3d.append([0, 0, 0, lid])
        # lid_left = lid_right = 0
        # while lid_left < len(lines_left) and lid_right < len(lines_right):
        #     # if len(lines_left[lid_left]) == 0 and len(lines_right[lid_right]) > 0 and lid_left == 0 and lid_right == 0:
        #     #     lid_left += 1
        #     #     continue
        #
        #     if lid_left < 4 and lid_right < 4:
        #         line_left = lines_left[lid_left]
        #         line_right = lines_right[lid_right]
        #         xl = []
        #         yl = []
        #         xr = []
        #         yr = []
        #         if len(line_left) == len(line_right):
        #             for j in range(len(line_left) - 1):
        #                 # scale = (len(line_left) - j) / 5
        #                 scale = 1
        #                 point_left = line_left[j]
        #                 point_right = line_right[j]
        #                 point_left_temp = line_left[j + 1]
        #                 point_right_temp = line_right[j + 1]
        #                 if point_left[0] != -1 and point_right[0] != -1 and point_left_temp[0] != -1 and \
        #                         point_right_temp[0] != -1:
        #                     for i in range(scale):
        #                         xl.append(point_left[0] + i * float(point_left_temp[0] - point_left[0]) / scale)
        #                         yl.append(point_left[1] + i * float(point_left_temp[1] - point_left[1]) / scale)
        #                         xr.append(point_right[0] + i * float(point_right_temp[0] - point_right[0]) / scale)
        #                         yr.append(point_right[1] + i * float(point_right_temp[1] - point_right[1]) / scale)
        #             if len(xl) == 0 or len(xr) == 0:
        #                 lid_left += 1
        #                 lid_right += 1
        #                 continue
        #             ptl = np.concatenate(
        #                 (np.asarray(xl, dtype=np.float).reshape(1, -1), np.asarray(yl, dtype=np.float).reshape(1, -1)),
        #                 axis=0)
        #             ptr = np.concatenate(
        #                 (np.asarray(xr, dtype=np.float).reshape(1, -1), np.asarray(yr, dtype=np.float).reshape(1, -1)),
        #                 axis=0)
        #             points4d = cv2.triangulatePoints(P1, P2, ptl, ptr)
        #             for j in range(points4d.shape[1]):
        #                 depth = points4d[2][j] / points4d[3][j]
        #                 height = points4d[1][j] / points4d[3][j]
        #                 width = points4d[0][j] / points4d[3][j]
        #                 if depth > 0 and depth < 150:
        #                     if max_depth < depth:
        #                         max_depth = depth
        #                     point4d = np.asarray([[width, height, depth, 1.0]], dtype=np.float)
        #                     world_point4d = np.matmul(Tr_cam_to_world, point4d.T)
        #                     world_point4d = world_point4d.squeeze()
        #                     # points3d.append([points4d[0][j] / points4d[3][j], points4d[1][j] / points4d[3][j],
        #                     #                  points4d[2][j] / points4d[3][j], lid])
        #                     points3d.append([world_point4d[0], world_point4d[1], world_point4d[2], lid_left])
        #                     # print lid+1, [xl[j], yl[j]], [xr[j], yr[j]], [points4d[0][j] / points4d[3][j], points4d[1][j] / points4d[3][j], points4d[2][j] / points4d[3][j]]
        #                     # colors.append([POINT_COLORS[lid][0], POINT_COLORS[lid][1], POINT_COLORS[lid][2]])
        #             # if max_depth <= 50 and lid_left == 3 and lid_right == 3:
        #             #     lid_left = 1
        #             #     lid_right = 0
        #             #     continue
        #     lid_left += 1
        #     lid_right += 1
        # # pose = []
        # # points3d = []
        # header = """VERSION 0.7
        #     FIELDS x y z intensity
        #     SIZE 4 4 4 4
        #     TYPE F F F F
        #     COUNT 1 1 1 1
        #     WIDTH %d
        #     HEIGHT 1
        #     VIEWPOINT 0 0 0 1 0 0 0
        #     POINTS %d
        #     DATA ascii
        #     """ % (len(points3d)+len(pose)+len(velo), len(points3d)+len(pose)+len(velo))
        # o.writelines(header)
        # for j in range(len(points3d)):
        #     o.write("%f %f %f %f\n" % (points3d[j][0], points3d[j][1], points3d[j][2], points3d[j][3]*30+30))
        # for j in range(len(pose)):
        #     o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], pose[j][3]))
        # for j in range(len(velo)):
        #     point4d = np.asarray([[velo[j][0], velo[j][1], velo[j][2], 1.0]], dtype=np.float)
        #     world_point4d = np.matmul(Tr_velo_to_world, point4d.T)
        #     world_point4d = world_point4d.squeeze()
        #     o.write("%f %f %f %f\n" % (world_point4d[0], world_point4d[1], world_point4d[2], velo[j][3]))
        # o.close()
        #
        # # frame_points.append(points3d)
        # # if len(frame_points) > 25:
        # #     frame_points.pop(0)
        # # height= disp_left.shape[0]*2
        # # bev_image, filter_points4d = gen_bev_frame(frame_points, pose, Tr_imu_to_world, height, height)
        # # disp_merged = np.concatenate([disp_left, disp_right], axis=0)
        # # bev_image = np.concatenate([disp_merged, bev_image], axis=1)
        # # cv2.imwrite(os.path.join(output_dir, "%.3f_bev.png" % (left_timestamp)), bev_image)
        # # o = open(os.path.join(output_dir, "%.3f_filter.pcd" % (left_timestamp)), "w")
        # # header = """VERSION 0.7
        # #             FIELDS x y z intensity
        # #             SIZE 4 4 4 4
        # #             TYPE F F F F
        # #             COUNT 1 1 1 1
        # #             WIDTH %d
        # #             HEIGHT 1
        # #             VIEWPOINT 0 0 0 1 0 0 0
        # #             POINTS %d
        # #             DATA ascii
        # #             """ % (len(filter_points4d) + len(pose), len(filter_points4d) + len(pose))
        # # o.writelines(header)
        # # for j in range(len(filter_points4d)):
        # #     o.write("%f %f %f %f\n" % (filter_points4d[j][0], filter_points4d[j][1], filter_points4d[j][2], filter_points4d[j][3]))
        # # for j in range(len(pose)):
        # #     o.write("%f %f %f %f\n" % (pose[j][0], pose[j][1], pose[j][2], pose[j][3]))
        # # o.close()
        #
        #
        # latency = latency_left + latency_right
        # print("timestamp: %.3f, latency_left: %.3f s, latency_right: %.3f s, latency_tot: %.3f s, distance: %.3f m" % (float(left_timestamp), latency_left, latency_right, latency, max_depth))
        #


        index = index + 1
        last_frame = frame_number

        if frame_limit > 0 and frame_number >= frame_limit:
            sys.stdout.flush()
            exit(0)

    return frame_number

def main():
    global args
    args = parser.parse_args()
    cfg.NUM_CLASSES = 3
    cfg.INPUT_MEAN = [103.939, 116.779, 123.68]
    cfg.INPUT_STD = [1., 1., 1.]

    if not os.path.isdir(args.out):
        os.mkdir(args.out)

    topics = []
    velo_topics = []

    odom_topics = []
    if args.odom_topics != None:
        for t in args.odom_topics.split(","):
            odom_topics.append(t)
            topics.append(t)

    cam_topics = []
    if args.cam_topics != None:
        for t in args.cam_topics.split(","):
            cam_topics.append(t)
            topics.append(t)

    if args.velo_topics != None:
        for t in args.velo_topics.split(","):
            velo_topics.append(t)
            topics.append(t)

    # model = load_model()
    model=[]
    offset = 0
    for b in args.bags.split(","):
        print("start bag", b)
        bag_name = b.split('/')[-1]
        car_name = bag_name.split('_')[1]
        date = int(bag_name.split('_')[0].split('T')[0])
        sys.stdout.flush()
        bag = rosbag.Bag(b)
        msg_it = iter(buffered_message_generator(bag, args.tolerance, topics, odom_topics))
        offset = msg_loop(args.out, args.rate, args.frame_limit, topics, velo_topics, cam_topics, odom_topics,
                          msg_it, offset, model, car_name, date)

if __name__ == '__main__':
    main()

