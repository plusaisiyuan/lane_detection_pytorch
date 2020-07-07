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
last_lane_percentage = 0

def msg_time(msg):
    # return msg.timestamp
    return msg.header.stamp


def msg_to_png(msgs, topics):
    for topic in topics:
        if 'left' in topic:
            left_img = bridge.compressed_imgmsg_to_cv2(msgs[topic][0], desired_encoding="bgr8")
            left_timestamp = msg_time(msgs[topic][0]).to_sec()
        else:
            exit(0)
    return left_img, left_timestamp


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


def reportLaneEvents(lane_result):
    global last_lane_percentage
    def getLanePercentage(cls_result):
        h, w = cls_result.shape
        lane_pixel_count = 0
        for i in range(0, h, 2):
            lane_each_row_count = 0
            first_lane_loc_x = 0
            for j in range(0, w, 2):
                if lane_each_row_count == 2:
                    break
                if cls_result[i, j]:
                    if lane_each_row_count == 0:
                        lane_each_row_count += 1
                        first_lane_loc_x = j
                    elif lane_each_row_count == 1:
                        offset = j - first_lane_loc_x
                        if offset > _param.lane_event_param().lane_offset():
                            lane_each_row_count += 1
        lane_pixel_count += lane_each_row_count

        lane_percentage = lane_pixel_count / h
        return lane_percentage

    lane_percentage = getLanePercentage(cls_result)
    recorded_topics = ""
    if lane_percentage < 0.1:
        recorded_topics = "lane_no_detection"
    elif lane_percentage < 0.25:
        recorded_topics = "lane_few_detection"
    if last_lane_percentage - lane_percentage > 0.4:
        recorded_topics = "lane_abnormal_detection"
    last_lane_percentage = lane_percentage
    return recorded_topics

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

        left_orig_image, left_timestamp = msg_to_png(m, cam_topics)

        cls_result = infer_model(model, left_orig_image)
        recorded_topics = reportLaneEvents(cls_result)

        if len(recorded_topics):
            cv2.imwrite(s.path.join(output_dir, recorded_topics, "#usb_cam_left#image_raw#compressed_%.3f.png" % (left_timestamp)), left_orig_image)

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

    model = load_model()
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

