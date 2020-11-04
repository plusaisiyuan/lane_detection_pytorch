#!/usr/bin/env python

import sys
import rosbag

from fastbag.readers import Reader
import math
import numpy as np
import sensor_msgs.point_cloud2
import argparse
import cv2
from genpy.rostime import Time
from cv_bridge import CvBridge
import os

TIME_INTERVAL_IN_SECOND = 1
GPS_INTERVAL_IN_METER = 10

bridge = CvBridge()
def msg_time(msg):
    #return msg.timestamp
    return msg.header.stamp
    
def msg_to_velo_file(topic, msg, path):
    velo = np.array([p for p in sensor_msgs.point_cloud2.read_points(msg)])
    velo = velo.astype(np.float64)
    header = """VERSION 0.7
    FIELDS x y z intensity timestamp ring
    SIZE 4 4 4 4 8 4
    TYPE F F F F F F
    COUNT 1 1 1 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    """ % (velo.shape[0], velo.shape[0])

    o = file(path, "w")
    o.write(header)
    if "inno" in topic:      #for innovation, intensity column = 4
        for p in velo:
            o.write("%f %f %f %f %f %e\n" % (p[0], p[1], p[2], p[4], p[3], p[5]))
    else:
        for p in velo:
            o.write("%f %f %f %f %f %e\n" % (p[0], p[1], p[2], p[3], p[4], p[5]))

def msg_to_png_file(msg, path, width, height):
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    if  img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    cv2.imwrite(path, img)

def msg_compressed_to_png_file(msg, path, width, height):
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    #if img.shape[0] != height or img.shape[1] != width:
    # img = cv2.resize(img, (img.shape[1]/2, img.shape[0]))
    #if img.shape[0] == 772 or img.shape[1] == 1032:
    cv2.imwrite(path, img)

def odom_msg_to_str(msg):
    msg_str = "header:\n"
    msg_str += "\tseq: %d\n" % msg.header.seq
    msg_str += "\tstamp: %f\n" % msg.header.stamp.to_sec()
    msg_str += "\tframe_id: %s\n" % msg.header.frame_id
    msg_str += "child_frame_id: %s\n" % msg.child_frame_id
    # pose
    msg_str += "pose:\n"
    msg_str += "\tpose:\n"
    msg_str += "\t\tposition:\n"
    msg_str += "\t\t\tx: %f\n" % msg.pose.pose.position.x
    msg_str += "\t\t\ty: %f\n" % msg.pose.pose.position.y
    msg_str += "\t\t\tz: %f\n" % msg.pose.pose.position.z
    msg_str += "\t\torientation:\n"
    msg_str += "\t\t\tx: %f\n" % msg.pose.pose.orientation.x
    msg_str += "\t\t\ty: %f\n" % msg.pose.pose.orientation.y
    msg_str += "\t\t\tz: %f\n" % msg.pose.pose.orientation.z
    msg_str += "\t\t\tw: %f\n" % msg.pose.pose.orientation.w
    msg_str += "\tcovariance[]\n"
    for i in range(len(msg.pose.covariance)):
        msg_str += "\t\tcovariance[%d]: %f\n" % (i, msg.pose.covariance[i])
    # twist
    msg_str += "twist:\n"
    msg_str += "\ttwist:\n"
    msg_str += "\t\tlinear:\n"
    msg_str += "\t\t\tx: %f\n" % msg.twist.twist.linear.x
    msg_str += "\t\t\ty: %f\n" % msg.twist.twist.linear.y
    msg_str += "\t\t\tz: %f\n" % msg.twist.twist.linear.z
    msg_str += "\t\tangular:\n"
    msg_str += "\t\t\tx: %f\n" % msg.twist.twist.angular.x
    msg_str += "\t\t\ty: %f\n" % msg.twist.twist.angular.y
    msg_str += "\t\t\tz: %f\n" % msg.twist.twist.angular.z
    msg_str += "\tcovariance[]\n"
    for i in range(len(msg.twist.covariance)):
        msg_str += "\t\tcovariance[%d]: %f\n" % (i, msg.twist.covariance[i])
    return msg_str

def msg_to_odom_file(msg, path):
    with open(path, 'w') as f:
        odom_str = odom_msg_to_str(msg)
        f.write(odom_str)

def radar_msg_to_str(msg):
    msg_str = "header:\n"
    msg_str += "\tseq: %d\n" % msg.header.seq
    msg_str += "\tstamp: %f\n" % msg.header.stamp.to_sec()
    msg_str += "\tframe_id: %s\n" % msg.header.frame_id
    msg_str += "tracks[]\n"
    for i in range(len(msg.tracks)):
        msg_str += "\ttracks[%d]:\n" % i
        msg_str += "\t\ttrack_id: %d\n" % msg.tracks[i].track_id
        msg_str += "\t\ttrack_shape:\n"
        msg_str += "\t\t\tpoints[]\n"
        for j in range(len(msg.tracks[i].track_shape.points)):
            msg_str += "\t\t\t\tpoints[%d]:\n" % j
            msg_str += "\t\t\t\t\tx: %f\n" % msg.tracks[i].track_shape.points[j].x
            msg_str += "\t\t\t\t\ty: %f\n" % msg.tracks[i].track_shape.points[j].y
            msg_str += "\t\t\t\t\tz: %f\n" % msg.tracks[i].track_shape.points[j].z
        msg_str += "\t\tlinear_velocity:\n"
        msg_str += "\t\t\tx: %f\n" % msg.tracks[i].linear_velocity.x
        msg_str += "\t\t\ty: %f\n" % msg.tracks[i].linear_velocity.y
        msg_str += "\t\t\tz: %f\n" % msg.tracks[i].linear_velocity.z

        msg_str += "\t\tlinear_acceleration:\n"
        msg_str += "\t\t\tx: %f\n" % msg.tracks[i].linear_acceleration.x
        msg_str += "\t\t\ty: %f\n" % msg.tracks[i].linear_acceleration.y
        msg_str += "\t\t\tz: %f\n" % msg.tracks[i].linear_acceleration.z 

    return msg_str

def msg_to_radar_file(msg, path):
    with open(path, 'w') as f:
        msg_str = radar_msg_to_str(msg)
        f.write(msg_str)

def buffered_message_generator(bag, tolerance, topics):
    buffers = dict([(t, []) for t in topics])
    skipcounts = dict([(t, 0) for t in topics])
    for msg in bag.read_messages(topics=topics):
        if msg.topic in topics:
            buffers[msg.topic].append(msg)
        else:
            continue
        while all(buffers.values()):
            time_and_bufs = sorted([(msg_time(b[0].message).to_sec(), b) for b in buffers.values()])
            if time_and_bufs[-1][0] - time_and_bufs[0][0] > tolerance:
                old_msg = time_and_bufs[0][1].pop(0)
                skipcounts[old_msg.topic] += 1
                continue
            msg_set = {}
            for topic, buf in buffers.iteritems():
                m = buf.pop(0).message
                msg_set[topic] = m
            yield msg_set
    for t, c in skipcounts.iteritems():
        print "skipped %d %s messages" % (c, t)
    sys.stdout.flush()

def distance(odom1, odom2):
    x1 = odom1.pose.pose.position.x
    y1 = odom1.pose.pose.position.y
    x2 = odom2.pose.pose.position.x
    y2 = odom2.pose.pose.position.y
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

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

def msg_loop(output_dir, rate, frame_limit, velo_topics, cam_topics, odom_topics, radar_topics, msg_it, width, height, offset):

    start_time = None
    last_frame = None
    timestamp_file = None
    frame_number = None
    index = 0

    for m in msg_it:
        if start_time is None:
            start_time = msg_time(m[topics[0]])
            sec = start_time.secs
            nsec = start_time.nsecs / (10 ** 6)
            frame_secs = sec * 1000 + nsec
            print(frame_secs)
        frame_number = int(((msg_time(m[topics[0]]) - start_time).to_sec() + (rate / 2.0)) / rate) + offset
        if last_frame == frame_number:
            continue
        sys.stdout.flush()
        sec = msg_time(m[topics[0]]).secs
        nsec = msg_time(m[topics[0]]).nsecs/(10**6)
        timestamp = '%d.%03d' % (sec, nsec)
		
        for topic in m.keys():
            file_name = ""
            if topic in velo_topics:
                #if is_time_ok and is_gps_ok:
                file_name = "%s/%s_%s.pcd" % (output_dir, timestamp, topic.split('/')[1])
                msg_to_velo_file(topic, m[topic], file_name)

            elif topic in cam_topics:
                #if is_time_ok and is_gps_ok:
                file_name = "%s/#%s#%s#%s_%.3f.png" % (output_dir, topic.split('/')[1], topic.split('/')[2], topic.split('/')[3], m[topic].header.stamp.to_sec())
                if 'compressed' in topic:
                    msg_compressed_to_png_file(m[topic], file_name, width, height)
                else:
                    msg_to_png_file(m[topic], file_name, width, height)
	    
            elif topic in odom_topics:
                file_name = "%s/%s_%s.txt" % (output_dir, timestamp, topic.split('/')[1])
                #msg_to_odom_file(m[topic], file_name)
                last_odom = m[topic]

            elif topic in radar_topics:
                file_name = "%s/%s_%s.txt" % (output_dir, timestamp, topic.split('/')[1])
                msg_to_radar_file(m[topic], file_name)

            if timestamp_file is None:
                timestamp_file = open("%s/timestamp.txt" % output_dir, 'a')
            
            timestamp_file.write("%s\t%.3f\n" % (file_name, m[topic].header.stamp.to_sec()))

        index = index + 1
        last_frame = frame_number
        if frame_limit > 0 and frame_number >= frame_limit:
            print("reach frame limit: %d, quit" % (frame_limit))
            sys.stdout.flush()
            exit(0)

    return frame_number

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags", type=str, help="path to bags (comma separated)")
    parser.add_argument("--out", type=str, help="output path (a directory)")
    parser.add_argument("--rate", type=float, help="desired sample rate in seconds", default=0.1)
    parser.add_argument("--tolerance", type=float, help="tolerance", default=0.05)
    parser.add_argument("--velo_topics", type=str, help="velodyne topic (comma separated, don't add space between topics)")
    parser.add_argument("--cam_topics", type=str, help="camera topics (comma separated, don't add space between topics)")
    parser.add_argument("--odom_topics", type=str, help="odometry topic (comma separated, don't add space between topics)")
    parser.add_argument("--radar_topics", type=str, help="radar topic (comma separated, don't add space between topics)")
    parser.add_argument("--frame_limit", type=int, help="frame limit if > 0", default=0)
    parser.add_argument("--width", type=int, help="full resolution image width", default=1032)
    parser.add_argument("--height", type=int, help="full resolution image height", default=772)

    args = parser.parse_args()
    if not os.path.isdir(args.out):
        os.mkdir(args.out)

    topics = []
    velo_topics = []
    if args.velo_topics != None:
        for t in args.velo_topics.split(","):
            velo_topics.append(t)
            topics.append(t)

    cam_topics = []
    if args.cam_topics != None:
        for t in args.cam_topics.split(","):
            cam_topics.append(t)
            topics.append(t)

    odom_topics = []
    if args.odom_topics != None:
        for t in args.odom_topics.split(","):
            odom_topics.append(t)
            topics.append(t)

    radar_topics = []
    if args.radar_topics != None:
        for t in args.radar_topics.split(","):
            radar_topics.append(t)
            topics.append(t)

    offset = 0
    for b in args.bags.split(","):
        print("start bag", b)
        sys.stdout.flush()
        if b.endswith('bag'):
            bag = rosbag.Bag(b)
        else:
            bag = Reader(b)
        msg_it = iter(buffered_message_generator(bag, args.tolerance, topics))
        offset = msg_loop(args.out, args.rate, args.frame_limit, velo_topics, cam_topics, odom_topics, radar_topics, msg_it, args.width, args.height, offset)
