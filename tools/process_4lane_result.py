#!/usr/bin/python

import cv2
import numpy as np
import json
import math
import os
import codecs
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label_list', type=str, default="/home/julian/data/lane_batch/L4E/json/json.list", help='the labeled file list')
parser.add_argument('--image_folder', type=str, default="/home/julian/data/lane_batch/L4E/", help='the root folder of the labeled images')
args = parser.parse_args()

cls_map = {
    ('solid', 'w'): 1,
    ('dashed', 'w'): 2,
    ('shoulder'): 3,
    ('cones'): 4,
    ('solid', 'y'): 5,
    ('dashed', 'y'): 6,
    ('seam'): 7,
    ('white', 's'): 1,
    ('white', 'd'): 2,
    ('yellow', 's'): 5,
    ('yellow', 'd'): 6,
    ('solid', True): 1,
    ('dashed', True): 2
}

color_map = {
    0 : (0, 0, 0),
    1 : (0, 0, 255),
    2 : (0, 255, 0),
    3 : (255, 0, 0),
    4 : (255, 255, 0)
}

def cls_mapping(line):
    try:
        if line['type'] == 'solid' or line['type'] == 'dashed':
            try:
                cls = cls_map[line['type'], line['color']]
            except:
                cls = cls_map[line['type'], line['seam']]
        elif line['type'] == 'yellow' or line['type'] == 'white':
            cls = cls_map[line['type'], line['cls']]
        else:
            cls = cls_map[line['type']]
    except:
        print("Error: type: " + line['type'])
        cls = 0
    return cls

def process_label(lanes, fpath):
    img = cv2.imread(fpath)
    h = img.shape[0]
    w = img.shape[1]
    disp_img = np.copy(img)
    label_img = np.zeros_like(img[:, :, 0])

    cls_list = []

    if len(lanes) == 0:
        return label_img, disp_img, cls_list
    for key in lanes:
        ins = int(key.split('_')[0])
        if ins > 3:
            return label_img, disp_img, cls_list

    for i in range(1, 5):
        cls_list.append(0)


    for key in lanes:
        line = lanes[key]
        cls = cls_mapping(line)
        if cls == 0:
            continue
        points = line['dots']
        if len(points) > 1:
            ins = int(key.split('_')[0])
            cls_list[ins] = cls
            for j in range(len(points)):
                if points[j][0] < 0:
                    points[j][0] = 0
                if points[j][0] > w-1:
                    points[j][0] = w-1
                if points[j][1] < 0:
                    points[j][1] = 0
                if points[j][1] > h-1:
                    points[j][1] = h-1
            for j in range(len(points) - 1):
                p1 = (int(points[j][0]), int(points[j][1]))
                p2 = (int(points[j + 1][0]), int(points[j + 1][1]))
                cv2.line(label_img, p1, p2, color=ins+1, thickness=3)
                cv2.line(disp_img, p1, p2, color=color_map[ins+1], thickness=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
    label_img = cv2.dilate(label_img, kernel, iterations=1)

    # for i in range(h):
    #     for j in range(w):
    #         disp_img[i, j, :] = color_map[label_img[i, j]]
    # disp_img = img*0.5+disp_img*0.5
    return label_img, disp_img,  cls_list

def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

def main():
    for i, line in enumerate(open(args.label_list)):
        label_file = codecs.open(line.split('.json')[-2] + '.json', 'r', encoding="utf-8").read()
        load_dict = json.loads(label_file)
        for obj in load_dict['labeling'][:]:
            try:
                fpath = obj['fileName']
            except:
                fpath = obj['filename']
            lanes = obj['lanes']
            # if lanes == {}:
            #     continue
            task_name = fpath.split('/')[-3]
            bag_name = fpath.split('/')[-2]
            image_name = fpath.split('/')[-1].replace('%23', '#')
            fpath = os.path.join(args.image_folder, task_name, 'sample', bag_name, image_name)

            if not os.path.exists(fpath):
                print("%s dosen't exist."% (fpath))
                continue

            outdirs = os.path.join(args.image_folder, task_name, '4lane')


            if fpath.find('.bag') != -1:
                label_img, disp_img, cls_list = process_label(lanes, i)
                if len(cls_list) == 0 and lanes != {}:
                    print(fpath, obj['ids'])
                else:
                    # if cls_list[1] !=0 and cls_list[2] != 0:
                    mkdir(os.path.join(outdirs, 'display', bag_name))
                    mkdir(os.path.join(outdirs, 'seglabel', bag_name))
                    disp_fn = os.path.join(outdirs, 'display', bag_name, image_name)
                    seglabel_fn = os.path.join(outdirs, 'seglabel', bag_name, image_name)
                    cls_fn = os.path.join(outdirs, 'seglabel', bag_name, image_name.replace('.png', '.txt'))

                    cv2.imwrite(disp_fn, disp_img)
                    cv2.imwrite(seglabel_fn, label_img)
                    f = open(cls_fn, 'w')
                    for cls in cls_list:
                        f.write(str(cls) + ' ')
                    f.close()

if __name__ == '__main__':
    main()
