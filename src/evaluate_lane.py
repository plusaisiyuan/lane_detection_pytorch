#!/bin/python

#Evaluate lane marker detection

import json
import sys
import numpy as np
import glob
import cv2
import os
from functools import reduce
import argparse
import math


def process(ins_img):
    def get_centroid(ary):
        ret = []
        seg = []
        for i in ary:
            if len(seg) == 0 or seg[-1] + 1 == i:
                seg.append(i)
            else:
                ret.append(seg[len(seg)//2])
                seg = [i]
        if len(seg) != 0:
            ret.append(seg[len(seg)//2])
        return ret

    h, w = ins_img.shape
    flat_ary = np.ravel(ins_img)
    ins_ids = { k:1 for k in flat_ary}.keys()
    lanes = {}
    for lid in ins_ids:
        #0 represents background, should be ignored
        if lid == 0:
            continue
        #ys and xs should be in increasing order
        ys, xs = np.where(ins_img == lid)
        ytox = {}
        for x, y in zip(xs, ys):
            ytox.setdefault(y, []).append(x)
        lane = {}
        for y in range(h):
            xs = ytox.get(y, [])
            #only use the center of consecutive pixels
            xs = get_centroid(xs)
            if len(xs) > 0:
                lane[y] = xs
        lanes[str(lid)] = lane
    ret = dict(
            lanes = lanes,
            img_size = ins_img.shape
    )
    return ret

def get_lane_points(lane_obj):
    xs = []
    ys = []
    for lid, lane in lane_obj['lanes'].items():
        for k, v in lane.items():
            y = int(k)
            for x in v:
                xs.append(x)
                ys.append(y)
    return np.array(ys, dtype=np.int32), np.array(xs, dtype=np.int32)



def evaluate_rmse(ret0, ret1, max_pixel_dis, mode='class'):
    #compare pixels within the image
    def coverage_xy(coords0, coords1, img_size):
        mask0 = np.zeros(img_size)
        lane_obj0 = dict(lanes = {'ANY_KEY': coords0})
        mask0[get_lane_points(lane_obj0)] = 1
        #count the baseline pixels
        gt_pixel_num = int(np.sum(mask0))
        #dilate this gt mask for later counting
        diameter = 2*max_pixel_dis + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter, diameter))
        mask0 = cv2.dilate(mask0, kernel, iterations = 1)

        #the compare mask
        mask1 = np.zeros(img_size)
        lane_obj1 = dict(lanes = {'ANY_KEY': coords1})
        mask1[get_lane_points(lane_obj1)] = 1
        #count the matching pixels with the dilated ground truth
        overlap = (mask0 == 1) & (mask1 == 1)
        correct_pixel_num = int(np.sum(overlap))

        sum_dis = 0

        for y, xs in coords0.items():
            comp_xs = coords1.get(y, [])
            for x in xs:
                dis = 5
                for comp_x in comp_xs:
                    if abs(x - comp_x) < dis:
                        dis = abs(x - comp_x)
                    if abs(x - comp_x) <= max_pixel_dis:
                        break
                sum_dis += dis
        if correct_pixel_num > gt_pixel_num :
            correct_pixel_num = gt_pixel_num - 1
        return dict(correct_pixel=correct_pixel_num, gt_pixel=gt_pixel_num, sum_dis=sum_dis)

    #compare pixels at the same y coordinate
    def coverage_y(coords0, coords1):
        correct_num = 0
        total_num = 0
        sum_dis = 0

        for y, xs in coords0.items():
            comp_xs = coords1.get(y, [])
            for x in xs:
                dis = 5
                for comp_x in comp_xs:
                    if abs(x - comp_x) < dis:
                        dis = abs(x - comp_x)
                    if abs(x - comp_x) <= max_pixel_dis:
                        correct_num += 1
                        break
                sum_dis += dis*dis
                total_num += 1
        return dict(correct_pixel=correct_num, gt_pixel=total_num, sum_dis=sum_dis)

    if mode == 'class':
        rets = [ret0['lanes'], ret1['lanes']]
        img_sizes = [ret0['img_size'], ret1['img_size']]
        assert img_sizes[0] == img_sizes[1]

        bi_stats = {}
        #do comparison both diretions
        for i in range(len(rets)):
            base = rets[i]
            comp = rets[1-i]
            stats = {}
            stats['1'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            stats['2'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            stats['3'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            stats['4'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            stats['5'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            stats['6'] = dict(correct_pixel=0, gt_pixel=0, sum_dis=0)
            #for each cls
            for cls, coords in base.items():
                comp_coords = comp.get(cls, {})
                stats[cls] = coverage_xy(coords, comp_coords, img_sizes[0])
                #stats[cls] = coverage_y(coords, comp_coords)
            #for all cls
            all_cvg = dict()
            for cls, cvg in stats.items():
                for k, v in cvg.items():
                    all_cvg.setdefault(k, 0)
                    all_cvg[k] += v
            stats['all'] = all_cvg
            bi_stats['%d-%d' % (i, 1-i)] = stats
        return bi_stats
    else:
        return None

def calculate(eval_ret):
    recall = {}
    precision = {}
    f1_score = {}
    rmse = {}
    metrics = []

    recall['1'] = eval_ret['0-1']['1']['correct_pixel'] / (eval_ret['0-1']['1']['gt_pixel'] + 0.0000001)
    recall['2'] = eval_ret['0-1']['2']['correct_pixel'] / (eval_ret['0-1']['2']['gt_pixel'] + 0.0000001)
    recall['3'] = eval_ret['0-1']['3']['correct_pixel'] / (eval_ret['0-1']['3']['gt_pixel'] + 0.0000001)
    recall['4'] = eval_ret['0-1']['4']['correct_pixel'] / (eval_ret['0-1']['4']['gt_pixel'] + 0.0000001)
    recall['5'] = eval_ret['0-1']['5']['correct_pixel'] / (eval_ret['0-1']['5']['gt_pixel'] + 0.0000001)
    recall['6'] = eval_ret['0-1']['6']['correct_pixel'] / (eval_ret['0-1']['6']['gt_pixel'] + 0.0000001)
    recall['all'] = eval_ret['0-1']['all']['correct_pixel'] / (eval_ret['0-1']['all']['gt_pixel'] + 0.0000001)
    metrics.append({'name': 'recall', 'category': 'white_solid', 'value': recall['1']})
    metrics.append({'name': 'recall', 'category': 'white_dash', 'value': recall['2']})
    metrics.append({'name': 'recall', 'category': 'curb', 'value': recall['3']})
    metrics.append({'name': 'recall', 'category': 'cone_line', 'value': recall['4']})
    metrics.append({'name': 'recall', 'category': 'yellow_solid', 'value': recall['5']})
    metrics.append({'name': 'recall', 'category': 'yellow_dash', 'value': recall['6']})
    metrics.append({'name': 'recall', 'category': 'all', 'value': recall['all']})

    precision['1'] = eval_ret['1-0']['1']['correct_pixel'] / (eval_ret['1-0']['1']['gt_pixel'] + 0.0000001)
    precision['2'] = eval_ret['1-0']['2']['correct_pixel'] / (eval_ret['1-0']['2']['gt_pixel'] + 0.0000001)
    precision['3'] = eval_ret['1-0']['3']['correct_pixel'] / (eval_ret['1-0']['3']['gt_pixel'] + 0.0000001)
    precision['4'] = eval_ret['1-0']['4']['correct_pixel'] / (eval_ret['1-0']['4']['gt_pixel'] + 0.0000001)
    precision['5'] = eval_ret['1-0']['5']['correct_pixel'] / (eval_ret['1-0']['5']['gt_pixel'] + 0.0000001)
    precision['6'] = eval_ret['0-1']['6']['correct_pixel'] / (eval_ret['1-0']['6']['gt_pixel'] + 0.0000001)
    precision['all'] = eval_ret['1-0']['all']['correct_pixel'] / (eval_ret['1-0']['all']['gt_pixel'] + 0.0000001)
    metrics.append({'name': 'precision', 'category': 'white_solid', 'value': precision['1']})
    metrics.append({'name': 'precision', 'category': 'white_dash', 'value': precision['2']})
    metrics.append({'name': 'precision', 'category': 'curb', 'value': precision['3']})
    metrics.append({'name': 'precision', 'category': 'cone_line', 'value': precision['4']})
    metrics.append({'name': 'precision', 'category': 'yellow_solid', 'value': precision['5']})
    metrics.append({'name': 'precision', 'category': 'yellow_dash', 'value': precision['6']})
    metrics.append({'name': 'precision', 'category': 'all', 'value': precision['all']})

    f1_score['1'] = 2 * recall['1'] * precision['1'] / (recall['1'] + precision['1'] + 0.0000001)
    f1_score['2'] = 2 * recall['2'] * precision['2'] / (recall['2'] + precision['2'] + 0.0000001)
    f1_score['3'] = 2 * recall['3'] * precision['3'] / (recall['3'] + precision['3'] + 0.0000001)
    f1_score['4'] = 2 * recall['4'] * precision['4'] / (recall['4'] + precision['4'] + 0.0000001)
    f1_score['5'] = 2 * recall['5'] * precision['5'] / (recall['5'] + precision['5'] + 0.0000001)
    f1_score['6'] = 2 * recall['6'] * precision['6'] / (recall['6'] + precision['6'] + 0.0000001)
    f1_score['all'] = 2 * recall['all'] * precision['all'] / (recall['all'] + precision['all'] + 0.0000001)
    metrics.append({'name': 'f1_score', 'category': 'white_solid', 'value': f1_score['1']})
    metrics.append({'name': 'f1_score', 'category': 'white_dash', 'value': f1_score['2']})
    metrics.append({'name': 'f1_score', 'category': 'curb', 'value': f1_score['3']})
    metrics.append({'name': 'f1_score', 'category': 'cone_line', 'value': f1_score['4']})
    metrics.append({'name': 'f1_score', 'category': 'yellow_solid', 'value': f1_score['5']})
    metrics.append({'name': 'f1_score', 'category': 'yellow_dash', 'value': f1_score['6']})
    metrics.append({'name': 'f1_score', 'category': 'all', 'value': f1_score['all']})

    rmse['1'] = math.sqrt(
        float(eval_ret['1-0']['1']['sum_dis']) / (eval_ret['1-0']['1']['gt_pixel'] + 0.0000001))
    rmse['2'] = math.sqrt(
        float(eval_ret['1-0']['2']['sum_dis']) / (eval_ret['1-0']['2']['gt_pixel'] + 0.0000001))
    rmse['3'] = math.sqrt(
        float(eval_ret['1-0']['3']['sum_dis']) / (eval_ret['1-0']['3']['gt_pixel'] + 0.0000001))
    rmse['4'] = math.sqrt(
        float(eval_ret['1-0']['4']['sum_dis']) / (eval_ret['1-0']['4']['gt_pixel'] + 0.0000001))
    rmse['5'] = math.sqrt(
        float(eval_ret['1-0']['5']['sum_dis']) / (eval_ret['1-0']['5']['gt_pixel'] + 0.0000001))
    rmse['6'] = math.sqrt(
        float(eval_ret['1-0']['6']['sum_dis']) / (eval_ret['1-0']['6']['gt_pixel'] + 0.0000001))
    rmse['all'] = math.sqrt(
        float(eval_ret['1-0']['all']['sum_dis']) / (eval_ret['1-0']['all']['gt_pixel'] + 0.0000001))
    metrics.append({'name': 'rmse', 'category': 'white_solid', 'value': rmse['1']})
    metrics.append({'name': 'rmse', 'category': 'white_dash', 'value': rmse['2']})
    metrics.append({'name': 'rmse', 'category': 'curb', 'value': rmse['3']})
    metrics.append({'name': 'rmse', 'category': 'cone_line', 'value': rmse['4']})
    metrics.append({'name': 'rmse', 'category': 'yellow_solid', 'value': rmse['5']})
    metrics.append({'name': 'rmse', 'category': 'yellow_dash', 'value': rmse['6']})
    metrics.append({'name': 'rmse', 'category': 'all', 'value': rmse['all']})

    return metrics

def evaluate_img(img0, img1, max_pixel_dis):
    return evaluate_rmse(process(img0), process(img1), max_pixel_dis)

def aggregate_results(rets):
    def aggregate_dict(obj1, obj2):
        obj = {}
        ks1 = set(obj1.keys())
        ks2 = set(obj2.keys())

        for k in ks1-ks2:
            obj[k] = obj1[k]
        for k in ks2-ks1:
            obj[k] = obj2[k]

        for k in ks1 & ks2:
            v = obj1[k]
            if type(v) == int or type(v) == np.int64 or type(v) == float:
                obj[k] = v + obj2[k]
            else:
                obj[k] = aggregate_dict(obj1[k], obj2[k])
        return obj
    return reduce(aggregate_dict, rets)




