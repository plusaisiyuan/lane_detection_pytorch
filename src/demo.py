import os
import time
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import net.erfnet as net
import dataset as ds
import utils.transforms as tf
import utils.prob_to_lines as ptl
from options.config import cfg

def load_model():

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in cfg.gpus)

    if cfg.dataset == 'VOCAug' or cfg.dataset == 'VOC2012' or cfg.dataset == 'COCO':
        num_ego = 21
        num_class = 2
        ignore_label = 255
    elif cfg.dataset == 'Cityscapes':
        num_ego = 19
        num_class = 2
        ignore_label = 255  # 0
    elif cfg.dataset == 'ApolloScape':
        num_ego = 37  # merge the noise and ignore labels
        num_class = 2
        ignore_label = 255
    elif cfg.dataset == 'CULane':
        num_ego = cfg.NUM_EGO
        num_class = 2
        ignore_label = 255
    else:
        num_ego = cfg.NUM_EGO
        num_class = cfg.NUM_CLASSES
        ignore_label = 255
    model = net.ERFNet(num_class, num_ego)
    model = torch.nn.DataParallel(model, device_ids=range(len(cfg.gpus))).cuda()

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(("=> loading checkpoint '{}'".format(cfg.resume)))
            checkpoint = torch.load(cfg.resume)
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(cfg.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(cfg.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    return model


def infer_model(model, image):
    model.eval()

    # Input
    h, w, c = image.shape
    # image_bgr = cv2.resize(image, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    # image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    # image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = (image_rgb - cfg.INPUT_MEAN) * cfg.INPUT_STD
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.float32)
    image_rgb = np.expand_dims(image_rgb, 0)

    rgb_tensor = torch.from_numpy(image_rgb)
    input_var = torch.autograd.Variable(rgb_tensor)

    # Comput
    start = time.time()
    output_cls, output_ego, output_exist = model(input_var)
    end = time.time()
    # output_cls = F.softmax(output_cls, dim=1)
    # output_ego = F.softmax(output_ego, dim=1)
    pred_cls = output_cls.data.cpu().numpy()  # BxCxHxW
    pred_ego = output_ego.data.cpu().numpy()  # BxCxHxW
    pred_exist = output_exist.data.cpu().numpy()

    maps = []
    exists = []

    threshold_cls = int(cfg.THRESHOLD_CLS * 255)
    threshold_ego = int(cfg.THRESHOLD_EGO * 255)
    result_cls = np.zeros((h, w)).astype(np.uint8)
    for num in range(cfg.NUM_CLASSES-1):
        prob_map = (pred_cls[0][num + 1] * 255).astype(np.uint8)

        map_bak = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH))
        map_bak[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(prob_map, (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                     interpolation=cv2.INTER_NEAREST)
        map_bak = cv2.resize(map_bak, (w, h), interpolation=cv2.INTER_NEAREST)
        result_cls[map_bak >= threshold_cls] = num + 1
    result_ego = np.zeros((h, w)).astype(np.uint8)
    for num in range(cfg.NUM_EGO):
        prob_map = (pred_ego[0][num + 1] * 255).astype(np.uint8)
        maps.append(prob_map)
        map_bak = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH))
        map_bak[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(prob_map, (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                     interpolation=cv2.INTER_NEAREST)
        map_bak = cv2.resize(map_bak, (w, h), interpolation=cv2.INTER_NEAREST)
        if pred_exist[0][num] > 0.5:
            result_ego[map_bak >= threshold_ego] = num + 1
        exists.append(pred_exist[0][num] > 0.7)
    lines = ptl.GetAllLines(exists, result_ego)

    for num in range(cfg.NUM_EGO):
        points = lines[num]
        for point in points:
            if point[0] != -1:
                cv2.circle(image, point, 1, cfg.EG0_POINT_COLORS[num], -1)

    latency = end - start

    return result_cls, result_ego, image, lines, latency


def main():
    cfg.INPUT_MEAN = [103.939, 116.779, 123.68]
    cfg.INPUT_STD = [1., 1., 1.]
    cfg.NUM_CLASSES = 3

    cfg.resume = 'trained/20200520223916/_erfnet_model_best.pth.tar'
    print("load model... ")
    model = load_model()
    image_file = '/home/jiangzh/lane/usb_cam_left/20200103T155346_j7-00006_0.bag/#usb_cam_left#image_raw#compressed_1578038027.086.png'
    image_bgr = cv2.imread(image_file)
    image_bgr = cv2.resize(image_bgr, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    h, w, c = image_bgr.shape
    cfg.LOAD_IMAGE_HEIGHT = h
    cfg.LOAD_IMAGE_WIDTH = w
    cfg.VERTICAL_CROP_SIZE = 0
    cfg.IN_IMAGE_H_AFTER_CROP = h
    result_cls_color = np.copy(image_bgr)
    result_ego_color = np.copy(image_bgr)
    print("predict image... ")
    result_cls, result_ego, result_points, lines, latency = infer_model(model, image_bgr)
    print("the latency of predicting image ", latency)

    for i in range(h):
        for j in range(w):
            if result_cls[i, j] != 0:
                result_cls_color[i, j, :] = cfg.EG0_POINT_COLORS[result_cls[i, j] - 1]
            if result_ego[i, j] != 0:
                result_ego_color[i, j, :] = cfg.EG0_POINT_COLORS[result_ego[i, j] - 1]

    cv2.imshow('result_cls', result_cls*40)
    cv2.waitKey()
    cv2.imshow('result_ego', result_ego_color)
    cv2.waitKey()
    cv2.imshow('result_points', result_points)
    cv2.waitKey()

if __name__ == '__main__':
    main()
