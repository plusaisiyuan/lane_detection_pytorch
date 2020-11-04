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

    num_ego = cfg.NUM_EGO
    num_class = cfg.NUM_CLASSES
    model = net.ERFNet(num_class, num_ego)
    model = torch.nn.DataParallel(model, device_ids=range(len(cfg.gpus))).cuda()

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(("=> loading checkpoint '{}'".format(cfg.resume)))
            checkpoint = torch.load(cfg.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(("=> no checkpoint found at '{}'".format(cfg.resume)))

    cudnn.benchmark = True
    cudnn.fastest = True

    return model


def infer_model(model, image):
    model.eval()

    # Input
    h, w, c = image.shape
    image_bgr = cv2.resize(image, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
    pred_cls = output_cls.data.cpu().numpy()  # BxCxHxW
    pred_ego = output_ego.data.cpu().numpy()  # BxCxHxW
    pred_exist = output_exist.data.cpu().numpy()

    exists = []

    threshold_cls = int(cfg.THRESHOLD_CLS * 255)
    threshold_ego = int(cfg.THRESHOLD_EGO * 255)
    result_cls = np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH)).astype(np.uint8)
    result_cls_orig = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH)).astype(np.uint8)
    for num in range(cfg.NUM_CLASSES-1):
        prob_map = (pred_cls[0][num + 1] * 255).astype(np.uint8)
        result_cls[prob_map >= threshold_cls] = num + 1
    result_cls_orig[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(result_cls,
                                                            (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                            interpolation=cv2.INTER_NEAREST)
    result_cls_orig = cv2.resize(result_cls_orig, (w, h), interpolation=cv2.INTER_NEAREST)

    result_ego = np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH)).astype(np.uint8)
    result_ego_orig = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH)).astype(np.uint8)
    for num in range(cfg.NUM_EGO):
        prob_map = (pred_ego[0][num + 1] * 255).astype(np.uint8)
        if pred_exist[0][num] > 0.5:
            result_ego[prob_map >= threshold_ego] = num + 1
        exists.append(pred_exist[0][num] > 0.5)
    result_ego_orig[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(result_ego,
                                                            (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                            interpolation=cv2.INTER_NEAREST)
    result_ego_orig = cv2.resize(result_ego_orig, (w, h), interpolation=cv2.INTER_NEAREST)
    lines = ptl.GetAllLines(exists, result_ego_orig)

    for num in range(cfg.NUM_EGO):
        points = lines[num]
        for point in points:
            if point[0] != -1:
                cv2.circle(image, point, 1, cfg.EG0_POINT_COLORS[num], -1)

    latency = end - start

    return result_cls_orig, result_ego_orig, image, lines, latency


def main():
    print("load model... ")
    model = load_model()
    image_file = '/home/jiangzh/lane/usb_cam_left/20200103T155346_j7-00006_0.bag/#usb_cam_left#image_raw#compressed_1578038027.086.png'
    image_bgr = cv2.imread(image_file)
    image_bgr = cv2.resize(image_bgr, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    h, w, c = image_bgr.shape
    # cfg.LOAD_IMAGE_HEIGHT = h
    # cfg.LOAD_IMAGE_WIDTH = w
    # cfg.VERTICAL_CROP_SIZE = 0
    # cfg.IN_IMAGE_H_AFTER_CROP = h
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
