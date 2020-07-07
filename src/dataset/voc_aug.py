import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from options.config import cfg

class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/home/julian/data/lane_batch/L4E', data_list='train', transform=None):

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list.find('test') != -1  # 'val'

        with open(os.path.join(dataset_path, 'list', data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.seglabel_list = []
            self.exist_list = []

            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path + line.strip().split(" ")[0])

                if not self.is_testing:
                    if cfg.NUM_CLASSES:
                        self.label_list.append(dataset_path + line.strip().split(" ")[1])
                    if cfg.NUM_EGO:
                        self.seglabel_list.append(dataset_path + line.strip().split(" ")[2])
                        self.exist_list.append(np.array([cfg.ego_mapping[int(line.strip().split(" ")[3])], cfg.ego_mapping[int(line.strip().split(" ")[4])], cfg.ego_mapping[int(line.strip().split(" ")[5])], cfg.ego_mapping[int(line.strip().split(" ")[6])]]))
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        image = cv2.resize(image, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        image = image[cfg.VERTICAL_CROP_SIZE:, :, :]
        if not self.is_testing:
            if cfg.NUM_CLASSES:
                label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
                label = cv2.resize(label, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label = label[cfg.VERTICAL_CROP_SIZE:, :]
                for i in range(len(cfg.cls_mapping)):
                    if i != cfg.cls_mapping[i]:
                        label[label == i] = cfg.cls_mapping[i]
            else:
                label = np.zeros((image.shape[0], image.shape[1]))
            label = label.squeeze()
            if cfg.NUM_EGO:
                seglabel = cv2.imread(os.path.join(self.gt_path, self.seglabel_list[idx]), cv2.IMREAD_UNCHANGED)
                seglabel = cv2.resize(seglabel, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
                seglabel = seglabel[cfg.VERTICAL_CROP_SIZE:, :]
                seglabel = seglabel.squeeze()
                exist = self.exist_list[idx]
            else:
                seglabel = np.zeros((image.shape[0], image.shape[1]))
                exist = []

        if self.transform:
            if self.is_testing:
                label = np.zeros((image.shape[0], image.shape[1]))
                seglabel = np.zeros((image.shape[0], image.shape[1]))
            image, label, seglabel = self.transform((image, label, seglabel))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
            seglabel = torch.from_numpy(seglabel).contiguous().long()

        if self.is_testing:
            return image, self.img[idx]
        else:
            return image, label, seglabel, exist
