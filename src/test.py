import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
import dataset as ds
from options.options import parser
import torch.nn.functional as F
from erf_settings import *
import prob_to_lines as ptl
from PIL import Image

best_mIoU = 0

def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 
        ignore_label = 255 
    elif args.dataset == 'CULane' or args.dataset == 'L4E':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet(3, num_class)
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, 'VOCAugDataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model, criterion, 0, evaluator)
    return


def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, img_name) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output, output_exist = model(input_var)

        # measure accuracy and record loss

        output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy() # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy() # BxO

        for cnt in range(len(img_name)):
            directory = 'predicts/ERFNet/' + img_name[cnt].split('/')[-4] + '/' + img_name[cnt].split('/')[-2]
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open('predicts/ERFNet'+img_name[cnt].replace('/sample', '').replace('.png', '_exist.txt'), 'w')
            cv_img = cv2.imread(dataset_path+img_name[cnt])
            in_frame = cv2.resize(cv_img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
            # croppedImage = in_frame[VERTICAL_CROP_SIZE:, :, :]  # FIX IT
            # croppedImageTrain = cv2.resize(croppedImage, (TRAIN_IMG_W, TRAIN_IMG_H))

            maps = []
            exists = []
            input_img = Image.fromarray(cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB))
            for num in range(4):
                prob_map = (pred[cnt][num + 1] * 255).astype(int)
                #prob_map = cv2.blur(prob_map, (9, 9))
                prob_map = prob_map.astype(np.uint8)
                prob_result = np.zeros((in_frame.shape[0], in_frame.shape[1]))
                prob_result[VERTICAL_CROP_SIZE:, :] = cv2.resize(prob_map, (IN_IMAGE_W, IN_IMAGE_H_AFTER_CROP), interpolation=cv2.INTER_NEAREST)
                maps.append(prob_map)
                if pred_exist[cnt][num] > 0.7:
                    input_img = ptl.AddMask(input_img, prob_result, COLORS[num], 0.75)  # Image with probability map
                    file_exist.write('1 ')
                else:
                    file_exist.write('0 ')
                exists.append(pred_exist[cnt][num] > 0.7)
                lines = ptl.GetLines(exists, maps, 0.75)
                points = lines[num]  # Points for the lane
                for point in points:
                    cv2.circle(in_frame, point, 5, POINT_COLORS[num], -1)

            res_img = cv2.cvtColor(np.asarray(input_img), cv2.COLOR_RGB2BGR)

            cv2.imwrite('predicts/ERFNet' + img_name[cnt].replace('/sample', '').replace('.png', '_prob.png'), res_img)
            cv2.imwrite('predicts/ERFNet' + img_name[cnt].replace('/sample', '').replace('.png', '_points.png'), in_frame)
            file_exist.close()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


if __name__ == '__main__':
    main()
