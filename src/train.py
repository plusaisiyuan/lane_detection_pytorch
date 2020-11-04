import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import cv2
from tensorboardX import SummaryWriter
import numpy as np

import net.erfnet as net
import dataset as ds
import utils.transforms as tf
from options.options import parser
from options.config import cfg
import json

best_mIoU_cls = 0
best_mIoU_ego = 0

start = time.time()

def main():
    global best_mIoU_cls, best_mIoU_ego, args
    args = parser.parse_args()
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

    print(json.dumps(cfg, sort_keys=True, indent=2))
    model = net.ERFNet(num_class, num_ego)
    model = torch.nn.DataParallel(model, device_ids=range(len(cfg.gpus))).cuda()

    if num_class:
        print(("=> train '{}' model".format('lane_cls')))
    if num_ego:
        print(("=> train '{}' model".format('lane_ego')))

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    resume_epoch = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(("=> loading checkpoint '{}'".format(cfg.resume)))
            checkpoint = torch.load(cfg.resume)
            if cfg.finetune:
                print('finetune from ', cfg.resume)
                state_all = checkpoint['state_dict']
                state_clip = {}  # only use backbone parameters
                for k, v in state_all.items():
                    if 'module' in k:
                        state_clip[k] = v
                        print(k)
                model.load_state_dict(state_clip, strict=False)
            else:
                print('==> Resume model from ' + cfg.resume)
                model.load_state_dict(checkpoint['state_dict'])
                if 'optimizer' in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epoch' in checkpoint.keys():
                    resume_epoch = int(checkpoint['epoch']) + 1
        else:
            print(("=> no checkpoint found at '{}'".format(cfg.resume)))
            model.apply(weights_init)
    else:
        model.apply(weights_init)


    # if cfg.resume:
    #     if os.path.isfile(cfg.resume):
    #         print(("=> loading checkpoint '{}'".format(cfg.resume)))
    #         checkpoint = torch.load(cfg.resume)
    #         cfg.start_epoch = checkpoint['epoch']
    #         # model = load_my_state_dict(model, checkpoint['state_dict'])
    #         torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
    #         print(("=> loaded checkpoint '{}' (epoch {})".format(cfg.evaluate, checkpoint['epoch'])))
    #     else:
    #         print(("=> no checkpoint found at '{}'".format(cfg.resume)))
    #         model.apply(weights_init)
    # else:
    #     model.apply(weights_init)

    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        getattr(ds, 'VOCAugDataSet')(dataset_path=cfg.dataset_path, data_list=cfg.train_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.695, 0.721),
                                interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT)),
            tf.GroupRandomRotation(degree=(-1, 1),
                                   interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                   padding=(cfg.INPUT_MEAN, (ignore_label,), (ignore_label,))),
            tf.GroupNormalize(mean=(cfg.INPUT_MEAN, (0,), (0,)), std=(cfg.INPUT_STD, (1,), (1,))),
        ])), batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        getattr(ds, 'VOCAugDataSet')(dataset_path=cfg.dataset_path, data_list=cfg.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.695, 0.721),
                                interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT)),
            tf.GroupNormalize(mean=(cfg.INPUT_MEAN, (0,), (0,)), std=(cfg.INPUT_STD, (1,), (1,))),
        ])), batch_size=cfg.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) optimizer and evaluator
    class_weights = torch.FloatTensor(cfg.CLASS_WEIGHT).cuda()
    weights = [1.0 for _ in range(num_ego+1)]
    weights[0] = 0.4
    ego_weights = torch.FloatTensor(weights).cuda()
    criterion_cls = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    criterion_ego = torch.nn.NLLLoss(ignore_index=ignore_label, weight=ego_weights).cuda()
    criterion_exist = torch.nn.BCELoss().cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg_loss = None
    if cfg.weight_decay > 0 and cfg.use_L1:
        reg_loss = Regularization(model, cfg.weight_decay, p=1).to(device)
    else:
        print("no regularization")

    if num_class:
        evaluator = EvalSegmentation(num_class, ignore_label)
    if num_ego:
        evaluator = EvalSegmentation(num_ego+1, ignore_label)


    # Tensorboard writer
    global writer
    writer = SummaryWriter(os.path.join(cfg.save_path, 'Tensorboard'))

    for epoch in range(cfg.epochs):  # args.start_epoch
        if epoch < resume_epoch:
            continue
        adjust_learning_rate(optimizer, epoch, cfg.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion_cls, criterion_ego, criterion_exist, optimizer, epoch, writer, reg_loss)

        # evaluate on validation set
        if (epoch + 1) % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            mIoU_cls, mIoU_ego = validate(val_loader, model, criterion_cls, criterion_ego, criterion_exist, epoch,
                                          evaluator, writer)
            # remember best mIoU and save checkpoint
            if num_class:
                is_best = mIoU_cls > best_mIoU_cls
            if num_ego:
                is_best = mIoU_ego > best_mIoU_ego
            best_mIoU_cls = max(mIoU_cls, best_mIoU_cls)
            best_mIoU_ego = max(mIoU_ego, best_mIoU_ego)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'state_dict': model.state_dict(),
                'best_mIoU': best_mIoU_ego,
            }, is_best)

    writer.close()

def train(train_loader, model, criterion_cls, criterion_ego, criterion_exist, optimizer, epoch, writer=None, reg_loss=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_reg = AverageMeter()
    losses_cls = AverageMeter()
    losses_ego = AverageMeter()
    losses_exist = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    loss_reg_avg = 0
    loss_cls_avg = 0
    loss_ego_avg = 0
    loss_exist_avg = 0
    loss_tot_avg = 0
    for i, (input, target_cls, target_ego, target_exist) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input)

        # compute output
        if cfg.NUM_CLASSES and cfg.NUM_EGO:
            output_cls, output_ego, output_exist = model(input_var)  # output_mid
        if cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
            output_cls = model(input_var)  # output_mid
        if cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
            output_ego, output_exist = model(input_var) # output_mid

        loss_tot = 0
        if cfg.weight_decay > 0 and cfg.use_L1:
            loss_reg = reg_loss(model)
            loss_tot += loss_reg
            loss_reg_avg += loss_reg.data.item()
            # measure accuracy and record loss
            losses_reg.update(loss_reg.data.item(), input.size(0))
        if cfg.NUM_CLASSES:
            target_cls = target_cls.cuda()
            target_cls_var = torch.autograd.Variable(target_cls)
            loss_cls = criterion_cls(torch.log(output_cls), target_cls_var)
            loss_tot += loss_cls

            loss_cls_avg += loss_cls.data.item()
            # measure accuracy and record loss
            losses_cls.update(loss_cls.data.item(), input.size(0))
        if cfg.NUM_EGO:
            target_ego = target_ego.cuda()
            target_exist = target_exist.float().cuda()
            target_ego_var = torch.autograd.Variable(target_ego)
            target_exist_var = torch.autograd.Variable(target_exist)
            loss_ego = criterion_ego(torch.log(output_ego), target_ego_var)
            loss_exist = criterion_exist(output_exist, target_exist_var)
            loss_tot += loss_ego + loss_exist * cfg.factor

            loss_ego_avg += loss_ego.data.item()
            loss_exist_avg += loss_exist.item()
            # measure accuracy and record loss
            losses_ego.update(loss_ego.data.item(), input.size(0))
            losses_exist.update(loss_exist.item(), input.size(0))
        loss_tot_avg += loss_tot.data.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg.print_freq == 0:
            if cfg.NUM_CLASSES and cfg.NUM_EGO:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                       # 'Loss_reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t'
                       'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                       'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t' 'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'
                       .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                               # loss_reg=losses_reg,
                               loss_cls=losses_cls, loss_ego=losses_ego,
                               loss_exist=losses_exist, lr=optimizer.param_groups[-1]['lr'])))
            if cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                       # 'Loss_reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t' 
                       'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                       .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                               # loss_reg=losses_reg,
                               loss_cls=losses_cls, lr=optimizer.param_groups[-1]['lr'])))
            if cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
                output_ego, output_exist = model(input_var)  # output_mid
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                       # 'Loss_reg {loss_reg.val:.4f} ({loss_reg.avg:.4f})\t' 
                       'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t' 'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'
                       .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                               # loss_reg=losses_reg,
                               loss_ego=losses_ego, loss_exist=losses_exist, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            if cfg.weight_decay > 0 and cfg.use_L1:
                losses_reg.reset()
            losses_cls.reset()
            losses_ego.reset()
            losses_exist.reset()

    if cfg.weight_decay > 0 and cfg.use_L1:
        writer.add_scalars('Loss_reg', {'Training': loss_reg_avg / len(train_loader)}, epoch)
    if cfg.NUM_CLASSES:
        writer.add_scalars('Loss_cls', {'Training': loss_cls_avg / len(train_loader)}, epoch)
    if cfg.NUM_EGO:
        writer.add_scalars('Loss_ego', {'Training': loss_ego_avg / len(train_loader)}, epoch)
        writer.add_scalars('Loss_exist', {'Training': loss_exist_avg / len(train_loader)}, epoch)
    writer.add_scalars('Loss_tot', {'Training': loss_tot_avg / len(train_loader)}, epoch)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(val_loader, model, criterion_cls, criterion_ego, criterion_exist, iter, evaluator, writer=None):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_ego = AverageMeter()
    IoU_cls = AverageMeter()
    IoU_ego = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    loss_cls_avg = 0
    loss_ego_avg = 0
    loss_exist_avg = 0
    loss_tot_avg = 0
    for i, (input, target_cls, target_ego, target_exist) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        if cfg.NUM_CLASSES and cfg.NUM_EGO:
            output_cls, output_ego, output_exist = model(input_var)  # output_mid
        if cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
            output_cls = model(input_var)  # output_mid
        if cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
            output_ego, output_exist = model(input_var)  # output_mid

        if cfg.NUM_CLASSES:
            target_cls = target_cls.cuda()
            target_cls_var = torch.autograd.Variable(target_cls)
            # measure accuracy and record loss
            pred_cls = output_cls.data.cpu().numpy().transpose(0, 2, 3, 1)
            pred_cls = np.argmax(pred_cls, axis=3).astype(np.uint8)
            loss_cls = criterion_cls(torch.log(output_cls), target_cls_var)
            IoU_cls.update(evaluator(pred_cls, target_cls.cpu().numpy()))
            losses_cls.update(loss_cls.data.item(), input.size(0))
            loss_cls_avg += loss_cls.data.item()
            loss_tot_avg += loss_cls.data.item()
        if cfg.NUM_EGO:
            target_ego = target_ego.cuda()
            target_exist = target_exist.float().cuda()
            target_ego_var = torch.autograd.Variable(target_ego)
            target_exist_var = torch.autograd.Variable(target_exist)
            # measure accuracy and record loss
            pred_ego = output_ego.data.cpu().numpy().transpose(0, 2, 3, 1)
            pred_ego = np.argmax(pred_ego, axis=3).astype(np.uint8)
            loss_ego = criterion_ego(torch.log(output_ego), target_ego_var)
            loss_exist = criterion_exist(output_exist, target_exist_var)
            IoU_ego.update(evaluator(pred_ego, target_ego.cpu().numpy()))
            losses_ego.update(loss_ego.data.item(), input.size(0))
            loss_ego_avg += loss_ego.data.item()
            loss_exist_avg += loss_exist.item()
            loss_tot_avg += loss_ego.data.item() + loss_exist.item() * cfg.factor

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg.print_freq == 0:
            if cfg.NUM_CLASSES:
                acc_cls = np.sum(np.diag(IoU_cls.sum)) / float(np.sum(IoU_cls.sum))
                mIoU_cls = np.diag(IoU_cls.sum) / (1e-20 + IoU_cls.sum.sum(1) + IoU_cls.sum.sum(0) - np.diag(IoU_cls.sum))
                mIoU_cls = np.sum(mIoU_cls) / len(mIoU_cls)
            if cfg.NUM_EGO:
                acc_ego = np.sum(np.diag(IoU_ego.sum)) / float(np.sum(IoU_ego.sum))
                mIoU_ego = np.diag(IoU_ego.sum) / (1e-20 + IoU_ego.sum.sum(1) + IoU_ego.sum.sum(0) - np.diag(IoU_ego.sum))
                mIoU_ego = np.sum(mIoU_ego) / len(mIoU_ego)
            if cfg.NUM_CLASSES and cfg.NUM_EGO:
                print((
                          'Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss_cls {loss_cls.val:.4f} '
                          '({loss_cls.avg:.4f})\t' 'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t' 'Pixels Acc_cls {acc_cls:.3f}\t'
                          'Pixels Acc_ego {acc_ego:.3f}\t' 'mIoU_cls {mIoU_cls:.3f}\t' 'mIoU_ego {mIoU_ego:.3f}'
                          .format(i, len(val_loader), batch_time=batch_time, loss_cls=losses_cls, loss_ego=losses_ego,
                                  acc_cls=acc_cls, acc_ego=acc_ego, mIoU_cls=mIoU_cls, mIoU_ego=mIoU_ego)))
            if cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
                print((
                          'Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss_cls {loss_cls.val:.4f} '
                          '({loss_cls.avg:.4f})\t' 'Pixels Acc_cls {acc_cls:.3f}\t' 'mIoU_cls {mIoU_cls:.3f}\t'
                          .format(i, len(val_loader), batch_time=batch_time, loss_cls=losses_cls,
                                  acc_cls=acc_cls, mIoU_cls=mIoU_cls)))
            if cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
                print((
                    'Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t'                   
                    'Pixels Acc_ego {acc_ego:.3f}\t' 'mIoU_ego {mIoU_ego:.3f}'
                        .format(i, len(val_loader), batch_time=batch_time, loss_ego=losses_ego,
                                acc_ego=acc_ego, mIoU_ego=mIoU_ego)))

    mIoU_cls = 0
    mIoU_ego = 0
    if cfg.NUM_CLASSES:
        acc_cls = np.sum(np.diag(IoU_cls.sum)) / float(np.sum(IoU_cls.sum))
        mIoU_cls = np.diag(IoU_cls.sum) / (1e-20 + IoU_cls.sum.sum(1) + IoU_cls.sum.sum(0) - np.diag(IoU_cls.sum))
        mIoU_cls = np.sum(mIoU_cls) / len(mIoU_cls)
        writer.add_scalars('Pixels Acc_cls', {'Validation': acc_cls}, iter)
        writer.add_scalars('mIoU_cls', {'Validation': mIoU_cls}, iter)
        writer.add_scalars('Loss_cls', {'Validation': loss_cls_avg / len(val_loader)}, iter)
    if cfg.NUM_EGO:
        acc_ego = np.sum(np.diag(IoU_ego.sum)) / float(np.sum(IoU_ego.sum))
        mIoU_ego = np.diag(IoU_ego.sum) / (1e-20 + IoU_ego.sum.sum(1) + IoU_ego.sum.sum(0) - np.diag(IoU_ego.sum))
        mIoU_ego = np.sum(mIoU_ego) / len(mIoU_ego)
        writer.add_scalars('Pixels Acc_ego', {'Validation': acc_ego}, iter)
        writer.add_scalars('mIoU_ego', {'Validation': mIoU_ego}, iter)
        writer.add_scalars('Loss_ego', {'Validation': loss_ego_avg/len(val_loader)}, iter)
        writer.add_scalars('Loss_exist', {'Validation': loss_exist_avg / len(val_loader)}, iter)
    writer.add_scalars('Loss_tot', {'Validation': loss_tot_avg / len(val_loader)}, iter)
    if cfg.NUM_CLASSES and cfg.NUM_EGO:
        print(('Testing Results: Pixels Acc_cls {acc_cls:.3f}\tmIoU_cls {mIoU_cls:.3f} ({bestmIoU_cls:.4f})\t'
               'Loss_cls {loss_cls.avg:.5f}\tPixels Acc_ego {acc_ego:.3f}\tmIoU_ego {mIoU_ego:.3f} ({bestmIoU_ego:.4f})\t'
               'Loss_ego {loss_ego.avg:.5f}'
               .format(acc_cls=acc_cls, mIoU_cls=mIoU_cls, bestmIoU_cls=max(mIoU_cls, best_mIoU_cls),
                       loss_cls=losses_cls,
                       acc_ego=acc_ego, mIoU_ego=mIoU_ego, bestmIoU_ego=max(mIoU_ego, best_mIoU_ego),
                       loss_ego=losses_ego)))
    if cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
        print(('Testing Results: Pixels Acc_cls {acc_cls:.3f}\tmIoU_cls {mIoU_cls:.3f} ({bestmIoU_cls:.4f})\t'
               'Loss_cls {loss_cls.avg:.5f}\t'
               .format(acc_cls=acc_cls, mIoU_cls=mIoU_cls, bestmIoU_cls=max(mIoU_cls, best_mIoU_cls),
                       loss_cls=losses_cls)))
    if cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
        print(('Testing Results: Pixels Acc_ego {acc_ego:.3f}\tmIoU_ego {mIoU_ego:.3f} ({bestmIoU_ego:.4f})\t'
               'Loss_ego {loss_ego.avg:.5f}'
               .format(acc_ego=acc_ego, mIoU_ego=mIoU_ego, bestmIoU_ego=max(mIoU_ego, best_mIoU_ego),
                       loss_ego=losses_ego)))

    return mIoU_cls, mIoU_ego

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_path = 'trained/' + time.strftime("%Y%m%d%H%M%S", time.localtime(start))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, '_'.join((cfg.snapshot_pref, cfg.method.lower(), filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(save_path, '_'.join((cfg.snapshot_pref, cfg.method.lower(), 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)

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
        hs = np.bincount(sumim[locs], minlength=self.num_class ** 2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):

    decay = ((1 - float(epoch) / cfg.epochs)**(0.9))
    # decay = 0.9 ** epoch
    # if decay < 1e-1:
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    lr = cfg.lr * decay
    decay = cfg.weight_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay

class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model: train model
        :param weight_decay: regularization params
        :param p:  when p = 0 is L2 regularization, p = 1 is L1 regularization
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        :param device: cuda or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p:
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay*reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except:
            print(m.__class__.__name__)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    main()
