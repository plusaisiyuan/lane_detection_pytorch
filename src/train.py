import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
from tensorboardX import SummaryWriter
import numpy as np

import net.erfnet as net
import dataset as ds
import utils.transforms as tf
from options.options import parser
from options.config import cfg

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
    model = net.ERFNet(num_class, num_ego)
    model = torch.nn.DataParallel(model, device_ids=range(len(cfg.gpus))).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        ckpt_name = []
        cnt = 0
        for name, param in state_dict.items():
            if name not in list(own_state.keys()) or 'output_conv' in name:
                 ckpt_name.append(name)
                 continue
            own_state[name].copy_(param)
            cnt += 1
        print('#reused param: {}'.format(cnt))
        return model

    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print(("=> loading checkpoint '{}'".format(cfg.resume)))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            model = load_my_state_dict(model, checkpoint['state_dict'])
            # torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(cfg.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(cfg.resume)))

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
        ])), batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        getattr(ds, 'VOCAugDataSet')(dataset_path=cfg.dataset_path, data_list=cfg.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.695, 0.721),
                                interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT)),
            tf.GroupNormalize(mean=(cfg.INPUT_MEAN, (0,), (0,)), std=(cfg.INPUT_STD, (1,), (1,))),
        ])), batch_size=cfg.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    class_weights = torch.FloatTensor(cfg.CLASS_WEIGHT).cuda()
    weights = [1.0 for _ in range(num_ego+1)]
    weights[0] = 0.4
    ego_weights = torch.FloatTensor(weights).cuda()
    criterion_cls = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    criterion_ego = torch.nn.NLLLoss(ignore_index=ignore_label, weight=ego_weights).cuda()
    criterion_exist = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    evaluator = EvalSegmentation(num_ego+1, ignore_label)

    # Tensorboard writer
    global writer
    writer = SummaryWriter(os.path.join(cfg.save_path, 'Tensorboard'))

    for epoch in range(cfg.epochs):  # args.start_epoch
        adjust_learning_rate(optimizer, epoch, cfg.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion_cls, criterion_ego, criterion_exist, optimizer, epoch, writer)

        # evaluate on validation set
        if (epoch + 1) % cfg.eval_freq == 0 or epoch == cfg.epochs - 1:
            mIoU_cls, mIoU_ego = validate(val_loader, model, criterion_cls, criterion_ego, criterion_exist, epoch,
                                          evaluator, writer)
            # remember best mIoU and save checkpoint
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

def train(train_loader, model, criterion_cls, criterion_ego, criterion_exist, optimizer, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_ego = AverageMeter()
    losses_exist = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    loss_cls_avg = 0
    loss_ego_avg = 0
    loss_exist_avg = 0
    loss_tot_avg = 0
    for i, (input, target_cls, target_ego, target_exist) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_cls = target_cls.cuda()
        target_ego = target_ego.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input)
        target_cls_var = torch.autograd.Variable(target_cls)
        target_ego_var = torch.autograd.Variable(target_ego)
        target_exist_var = torch.autograd.Variable(target_exist)

        # compute output
        output_cls, output_ego, output_exist = model(input_var)  # output_mid
        loss_cls = criterion_cls(torch.log(output_cls), target_cls_var)
        loss_ego = criterion_ego(torch.log(output_ego), target_ego_var)
        loss_exist = criterion_exist(output_exist, target_exist_var)
        loss_tot = loss_cls + loss_ego + loss_exist * 0.1

        # measure accuracy and record loss
        losses_cls.update(loss_cls.data.item(), input.size(0))
        losses_ego.update(loss_ego.data.item(), input.size(0))
        losses_exist.update(loss_exist.item(), input.size(0))

        loss_cls_avg += loss_cls.data.item()
        loss_ego_avg += loss_ego.data.item()
        loss_exist_avg += loss_exist.item()
        loss_tot_avg += loss_cls.data.item() + loss_ego.data.item() + loss_exist.item() * 0.1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % cfg.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t' 
                   'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t' 'Loss_exist {loss_exist.val:.4f} ({loss_exist.avg:.4f})\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss_cls=losses_cls, loss_ego=losses_ego,
                        loss_exist=losses_exist, lr=optimizer.param_groups[-1]['lr'])))
            batch_time.reset()
            data_time.reset()
            losses_cls.reset()
            losses_ego.reset()
            losses_exist.reset()
    writer.add_scalars('Loss_cls', {'Training': loss_cls_avg / len(train_loader)}, epoch)
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
        target_cls = target_cls.cuda()
        target_ego = target_ego.cuda()
        target_exist = target_exist.float().cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_cls_var = torch.autograd.Variable(target_cls)
        target_ego_var = torch.autograd.Variable(target_ego)
        target_exist_var = torch.autograd.Variable(target_exist)

        # compute output
        output_cls, output_ego, output_exist = model(input_var)
        loss_cls = criterion_cls(torch.log(output_cls), target_cls_var)
        loss_ego = criterion_ego(torch.log(output_ego), target_ego_var)
        loss_exist = criterion_exist(output_exist, target_exist_var)

        # measure accuracy and record loss
        pred_cls = output_cls.data.cpu().numpy().transpose(0, 2, 3, 1)
        pred_cls = np.argmax(pred_cls, axis=3).astype(np.uint8)
        pred_ego = output_ego.data.cpu().numpy().transpose(0, 2, 3, 1)
        pred_ego = np.argmax(pred_ego, axis=3).astype(np.uint8)
        IoU_cls.update(evaluator(pred_cls, target_cls.cpu().numpy()))
        IoU_ego.update(evaluator(pred_ego, target_ego.cpu().numpy()))
        losses_cls.update(loss_cls.data.item(), input.size(0))
        losses_ego.update(loss_ego.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        loss_cls_avg += loss_cls.data.item()
        loss_ego_avg += loss_ego.data.item()
        loss_exist_avg += loss_exist.item()
        loss_tot_avg += loss_cls.data.item() + loss_ego.data.item() + loss_exist.item() * 0.1

        if (i + 1) % cfg.print_freq == 0:
            acc_cls = np.sum(np.diag(IoU_cls.sum)) / float(np.sum(IoU_cls.sum))
            mIoU_cls = np.diag(IoU_cls.sum) / (1e-20 + IoU_cls.sum.sum(1) + IoU_cls.sum.sum(0) - np.diag(IoU_cls.sum))
            mIoU_cls = np.sum(mIoU_cls) / len(mIoU_cls)

            acc_ego = np.sum(np.diag(IoU_ego.sum)) / float(np.sum(IoU_ego.sum))
            mIoU_ego = np.diag(IoU_ego.sum) / (1e-20 + IoU_ego.sum.sum(1) + IoU_ego.sum.sum(0) - np.diag(IoU_ego.sum))
            mIoU_ego = np.sum(mIoU_ego) / len(mIoU_ego)
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Loss_cls {loss_cls.val:.4f} '
                   '({loss_cls.avg:.4f})\t' 'Loss_ego {loss_ego.val:.4f} ({loss_ego.avg:.4f})\t' 'Pixels Acc_cls {acc_cls:.3f}\t' 
                   'Pixels Acc_ego {acc_ego:.3f}\t' 'mIoU_cls {mIoU_cls:.3f}\t' 'mIoU_ego {mIoU_ego:.3f}'
                   .format(i, len(val_loader), batch_time=batch_time, loss_cls=losses_cls, loss_ego=losses_ego,
                           acc_cls=acc_cls, acc_ego=acc_ego, mIoU_cls=mIoU_cls, mIoU_ego=mIoU_ego)))

    acc_cls = np.sum(np.diag(IoU_cls.sum)) / float(np.sum(IoU_cls.sum))
    mIoU_cls = np.diag(IoU_cls.sum) / (1e-20 + IoU_cls.sum.sum(1) + IoU_cls.sum.sum(0) - np.diag(IoU_cls.sum))
    mIoU_cls = np.sum(mIoU_cls) / len(mIoU_cls)
    acc_ego = np.sum(np.diag(IoU_ego.sum)) / float(np.sum(IoU_ego.sum))
    mIoU_ego = np.diag(IoU_ego.sum) / (1e-20 + IoU_ego.sum.sum(1) + IoU_ego.sum.sum(0) - np.diag(IoU_ego.sum))
    mIoU_ego = np.sum(mIoU_ego) / len(mIoU_ego)
    writer.add_scalars('Pixels Acc_cls', {'Validation': acc_cls}, iter)
    writer.add_scalars('mIoU_cls', {'Validation': mIoU_cls}, iter)
    writer.add_scalars('Pixels Acc_ego', {'Validation': acc_ego}, iter)
    writer.add_scalars('mIoU_ego', {'Validation': mIoU_ego}, iter)
    writer.add_scalars('Loss_cls', {'Validation': loss_cls_avg/len(val_loader)}, iter)
    writer.add_scalars('Loss_ego', {'Validation': loss_ego_avg/len(val_loader)}, iter)
    writer.add_scalars('Loss_exist', {'Validation': loss_exist_avg / len(val_loader)}, iter)
    writer.add_scalars('Loss_tot', {'Validation': loss_tot_avg / len(val_loader)}, iter)
    print(('Testing Results: Pixels Acc_cls {acc_cls:.3f}\tmIoU_cls {mIoU_cls:.3f} ({bestmIoU_cls:.4f})\t'
           'Loss_cls {loss_cls.avg:.5f}\tPixels Acc_ego {acc_ego:.3f}\tmIoU_ego {mIoU_ego:.3f} ({bestmIoU_ego:.4f})\t'
           'Loss_ego {loss_ego.avg:.5f}'
           .format(acc_cls=acc_cls, mIoU_cls=mIoU_cls, bestmIoU_cls=max(mIoU_cls, best_mIoU_cls), loss_cls=losses_cls,
                   acc_ego=acc_ego, mIoU_ego=mIoU_ego, bestmIoU_ego=max(mIoU_ego, best_mIoU_ego), loss_ego=losses_ego)))

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

    # decay = ((1 - float(epoch) / cfg.epochs)**(0.9))
    # if decay < 1e-1:
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    lr = cfg.lr * decay
    decay = cfg.weight_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay


if __name__ == '__main__':
    main()
