class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, threshold=0.5):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    output = output > threshold
    target = target.squeeze()
    correct = (target == output)
    #print("C: ", correct)
    correct = torch.all(correct, dim=1)

    predicted_positive_indexes = (output[:,0].nonzero(as_tuple=True)[0])
    true_positive = (target[:,0])[predicted_positive_indexes]

    crucial_fail = (target[:,1])[predicted_positive_indexes]
    soso_fail = (target[:,2])[predicted_positive_indexes]

    #print(predicted_positive_indexes)
    #print(true_positive)
    #print("Precision: ", precision)
    

    #print("T: ", target)
    #print("O: ", output)
    #print("C: ", correct)
    #print("acc: ", correct.sum() / output.shape[0])
    
    return correct.sum() / output.shape[0], true_positive.sum() / predicted_positive_indexes.shape[0], crucial_fail.sum() / predicted_positive_indexes.shape[0], soso_fail.sum() / predicted_positive_indexes.shape[0]
  
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import shutil


def save_ckpt(state, is_best):
  f_path = 'stock_predict_latest_old_channel5.pth'
  torch.save(state, f_path)
  if is_best:
    best_fpath  = 'stock_predict_best_old_channel5.pt'
    shutil.copyfile(f_path, best_fpath)



def load_ckpt(model, optimizer):
  checkpoint_fpath = 'stock_predict_best_old_channel5.pt'
  checkpoint = torch.load(checkpoint_fpath)
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  epoch = checkpoint['epoch']
  return model, optimizer, epoch

def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()

    accuracies = np.array([])
    precisions = np.array([])
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input
        target = target

        output = model(input)
        #if epoch % 25 == 24:
        #    print("target:", target[:,0])
        #    print("out:", output)
        loss = criterion(output, target[:,0])
        acc, pre, _, _ = accuracy(output, target)
        accuracies = np.append(accuracies, acc)
        if not pre is None:
            precisions = np.append(precisions, pre)
        
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINTFREQ == 0:
            progress.print(i)

    avg_acc = accuracies.mean()
    avg_pres = precisions.mean()
    print("TRAIN ACCURACY: ",avg_acc)
    print("TRAIN PRECISION: ",avg_pres)
    #print('=> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #      .format(top1=top1, top5=top5))
    return avg_acc, avg_pres


def validate(val_loader, model, criterion, threshold=0.5):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), batch_time, losses, top1, top5, prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    total_loss = 0.0

    accuracies = np.array([])
    precisions = np.array([])
    c_fails = np.array([])
    s_fails = np.array([])
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input
            target = target

            # compute output
            output = model(input)
            loss = criterion(output, target[:,0])
            acc, pre, c_fail, s_fail = accuracy(output, target, threshold)
            accuracies = np.append(accuracies, acc)
            if not pre is None:
                precisions = np.append(precisions, pre)
            if not c_fail is None:
                c_fails = np.append(c_fails, c_fail)
            if not s_fail is None:
                s_fails = np.append(s_fails, s_fail)

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            total_loss += loss.item()

            if i % PRINTFREQ == 0:
                progress.print(i)
                print(output)

            end = time.time()

        print(
            "====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
        )
        total_loss = total_loss / len(val_loader)
        
    avg_acc = accuracies.mean()
    avg_pres = precisions.mean()
    avg_c_fails = c_fails.mean()
    avg_s_fails = s_fails.mean()
    print("VAL ACCURACY: ", avg_acc)
    print("VAL PRECISION: ",avg_pres)
    print("VAL C FAILS: ",avg_c_fails)
    print("VAL S FAILS: ",avg_s_fails)
    return avg_acc, avg_pres
