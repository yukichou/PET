# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import time
import logging
from sklearn.metrics import f1_score
import torch
import numpy as np
from core.evaluate import accuracy
from sklearn.metrics import accuracy_score
import pandas as pd
logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    score=[]
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        loss = criterion(output, target)


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        print("target:",target.cpu().detach().numpy())
        print("output",torch.max(output, 1)[1].cpu().detach().numpy())
        # F1 = f1_score(target.cpu().detach().numpy(), torch.max(output, 1)[1].cpu().detach().numpy(), labels=None)
        acc_score = accuracy_score(target.cpu().detach().numpy(),torch.max(output, 1)[1].cpu().detach().numpy())
        # score.append(F1)
        print("train acc_score:", acc_score)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()


    # switch to evaluate mode
    model.eval()
    score = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # print("target:",target.cpu().detach().numpy())
            # print("output",torch.max(output, 1)[1].cpu().detach().numpy())
            # F1 = f1_score(target.cpu().detach().numpy(), torch.max(output, 1)[1].cpu().detach().numpy(), labels=None, pos_label=1, average='binary', sample_weight=None)
            acc_score = accuracy_score(target.cpu().detach().numpy(),torch.max(output, 1)[1].cpu().detach().numpy())
            print("Test acc_score:",acc_score)
        # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t'.format(
                  batch_time=batch_time, loss=losses)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return acc_score

def pred(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    model.eval()
    with open('reslut.csv', 'w') as csvfile:
        fieldnames = ['uuid', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for i, (input, idx) in enumerate(val_loader):
                # compute output
                output = torch.max(model(input), 1)[1].cpu().detach().numpy()
                for j in range(len(input)):
                    writer.writerow({"uuid":idx[j], "label": pred2label(output[j])})

def pred2label(pred):
    if pred == 0:
        x = 'CN'
    elif pred == 1:
        x = 'AD'
    return x
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
