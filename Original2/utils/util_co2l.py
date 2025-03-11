import os
import csv
import numpy as np
import math


import torch



# AverageMeter
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

# 対照学習のデータ拡張
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    

# 学習率調整
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# warm-up
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# モデルの保存
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...'+save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# csvファイルに値を書き込む
def write_csv(value, path, file_name):


    # パスの存在確認（存在しなければファイルを作成）
    file_path = f"{path}/{file_name}.csv"
    if not os.path.isfile(file_path):
        
        # ファイルが存在しない場合は新規作成
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

    # ファイルにvalueの値を書き込む
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # value がリストの場合はそのまま書き込む。単一の値の場合はリスト化する
        if isinstance(value, list):
            writer.writerow(value)
        else:
            writer.writerow([value])