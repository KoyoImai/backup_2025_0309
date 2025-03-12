import os
import sys
import argparse
import random
import numpy as np
import math
import time

from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import torch
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn


from networks.resnet_multilevel_cifar import resnet18_sd, LinearClassifier
from optimizer.optimizer_multilevel import set_optimizer
from utils.util_multilevel import adjust_learning_rate, warmup_learning_rate
from utils.util_multilevel import AverageMeter




def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # （継続学習の）問題設定
    parser.add_argument('--target_task', type=int, default=0,
                        help='Use all classes if None else learned tasks so far')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # model関連
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--num_pt', type=int, default=300)

    # 最適化手法
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # データセット
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/ContinualLearning/CIL/Datasets',
                        help='path to custom dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')

    # モデルのパラメータ＆バッファ内のデータなどのパス
    parser.add_argument('--orig', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--logpt', type=str, default='',
                        help='path to pre-trained model')
    
    # その他
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')

    opt = parser.parse_args()


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # model名
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # 大規模バッチのwarm-up
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    # 各データセットの設定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 20
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # 基本パスの作成
    opt.origin_ckpt = opt.orig

    # 学習済みパラメータまでのパス
    opt.ckpt = opt.ckpt = f"{opt.ckpt}/last_{opt.target_task}.pth"

    # リプレイバッファ内のデータまでのパス
    opt.logpt = os.path.join(opt.logpt, 'replay_indices_{target_task}.npy'.format(target_task=opt.target_task))

    # 評価結果を保存するディレクトリ
    if not os.path.isdir(opt.origin_ckpt):
        os.makedirs(opt.origin_ckpt)
    
    return opt


# データローダーの作成
def set_loader(opt, replay_indices):

    # 平均・標準偏差の定義
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':       # scaleから
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # データ拡張の定義
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size,opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 検証用クラス
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))
    print("target_classes: ", target_classes)   # target_classes:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # CIFAR10の場合
    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)

        ## target_task（学習直後のタスク？）のクラスのデータのインデックスのみを取り出す
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        
        ## target_taskのデータのインデックスにバッファ内データのインデックスを追加
        subset_indices += replay_indices.tolist()   
        # print("len(subset_indices): ", len(subset_indices))    # len(subset_indices):  10200

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)    # [0 1 2 3 4 5 6 7 8 9]
        print(uc)    # [  25   25   25   25   25   25   25   25 5000 5000]

        ## クラス不均衡緩和のため，数が少ないクラスのデータに対する重みを大きくする
        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)
    
    # CIFAR100の場合
    if opt.dataset == 'cifar100':
        subset_indices = []
        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                           transform=train_transform,
                                           download=True)

        _train_targets = np.array(_train_dataset.targets)

        ## target_task（学習直後のタスク？）のクラスのデータのインデックスのみを取り出す
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        
        ## target_taskのデータのインデックスにバッファ内データのインデックスを追加
        subset_indices += replay_indices.tolist()   
        # print("len(subset_indices): ", len(subset_indices))    # len(subset_indices):  10200

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)    # [0 1 2 3 4 5 6 7 8 9]
        print(uc)    # [  25   25   25   25   25   25   25   25 5000 5000]

        ## クラス不均衡緩和のため，数が少ないクラスのデータに対する重みを大きくする
        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)

        _train_targets = np.array(_train_dataset.targets)
        for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        subset_indices += replay_indices.tolist()

        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
        print(ut)
        print(uc)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1./c

        train_dataset =  Subset(_train_dataset, subset_indices)

        subset_indices = []
        _val_dataset = TinyImagenet(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

    else:
        raise ValueError(opt.dataset)
    
    # data_loaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
    
    return train_loader, val_loader, uc


# modelと損失関数の作成
def set_model(opt, cls_num_list):

    # モデル，損失関数，分類器の作成
    # modelの定義
    if opt.dataset in ['cifar10', 'cifar100', 'tiny-imagenet']:
        model = resnet18_sd(nclasses=opt.n_cls, nf=64)
    elif opt.dataset in ['imagenet100']:
        model = resnet18_sd()
        assert False
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    # 学習済みパラメータの読み込み
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    # print("state_dict: ", state_dict)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


# 1エポック分の学習
def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    acc = 0.0
    cnt = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            feat_list = model.features(images)
            features = feat_list[3]
        # output = classifier(features.detach())
        output = model.head(feat_list, use_proj=False)[3]
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc += (output.argmax(1) == labels).float().sum().item()
        cnt += bsz

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=acc/cnt*100.))
            sys.stdout.flush()

    return losses.avg, acc/cnt*100.


# 検証
def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    # corr = [0.] * (4 + 1) * opt.cls_per_task
    # cnt  = [0.] * (4 + 1) * opt.cls_per_task

    correct_task = 0.0


    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            feat_list = model.features(images)
            output = model.head(feat_list, use_proj=False)[3]
            # output = classifier(model.features(images)[3])
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #
            cls_list = np.unique(labels.cpu())
            correct_all = (output.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (output[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))

    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
    return losses.avg, top1.avg, corr, cnt, correct_task/np.sum(cnt)*100.




def main():

    best_acc = 0
    task_acc = 0

    # コマンドライン引数の処理
    opt = parse_option()

    # seed値の固定
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # バッファ内データのインデックスを読み込む
    if opt.target_task is not None:
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            replay_indices = np.load(opt.logpt)
        print(len(replay_indices))
    
    # データローダの作成
    train_loader, val_loader, cls_num_list = set_loader(opt, replay_indices)

    # モデルと損失関数の作成
    model, classifier, criterion = set_model(opt, cls_num_list)

    # optimizerの作成
    # optimizer = set_optimizer(opt, classifier)
    optimizer = set_optimizer(opt, model)

    # 学習と検証
    for epoch in range(1, opt.epochs+1):

        # 学習率の調整
        adjust_learning_rate(opt, optimizer, epoch)

        # 1エポック分の学習
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f} {:.3f}'.format(
            epoch, time2 - time1, acc, optimizer.param_groups[0]['lr']))
        
        # 検証
        loss, val_acc, val_corr, val_cnt, task_acc = validate(val_loader, model, classifier, criterion, opt)
        val_acc = np.sum(val_corr)/np.sum(val_cnt)*100.
        if val_acc > best_acc:
            best_acc = val_acc

        val_acc_stats = {}
        for cls, (cr, c) in enumerate(zip(val_corr, val_cnt)):
            if c > 0:
                val_acc_stats[str(cls)] = cr / c * 100.
        # writer.add_scalars('val_acc', val_acc_stats, epoch)

    with open(os.path.join(opt.origin_ckpt, 'acc_buffer_{}.txt'.format(opt.target_task)), 'w') as f:
        out = 'best accuracy: {:.2f}\n'.format(best_acc)
        out += '{:.2f} {:.2f}'.format(val_acc, task_acc)
        print(out)
        out += '\n'
        for k, v in val_acc_stats.items():
            print(v)
            out += '{}\n'.format(v)
        f.write(out)

    save_file = os.path.join(
        opt.origin_ckpt, 'linear_{target_task}.pth'.format(target_task=opt.target_task))
    print('==> Saving...'+save_file)
    torch.save({
        'opt': opt,
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_file)






if __name__ == '__main__':

    main()