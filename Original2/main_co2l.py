import os
import time
import copy
import math
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from datasets.dataset_co2l import TinyImagenet
from networks.resnet_co2l_cifar import SupConResNet
from networks.resnet_co2l_imnet import resnet18
from losses.loss_co2l import SupConLoss
from optimizer.optimizer_co2l import set_optimizer
from utils.util_co2l import TwoCropTransform
from utils.util_co2l import adjust_learning_rate
from train.train_co2l import train
from utils.util_co2l import save_model




def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # 継続学習の問題設定
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--end_task', type=int, default=None)
    parser.add_argument('--cls_per_task', type=int, default=2)
    parser.add_argument('--mem_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # modelのパラメータなど
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--feat_dim', type=int, default=128)

    # データセット
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tiny-imagenet', 'path'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/ContinualLearning/CIL/Datasets',
                        help='path to custom dataset')
    
    # 損失関数
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')
    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')
    parser.add_argument('--distill_power', type=float, default=1.0)

    # 最適化
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    

    # その他
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--original_name', type=str, default="practice")
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')

    # デバック
    parser.add_argument('--debug_mode', action='store_true')

    opt = parser.parse_args()

    # データセット毎にパラメータを設定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    elif opt.dataset == 'cifar100':
        opt.n.cls = 100
        opt.cls_per_task = 20
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    elif opt.dataset == 'imagenet100':
        opt.n_cls = 100
        opt.cls_per_task = 10
        opt.size = 224
    else:
        pass

    # modelのパラメータや記録の保存作
    opt.model_path = f'./exp_logs/checkpoints_co2l/{opt.original_name}/model_param/'
    opt.log_path   = f'./exp_logs/checkpoints_co2l/{opt.original_name}/buffer_log/'
    opt.learning_path = f'./exp_logs/checkpoints_co2l/{opt.original_name}/learning_log/'

    # modelの名前
    opt.model_name = '{}_{}'.\
        format(opt.dataset, opt.model)
    
    # 大規模バッチの場合のwarm-up
    if opt.batch_size > 256:
        opt.warm = True
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
    
    # modelのパラメータを保存するディレクトリ
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # リプレイバッファを保存するディレクトリ
    opt.log_folder = os.path.join(opt.log_path, opt.model_name)
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    # 実験記録を保存するディレクトリ
    opt.learninglog_folder = os.path.join(opt.learning_path, opt.model_name)
    if not os.path.isdir(opt.learninglog_folder):
        os.makedirs(opt.learninglog_folder)


    return opt



# model，損失関数の作成
def set_model(opt):

    # modelの定義
    if opt.dataset in ['cifar10', 'cifar100', 'tiny-imagenet']:
        model = SupConResNet(name=opt.model)
    elif opt.dataset in ['imagenet100']:
        model = resnet18()

    # 損失関数の定義
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

# リプレイバッファ内のインデックス
def set_replay_samples(opt, model, prev_indices=None):

    # modelの状態を保存・evalモードに変更
    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]
    

    # データセットの作成
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # 前タスクのデータのインデックス
    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
        # print("observed_classes: ", observed_classes)
    
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices

    # 学習済みクラスのインデックスを取得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()
    # print("len(observed_indices): ", len(observed_indices))


    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)


    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))


    model.is_training = is_training

    return prev_indices + selected_observed_indices

# データローダの作成
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
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
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

    # 現在タスクのクラスを取得
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    # データセットの作成
    if opt.dataset == 'cifar10':
        subset_indices = []
        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
        
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'cifar100':
        subset_indices = []
        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                           transform=TwoCropTransform(train_transform),
                                           download=True)
        
        for tc in target_classes:
            target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        _train_dataset = TinyImagenet(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        for tc in target_classes:
            target_class_indices = np.where(_train_dataset.targets == tc)[0]
            subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()

        subset_indices += replay_indices

        train_dataset =  Subset(_train_dataset, subset_indices)
        print('Dataset size: {}'.format(len(subset_indices)))
        uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
        print(uc[np.argsort(uk)])

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices





def main():

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

    # ターゲットタスク
    target_task = opt.target_task

    # model，損失関数の作成
    model, criterion = set_model(opt)
    model2, _ = set_model(opt)
    model2.eval()

    # Optimierの定義
    optimizer = set_optimizer(opt, model)

    # リプレイ対象のデータのインデックス
    replay_indices = None

    # 通常学習時のエポック数
    original_epochs = opt.epochs

    # タスクの範囲を設定
    if opt.end_task is not None:
        opt.end_task = min(opt.end_task+1, opt.n_cls // opt.cls_per_task)
    else:
        opt.end_task = opt.n_cls // opt.cls_per_task
    
    
    # タスク毎に順番に学習
    for target_task in range(0, opt.end_task):

        # 学習中のタスク
        opt.target_task = target_task

        # modelのコピー（deepcopyによってmodelとmodel2は完全に独立）
        model2 = copy.deepcopy(model)

        # リプレイバッファ内データのインデックス
        replay_indices = set_replay_samples(opt, model)
        # print("len(replay_indices): ", len(replay_indices))
        np.save(
          os.path.join(opt.log_folder, 'replay_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(replay_indices))
        
        # データローダーの作成
        train_loader, subset_indices = set_loader(opt, replay_indices)
        np.save(
          os.path.join(opt.log_folder, 'subset_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(subset_indices))
        
        # 何エポック学習するかの決定
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs


        # 訓練
        for epoch in range(1, opt.epochs + 1):

            # 学習率の調整
            adjust_learning_rate(opt, optimizer, epoch)

            # 1エポックの学習
            time1 = time.time()
            loss, model2 = train(train_loader, model, model2, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # 最後のモデルを保存
        save_file = os.path.join(
            opt.save_folder, 'last_{policy}_{target_task}.pth'.format(policy=opt.replay_policy ,target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':

    main()
