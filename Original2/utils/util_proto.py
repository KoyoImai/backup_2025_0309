from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Sampler
from scipy.stats import multivariate_normal
# from optimizer.sololearn_lars import LARS


from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment



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

# 正規化
def normalize(x): 
    x = x / torch.norm(x, dim=1, p=2, keepdim=True)
    return x

# 対照学習のデータ拡張
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
# プロトタイプベクトル同士の角度計算
def compute_angles(vectors):
    proto = vectors.cpu().numpy()
    dot = np.matmul(proto, proto.T)
    dot = dot.clip(min=0, max=1)
    theta = np.arccos(dot)
    np.fill_diagonal(theta, np.nan)
    theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
    
    avg_angle_close = theta.min(axis = 1).mean()
    avg_angle = theta.mean()

    return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

# 各クラスの特徴量の平均
def get_prototypes(model, train_loader, prototype_loader, target_task, n_cls, cls_per_task):
    
    # evalモードに変更
    model.eval()

    features = []
    labels = []

    with torch.no_grad():

        for idx, (images, label) in enumerate(prototype_loader):
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            # print("images.shape: ", images.shape)
            # print("label.shape: ", label.shape)

            feature, _ = model(images, return_feat=True)

            features += [feature.detach().cpu()]
            labels += [label.detach().cpu()]
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        # print("features.shape: ", features.shape)
        # print("labels.shape: ", labels.shape)

        task_labels = torch.unique(labels)
        # print("task_labels: ", task_labels)

        proto_list = []

        for class_index in task_labels:
            data_index = (labels == class_index)
            # print("len(data_index): ", len(data_index))

            embeddings = features[data_index]
            # print("embeddings.shape: ", embeddings.shape)

            embeddings_mean = embeddings.mean(dim=0)
            # print("embeddings_mean.shape: ", embeddings_mean.shape)

            proto_list.append(embeddings_mean)
        
        prototypes = torch.stack(proto_list, dim=0)
        # print("prototypes.shape: ", prototypes.shape)

    # trainモードに戻す
    model.train()

    return prototypes, task_labels

# プロトタイプを割り当て順に並び替え
def reorder_pseudo_target_params_and_optimizer_state(model, optimizer, new_indices):
    """
    model.pseudo_targets.base_fc.weight (形状 [num_pt, in_features]) の並びを
    new_indices に従って再配置し、optimizer.state 内の対応するバッファも同様に並び替える。
    """
    with torch.no_grad():
        
        """  モデルの重みパラメータ入れ替え  """
        # ターゲットパラメータへの参照
        weight_param = model.pseudo_targets.base_fc.weight  # nn.Parameter
        print("weight_param.shape: ", weight_param.shape)   # weight_param.shape:  torch.Size([300, 512])
        
        old_weight_data = weight_param.data.clone()
        weight_param.data = old_weight_data[new_indices, :]

        # print("weight_param.data[0]: ", weight_param.data[0])
        # print("old_weight_data[new_indices[0]]: ", old_weight_data[new_indices[0]])

        """  optimizerのモメンタムの入れ替え  """
        # （2）optimizer の state も並び替える
        #      まずは weight_param に対応する state を取り出す
        # print("optimizer.param_groups[0]['params'][-1].shape: ", optimizer.param_groups[0]['params'][-1].shape)  
        # optimizer.param_groups[0]['params'][-1].shape:  torch.Size([300, 512])
        
        # for param_idx, param in enumerate(optimizer.param_groups[0]['params']):
        #     print("param_idx: ", param_idx)
        
        # ## モメンタムと対応パラメータを確認
        # ## 最後のパラメータ
        # param_names = []
        # for param_idx, param in enumerate(optimizer.param_groups[0]['params']):
        #     state = optimizer.state[param]
        #     param_name = None
        #     # モデル内のパラメータ名を取得
        #     for name, p in model.named_parameters():
        #         if param is p:
        #             param_name = name
        #             param_names += [name]
        # print("param_names: ", param_names)
        # print("len(param_names): ", len(param_names))
        # assert False

        # print("optimizer.state_dict().keys(): ", optimizer.state_dict().keys())   # optimizer.state_dict().keys():  dict_keys(['state', 'param_groups'])
        ####print("optimizer.state_dict()['state']: ", optimizer.state_dict()['state'])
        # print("optimizer.state_dict()['state'][62]['momentum_buffer'].shape: ", optimizer.state_dict()['state'][62]['momentum_buffer'].shape)
        # # optimizer.state_dict()['state'][62]['momentum_buffer'].shape:  torch.Size([512, 512])
        #####print("optimizer.state_dict()['state'][63]['momentum_buffer'].shape: ", optimizer.state_dict()['state'][63]['momentum_buffer'].shape)
        # # optimizer.state_dict()['state'][63]['momentum_buffer'].shape:  torch.Size([512])
        # print("optimizer.state_dict()['state'][64]['momentum_buffer'].shape: ", optimizer.state_dict()['state'][64]['momentum_buffer'].shape)
        ####print("optimizer.state_dict()['state'].keys(): ", optimizer.state_dict()['state'].keys())
        # assert False

        if not (64 in optimizer.state_dict()['state'].keys()):
            return

        # print("Optimizer State:")
        # for param_idx, param in enumerate(optimizer.param_groups[0]['params']):
        #     print(f"Parameter {param_idx}:")
        #     print(optimizer.state[param])

        param_state = optimizer.state["weight_param"]
        # print("param_state: ", param_state)

        param_state = optimizer.state_dict()['state'][64]    #['momentum_buffer']
        # print("param_state: ", param_state)
        # print("param_state.shape: ", param_state.shape)  # param_state.shape:  torch.Size([30, 128])
        # print("new_indices: ", new_indices)


        # SGD(momentum)の場合
        if 'momentum_buffer' in param_state:
            old_momentum = param_state['momentum_buffer'].clone()
            param_state['momentum_buffer'] = old_momentum[new_indices, :]

# プロトタイプの割り当て
def get_assignment(model, train_loader, prototype_loader, optimizer, target_task, n_cls, cls_per_task, opt):
    
    ## 入れ替え順序の初期化
    new_indices = torch.arange(opt.num_pt)
    # print("new_indices: ", new_indices)

    ## train_loaderに含まれる新クラスデータの特徴量の平均を計算
    prototypes, task_labels = get_prototypes(model, train_loader, prototype_loader, target_task, n_cls, cls_per_task)
    prototypes = normalize(prototypes)
    # print("prototypes.shape: ", prototypes.shape)   # prototypes.shape:  torch.Size([20, 512])

    ## 擬似ターゲットを取り出す
    pseudo_targets = model.prototypes.return_values()
    pseudo_targets = pseudo_targets[cls_per_task*target_task:]
    # print("pseudo_targets.shape: ", pseudo_targets.shape)   # pseudo_targets.shape:  torch.Size([300, 512])
                                                            # pseudo_targets.shape:  torch.Size([280, 512])
    
    ## prototypes と model.pseudo_targets.base_fc[cls_per_task*target_task:]の類似度を計算し，
    ## 計算結果をコストにscipy.optimize.linear_sum_assignmentを実行
    cost = cosine_similarity(prototypes.detach().cpu().numpy(), pseudo_targets.detach().cpu().numpy())
    # print("cost.shape: ", cost.shape)   # cost.shape:  (20, 300)
                                        # cost.shape:  (20, 280)

    _, col_ind = linear_sum_assignment(cost, maximize=True)
    # print("col_ind: ", col_ind)               # 
    # print("col_ind.shape: ", col_ind.shape)   # col_ind.shape:  (20,)
    # assert False
    # print("cost[0][col_ind[0]:col_ind[0]+3]: ", cost[0][col_ind[0]:col_ind[0]+3])
    # print("max(cost[0]): ", max(cost[0]))
    # print("max(cost[1]): ", max(cost[1]))
    
    # col_ind[0] 番目の擬似ターゲット pseudo_targets[col_ind[0]] と 
    # (cls_per_task*target_task)番目の擬似ターゲット pseudo_targets[cls_per_task*target_task] 
    # を交代するために new_inidices の順番を入れ替える
    for i, j in enumerate(range(cls_per_task*target_task, cls_per_task*(target_task+1))):
        
        # cls_per_task*target_task+col_ind[i]番目と (cls_per_task*target_task+j)番目を入れ替える
        # print(cls_per_task*target_task+col_ind[i], j)
        new_indices[j] = cls_per_task*target_task+col_ind[i]
        new_indices[cls_per_task*target_task+col_ind[i]] = j
    
    # print("new_indices: ", new_indices)
    # print("torch.sort(new_indices): ", torch.sort(new_indices))

    # for i in range(15):
    #     print(f"new_indices[{20*i}:{20*(i+1)}]: ", new_indices[20*i:20*(i+1)])

    reorder_pseudo_target_params_and_optimizer_state(model, optimizer, new_indices)

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



