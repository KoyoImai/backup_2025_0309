from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''


def sup_con_loss(features, temperature=0.1, labels=None, mask=None):
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))
    features_norm = F.normalize(features, p=2, dim=1)
    batch_size = features_norm.shape[0]

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features_norm, features_norm.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits)

    logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
    positives_mask = mask * logits_mask
    negatives_mask = 1. - mask

    num_positives_per_row = torch.sum(positives_mask, axis=1)
    denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
        exp_logits * positives_mask, axis=1, keepdims=True)

    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan!")

    log_probs = torch.sum(
        log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                    num_positives_per_row > 0]

    # loss
    loss = -log_probs
    loss = loss.mean()
    return loss


# MOSE用に改良
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature


    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # print("labels: ", labels)
        # print("features.shape: ", features.shape)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        contrast_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # 特徴量の正規化
        features = F.normalize(features, p=2, dim=1)

        batch_size = features.shape[0]
        labels = labels.repeat(2)
        # print(labels.shape)
        # print("labels[0:10]: ", labels[0:10])
        # print("labels[512:522]: ", labels[512:522])
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]

        # loss
        loss = -log_probs
        loss = loss.mean()
        return loss

        # # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        #     0
        # )
        # mask = mask * logits_mask

        # # print("mask.shape: ", mask.shape)                 # mask.shape:  torch.Size([1024, 1024])
        # # print("logits_mask.shape: ", logits_mask.shape)   # logits_mask.shape:  torch.Size([1024, 1024])
        # # assert False

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # # compute mean of log-likelihood over positive
        # # modified to handle edge cases when there is no positive pair
        # # for an anchor point. 
        # # Edge case e.g.:- 
        # # features of shape: [4,1,...]
        # # labels:            [0,1,1,2]
        # # loss before mean:  [nan, ..., ..., nan] 
        # mask_pos_pairs = mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()

        # return loss




# Co2Lの教師あり対照損失
# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
#         assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
        
#         # 特徴量の正規化
#         features = F.normalize(features, p=2, dim=1)

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # print("mask.shape: ", mask.shape)                 # mask.shape:  torch.Size([1024, 1024])
#         # print("logits_mask.shape: ", logits_mask.shape)   # logits_mask.shape:  torch.Size([1024, 1024])
#         # assert False

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

#         curr_class_mask = torch.zeros_like(labels)
#         for tc in target_labels:
#             curr_class_mask += (labels == tc)
#         curr_class_mask = curr_class_mask.view(-1).to(device)
#         loss = curr_class_mask * loss.view(anchor_count, batch_size)

#         if reduction == 'mean':
#             loss = loss.mean()
#         elif reduction == 'none':
#             loss = loss.mean(0)
#         else:
#             raise ValueError('loss reduction not supported: {}'.
#                              format(reduction))

#         return loss
