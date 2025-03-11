import sys
import time
import copy


import torch
import torch.nn.functional as F


from utils.util_co2l import AverageMeter
from utils.util_co2l import warmup_learning_rate
from utils.util_co2l import write_csv
from losses.loss_multilevel import sup_con_loss


def train(train_loader, model, model2, criterion, optimizer, epoch, opt):


    ## modelを訓練モードに変更
    model.train()

    ## メータ
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_scl = AverageMeter()
    losses_ce = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # 画像の取得
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # 前タスクのデータかどうかを判定
        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)
        
        # warm-up
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # モデルに画像を入力
        feat_list = model.features(images)
        proj_list = model.head(feat_list, use_proj=True)
        pred_list = model.head(feat_list, use_proj=False)

        # MOSEの蒸留損失における生徒出力の番号
        stu_feat = feat_list[model.expert_number]
        stu_feat = model.final_addaption_layer(stu_feat)
        
        
        # 形状確認
        # fea1, fea2, fea3, fea4 = feat_list
        # print("fea1.shape: ", fea1.shape)      # fea1.shape:  torch.Size([1024, 512])
        # print("fea2.shape: ", fea2.shape)      # fea2.shape:  torch.Size([1024, 512])
        # print("fea3.shape: ", fea3.shape)      # fea3.shape:  torch.Size([1024, 512])
        # print("fea4.shape: ", fea4.shape)      # fea4.shape:  torch.Size([1024, 512])

        # fea1, fea2, fea3, fea4 = proj_list
        # print("fea1.shape: ", fea1.shape)      # fea1.shape:  torch.Size([1024, 128])
        # print("fea2.shape: ", fea2.shape)      # fea2.shape:  torch.Size([1024, 128])
        # print("fea3.shape: ", fea3.shape)      # fea3.shape:  torch.Size([1024, 128])
        # print("fea4.shape: ", fea4.shape)      # fea4.shape:  torch.Size([1024, 128])

        # fea1, fea2, fea3, fea4 = pred_list
        # print("fea1.shape: ", fea1.shape)      # fea1.shape:  torch.Size([1024, 100])
        # print("fea2.shape: ", fea2.shape)      # fea2.shape:  torch.Size([1024, 100])
        # print("fea3.shape: ", fea3.shape)      # fea3.shape:  torch.Size([1024, 100])
        # print("fea4.shape: ", fea4.shape)      # fea4.shape:  torch.Size([1024, 100])

        # 損失の初期化
        loss_scl = 0
        loss_ce = 0
        loss = 0

        for i in range(len(feat_list)):

            feat = feat_list[i]
            proj = proj_list[i]
            pred = pred_list[i]

            # 教師あり対照損失
            pro1, pro2 = torch.split(proj, [bsz, bsz], dim=0)
            features = torch.cat([pro1.unsqueeze(1), pro2.unsqueeze(1)], dim=1)
            # loss_scl += criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
            loss_scl += criterion(features, labels)
            # loss_scl += sup_con_loss(proj, opt.temp, labels=labels)
            # print("loss_scl: ", loss_scl)

            # 交差エントロピー損失
            labels_ce = labels.repeat(2)
            loss_ce += F.cross_entropy(pred, labels_ce)
            # print("loss_ce: ", loss_ce)

        # 損失を合計
        loss += loss_scl + loss_ce

        """   教師あり対照損失の勾配を確認    """
        optimizer.zero_grad()
        loss_scl.backward(retain_graph=True)  # グラフを保持
        scl_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        """"""""""""""""""""""""""""""""""""

        """    交差エントロピー損失の勾配を確認     """
        optimizer.zero_grad()
        loss_ce.backward(retain_graph=True)  # グラフを保持
        ce_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        """"""""""""""""""""""""""""""""""""""""""

        # lossesの更新
        losses.update(loss.item(), bsz)
        losses_scl.update(loss_scl.item(), bsz)
        losses_ce.update(loss_ce.item(), bsz)

        # 最適化の実行 & 全体損失の勾配を計算
        optimizer.zero_grad()
        loss.backward()
        loss_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f} {scl.avg:.3f} {ce.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, scl=losses_scl, ce=losses_ce))
            sys.stdout.flush()
        
        # 学習記録
        write_csv(value=loss.detach().item(), path=opt.learninglog_folder, file_name="loss")
        write_csv(value=loss_scl.detach().item(), path=opt.learninglog_folder, file_name="loss_scl")
        write_csv(value=loss_ce.detach().item(), path=opt.learninglog_folder, file_name="loss_ce")

        write_csv(value=loss_grad_norm, path=opt.learninglog_folder, file_name="loss_grad_norm")
        write_csv(value=scl_grad_norm, path=opt.learninglog_folder, file_name="scl_grad_norm")
        write_csv(value=ce_grad_norm, path=opt.learninglog_folder, file_name="ce_grad_norm")

        
        
    return loss, model2
            


            


        