import sys
import time
import copy
import torch


from utils.util_co2l import AverageMeter
from utils.util_co2l import warmup_learning_rate
from utils.util_co2l import write_csv


def train(train_loader, model, model2, criterion, optimizer, epoch, opt):


    ## modelを訓練モードに変更
    model.train()

    ## メータ
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()

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
        features, encoded = model(images, return_feat=True)

        # IRD (current modelのlogitsを計算)
        if opt.target_task > 0:
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
            logits_mask = torch.scatter(
                torch.ones_like(features1_sim),
                1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                0
            )
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
        
        # Asym SupCon損失の計算
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
        loss_scl = copy.deepcopy(loss.detach().cpu())

        """   教師あり対照損失の勾配を確認    """
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # グラフを保持
        scl_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        # print("scl_grad_norm: ", scl_grad_norm)
        """"""""""""""""""""""""""""""""""""


        # IRD損失の計算 (past modelのlogitsの計算)
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)


        """      蒸留損失の勾配を確認      """
        if opt.target_task > 0:
            optimizer.zero_grad()
            (opt.distill_power * loss_distill).backward(retain_graph=True)  # グラフを保持
            distill_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        """"""""""""""""""""""""""""""""""""

        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, distill=distill))
            sys.stdout.flush()

        
        # 学習記録
        write_csv(value=loss.detach().item(), path=opt.learninglog_folder, file_name="loss")
        write_csv(value=loss_scl.detach().item(), path=opt.learninglog_folder, file_name="loss_scl")

        write_csv(value=loss_grad_norm, path=opt.learninglog_folder, file_name="loss_grad_norm")
        write_csv(value=scl_grad_norm, path=opt.learninglog_folder, file_name="scl_grad_norm")

        if opt.target_task > 0:
            write_csv(value=loss_distill.detach().item(), path=opt.learninglog_folder, file_name="loss_distill")
            write_csv(value=distill_grad_norm, path=opt.learninglog_folder, file_name="distill_grad_norm")



    return losses.avg, model2