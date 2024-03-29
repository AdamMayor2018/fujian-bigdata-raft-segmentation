
# 训练的多卡版本
import os.path
import random
import sys
sys.path.append("../../")
import torch
from os.path import join as opj
import numpy as np
from tqdm import tqdm
from raft_baseline.util.common import fix_seed
from raft_baseline.config.conf_loader import YamlConfigLoader
from raft_baseline.train.dataset import RandomBalancedSampler, RaftDataset, RaftTrainDataset,BucketedDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from torch import nn, optim
import logging
import segmentation_models_pytorch as smp
import pandas as pd
import torch.distributed as dist
import torchmetrics
from torchsummary import summary
logger = logging.getLogger('train')
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def cal_np_f1_score(targets, logits):
    # 计算TP、FP、FN
    true_positives = np.sum((logits == 1) & (targets == 1))
    predicted_positives = np.sum(logits == 1)
    actual_positives = np.sum(targets == 1)

    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) + 1e-6
    return f1_score


def my_collate(batch):
    inputs = torch.stack([data["image"] for data in batch])
    targets = torch.stack([data["mask"] for data in batch])
    # info = [data["info"] for data in batch]
    return inputs, targets

def train_my_collate(batch):
    inputs = torch.stack([data["image"] for data in batch])
    targets = torch.stack([data["mask"] for data in batch])
    info = [data["info"] for data in batch]
    return inputs, targets, info


if __name__ == '__main__':
    # fix seed
    seed = 2022
    fix_seed(seed)
    conf_loader = YamlConfigLoader(yaml_path="../config/raft_baseline_dist_train_config.yaml")
    gpus = conf_loader.attempt_load_param("device").split(",")
    local_rank = os.environ["LOCAL_RANK"]
    device_id = gpus[int(local_rank)]
    # load global config

    dist.init_process_group(backend="nccl")
    device = torch.device('cuda:{}'.format(device_id))


    # augmentation
    aug = AugmentationTool(conf_loader)
    # dataset
    train_dataset = BucketedDataset(conf_loader=conf_loader, mode="train", aug=aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # random_sampler = RandomBalancedSampler(conf_loader)

    valid_dataset = RaftDataset(conf_loader, mode="val", aug=aug)
    train_loader = DataLoader(train_dataset, batch_size=conf_loader.attempt_load_param("train_batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True, collate_fn=my_collate, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=conf_loader.attempt_load_param("val_batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True, collate_fn=my_collate)
    num_epochs = conf_loader.attempt_load_param("num_epochs")

    # model
    resolution = (conf_loader.attempt_load_param("train_width"), conf_loader.attempt_load_param("train_height"))
    # model_params = conf_loader.attempt_load_param("model_params")
    # model = build_model(model_name=conf_loader.attempt_load_param("backbone_name"), resolution=resolution,
    #                     deep_supervision=model_params["deep_supervision"], clf_head=model_params["clf_head"],
    #                     clf_threshold=eval(model_params["clf_threshold"]),
    #                     load_weights=model_params["load_backbone_weights"])
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    if int(local_rank) == 0:
        summary(model, (3, resolution[0], resolution[1]), device="cpu")

    # load pretrained
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(conf_loader.attempt_load_param("pretrained_path")))
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # critirion optimizer scheduler
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(device)
    criterion2 = nn.BCEWithLogitsLoss().to(device)
    optim_params = conf_loader.attempt_load_param("optim_params")
    for k, v in optim_params.items():
        if isinstance(v, str):
            optim_params[k] = eval(v)
    # optim_params = {k: eval(v) for k, v in optim_params.items() if type(v) == "str"}
    optimizer = optim.AdamW(model.parameters(), **optim_params)
    sched_params = conf_loader.attempt_load_param("sched_params")
    for k, v in sched_params[conf_loader.attempt_load_param("lr_scheduler_name")].items():
        if isinstance(v, str):
            sched_params[conf_loader.attempt_load_param("lr_scheduler_name")][k] = eval(v)
    if conf_loader.attempt_load_param("lr_scheduler_name") == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params['ReduceLROnPlateau'])
    elif conf_loader.attempt_load_param("lr_scheduler_name") == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params['CosineAnnealingLR'])
    elif conf_loader.attempt_load_param('lr_scheduler_name') == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_params['StepLR'])
    # grad scaler
    scaler = torch.cuda.amp.GradScaler()
    log_cols = ['epoch', 'lr', 'loss_trn', 'loss_val', 'f1_train', 'f1_val']
    # save best models
    best_k = conf_loader.attempt_load_param("save_best_num")
    best_scores = np.zeros((best_k, 2))
    best_scores[:, 1] += 0
    result_dict = {}
    train_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    #if int(local_rank) == 0:
    #val_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)

    # train
    record_df = pd.DataFrame(columns=log_cols, dtype=object)
    for epoch in range(1, conf_loader.attempt_load_param("num_epochs") + 1):
        # samples = random_sampler.sample(frac=0, save=True)
        # random_sampler.describe_distribution(samples)
        # train_dataset = RaftTrainDataset(conf_loader, aug, samples)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=conf_loader.attempt_load_param("train_batch_size"),
        #                           shuffle=False, num_workers=4, pin_memory=True, collate_fn=train_my_collate, drop_last=True, sampler=train_sampler)

        train_sampler.set_epoch(epoch)
        train_epoch_loss = 0
        dice_epoch_loss = 0
        bce_epoch_loss = 0
        valid_epoch_loss = 0
        val_targets_all = []
        val_logits_all = []
        train_cal_sum = 0
        if int(local_rank) == 0 or int(local_rank) == 1 or int(local_rank) == 2 or int(local_rank) == 3:
            logger.info(f"start trainning epoch {epoch}.")
            logger.info(f"lr: {[group['lr'] for group in optimizer.param_groups]}")
        model.train()
        train_epoch = tqdm(train_loader, total=int(len(train_loader)))
        valid_epoch_f1_scores = []
        # FI score container
        with (torch.cuda.amp.autocast()):
            for i, data in enumerate(train_epoch):
                inputs = data[0]
                targets = data[1]

                # print(f"mask ratio over batch {i}: {targets[targets==1].sum() / targets.numel()}")
                # if i == 0 or i == 1:
                #     info = data[2]
                #     print(info)
                #     print(f"mask ratio over batch {i}: {targets[targets == 1].sum() / targets.numel()}")
                #     import pdb;pdb.set_trace()
                train_batch = inputs.shape[0]
                logits = model(inputs.to(device, torch.float32, non_blocking=True))
                y_true = targets.to(device, torch.float32, non_blocking=True)
                logits = logits.squeeze(1)
                #train_batch_loss = criterion(logits, y_true)
                dice_loss = criterion(logits, y_true)
                bce_loss = criterion2(logits, y_true)
                train_batch_loss = 0.5 * dice_loss + 0.5 * bce_loss
                # logger.info(f"train batch : {i}, dice_loss: {0.5 * criterion(logits, y_true)}, bce_loss: {0.5 * criterion2(logits, y_true)}")
                dist.barrier()
                train_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_epoch_loss += train_batch_loss.item() * train_batch
                dice_epoch_loss += dice_loss.item() * train_batch
                bce_epoch_loss += bce_loss.item() * train_batch
                train_cal_sum += train_batch
                logits = torch.sigmoid(logits)
                train_batch_f1_score = train_f1_metric.update(logits, y_true)
            train_epoch_f1_score = train_f1_metric.compute().cpu().numpy()


            # Resetting internal state such that metric ready for new data
            train_f1_metric.reset()
                # logits[logits >= 0.5] = 1
                # logits[logits < 0.5] = 0
                # batch_train_f1_score = f1_score(targets.flatten().tolist(), logits.flatten().tolist())
                # np_train_f1_score = cal_np_f1_score(targets.flatten().numpy(), logits.flatten())
                # train_epoch_f1_scores.append(batch_train_f1_score)

            if int(local_rank) == 0 or int(local_rank) == 1 or int(local_rank) == 2 or int(local_rank) == 3:
                avg_train_loss = train_epoch_loss / train_cal_sum
                avg_dice_loss = dice_epoch_loss / train_cal_sum
                avg_bce_loss = bce_epoch_loss / train_cal_sum
                # train_epoch_f1_score = sum(train_epoch_f1_scores) / len(train_epoch_f1_scores)
                logger.info(f"epoch {epoch}, localRank: {int(local_rank)}, train loss: {avg_train_loss}, dice loss: {avg_dice_loss}, bce loss: {avg_bce_loss} train F1_score: {train_epoch_f1_score}")
            # validation
            # del data, loss, logits, y_true, inputs, targets
            model.eval()
            # torch.cuda.empty_cache()
            # gc.collect()
            val_epoch = tqdm(valid_loader, total=int(len(valid_loader)))
            for i, data in enumerate(val_epoch):
                inputs = data[0]
                targets = data[1]
                with torch.no_grad():
                    val_batch = inputs.shape[0]
                    logits = model(inputs.to(device, torch.float32, non_blocking=True))
                    logits = logits.squeeze(1)
                    y_true = targets.to(device, torch.float32, non_blocking=True)
                    # val_batch_loss = criterion(logits.squeeze(1), y_true).item() * val_batch
                    val_batch_loss = 0.5 * criterion(logits, y_true) + 0.5 * criterion2(logits, y_true)
                    #val_batch_loss = criterion(logits.squeeze(1), y_true).item() * val_batch
                    # logger.info(
                    #     f"val batch : {i}, dice_loss: {0.5 * criterion(logits, y_true)}, bce_loss: {0.5 * criterion2(logits, y_true)}")
                    valid_epoch_loss += val_batch_loss.item() * val_batch
                    logits = torch.sigmoid(logits).detach().cpu().numpy()
                    #val_batch_f1_score = val_f1_metric.update(logits, y_true)
                    logits[logits >= 0.5] = 1
                    logits[logits < 0.5] = 0
                    val_logits_all.extend(logits.flatten().tolist())
                    val_targets_all.extend(targets.numpy().flatten().tolist())
                    # batch_val_f1_score = f1_score(targets.flatten().tolist(), logits.flatten().tolist())
                    # valid_epoch_f1_scores.append(batch_val_f1_score)
                # release GPU memory cache
                # del data, loss, logits, y_true, inputs, targets
                # torch.cuda.empty_cache()
                # gc.collect()

            avg_val_loss = valid_epoch_loss / len(valid_dataset)
            avg_val_loss = torch.tensor(avg_val_loss).to(device)
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
            dist_avg_val_loss = avg_val_loss.item() / dist.get_world_size()

            scheduler.step(dist_avg_val_loss)
            valid_epoch_f1_score = cal_np_f1_score(np.array(val_targets_all), np.array(val_logits_all))
            valid_epoch_f1_score = torch.tensor(valid_epoch_f1_score).to(device)
            dist.all_reduce(valid_epoch_f1_score, op=dist.ReduceOp.SUM)
            dis_val_epoch_f1_score = valid_epoch_f1_score.item() / dist.get_world_size()
            #valid_epoch_f1_score = val_f1_metric.compute()
            #valid_epoch_f1_score = sum(valid_epoch_f1_scores) / len(valid_epoch_f1_scores)
            if os.environ['LOCAL_RANK'] == '0':
                logger.info(f"epoch {epoch}, val loss: {dist_avg_val_loss}, valid F1_score： {dis_val_epoch_f1_score}")
            #val_f1_metric.reset()
            # save topk val loss model weights
            save_dir = conf_loader.attempt_load_param("weight_save_path")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # 小到大排序
            weight_save_path = opj(save_dir,
                                   f'model_seed{seed}_fold0_epoch{epoch}_val{round(dist_avg_val_loss, 4)}_f1_score_{round(dis_val_epoch_f1_score, 4)}.pth')
            result_dict[epoch] = weight_save_path

            if dis_val_epoch_f1_score > best_scores[-1, 1]:
                if os.environ['LOCAL_RANK'] == '0':
                    # topk
                    torch.save(model.state_dict(), weight_save_path)  # save
                    prepare_del = best_scores[-1][0]
                    best_scores[-1] = [epoch, round(dis_val_epoch_f1_score, 4)]
                    # delete
                    if prepare_del != 0:
                        logger.info(f"delete worse weight: {result_dict[prepare_del]}")
                        os.remove(result_dict[prepare_del])
                    best_scores = best_scores[np.argsort(-best_scores[:, 1])]
                    logger.info(f"current best scores: {best_scores}")

                    record_df.loc[epoch - 1, log_cols] = np.array([epoch,
                                                               [group['lr'] for group in optimizer.param_groups],
                                                               avg_train_loss, dist_avg_val_loss, train_epoch_f1_score,
                                                                dis_val_epoch_f1_score], dtype='object')
    record_df.to_csv(conf_loader.attempt_load_param("result_csv_path") + f'log_seed{seed}_ddp_result.csv', index=False)
    #dist.destroy_process_group()
