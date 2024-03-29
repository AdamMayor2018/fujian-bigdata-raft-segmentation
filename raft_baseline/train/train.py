import os.path
import sys
sys.path.append("../../")
import torch
from os.path import join as opj
import numpy as np
import argparse
from tqdm import tqdm
from raft_baseline.util.common import fix_seed
from raft_baseline.config.conf_loader import YamlConfigLoader
from raft_baseline.train.dataset import RaftDataset, BucketedDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from torch import nn, optim
import logging
import segmentation_models_pytorch as smp
import pandas as pd
from torchsummary import summary
import torchmetrics

logger = logging.getLogger('train')
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 使用argparse解析命令行参数
def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Path to training config')
    parser.add_argument('--config', '-c', type=str, default='',
                        help='Path to the experiment configuration YAML file.')

    args = parser.parse_args()
    return args.config

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
    return inputs, targets


if __name__ == '__main__':
    # fix seed
    seed = 2022
    fix_seed(seed)
    # load global config
    config = parse_command_line_args() or "../config/raft_baseline_config.yaml"
    conf_loader = YamlConfigLoader(yaml_path=config)
    device = conf_loader.attempt_load_param("device")
    device = device if torch.cuda.is_available() else "cpu"

    # augmentation
    aug = AugmentationTool(conf_loader)
    # dataset
    train_dataset = BucketedDataset(conf_loader=conf_loader, mode="train", aug=aug)
    valid_dataset = RaftDataset(conf_loader, mode="val", aug=aug)
    train_loader = DataLoader(train_dataset, batch_size=conf_loader.attempt_load_param("train_batch_size"),
                              shuffle=True, num_workers=4, pin_memory=True, collate_fn=my_collate, drop_last=True)
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
    model = smp.PAN(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights='imagenet',
        in_channels=3,
        encoder_output_stride=32,
        classes=1,
        activation=None
    )
    #summary(model, (3, resolution[0], resolution[1]), device="cpu")


    # load pretrained
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(conf_loader.attempt_load_param("pretrained_path")))
    model = model.to(device)
    # critirion optimizer scheduler

    # criterion = nn.BCEWithLogitsLoss().to(device)
    #criterion = dice_bce_loss_with_logits(device=device).to(device)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(device)
    criterion2 = smp.losses.SoftBCEWithLogitsLoss().to(device)
    #criterion2 = nn.BCEWithLogitsLoss().to(device)
    optim_params = conf_loader.attempt_load_param("optim_params")
    for k, v in optim_params.items():
        if isinstance(v, str):
            optim_params[k] = eval(v)
    # optim_params = {k: eval(v) for k, v in optim_params.items() if type(v) == "str"}
    optimizer = optim.AdamW(model.parameters(), **optim_params)
    sched_params = conf_loader.attempt_load_param("sched_params")
    for k, v in sched_params.items():
        if isinstance(v, str):
            sched_params[k] = eval(v)
    if conf_loader.attempt_load_param("lr_scheduler_name") == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)
    elif conf_loader.attempt_load_param("lr_scheduler_name") == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
    elif conf_loader.attempt_load_param('lr_scheduler_name') == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_params)
    # grad scaler
    scaler = torch.cuda.amp.GradScaler()
    log_cols = ['epoch', 'lr', 'loss_trn', 'loss_val', 'f1_trn', 'f1_val']
    # save best models
    best_k = conf_loader.attempt_load_param("save_best_num")
    best_scores = np.zeros((best_k, 2))
    #best_scores[:, 1] += 1e6
    result_dict = {}
    # train
    record_df = pd.DataFrame(columns=log_cols, dtype=object)
    train_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    for epoch in range(1, conf_loader.attempt_load_param("num_epochs") + 1):
        # if epoch >= 70:
        #     transform = aug.get_transforms_valid()
        #     train_dataset.set_transform(transform)
        # if epoch == 2:
        #     import pdb
        #     pdb.set_trace()
        train_epoch_loss = 0
        valid_epoch_loss = 0
        train_targets_all = []
        train_logits_all = []
        val_targets_all = []
        val_logits_all = []

        logger.info(f"start trainning epoch : {epoch}.")
        logger.info(f"lr: {[group['lr'] for group in optimizer.param_groups]}")
        #train_loader.sampler.set_epoch(epoch)
        model.train()
        train_epoch = tqdm(train_loader, total=int(len(train_loader)))
        train_epoch_f1_scores = []
        valid_epoch_f1_scores = []
        # FI score container
        with (torch.cuda.amp.autocast()):
            for i, data in enumerate(train_epoch):

                inputs = data[0]
                targets = data[1]
                train_batch = inputs.shape[0]
                logits = model(inputs.to(device, torch.float32, non_blocking=True))
                y_true = targets.to(device, torch.float32, non_blocking=True)
                # logger.info(f"{logits.shape}, {y_true.shape}")
                logits = logits.squeeze(1)
                #train_batch_loss = criterion(logits, y_true)
                train_batch_loss = 0.5 * criterion(logits, y_true) + 0.5 * criterion2(logits, y_true)
                #train_batch_loss = criterion(logits, y_true)
                ##train_batch_loss = (0.5 - epoch / num_epochs * 1/2) * criterion(logits, y_true) + (0.5 + epoch / num_epochs * 1/2) * criterion2(logits, y_true)
                # logger.info(f"train batch : {i}, dice_loss: {0.5 * criterion(logits, y_true)}, bce_loss: {0.5 * criterion2(logits, y_true)}")
                #logger.info(f"train batch : {i}, f1 loss: {train_batch_loss}")
                train_batch_f1_score = train_f1_metric.update(logits, y_true)
                train_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_epoch_loss += train_batch_loss.item() * train_batch
                # logits = torch.sigmoid(logits).detach().cpu().numpy()
                # logits[logits >= 0.5] = 1
                # logits[logits < 0.5] = 0

                # train_targets_all.extend(targets.numpy().flatten().tolist())
                # train_logits_all.extend(logits.flatten().tolist())
                # batch_train_f1_score = f1_score(targets.flatten().tolist(), logits.flatten().tolist())
                # np_train_f1_score = cal_np_f1_score(targets.flatten().numpy(), logits.flatten())
                # train_epoch_f1_scores.append(batch_train_f1_score)
                # logger.info(f"batch f1 score: {batch_train_f1_score}, np f1 score:{np_train_f1_score}")

            avg_train_loss = train_epoch_loss / len(train_dataset)
            # train_epoch_f1_score = sum(train_epoch_f1_scores) / len(train_epoch_f1_scores)
            train_epoch_f1_score = train_f1_metric.compute().cpu().numpy()
            # train_epoch_f1_score = cal_np_f1_score(np.array(train_targets_all), np.array(train_logits_all))
            logger.info(f"epoch {epoch}, train loss: {avg_train_loss}, train F1_score： {train_epoch_f1_score}")
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
            scheduler.step(valid_epoch_loss)
            valid_epoch_f1_score = cal_np_f1_score(np.array(val_targets_all), np.array(val_logits_all))
            # valid_epoch_f1_score = sum(valid_epoch_f1_scores) / len(valid_epoch_f1_scores)
            logger.info(f"epoch {epoch}, val loss: {avg_val_loss}, valid F1_score： {valid_epoch_f1_score}")

            # save topk val loss model weights
            save_dir = conf_loader.attempt_load_param("weights_path")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # 小到大排序
            weight_save_path = opj(save_dir,
                                   f'model_seed{seed}_fold0_epoch{epoch}_val{round(avg_val_loss, 4)}_f1_score_{round(valid_epoch_f1_score, 4)}.pth')
            result_dict[epoch] = weight_save_path
            if valid_epoch_f1_score > best_scores[-1, 1]:
                # topk
                torch.save(model.state_dict(), weight_save_path)  # save
                prepare_del = best_scores[-1][0]
                best_scores[-1] = [epoch, round(valid_epoch_f1_score, 4)]
                # delete
                if prepare_del != 0:
                    logger.info(f"delete worse weight: {result_dict[prepare_del]}")
                    os.remove(result_dict[prepare_del])
            best_scores = best_scores[np.argsort(-best_scores[:, 1])]
            logger.info(f"current best scores: {best_scores}")

        record_df.loc[epoch - 1, log_cols] = np.array([epoch,
                                                       [group['lr'] for group in optimizer.param_groups],
                                                       avg_train_loss, avg_val_loss,
                                                       train_epoch_f1_score, valid_epoch_f1_score], dtype='object')
    record_df.to_csv(conf_loader.attempt_load_param("results_path") + f'log_seed{seed}_retrain_result.csv', index=False)
