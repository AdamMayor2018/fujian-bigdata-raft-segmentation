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
from raft_baseline.train.dataset import RaftDataset, BucketedDataset, RaftPostDataset, BucketedPostDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from torch import nn, optim
import logging
import segmentation_models_pytorch as smp
import pandas as pd
import torchmetrics
from torchsummary import summary
import matplotlib.pyplot as plt

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
    inputs = np.stack([data["image"] for data in batch])
    targets = np.stack([data["mask"] for data in batch])
    return inputs, targets


if __name__ == '__main__':
    # fix seed
    seed = 2022
    fix_seed(seed)
    # load global config
    config = parse_command_line_args() or "../config/raft_post_v2_config.yaml"
    conf_loader = YamlConfigLoader(yaml_path=config)
    device = conf_loader.attempt_load_param("device")
    device = device if torch.cuda.is_available() else "cpu"
    # v2 post train process dataset -> model1 predict > model2 train
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
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("model_backbone"),
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None
    )
    model_p = smp.Unet(
        encoder_name=conf_loader.attempt_load_param("post_backbone"),
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        activation=None
    )

    # summary(model, (3, resolution[0], resolution[1]), device="cpu")
    train_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    val_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    model1_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)

    # load pretrained
    if conf_loader.attempt_load_param("model_pretrained") and conf_loader.attempt_load_param("model_pretrained_path"):
        model.load_state_dict(torch.load(
            os.path.join("../../model/", conf_loader.attempt_load_param("model_pretrained_path"))))

    model = model.to(device)
    model.eval()
    model_p = model_p.to(device)

    # critirion optimizer scheduler
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True).to(device)
    criterion2 = smp.losses.SoftBCEWithLogitsLoss().to(device)
    optim_params = conf_loader.attempt_load_param("optim_params")
    for k, v in optim_params.items():
        if isinstance(v, str):
            optim_params[k] = eval(v)
    # optim_params = {k: eval(v) for k, v in optim_params.items() if type(v) == "str"}
    optimizer = optim.SGD(model_p.parameters(), **optim_params)
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
    for epoch in range(1, conf_loader.attempt_load_param("num_epochs") + 1):
        train_epoch_loss = 0
        valid_epoch_loss = 0
        train_targets_all = []
        train_logits_all = []
        val_targets_all = []
        val_logits_all = []

        logger.info(f"start trainning epoch : {epoch}.")
        logger.info(f"lr: {[group['lr'] for group in optimizer.param_groups]}")

        #train_loader.sampler.set_epoch(epoch)
        model_p.train()
        train_epoch = tqdm(train_loader, total=int(len(train_loader)))
        train_epoch_f1_scores = []
        valid_epoch_f1_scores = []
        # FI score container
        with (torch.cuda.amp.autocast()):
            for i, data in enumerate(train_epoch):

                inputs = data[0]
                targets = data[1]
                train_batch = inputs.shape[0]
                inputs = torch.tensor(inputs)
                targets = torch.tensor(targets)
                # model1 predict , model2 train
                logit_1 = model.predict(inputs.to(device, torch.float32, non_blocking=True))  # (batch, 1, 512, 512), device='cuda'

                output = model_p(logit_1.to(torch.float32, non_blocking=True))  # (batch, 1, 512, 512), device='cuda'
                y_true = targets.to(device, torch.float32, non_blocking=True)
                # logger.info(f"{logits.shape}, {y_true.shape}")
                output = output.squeeze(1)

                #train_batch_loss = criterion(logits, y_true)
                train_batch_loss = criterion(output, y_true) + criterion2(output, y_true)

                train_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_epoch_loss += train_batch_loss.item() * train_batch
                # logits = torch.sigmoid(logit_1).detach().cpu().numpy()
                train_batch_f1_score = train_f1_metric.update(output, y_true)
                model1_batch_f1_score = model1_f1_metric.update(logit_1.squeeze(1), y_true)

            avg_train_loss = train_epoch_loss / len(train_dataset)
            train_epoch_f1_score = train_f1_metric.compute().cpu().numpy()
            model1_epoch_f1_score = model1_f1_metric.compute().cpu().numpy()
            # train_epoch_f1_score = sum(train_epoch_f1_scores) / len(train_epoch_f1_scores)
            # train_epoch_f1_score = cal_np_f1_score(np.array(train_targets_all), np.array(train_logits_all))
            logger.info(f"epoch {epoch}, train loss: {avg_train_loss}, train F1_score： {train_epoch_f1_score}")
            logger.info(f"epoch {epoch}, train model1 F1_score： {model1_epoch_f1_score}")
            # validation
            # del data, loss, logits, y_true, inputs, targets
            model_p.eval()
            # torch.cuda.empty_cache()
            # gc.collect()
            val_epoch = tqdm(valid_loader, total=int(len(valid_loader)))
            for i, data in enumerate(val_epoch):
                inputs = data[0]
                targets_v = data[1]
                inputs = torch.tensor(inputs)
                targets_v = torch.tensor(targets_v)
                with torch.no_grad():
                    val_batch = inputs.shape[0]
                    logits_v = model.predict(inputs.to(device, torch.float32, non_blocking=True))
                    output_v = model_p(logits_v.to(torch.float32, non_blocking=True))
                    output_v = output_v.squeeze(1)
                    y_true_v = targets_v.to(device, torch.float32, non_blocking=True)
                    # val_batch_loss = criterion(logits.squeeze(1), y_true).item() * val_batch
                    try:
                        val_batch_loss = criterion(output_v, y_true_v) + criterion2(output_v, y_true_v)
                    except Exception as e:
                        import pdb;pdb.set_trace()
                    #val_batch_loss = criterion(logits.squeeze(1), y_true).item() * val_batch
                    # logger.info(
                    #     f"val batch : {i}, dice_loss: {0.5 * criterion(logits, y_true)}, bce_loss: {0.5 * criterion2(logits, y_true)}")
                    val_batch_f1_score = val_f1_metric.update(output_v, y_true_v)
                    valid_epoch_loss += val_batch_loss.item() * val_batch
                    # output_v = torch.sigmoid(output_v).detach().cpu().numpy()
                    # output_v[output_v >= 0.5] = 1
                    # output_v[output_v < 0.5] = 0
                    # val_logits_all.extend(output_v.flatten().tolist())
                    # val_targets_all.extend(targets_v.numpy().flatten().tolist())
                    # batch_val_f1_score = f1_score(targets.flatten().tolist(), logits.flatten().tolist())
                    # valid_epoch_f1_scores.append(batch_val_f1_score)
                # release GPU memory cache
                # del data, loss, logits, y_true, inputs, targets
                # torch.cuda.empty_cache()
                # gc.collect()
            valid_epoch_f1_score = val_f1_metric.compute().cpu()
            avg_val_loss = valid_epoch_loss / len(valid_dataset)
            scheduler.step(valid_epoch_loss)
            # valid_epoch_f1_score = cal_np_f1_score(np.array(val_targets_all), np.array(val_logits_all))
            # valid_epoch_f1_score = sum(valid_epoch_f1_scores) / len(valid_epoch_f1_scores)
            logger.info(f"epoch {epoch}, val loss: {avg_val_loss}, valid F1_score： {valid_epoch_f1_score}")

            # save topk val loss model weights
            save_dir = conf_loader.attempt_load_param("weights_path")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # 小到大排序
            weight_save_path = opj(save_dir,
                                   f'model_seed{seed}_fold0_epoch{epoch}_val{round(avg_val_loss, 4)}_f1_score_{round(valid_epoch_f1_score.item(), 4)}.pth')
            result_dict[epoch] = weight_save_path
            if valid_epoch_f1_score.item() > best_scores[-1, 1]:
                # topk
                torch.save(model_p.eval().state_dict(), weight_save_path)  # save
                prepare_del = best_scores[-1][0]
                best_scores[-1] = [epoch, round(valid_epoch_f1_score.item(), 4)]
                # delete
                if prepare_del != 0:
                    logger.info(f"delete worse weight: {result_dict[prepare_del]}")
                    os.remove(result_dict[prepare_del])
            best_scores = best_scores[np.argsort(-best_scores[:, 1])]
            logger.info(f"current best scores: {best_scores}")

        record_df.loc[epoch - 1, log_cols] = np.array([epoch,
                                                       [group['lr'] for group in optimizer.param_groups],
                                                       avg_train_loss, avg_val_loss,
                                                       train_epoch_f1_score, valid_epoch_f1_score.item()], dtype='object')
    record_df.to_csv(conf_loader.attempt_load_param("results_path") + f'log_seed{seed}_retrain_result.csv', index=False)
