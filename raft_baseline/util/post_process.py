import os, sys
from PIL import Image, ImageFile
import os.path
import sys
sys.path.append("../../")
import torch
import numpy as np
from os.path import join as opj
import argparse
from tqdm import tqdm
from raft_baseline.util.common import fix_seed
from raft_baseline.config.conf_loader import YamlConfigLoader
from raft_baseline.train.dataset import RaftDataset, BucketedDataset, RaftPostDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from torch import nn, optim
import logging
import segmentation_models_pytorch as smp
import pandas as pd
from torchsummary import summary
# import matplotlib.pyplot as plt
# ! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torchsummary import summary
import torchmetrics
import shutil
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('train')
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def make_dataset(to_pred_dir, model_dir, mode):
    # run_py_path = os.path.abspath(__file__)
    # model_dir = os.path.dirname(run_py_path)
    # model_dir = '/data/user/zhaozeming/competition/fujian-bigdata-raft-segmentation/raft_baseline'
    config_path = '../config/raft_baseline_v2_config.yaml'
    conf_loader = YamlConfigLoader(yaml_path=config_path)
    ratio = conf_loader.attempt_load_param("ratio")
    aug = AugmentationTool(conf_loader)
    device = conf_loader.attempt_load_param("device")
    to_pred_dir = os.path.join(to_pred_dir, f"{mode}_post")
    from_pred_dir = conf_loader.attempt_load_param("train_dir")
    # model
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        try:
            model.load_state_dict(torch.load(
                os.path.join(model_dir, "ddp_weights", conf_loader.attempt_load_param("pretrained_path"))))
        except Exception as e:
            model.load_state_dict({k.replace('module.', ''): v for k, v in
                                   torch.load(os.path.join(model_dir, "ddp_weights",
                                                           conf_loader.attempt_load_param("pretrained_path"))).items()})


    summary(model, input_size=(3, 512, 512), device="cpu")
    model = model.to(device)
    model.eval()
    train_dataset = RaftPostDataset(conf_loader=conf_loader, mode=mode, aug=aug)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)

    with torch.no_grad():
        train_epoch = tqdm(train_dataloader, total=int(len(train_dataloader)))
        for i, data in enumerate(train_epoch):
            inputs = data['image']
            targets = data['mask'].squeeze(0)
            input_path = data['image_path'][0].split('/')[-1]
            # mask_path = data['mask_path'][0].split('/')[-1]
            # mask_path = data[3]
            y_true = targets.to(device, torch.float32, non_blocking=True)
            logits = model.predict(inputs.to(device, torch.float32, non_blocking=True))
            logits = torch.sigmoid(logits)
            logits = logits.squeeze(0).squeeze(0)

            train_batch_f1_score = train_f1_metric.update(logits, y_true)
            logits = logits.squeeze(0).squeeze(0)
            logits[logits >= ratio] = 1
            logits[logits < ratio] = 0

            logits = logits.cpu().detach().numpy().astype(np.uint8)
            new_img = Image.fromarray(logits)
            new_mask = Image.fromarray(targets.cpu().detach().numpy().astype(np.uint8))
            # input_show_img = inputs.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            # input_show_img = Image.fromarray(input_show_img)
            # new_input_show_img = Image.open(data['image_path'][0])
            # plt.subplot(1, 4, 1)
            # plt.imshow(new_img)
            # plt.subplot(1, 4, 2)
            # plt.imshow(new_mask)
            # plt.subplot(1, 4, 3)
            # plt.imshow(input_show_img)
            # plt.subplot(1, 4, 4)
            # plt.imshow(new_input_show_img)
            # plt.show()
            #
            new_img_path = os.path.join(to_pred_dir, f'images', input_path)
            if not os.path.exists(os.path.dirname(new_img_path)):
                os.makedirs(os.path.dirname(new_img_path))
            new_img.save(new_img_path)
            mask_path = data['image_path'][0].replace('images', 'labels')

            if os.path.exists(mask_path):
                new_mask_path = os.path.join(to_pred_dir, f'labels', input_path)
                if not os.path.exists(os.path.dirname(new_mask_path)):
                    os.makedirs(os.path.dirname(new_mask_path))
                new_mask.save(new_mask_path)
            else:
                raise Exception(f'{mask_path} not exists')
        train_epoch_f1_score = train_f1_metric.compute().cpu().numpy()
        logger.info(f"train F1_score： {train_epoch_f1_score}")


if __name__ == '__main__':
    to_predict_dir = "/data/cx/datasets/fujian_gis_data"
    model_dir = '/data/user/zhaozeming/competition/fujian-bigdata-raft-segmentation/raft_baseline'
    make_dataset(to_predict_dir, model_dir, mode='train')
    make_dataset(to_predict_dir, model_dir, mode='val')