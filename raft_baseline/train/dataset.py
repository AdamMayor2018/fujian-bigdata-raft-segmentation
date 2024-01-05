# 训练数据处理
import numpy as np
import cv2
from collections import defaultdict
import os
import cv2
from os.path import join as opj
from torch.utils.data import Dataset
from raft_baseline.config.conf_loader import YamlConfigLoader
from glob import glob
from raft_baseline.train.transform import AugmentationTool


class RaftDataset(Dataset):
    def __init__(self, conf_loader: YamlConfigLoader, mode: str, aug:AugmentationTool):
        self.conf_loader = conf_loader
        self.mode = mode
        self.data_dir = conf_loader.attempt_load_param("train_dir") \
            if self.mode == "train" else conf_loader.attempt_load_param("val_dir")
        images = glob(opj(self.data_dir, "images", "*.jpg"))
        # self.pairs = {"images:": [], "labels": []}
        self.pairs = defaultdict(list)
        for path in images:
            img_name = os.path.basename(path)
            label_path = opj(self.data_dir, "labels", img_name)
            if os.path.exists(label_path):
                self.pairs["images"].append(path)
                self.pairs["labels"].append(label_path)
        self.transform = aug.get_transforms_train() if self.mode == "train" else aug.get_transforms_valid()
        self.width = self.conf_loader.attempt_load_param("train_width") \
            if self.mode == "train" else self.conf_loader.attempt_load_param("val_width")
        self.height = self.conf_loader.attempt_load_param("train_height") \
            if self.mode == "train" else self.conf_loader.attempt_load_param("val_height")

    def __getitem__(self, idx):
        img_path = self.pairs["images"][idx]
        label_path = self.pairs["labels"][idx]
        #print(img_path, label_path)
        img = cv2.imread(img_path)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        h, w, c = img.shape
        # 如果尺寸和配置不通进行缩放
        if h != self.height or w != self.width:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # 原图给二值化的1对应其他值
        mask[mask != 0] = 1
        if self.conf_loader.attempt_load_param("transform") and self.transform:
            augmented = self.transform(image=img.astype(np.uint8),
                                       mask=mask.astype(np.uint8))
            img = augmented['image']
            mask = augmented['mask']
        return {'image': img, 'mask': mask}

    def __len__(self):
        return len(self.pairs["images"])


if __name__ == '__main__':

    conf_loader = YamlConfigLoader(yaml_path="../config/raft_baseline_config.yaml")
    aug = AugmentationTool(conf_loader)
    train_set = RaftDataset(conf_loader, mode="train", aug=aug)
    print(len(train_set))
    print(train_set[0]["image"].shape, train_set[0]["mask"].shape)
