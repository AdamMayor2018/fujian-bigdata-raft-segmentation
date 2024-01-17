# 训练数据处理
import numpy as np
from collections import defaultdict
import os
import cv2
from os.path import join as opj
from torch.utils.data import Dataset
from raft_baseline.config.conf_loader import YamlConfigLoader
from glob import glob
from model.transform import AugmentationTool
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random


class BucketedDataset(Dataset):
    def __init__(self, conf_loader: YamlConfigLoader, mode: str, aug: AugmentationTool):
        self.conf_loader = conf_loader
        self.world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ else 1
        self.mode = mode
        self.data_dir = self.conf_loader.attempt_load_param("train_dir") \
            if self.mode == "train" else conf_loader.attempt_load_param("val_dir")
        self.transform = aug.get_transforms_train() if self.mode == "train" else aug.get_transforms_valid()
        self.width = self.conf_loader.attempt_load_param("train_width") \
            if self.mode == "train" else self.conf_loader.attempt_load_param("val_width")
        self.height = self.conf_loader.attempt_load_param("train_height") \
            if self.mode == "train" else self.conf_loader.attempt_load_param("val_height")

        self.bucket_dir = self.conf_loader.attempt_load_param("bucket_dir")

        # check if self.bucket_dir exist, if self.bucket
        if not os.path.exists(self.bucket_dir): os.makedirs(self.bucket_dir)

        self.bucket_count = self.conf_loader.attempt_load_param("bucket_count")
        self.bucket_random_seed = self.conf_loader.attempt_load_param("bucket_random_seed")
        self.min_bucket_size = self.conf_loader.attempt_load_param("min_bucket_size")
        self.min_mask_ratio = self.conf_loader.attempt_load_param("min_mask_ratio")
        self.shuffle_within_buckets = self.conf_loader.attempt_load_param("shuffle_within_buckets")
        # np.random.seed(self.bucket_random_seed)

        self.images = glob(opj(self.data_dir, "images", "*.jpg"))

        if 'bucket_0.csv' not in os.listdir(self.bucket_dir):
            self.data = self.create_data()
            self.buckets = self.create_buckets()
            self.buckets = self.filter_buckets()
            self.describe_buckets()
            self.save_buckets()
        else:
            self.buckets = self.read_buckets()

        if self.shuffle_within_buckets:
            for samples in self.buckets.values():
                samples.sample(frac=1, random_state=self.bucket_random_seed)

        self.bucket_num = len(self.buckets)
        self.bucket_size = self.get_bucket_size()

    def _to_np(self, image_path, gray=False):
        if not gray:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape

        # 如果尺寸和配置不通进行缩放
        if h != self.height or w != self.width:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if gray: image[image != 0] = 1

        return image

    def create_data(self):
        data = pd.DataFrame()
        data['image_path'] = self.images
        data['label_path'] = data['image_path'].apply(lambda x: opj(self.data_dir, "labels", os.path.basename(x)))
        data['image'] = data['image_path'].apply(lambda x: self._to_np(x))
        data['label'] = data['label_path'].apply(lambda x: self._to_np(x))
        data['label_mean'] = data['label'].apply(lambda x: x.mean())
        return data

    def create_buckets(self):
        max_mask_ratio = self.data['label_mean'].max()
        bins = np.linspace(self.min_mask_ratio, max_mask_ratio, self.bucket_count + 1)

        bucketed_data = {0: self.data[
            (self.data['label_mean'] >= 0) &
            (self.data['label_mean'] < self.min_mask_ratio)
            ].reset_index(drop=True)}

        for i in range(1, self.bucket_count):
            bucketed_data[i] = self.data[
                (self.data['label_mean'] >= bins[i - 1]) &
                (self.data['label_mean'] < bins[i])
                ].reset_index(drop=True)

        bucketed_data[i+1] = self.data[
            (self.data['label_mean'] >= bins[i]) &
            (self.data['label_mean'] <= max_mask_ratio)
            ].reset_index(drop=True)

        return bucketed_data

    def filter_buckets(self):
        buckets = {}
        bin_num = 0

        for k, v in self.buckets.items():
            if len(v) >= self.min_bucket_size:
                buckets[bin_num] = v
                bin_num += 1

        return buckets

    def describe_buckets(self):
        feature_column = 'label_mean'
        for k, v in self.buckets.items():
            print(
                f'桶{k} - size:{len(v)} - mask range: {round(v[feature_column].min(), 2)} to {round(v[feature_column].max(), 2)} - percent: {round(len(v) / len(self.images), 4) * 100}%')

    def save_buckets(self):
        for i, data in self.buckets.items():
            data[['image_path', 'label_path', 'label_mean']].to_csv(f'{self.bucket_dir}/bucket_{i}.csv', index=False)

    def read_buckets(self):
        buckets = {}
        bucket_path_list = glob(opj(self.bucket_dir, "*.csv"))

        for bucket_path in bucket_path_list:
            bucket_num = bucket_path.split('_')[-1][0]
            buckets[int(bucket_num)] = pd.read_csv(bucket_path)

        return buckets

    def get_bucket_size(self):
        bucket_size = 0
        for i in range(1, len(self.buckets)):
            bucket_size = max(bucket_size, len(self.buckets[i]))
        return bucket_size

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        bucket_id = (idx // int(self.world_size)) % self.bucket_num
        # print("idx:", idx, "local_rank:", os.environ['LOCAL_RANK'], "bucket_id:", bucket_id, '\n')
        # 获取桶id
        # if idx == 0:
        #     bucket_id = 0
        # else:
        #     bucket_id = idx % self.bucket_num
        # 获取桶内数据
        samples = self.buckets[bucket_id]
        sample_idx = np.random.choice(samples.index, size=1)[0]

        img_path = samples.loc[sample_idx, 'image_path']
        label_path = samples.loc[sample_idx, 'label_path']
        # print(img_path, label_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        return {'image': img, 'mask': mask, 'info': 'idx:' +str(idx) + 'bucket: ' + str(bucket_id) + 'local:' + img_path}

    def __getbuckets__(self):
        return self.buckets

    def __getbucket__(self, bucket_id):
        return self.buckets.get(bucket_id, f'Bucket {bucket_id} does not exist')

    def __len__(self):
        return self.bucket_size * (self.bucket_count + 1)

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


class RaftInferExpansionDataset(Dataset):
    def __init__(self, file_path, conf_loader: YamlConfigLoader, aug: AugmentationTool):
        self.image = np.array(Image.open(file_path)).astype(np.float32).transpose((2, 0, 1))
        self.image = torch.from_numpy(self.image)
        self.transform = aug.get_transforms_valid()
        # print(self.image.shape)
        _, self.height, self.width = self.image.shape
        self.tile_size = conf_loader.attempt_load_param("tile_size")
        self.pad_size = conf_loader.attempt_load_param("pad_size")
        self.matting_size = conf_loader.attempt_load_param("tile_size") - 2 * self.pad_size
        # 先计算在原图上下左右都按照expansion尺寸pad一圈之后再进行滑窗
        # pad image
        self.pad_image = F.pad(self.image, (self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode='reflect').numpy().astype(np.uint8)
        # self.result_blank = np.ones((2000 - 2*self.pad_size, 2000 - 2*self.pad_size, 3)).astype(np.uint8) * 255
        # self.height, self.width, _ = self.result_blank.shape
        # self.pad_image = np.ones((2000, 2000, 3)).astype(np.uint8)
        # self.blank = np.ones_like(self.pad_image).astype(np.uint8) * 255
        #_, self.pad_height, self.pad_width = self.image.shape
        _, self.pad_height, self.pad_width= self.pad_image.shape
        self.num_h = self.height // self.matting_size  # 横着有多少块
        self.num_w = self.width // self.matting_size # 竖着有多少块
        self.num_h += 1 if (self.pad_height % self.tile_size) != 0 else self.num_h
        self.num_w += 1 if (self.pad_width % self.tile_size) != 0 else self.num_w

    def __len__(self):
        return self.num_w * self.num_h

    def __getitem__(self, idx):
        #返回的应该是pad之后的切片图像（给到模型）和在原始图mask上的填充索引
        i_h = idx // self.num_w
        i_w = idx % self.num_w

        # pad crop 左上角顶点
        pad_ymin = i_h * self.tile_size if i_h == 0 else i_h * self.tile_size - 2 * self.pad_size * i_h
        pad_xmin = i_w * self.tile_size if i_w == 0 else i_w * self.tile_size - 2 * self.pad_size * i_w

        # pad crop idx
        pad_ymax = min(pad_ymin + self.tile_size, self.pad_height)

        pad_xmax = min(pad_xmin + self.tile_size, self.pad_width)
        pad_xmin = pad_xmin if pad_xmin + self.tile_size < self.pad_width else self.pad_width - self.tile_size
        pad_ymin = pad_ymin if pad_ymin + self.tile_size < self.pad_height else self.pad_height - self.tile_size
        # self.blank = draw_box(self.blank, cords=(pad_xmin, pad_ymin, pad_xmax, pad_ymax), color=(255, 0, 0), thickness=3)

        # 切片图像
        crop_image = self.pad_image[:, pad_ymin: pad_ymax, pad_xmin: pad_xmax]
        crop_image = crop_image.transpose((1, 2, 0))

        #计算里面去除pad后小框的位置
        orgin_ymin = i_h * self.matting_size
        orgin_xmin = i_w * self.matting_size
        orgin_xmax = min(orgin_xmin + self.matting_size, self.width)
        orgin_ymax = min(orgin_ymin + self.matting_size, self.height)
        orgin_xmin = orgin_xmin if orgin_xmin + self.matting_size < self.width else self.width - self.matting_size
        orgin_ymin = orgin_ymin if orgin_ymin + self.matting_size < self.height else self.height - self.matting_size
        # self.result_blank = draw_box(self.result_blank, cords=(orgin_xmin, orgin_ymin, orgin_xmax, orgin_ymax), color=(0, 255, 0),
        #                       thickness=3)
        # origin crop idx
        if self.transform:
            crop_image = self.transform(image=crop_image)["image"]

        return crop_image,  [pad_xmin, pad_ymin, pad_xmax, pad_ymax], [orgin_xmin, orgin_ymin, orgin_xmax, orgin_ymax]


if __name__ == '__main__':

    conf_loader = YamlConfigLoader(yaml_path="../config/raft_baseline_config.yaml")
    aug = AugmentationTool(conf_loader)
    train_set = RaftDataset(conf_loader, mode="train", aug=aug)
    print(len(train_set))
    print(train_set[0]["image"].shape, train_set[0]["mask"].shape)
