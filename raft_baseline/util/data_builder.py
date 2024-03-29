# -- coding: utf-8 --
# @Time : 2023/12/28 11:05
# @Author : caoxiang
# @File : data_builder.py
# @Software: PyCharm
# @Description:
import os.path
import random

import cv2
import torch
import time
from PIL import Image, ImageFile
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import numpy as np
from os.path import join as opj
from raft_baseline.config.conf_loader import YamlConfigLoader
import glob
from model.draw import draw_box
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random

def generate_data(filename, i, img_patch, mask_patch, conf_loader: YamlConfigLoader):
    img_save_path = opj(conf_loader.attempt_load_param('OUTPUT_PATH'), filename + f'_img_{i:04d}.jpg')
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)  # rgb -> bgr
    cv2.imwrite(img_save_path, img_patch)  # bgr -> rgb

    mask_save_path = opj(conf_loader.attempt_load_param('OUTPUT_PATH'), filename + f'_rle_{i:04d}')

    num_masked_pixels = mask_patch.sum()
    ratio_masked_area = mask_patch.sum() / (mask_patch.shape[0] * mask_patch.shape[1])
    std_img = img_patch.std()
    data = [img_save_path.split('/')[-1], mask_save_path.split('/')[-1],
            num_masked_pixels, ratio_masked_area, std_img]
    return data


def split_tiffs(tif_path: str, conf_loader: YamlConfigLoader):
    """
        拆分tiff文件, 一分为2， 按照图像的长边按照ratio来进行拆分， 避免overlap tile的时候出现数据泄露
    """
    assert tif_path.endswith(".tif"), "tif file must ends with .tif"
    print(tif_path)
    name = os.path.basename(tif_path)
    par_dir = os.path.dirname(tif_path)
    print(par_dir, name)
    ratio = conf_loader.attempt_load_param("split_ratio")
    # img = cv2.imread(tif_path, 1)
    img = Image.open(tif_path)
    img = np.array(img)
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
    # longer side
    longer_side = max(h, w)
    split_idx = int(longer_side * ratio)
    if h > w:
        if img.ndim == 3:
            train_tif = img[: split_idx, :, :]
            val_tif = img[split_idx:, :, :]
        elif img.ndim == 2:
            train_tif = img[: split_idx, :]
            val_tif = img[split_idx:, :]
    else:
        if img.ndim == 3:
            train_tif = img[:, : split_idx, :]
            val_tif = img[:, split_idx:, :]
        elif img.ndim == 2:
            train_tif = img[:, : split_idx]
            val_tif = img[:, split_idx:]
    print(os.path.join(par_dir, f"train_{name}"))
    train_tif = Image.fromarray(train_tif)
    val_tif = Image.fromarray(val_tif)
    train_tif.save(os.path.join(par_dir, f"train_{name}"), "TIFF")
    val_tif.save(os.path.join(par_dir, f"val_{name}"), "TIFF")


class RaftTileDataset(Dataset):
    def __init__(self, filename, conf_loader: YamlConfigLoader, mode="train"):
        super().__init__()
        self.conf_loader = conf_loader
        self.mode = mode
        if os.path.exists(filename):
            path = filename
        else:
            path = opj(self.conf_loader.attempt_load_param("raw_train_dir"), filename + '.tif')
        mask_path = path.replace("img", "mask")
        self.image = rasterio.open(path)
        self.mask = rasterio.open(mask_path)

        if self.image.count != 3:
            subdatasets = self.image.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.image.height, self.image.width
        self.tile_size = self.conf_loader.attempt_load_param('tile_size')
        self.overlap_size = self.conf_loader.attempt_load_param('train_overlap_size') if self.mode == "train" else self.conf_loader.attempt_load_param('val_overlap_size')
        # self.pad_h = self.tile_size - self.h % self.tile_size  # add to whole slide
        # self.pad_w = self.tile_size - self.w % self.tile_size  # add to whole slide
        self.pad_h = (self.tile_size - self.overlap_size) - (self.h - self.overlap_size) % (self.tile_size - self.overlap_size)
        self.pad_w = (self.tile_size - self.overlap_size) - (self.h - self.overlap_size) % (self.tile_size - self.overlap_size)
        # self.num_h = (self.h + self.pad_h) // self.tile_size
        # self.num_w = (self.w + self.pad_w) // self.tile_size

        self.image = torch.from_numpy(self.image.read([1, 2, 3])).float()
        self.mask = torch.from_numpy(self.mask.read([1])).float()
        #pad image and mask
        pad_left = self.pad_w // 2
        pad_right = self.pad_w - pad_left
        pad_top = self.pad_h // 2
        pad_bottom = self.pad_h - pad_top
        self.image = F.pad(self.image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(np.uint8).transpose((1, 2, 0))
        self.mask = F.pad(self.mask, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(np.uint8).transpose((1, 2, 0))
        self.pad_h, self.pad_w, _ = self.image.shape
       #self.result_blank = np.ones((self.pad_h, self.pad_w, 3)).astype(np.uint8) * 255
        # self.num_h = self.pad_h // (self.tile_size - self.overlap_size)  # 横着有多少块
        # self.num_w = self.pad_w // (self.tile_size - self.overlap_size)  # 竖着有多少块
        # self.num_h += 1 if (self.pad_h % self.tile_size) != 0 else self.num_h
        # self.num_w += 1 if (self.pad_w % self.tile_size) != 0 else self.num_w
        self.num_h = (self.pad_h - self.overlap_size) // (self.tile_size - self.overlap_size)
        self.num_w = (self.pad_w - self.overlap_size) // (self.tile_size - self.overlap_size)

    def __len__(self):
        return self.num_h * self.num_w

    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        pad_ymin = i_h * self.tile_size if i_h == 0 else i_h * self.tile_size - self.overlap_size * i_h
        pad_xmin = i_w * self.tile_size if i_w == 0 else i_w * self.tile_size - self.overlap_size * i_w
        # pad crop idx
        pad_ymax = min(pad_ymin + self.tile_size, self.pad_h)

        pad_xmax = min(pad_xmin + self.tile_size, self.pad_w)
        pad_xmin = pad_xmin if pad_xmin + self.tile_size < self.pad_w else self.pad_w - self.tile_size
        pad_ymin = pad_ymin if pad_ymin + self.tile_size < self.pad_h else self.pad_h - self.tile_size

        # self.result_blank = draw_box(self.result_blank, cords=(pad_xmin, pad_ymin, pad_xmax, pad_ymax), color=(255, 0, 0),
        #                       thickness=3)
        # placeholder for input tile (before resize)
        img_patch = np.zeros((self.tile_size, self.tile_size, 3), np.uint8)
        mask_patch = np.zeros((self.tile_size, self.tile_size, 1), np.uint8)
        # replace the value for img patch
        if self.image.shape[-1] == 3:
            img_patch[0:pad_ymax - pad_ymin, 0:pad_xmax - pad_xmin] = self.image[pad_ymin: pad_ymax, pad_xmin: pad_xmax, :]
                #np.moveaxis(self.image.read([1, 2, 3], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)

        # replace the value for mask patch
        if self.mask.shape[-1] == 1:
            mask_patch[0:pad_ymax - pad_ymin, 0:pad_xmax - pad_xmin] = self.mask[pad_ymin: pad_ymax, pad_xmin: pad_xmax, :]
            mask_patch[mask_patch != 0] = 1
                #np.moveaxis(self.mask.read([1], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)
        return {'img': img_patch, 'mask': mask_patch}


if __name__ == '__main__':
    conf_loader = YamlConfigLoader("../config/raft_baseline_config.yaml")
    raw_train_dir = conf_loader.attempt_load_param("raw_train_dir")
    train_dir = conf_loader.attempt_load_param("train_dir")
    raw_val_dir = conf_loader.attempt_load_param("raw_val_dir")
    val_dir = conf_loader.attempt_load_param("val_dir")
    copy_and_paste = conf_loader.attempt_load_param("copy_and_paste")
    copy_and_paste_prob = conf_loader.attempt_load_param("copy_and_paste_prob")
    train_raw_img_paths = glob.glob(opj(raw_train_dir, "img*.tif"))
    val_raw_img_paths = glob.glob(opj(raw_val_dir, "img*.tif"))
    print(train_raw_img_paths)
    masked_imgs = []
    background_imgs = []
    #step1 执行测试集基础切图
    for j, i_path in enumerate(train_raw_img_paths):
        #split_tiffs(i_path, conf_loader)
        dataset = RaftTileDataset(i_path, conf_loader, mode="train")
        for i in range(len(dataset)):
            print(f"making train ({j}_{i}) data pair.")
            pair = dataset[i]
            image = pair["img"]
            mask = pair["mask"]
            if image.sum() == 0:
                continue
            # if (mask == 0).sum() / mask.size > 0.9:
            #     continue
            # if mask.sum() / mask.size > 0.3:
            #     masked_imgs.append(opj(f"{train_dir}", "images", f"train_{j}_{i}.png"))
            # if mask.sum() / mask.size == 0:
            #     # 纯背景
            #     background_imgs.append(opj(f"{train_dir}", "images", f"train_{j}_{i}.png"))
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.squeeze(-1))
            mask = mask.convert("L")
            image.save(opj(f"{train_dir}", "images", f"train_{j}_{i}.png"))
            mask.save(opj(f"{train_dir}", "labels", f"train_{j}_{i}.png"))
        #plt.imsave(f"tile_show_{i}.jpg", dataset.result_blank)
    # step2 执行copy and paste 数据增广

    # if copy_and_paste:
    #     print("start copy and paste augmentation...")
    #     for i, img_path in enumerate(masked_imgs):
    #         if random.random() > copy_and_paste_prob:
    #             continue
    #         print(f"start copy and paste augmentation for {img_path}")
    #         foreground_img = np.array(Image.open(img_path))
    #         foreground_mask_path = img_path.replace("images", "labels")
    #         foreground_mask = np.array(Image.open(foreground_mask_path))
    #         background_img_path = random.choice(background_imgs)
    #         background_img = np.array(Image.open(background_img_path))
    #         background_mask = np.array(Image.open(background_img_path.replace("images", "labels")))
    #         mix_img = foreground_img * 0.5 + background_img * 0.5
    #         mix_img = mix_img.astype(np.uint8)
    #         mix_mask = foreground_mask + background_mask
    #         mix_img = Image.fromarray(mix_img)
    #         mix_mask = Image.fromarray(mix_mask)
    #         mix_img.save(img_path.replace(".png", "_cnp.png"))
    #         mix_mask.save(foreground_mask_path.replace(".png", "_cnp.png"))

    # step3 执行验证集基础切图

    for j, i_path in enumerate(val_raw_img_paths):
        #split_tiffs(i_path, conf_loader)
        dataset = RaftTileDataset(i_path, conf_loader, mode="val")
        for i in range(len(dataset)):
            print(f"making val ({j}_{i}) data pair.")
            pair = dataset[i]
            image = pair["img"]
            mask = pair["mask"]
            if image.sum() == 0:
                continue
            # if (mask == 0).sum() / mask.size > 0.9:
            #     continue
            #print(image.shape, mask.shape)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask.squeeze(-1))
            image.save(opj(f"{val_dir}", "images", f"val_{j}_{i}.png"))
            mask.save(opj(f"{val_dir}", "labels", f"val_{j}_{i}.png"))