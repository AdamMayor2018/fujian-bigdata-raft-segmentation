# -- coding: utf-8 --
# @Time : 2023/12/28 11:05
# @Author : caoxiang
# @File : data_builder.py
# @Software: PyCharm
# @Description:
import os.path
import cv2
from PIL import Image, ImageFile
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import numpy as np
from os.path import join as opj
from raft_baseline.config.conf_loader import YamlConfigLoader
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


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


class HuBMAPDataset(Dataset):
    def __init__(self, filename, conf_loader: YamlConfigLoader):
        super().__init__()
        self.conf_loader = conf_loader
        if os.path.exists(filename):
            path = filename
        else:
            path = opj(self.conf_loader.attempt_load_param("train_dir"), filename + '.tiff')
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
        self.sz = self.conf_loader.attempt_load_param('tile_size')
        self.shift_h = self.conf_loader.attempt_load_param('shift_h')
        self.shift_w = self.conf_loader.attempt_load_param('shift_w')
        self.pad_h = self.sz - self.h % self.sz  # add to whole slide
        self.pad_w = self.sz - self.w % self.sz  # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.sz
        self.num_w = (self.w + self.pad_w) // self.sz

        if self.h % self.sz < self.shift_h:
            self.num_h -= 1
        if self.w % self.sz < self.shift_w:
            self.num_w -= 1

    def __len__(self):
        return self.num_h * self.num_w

    def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h * self.sz + self.shift_h
        x = i_w * self.sz + self.shift_w
        py0, py1 = max(0, y), min(y + self.sz, self.h)
        px0, px1 = max(0, x), min(x + self.sz, self.w)

        # placeholder for input tile (before resize)
        img_patch = np.zeros((self.sz, self.sz, 3), np.uint8)
        mask_patch = np.zeros((self.sz, self.sz, 1), np.uint8)

        # replace the value for img patch
        if self.image.count == 3:
            img_patch[0:py1 - py0, 0:px1 - px0] = \
                np.moveaxis(self.image.read([1, 2, 3], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img_patch[0:py1 - py0, 0:px1 - px0, i] = \
                    layer.read(1, window=Window.from_slices((py0, py1), (px0, px1)))

        # replace the value for mask patch
        if self.mask.count == 1:
            mask_patch[0:py1 - py0, 0:px1 - px0] = \
                np.moveaxis(self.mask.read([1], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)
        return {'img': img_patch, 'mask': mask_patch}


if __name__ == '__main__':
    conf_loader = YamlConfigLoader("../config/raft_baseline_config.yaml")
    raw_dir = conf_loader.attempt_load_param("raw_dir")
    print(raw_dir)
    img_paths = glob.glob(opj(raw_dir, "mask*.tif"))
    print(img_paths)
    for i_path in img_paths:
        split_tiffs(i_path, conf_loader)
        # dataset = HuBMAPDataset(i_path, conf_loader)
        # for _ in dataset:
        #     pass
    # dataset = HuBMAPDataset()
