# -- coding: utf-8 --
# @Time : 2024/1/8 14:49
# @Author : caoxiang
# @File : data_util.py
# @Software: PyCharm
# @Description: 滑窗推理类
from torch.utils.data import Dataset
from conf_loader import YamlConfigLoader
from os.path import join as opj
from PIL import Image, ImageFile
import torch.nn.functional as F
# ! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torch
import rasterio
from rasterio.windows import Window`

import cv2
import os

sz = 1024   #the size of tiles
sz_reduction = 2  #reduce the original images by 4 times
expansion = 256

class RaftInferExpansionDataset(Dataset):
    def __init__(self, idx, sz=sz, sz_reduction=sz_reduction, expansion=expansion):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.sz_reduction = sz_reduction
        self.sz = sz_reduction * sz
        self.expansion = sz_reduction * expansion
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz - self.expansion // 2, -self.pad1 // 2 + n1 * self.sz - self.expansion // 2
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz + self.expansion, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz + self.expansion, self.shape[1])
        img = np.zeros((self.sz + self.expansion, self.sz + self.expansion, 3), np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                                           window=Window.from_slices(
                                                                                               (p00, p01), (p10, p11))),
                                                                            0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = \
                    layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))

        if self.sz_reduction != 1:
            img = cv2.resize(img, (
            (self.sz + self.expansion) // self.sz_reduction, (self.sz + self.expansion) // self.sz_reduction),
                             interpolation=cv2.INTER_AREA)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx

# class RaftTileDataset(Dataset):
#     def __init__(self, file_path: str, conf_loader: YamlConfigLoader):
#         super().__init__()
#         self.conf_loader = conf_loader
#         self.path = file_path
#         self.image = np.array(Image.open(self.path))
#         self.h, self.w, _ = self.image.shape
#         self.sz = self.conf_loader.attempt_load_param('tile_size')
#         self.shift_h = self.conf_loader.attempt_load_param(
#             'shift_h') if self.mode == "train" else self.conf_loader.attempt_load_param('val_shift_h')
#         self.shift_w = self.conf_loader.attempt_load_param(
#             'shift_w') if self.mode == "train" else self.conf_loader.attempt_load_param('val_shift_w')
#         self.pad_h = self.sz - self.h % self.sz  # add to whole slide
#         self.pad_w = self.sz - self.w % self.sz  # add to whole slide
#         self.num_h = (self.h + self.pad_h) // self.sz
#         self.num_w = (self.w + self.pad_w) // self.sz
#         self.image = torch.from_numpy(self.image.read([1, 2, 3])).float()
#         self.mask = torch.from_numpy(self.mask.read([1])).float()
#         # pad image and mask
#         pad_left = self.pad_w // 2
#         pad_right = self.pad_w - pad_left
#         pad_top = self.pad_h // 2
#         pad_bottom = self.pad_h - pad_top
#         self.image = F.pad(self.image, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(
#             np.uint8).transpose((1, 2, 0))
#         self.mask = F.pad(self.mask, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(
#             np.uint8).transpose((1, 2, 0))
#
#         if self.h % self.sz < self.shift_h:
#             self.num_h -= 1
#         if self.w % self.sz < self.shift_w:
#             self.num_w -= 1
#         self.new_h, self.new_w, _ = self.image.shape
#
#     def __len__(self):
#         return self.num_h * self.num_w
#
#     def __getitem__(self, idx):  # idx = i_h * self.num_w + i_w
#         # prepare coordinates for rasterio
#         i_h = idx // self.num_w
#         i_w = idx % self.num_w
#         y = i_h * self.sz + self.shift_h
#         x = i_w * self.sz + self.shift_w
#         py0, py1 = max(0, y), min(y + self.sz, self.new_h)
#         px0, px1 = max(0, x), min(x + self.sz, self.new_w)
#
#         # placeholder for input tile (before resize)
#         img_patch = np.zeros((self.sz, self.sz, 3), np.uint8)
#         mask_patch = np.zeros((self.sz, self.sz, 1), np.uint8)
#
#         # replace the value for img patch
#         if self.image.shape[-1] == 3:
#             img_patch[0:py1 - py0, 0:px1 - px0] = self.image[py0: py1, px0: px1, :]
#             # np.moveaxis(self.image.read([1, 2, 3], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)
#
#         # replace the value for mask patch
#         if self.mask.shape[-1] == 1:
#             mask_patch[0:py1 - py0, 0:px1 - px0] = self.mask[py0: py1, px0: px1, :]
#             # np.moveaxis(self.mask.read([1], window=Window.from_slices((py0, py1), (px0, px1))), 0, -1)
#         return {'img': img_patch, 'mask': mask_patch}
