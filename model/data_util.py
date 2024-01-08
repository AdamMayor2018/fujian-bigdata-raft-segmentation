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
from transform import AugmentationTool
import numpy as np
import torch
import cv2
import os

class RaftInferExpansionDataset(Dataset):
    def __init__(self, file_path, conf_loader: YamlConfigLoader,  aug:AugmentationTool):
        self.image = np.array(Image.open(file_path)).astype(np.uint8)
        self.image = torch.from_numpy(self.image)
        self.height, self.width, _ = self.image.shape
        self.tile_size = conf_loader.attempt_load_param("tile_size")
        self.expansion = conf_loader.attempt_load_param("expansion")
        # 先计算在原图上下左右都按照expansion尺寸pad一圈之后再进行滑窗
        # pad image
        self.image = F.pad(self.image, (self.expansion, self.expansion, self.expansion, self.expansion), mode='reflect').numpy().astype(
            np.uint8)


    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    pass
