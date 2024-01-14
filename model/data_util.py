# -- coding: utf-8 --
# @Time : 2024/1/8 14:49
# @Author : caoxiang
# @File : data_util.py
# @Software: PyCharm
# @Description: 滑窗推理类
from torch.utils.data import Dataset
from conf_loader import YamlConfigLoader
import time
from os.path import join as opj
from PIL import Image, ImageFile
import torch.nn.functional as F
from draw import draw_box
# ! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from transform import AugmentationTool
import numpy as np
import torch
# import cv2
# import os


class RaftInferExpansionDataset(Dataset):
    def __init__(self, file_path, conf_loader: YamlConfigLoader, aug: AugmentationTool):
        self.image = np.array(Image.open(file_path), dtype=np.uint8).transpose((2, 0, 1))
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
                           mode='constant').numpy().astype(np.uint8)
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
    import matplotlib.pyplot as plt
    file_path = "/data/cx/datasets/fujian_gis_data/raw/raw_image/img1.tif"
    conf_loader = YamlConfigLoader("infer_config.yaml")
    aug_tool = AugmentationTool(conf_loader)
    dataset = RaftInferExpansionDataset(file_path, conf_loader, aug_tool)
    for i in range(len(dataset)):
        pair = dataset[i]
        crop_image = pair[0]
        indices = pair[1]
        origin_indices = pair[2]
        print(crop_image.shape, indices, origin_indices)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.title("padded croping")
        # plt.imshow(dataset.blank)
        # plt.subplot(1, 2, 2)
        # plt.title("origin matting")
        # plt.imshow(dataset.result_blank)
        # plt.show()

