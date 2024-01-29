# -- coding: utf-8 --
# @Time : 2024/1/29 11:40
# @Author : caoxiang
# @File : cross_cut.py
# @Software: PyCharm
# @Description: 为了平衡数据分布 将数据集切分成十个条

from PIL import  Image
import rasterio
import torch
import numpy as np
import torch.nn.functional as F

if __name__ == '__main__':
    img = rasterio.open("/data/cx/datasets/fujian_gis_data/raw/raw_image/img2.tif")
    img = torch.from_numpy(img.read([1, 2, 3])).float()
    mask = rasterio.open("/data/cx/datasets/fujian_gis_data/raw/raw_image/mask2.tif")
    mask = torch.from_numpy(mask.read([1])).float()
    splits = 10
    c, h, w = img.shape
    if h > w:
        pad_w = 0
        pad_h = splits - h % splits
    else:
        pad_h = 0
        pad_w = splits - w % splits
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(
        np.uint8).transpose((1, 2, 0))
    mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect').numpy().astype(
        np.uint8).transpose((1, 2, 0))

    if h > w:
        imgs = np.split(img, splits, axis=0)
        masks = np.split(mask, splits, axis=0)
    else:
        imgs = np.split(img, splits, axis=1)
        masks = np.split(mask, splits, axis=1)

    for i, (i_image, imask) in enumerate(zip(imgs, masks)):
        i_image = Image.fromarray(i_image)
        imask = Image.fromarray(imask.squeeze(2))
        i_image.save(f"/data/cx/datasets/fujian_gis_data/cross_balance_data/raw_image/img2_{i}.tif", "TIFF")
        imask.save(f"/data/cx/datasets/fujian_gis_data/cross_balance_data/raw_mask/mask2_{i}.tif", "TIFF")
