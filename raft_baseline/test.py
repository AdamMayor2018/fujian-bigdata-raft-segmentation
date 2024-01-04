# -- coding: utf-8 --
# @Time : 2024/1/3 15:00
# @Author : caoxiang
# @File : test.py
# @Software: PyCharm
# @Description:
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
mask_path = "/data/cx/datasets/fujian_gis_data/train/labels/train_1_935.jpg"

mask = Image.open(mask_path)
mask = mask.convert("L")
plt.imshow(mask)
plt.show()
print(set(np.array(mask).flatten().tolist()))