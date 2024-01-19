# 测算鱼排的灰度取值
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio

mask = Image.open("/data/cx/datasets/fujian_gis_data/val/labels/val_0_120.png")
#mask1 = mask.read(1)
mask = mask.convert("L")
mask1 = np.array(mask).astype(np.uint8)
print(set(mask1.flatten().tolist()))
# mask = Image.open("/data/cx/datasets/fujian_gis_data/val/labels/val_0_120.jpg")
# mask = np.array(mask)
# mask[mask !=0 ] = 1
# plt.imshow(mask)
# plt.show()
# print(np.array(mask))
# print(np.array(mask).shape)
# print(cv2.imread("/data/cx/datasets/fujian_gis_data/val/labels/val_0_120.jpg", 1))
