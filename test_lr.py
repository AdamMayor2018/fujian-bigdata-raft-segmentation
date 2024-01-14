# 测算鱼排的灰度取值
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np


img = Image.open("/data/cx/datasets/fujian_gis_data/raw/raw_image/img1.tif")
mask = Image.open("/data/cx/datasets/fujian_gis_data/raw/raw_image/mask1.tif")

img = np.array(img).astype(np.uint8)
mask = np.array(mask).astype(np.uint8)
mask[mask >= 1] = 1
img_sum = img.sum(axis=2)
print(img_sum.shape, mask.shape)
img_sum = img_sum * mask
print(img_sum[img_sum != 0].min())