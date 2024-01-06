# -- coding: utf-8 --
# @Time : 2024/1/3 15:00
# @Author : caoxiang
# @File : test.py
# @Software: PyCharm
# @Description:
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import numpy as np
import torch
import copy
#from raft_baseline.models.model import build_model
from raft_baseline.config.conf_loader import YamlConfigLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
# mask_path = "/data/cx/datasets/fujian_gis_data/train/labels/train_1_935.jpg"
# mask = Image.open(mask_path)
# mask = mask.convert("L")
# plt.imshow(mask)
# plt.show()
from raft_baseline.train.transform import AugmentationTool

conf_loader = YamlConfigLoader(yaml_path="config/raft_baseline_infer_config.yaml")
device = conf_loader.attempt_load_param("device")
device = device if torch.cuda.is_available() else "cpu"
resolution = (conf_loader.attempt_load_param("train_width"), conf_loader.attempt_load_param("train_height"))
# model_params = conf_loader.attempt_load_param("model_params")
# model = build_model(model_name=conf_loader.attempt_load_param("backbone_name"), resolution=resolution,
#                         deep_supervision=model_params["deep_supervision"], clf_head=model_params["clf_head"],
#                         clf_threshold=eval(model_params["clf_threshold"]),
#                         load_weights=model_params["load_backbone_weights"])
# load pretrained
model = smp.DeepLabV3Plus(
    encoder_name='resnet18',
    # encoder_weights='noisy-student',
    in_channels=3,
    classes=1,
    activation=None
)
if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
    model.load_state_dict(torch.load(conf_loader.attempt_load_param("pretrained_path")))
model = model.to(device)
model.eval()
transform = AugmentationTool(conf_loader).get_transforms_valid()
image_path = "/data/cx/datasets/fujian_gis_data/val/images/val_0_410.jpg"
mask_path = "/data/cx/datasets/fujian_gis_data/val/labels/val_0_410.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
origin_image = copy.deepcopy(image)
image = transform(image=image)["image"].unsqueeze(0)
image = image.to(device)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask[mask!=0] = 1
print(image.shape)
logits = model.predict(image)
logits = torch.sigmoid(logits)
logits[logits >= 0.7] = 1
logits[logits < 0.7] = 0
logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy()
print(logits.shape)
plt.subplot(1, 3, 1)
plt.imshow(mask)
plt.subplot(1, 3, 2)
plt.imshow(logits)
plt.subplot(1, 3, 3)
plt.imshow(origin_image)
plt.show()
