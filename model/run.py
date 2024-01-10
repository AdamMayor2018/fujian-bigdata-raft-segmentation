# -- coding: utf-8 --
# @Time : 2024/1/8 14:38
# @Author : caoxiang
# @File : run.py.py
# @Software: PyCharm
# @Description: 按照官方要求提供的推理代码

from conf_loader import YamlConfigLoader
import os, sys
import cv2
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image, ImageFile
from data_util import RaftInferExpansionDataset
from transform import AugmentationTool

# ! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def main(to_pred_dir, result_save_path):
    run_py_path = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py_path)
    conf_loader = YamlConfigLoader(yaml_path=os.path.join(model_dir, "infer_config.yaml"))
    ratio = conf_loader.attempt_load_param("ratio")
    aug = AugmentationTool(conf_loader)
    # model
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        # encoder_weights='noisy-student',
        in_channels=3,
        classes=1,
        activation=None
    )
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(os.path.join(model_dir, conf_loader.attempt_load_param("pretrained_path"))))
    model = model.to("cuda:0")
    model.eval()
    # data
    pred_imgs_paths = os.listdir(to_pred_dir)
    pred_img_path = os.path.join(to_pred_dir, pred_imgs_paths[0])  # ! 测试集只有一张图片
    image = Image.open(pred_img_path)
    image = np.array(image)
    height, width , _ = image.shape
    result_mask = np.zeros((height, width), dtype=np.uint8)  # ! 结果mask
    dataset = RaftInferExpansionDataset(file_path=None, conf_loader=conf_loader, aug=aug)
    with torch.no_grad():
        for i in range(len(dataset)):
            crop_image, pad_indices, origin_indices = dataset[i]
            logits = model.predict(crop_image)
            logits = torch.sigmoid(logits)
            logits[logits >= ratio] = 1
            logits[logits < ratio] = 0
            logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy()

    #! PIL保存
    pred = Image.fromarray(result_mask)
    pred.save(result_save_path)


if __name__ == "__main__":
    # to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    # result_save_path = sys.argv[2]  # 预测结果保存文件路径
    to_pred_dir = "/data/cx/datasets/fujian_gis_data/pred"
    result_save_path = "/data/cx/datasets/fujian_gis_data/pred"
    main(to_pred_dir, result_save_path)
