# -- coding: utf-8 --
# @Time : 2024/1/8 14:38
# @Author : caoxiang
# @File : run.py.py
# @Software: PyCharm
# @Description: 按照官方要求提供的推理代码

from raft_baseline.config.conf_loader import YamlConfigLoader
import os, sys
import cv2
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image, ImageFile
from raft_baseline.train.dataset import RaftInferExpansionDataset
from raft_baseline.train.transform import AugmentationTool
from tqdm import tqdm
import matplotlib.pyplot as plt
# ! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def cut_img(logits, result_mask, padding_size, matting_size, origin_indices):
    if padding_size + matting_size > logits.shape[0]:
        import pdb;pdb.set_trace()
    try:
        new_img = logits[padding_size:padding_size + matting_size, padding_size:padding_size + matting_size]
        result_mask[origin_indices[1]: origin_indices[3], origin_indices[0]: origin_indices[2]] = new_img
    except:
        import pdb;pdb.set_trace()
    return result_mask


def main(to_pred_dir, result_save_path):
    run_py_path = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py_path)
    conf_loader = YamlConfigLoader(yaml_path=os.path.join(model_dir, "config", "infer_config.yaml"))
    ratio = conf_loader.attempt_load_param("ratio")
    aug = AugmentationTool(conf_loader)
    # model
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(os.path.join(model_dir, "weights", conf_loader.attempt_load_param("pretrained_path"))))
    model = model.to("cuda:0")
    model.eval()
    # data
    pred_imgs_paths = os.listdir(to_pred_dir)
    pred_img_path = os.path.join(to_pred_dir, pred_imgs_paths[0])  # ! 测试集只有一张图片
    image = Image.open(pred_img_path)
    image = np.array(image)
    height, width , _ = image.shape
    result_mask = np.zeros((height, width), dtype=np.uint8)  # ! 结果mask
    dataset = RaftInferExpansionDataset(file_path=pred_img_path, conf_loader=conf_loader, aug=aug)
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), total=int(len(dataset))):
            crop_image, pad_indices, origin_indices = dataset[i]
            crop_image = crop_image.to("cuda:0")
            crop_image = crop_image.unsqueeze(0)
            logits = model.predict(crop_image)
            logits = torch.sigmoid(logits)
            logits[logits >= ratio] = 1
            logits[logits < ratio] = 0
            logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
            cut_img(logits, result_mask, dataset.pad_size, dataset.matting_size, origin_indices)
        # 腐蚀
        #result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel=(3, 3), iterations=3)
        image_mask = Image.open(os.path.join(to_pred_dir, "..", 'val_mask1.tif'))
        image_mask = np.array(image_mask)
        image_mask[image_mask >= 1] = 1
        TP = np.sum(np.logical_and(result_mask == 1, image_mask == 1))
        FP = np.sum(np.logical_and(result_mask == 1, image_mask == 0))
        FN = np.sum(np.logical_and(result_mask == 0, image_mask == 1))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f1)
        plt.subplot(1, 2, 1)
        plt.imshow(result_mask)
        plt.subplot(1, 2, 2)
        plt.imshow(image_mask)
        plt.show()


    #! PIL保存
    # pred = Image.fromarray(result_mask)
    # pred.save(result_save_path)


if __name__ == "__main__":
    # to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    # result_save_path = sys.argv[2]  # 预测结果保存文件路径
    to_pred_dir = "/data/cx/datasets/fujian_gis_data/pred"
    result_save_path = "/data/user/zhaozeming/competition/result.tif"
    main(to_pred_dir, result_save_path)
