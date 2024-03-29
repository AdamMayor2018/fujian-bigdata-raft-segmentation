# -- coding: utf-8 --
# @Time : 2024/1/8 14:38
# @Author : caoxiang
# @File : run.py.py
# @Software: PyCharm
# @Description: 按照官方要求提供的推理代码
import copy
import glob

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
from torchsummary import summary
from sklearn.metrics import roc_curve



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
    conf_loader = YamlConfigLoader(yaml_path=os.path.join(model_dir, "config", "raft_baseline_val_config.yaml"))
    ratio = conf_loader.attempt_load_param("ratio")
    aug = AugmentationTool(conf_loader)
    device = conf_loader.attempt_load_param("device")
    # model
    model = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        try:
            model.load_state_dict(torch.load(os.path.join(model_dir, "experiments/experiment_resnext50_dice_lovaz_loss/weights",  conf_loader.attempt_load_param("pretrained_path"))))
        except Exception as e:
            model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(os.path.join(model_dir, "experiments/experiment_resnext50_dice_lovaz_loss/weights",  conf_loader.attempt_load_param("pretrained_path"))).items()})
    summary(model, input_size=(3, 512, 512), device="cpu")
    model = model.to(device)
    model.eval()

    model2 = smp.DeepLabV3Plus(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("model2_pretrained_path"):
        try:
            model2.load_state_dict(torch.load(os.path.join(model_dir, "baseline_weights_new",
                                                           conf_loader.attempt_load_param("model2_pretrained_path"))))
        except Exception as e:
            model2.load_state_dict({k.replace('module.', ''): v for k, v in
                                    torch.load(os.path.join(model_dir, "baseline_weights_new",
                                                            conf_loader.attempt_load_param(
                                                                "model2_pretrained_path"))).items()})
    summary(model2, input_size=(3, 512, 512), device="cpu")
    model2 = model2.to(device)
    model2.eval()

    # data

    pred_imgs_paths = os.listdir(to_pred_dir)
    mean_f1_scores = []
    ratios = []
    # for ratio in range(0, 100, 1):
    #     scores = []
    #     ratio = ratio / 100
    #     ratios.append(ratio)
    scores = []
    for pred_name in pred_imgs_paths:
        pred_img_path = os.path.join(to_pred_dir, pred_name)
        image = Image.open(pred_img_path)
        image = np.array(image)
        height, width, _ = image.shape
        result_mask = np.zeros((height, width), dtype=np.uint8)  # ! 结果mask
        dataset = RaftInferExpansionDataset(file_path=pred_img_path, conf_loader=conf_loader, aug=aug)
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), total=int(len(dataset))):
                crop_image, pad_indices, origin_indices = dataset[i]
                crop_image = crop_image.to(device)
                crop_image = crop_image.unsqueeze(0)
                logits = model.predict(crop_image)
                logits = torch.sigmoid(logits)
                logits2 = model2.predict(crop_image)
                logits2 = torch.sigmoid(logits2)
                logits = 0.5 * logits + 0.5 * logits2
                logits[logits >= ratio] = 1
                logits[logits < ratio] = 0
                logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)

                # logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy()
                result_mask = cut_img(logits, result_mask, dataset.pad_size, dataset.matting_size, origin_indices)
            # 开运算
            result_mask = result_mask.astype(np.uint8)
            # 找到轮廓
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
            #
            # # 查看各个返回值
            # # 连通域数量
            # print('num_labels = ', num_labels)
            # # 连通域的信息：对应各个轮廓的x、y、width、height和面积
            # print('stats = ', stats)
            # # 连通域的中心点
            # print('centroids = ', centroids)
            # # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
            #
            # for i in range(1, num_labels):
            #     mask = labels == i
            #     # print(i, " prob avg:", result_mask[mask].sum() / result_mask[mask].size, "mask size: ", result_mask[mask].size)
            #     # #填平平均置信度低的label都改为0
            #     # if result_mask[mask].sum() / result_mask[mask].size < 0.8:
            #     #     result_mask[mask] = 0
            #     # 挖去小点
            #     if result_mask[mask].size < 100:
            #         result_mask[mask] = 0
            # # result_mask[result_mask >= ratio] = 1
            # # result_mask[result_mask < ratio] = 0
            #
            # #填上小洞
            # reverse_result_mask = 1 - result_mask
            # reverse_result_mask = reverse_result_mask.astype(np.uint8)
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(reverse_result_mask,
            #                                                                         connectivity=8)
            # for i in range(1, num_labels):
            #     mask = labels == i
            #     # 填上小洞
            #     if reverse_result_mask[mask].size < 200:
            #         reverse_result_mask[mask] = 0
            # final_result_mask = 1 - reverse_result_mask
            #
            image_mask = Image.open(os.path.join(to_pred_dir, "..", f'{pred_name.replace("img", "mask")}'))
            image_mask = np.array(image_mask)
            image_mask[image_mask >= 1] = 1
            # fpr, tpr, thresholds = roc_curve(image_mask.flatten(), result_mask.flatten(), pos_label=1, sample_weight=None, drop_intermediate=True)
            # plt.plot(fpr, tpr, marker='o')
            # plt.show()
            # from sklearn.metrics import auc
            # AUC = auc(fpr, tpr)
            #print(f"{pred_name} fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}, AUC:{AUC}")
            final_result_mask = result_mask
            TP = np.sum(np.logical_and(final_result_mask == 1, image_mask == 1))
            FP = np.sum(np.logical_and(final_result_mask == 1, image_mask == 0))
            FN = np.sum(np.logical_and(final_result_mask == 0, image_mask == 1))
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            print(f"{pred_name} f1: {f1}")
            scores.append(f1)
            plt.subplot(1, 2, 1)
            plt.imshow(final_result_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(image_mask)
            plt.show()
    mean_f1 = np.mean(scores)
    mean_f1_scores.append(mean_f1)
    print(f"ratio : {ratio}, mean f1: {mean_f1}")

    # plt.plot(ratios, mean_f1_scores, marker='o')
    # plt.show()
    #! PIL保存
    # pred = Image.fromarray(result_mask)
    # pred.save(result_save_path)


if __name__ == "__main__":
    # to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    # result_save_path = sys.argv[2]  # 预测结果保存文件路径
    to_pred_dir = "/data/cx/datasets/fujian_gis_data/pred"
    result_save_path = "/data/user/zhaozeming/competition/result.tif"
    main(to_pred_dir, result_save_path)
