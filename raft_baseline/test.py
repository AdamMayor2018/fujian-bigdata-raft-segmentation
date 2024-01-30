# -- coding: utf-8 --
# @Time : 2024/1/8 14:38
# @Author : caoxiang
# @File : run.py.py
# @Software: PyCharm
# @Description: 按照官方要求提供的推理代码
from raft_baseline.config.conf_loader import YamlConfigLoader
import os, sys
import cv2
from glob import glob
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
from thop import profile
import torchmetrics


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
    model = smp.PAN(
        encoder_name=conf_loader.attempt_load_param("backbone"),
        encoder_weights=None,
        in_channels=3,
        classes=1,
        encoder_output_stride=32,
        activation=None
    )
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        try:
            model.load_state_dict(torch.load(os.path.join(model_dir, "experiments/experiment_hrnet48_ssld_with_10balance_data/weights",  conf_loader.attempt_load_param("pretrained_path"))))
        except Exception as e:
            model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load(os.path.join(model_dir, "experiments/experiment_hrnet48_ssld_with_10balance_data/weights",  conf_loader.attempt_load_param("pretrained_path"))).items()})
    #summary(model, input_size=(3, 512, 512), device="cpu")
    dummy_input = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    model = model.to(device)
    model.eval()
    test_f1_metric = torchmetrics.classification.BinaryF1Score().to(device)
    # data
    pred_imgs_paths = glob(os.path.join(to_pred_dir, "img*.tif"))
    #pred_imgs_paths = os.listdir(to_pred_dir)
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
        height, width , _ = image.shape
        result_mask = np.zeros((height, width), dtype=np.float32)  # ! 结果mask
        dataset = RaftInferExpansionDataset(file_path=pred_img_path, conf_loader=conf_loader, aug=aug)
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), total=int(len(dataset))):
                crop_image, pad_indices, origin_indices = dataset[i]
                crop_image = crop_image.to(device)
                crop_image = crop_image.unsqueeze(0)
                logits = model.predict(crop_image)
                logits = torch.sigmoid(logits)
                logits[logits >= ratio] = 1
                logits[logits < ratio] = 0
                logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
                #logits = logits.squeeze(0).squeeze(0).cpu().detach().numpy()
                result_mask = cut_img(logits, result_mask, dataset.pad_size, dataset.matting_size, origin_indices)
            # 开运算
            #result_mask = cv2.dilate(result_mask, kernel=(3, 3), iterations=2)
            #result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel=(3, 3), iterations=2)
            image_mask = Image.open(os.path.join(to_pred_dir, "..", f'{pred_name.replace("img", "mask")}'))
            image_mask = np.array(image_mask)
            image_mask[image_mask >= 1] = 1
            result_mask = result_mask.astype(np.float32)
            image_mask = image_mask.astype(np.float32)
            test_f1_per_image_score = test_f1_metric.update(torch.from_numpy(result_mask), torch.from_numpy(image_mask))

            # fpr, tpr, thresholds = roc_curve(image_mask.flatten(), result_mask.flatten(), pos_label=1, sample_weight=None, drop_intermediate=True)
            # plt.plot(fpr, tpr, marker='o')
            # plt.show()
            # from sklearn.metrics import auc
            # AUC = auc(fpr, tpr)
            #print(f"{pred_name} fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}, AUC:{AUC}")
            # TP = np.sum(np.logical_and(result_mask == 1, image_mask == 1))
            # FP = np.sum(np.logical_and(result_mask == 1, image_mask == 0))
            # FN = np.sum(np.logical_and(result_mask == 0, image_mask == 1))
            # precision = TP / (TP + FP)
            # recall = TP / (TP + FN)
            # f1 = 2 * precision * recall / (precision + recall)
            print(f"{pred_name} f1: {test_f1_per_image_score}")
            #scores.append(f1)
            plt.subplot(1, 2, 1)
            plt.imshow(result_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(image_mask)
            plt.show()
    # mean_f1 = np.mean(scores)
    # mean_f1_scores.append(mean_f1)
    test_f1_score = test_f1_metric.compute().cpu().numpy()
    print(f"ratio : {ratio}, mean f1: {test_f1_score}")

    # plt.plot(ratios, mean_f1_scores, marker='o')
    # plt.show()
    #! PIL保存
    # pred = Image.fromarray(result_mask)
    # pred.save(result_save_path)


if __name__ == "__main__":
    # to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    # result_save_path = sys.argv[2]  # 预测结果保存文件路径
    to_pred_dir = "/data/cx/datasets/fujian_gis_data/new_pred"
    result_save_path = "/data/user/zhaozeming/competition/result.tif"
    main(to_pred_dir, result_save_path)
