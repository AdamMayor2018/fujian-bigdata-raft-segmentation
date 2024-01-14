import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur, CLAHE,channel_shuffle,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations.pytorch import ToTensorV2
from raft_baseline.config.conf_loader import YamlConfigLoader

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class AugmentationTool:
    def __init__(self, conf_loader: YamlConfigLoader):
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        self.conf_loader = conf_loader

    def get_transforms_train(self):
        transform_train = Compose([
            # Basic
            RandomRotate90(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            # ChannelShuffle(p=0.3),
            # # #Morphology
            # ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
            #                  interpolation=1, border_mode=0, value=(0, 0, 0), p=0.5),
            # GaussNoise(var_limit=(0, 50.0), mean=0, p=0.5),
            # GaussianBlur(blur_limit=(3, 7), p=0.5),
            #
            # #Color
            RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                     brightness_by_max=True, p=0.5),
            HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                               val_shift_limit=0, p=0.5),

            CoarseDropout(max_holes=2,
                          max_height=self.conf_loader.attempt_load_param('train_height') // 4, max_width=self.conf_loader.attempt_load_param("train_width") // 4,
                          min_holes=1,
                          min_height=self.conf_loader.attempt_load_param('train_height') // 16, min_width=self.conf_loader.attempt_load_param('train_width') // 16,
                          fill_value=0, mask_fill_value=0, p=0.5),

            Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]),
                      std=(STD[0], STD[1], STD[2])),
            ToTensorV2(),
        ])
        return transform_train

    def get_transforms_valid(self):
        transform_valid = Compose([
            Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]),
                      std=(STD[0], STD[1], STD[2])),
            ToTensorV2(),
        ])
        return transform_valid

    def denormalize(z, mean=MEAN.reshape(-1, 1, 1), std=STD.reshape(-1, 1, 1)):
        return std * z + mean
