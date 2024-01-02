import torch

from raft_baseline.util.common import fix_seed
from raft_baseline.config.conf_loader import YamlConfigLoader
from raft_baseline.train.dataset import RaftDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from raft_baseline.models.model import build_model

if __name__ == '__main__':
    #fix seed
    fix_seed(2024)
    #load global config
    conf_loader = YamlConfigLoader(yaml_path="../config/raft_baseline_config.yaml")
    print(conf_loader)
    device = conf_loader.attempt_load_param("device")
    device = device if torch.cuda.is_available() else "cpu"

    #augmentation
    aug = AugmentationTool(conf_loader)
    #dataset
    train_dataset = RaftDataset(conf_loader=conf_loader, mode="train", aug=aug)
    valid_dataset = RaftDataset(conf_loader, mode="val", aug=aug)
    print(train_dataset[0], valid_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=conf_loader.attempt_load_param("batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=conf_loader.attempt_load_param("val_batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True)
    num_epochs = conf_loader.attempt_load_param("num_epochs")

    #model
    resolution = (conf_loader.attempt_load_param("train_width"), conf_loader.attempt_load_param("train_height"))
    model_params = conf_loader.attempt_load_param("model_params")
    model = build_model(model_name=conf_loader.attempt_load_param("backbone_name"), resolution=resolution,
                        deep_supervision=model_params["deep_supervision"],clf_head= model_params["clf_head"],
                        clf_threshold=model_params["clf_threshold"], load_weights=model_params["load_backbone_weights"])
    #load pretrained
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(conf_loader.attempt_load_param("pretrained_path")))





