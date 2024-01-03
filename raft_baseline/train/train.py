import torch
import gc
from tqdm import tqdm
from raft_baseline.util.common import fix_seed
from raft_baseline.config.conf_loader import YamlConfigLoader
from raft_baseline.train.dataset import RaftDataset
from raft_baseline.train.transform import AugmentationTool
from torch.utils.data import DataLoader
from raft_baseline.models.model import build_model
from torch import nn, optim
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger('train')
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def my_collate(batch):
    inputs = torch.stack([data["image"] for data in batch])
    targets = torch.stack([data["mask"] for data in batch])
    return inputs, targets


if __name__ == '__main__':
    # fix seed
    fix_seed(2024)
    # load global config
    conf_loader = YamlConfigLoader(yaml_path="../config/raft_baseline_config.yaml")
    device = conf_loader.attempt_load_param("device")
    device = device if torch.cuda.is_available() else "cpu"

    # augmentation
    aug = AugmentationTool(conf_loader)
    # dataset
    train_dataset = RaftDataset(conf_loader=conf_loader, mode="train", aug=aug)
    valid_dataset = RaftDataset(conf_loader, mode="val", aug=aug)
    train_loader = DataLoader(train_dataset, batch_size=conf_loader.attempt_load_param("train_batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True, collate_fn=my_collate, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=conf_loader.attempt_load_param("val_batch_size"),
                              shuffle=False, num_workers=4, pin_memory=True, collate_fn=my_collate)
    num_epochs = conf_loader.attempt_load_param("num_epochs")

    # model
    resolution = (conf_loader.attempt_load_param("train_width"), conf_loader.attempt_load_param("train_height"))
    model_params = conf_loader.attempt_load_param("model_params")
    model = build_model(model_name=conf_loader.attempt_load_param("backbone_name"), resolution=resolution,
                        deep_supervision=model_params["deep_supervision"], clf_head=model_params["clf_head"],
                        clf_threshold=model_params["clf_threshold"], load_weights=model_params["load_backbone_weights"])
    # load pretrained
    if conf_loader.attempt_load_param("pretrained") and conf_loader.attempt_load_param("pretrained_path"):
        model.load_state_dict(torch.load(conf_loader.attempt_load_param("pretrained_path")))

    # critirion optimizer scheduler
    criterion = nn.BCEWithLogitsLoss().to(device)
    optim_params = conf_loader.attempt_load_param("optim_params")
    optim_params = {k: eval(v) for k, v in optim_params.items() if type(v) == "str"}
    optimizer = optim.Adam(model.parameters(), **optim_params)
    sched_params = conf_loader.attempt_load_param("sched_params")
    sched_params = {k: eval(v) for k, v in sched_params.items() if type(v) == "str"}

    if conf_loader.attempt_load_param("lr_scheduler_name") == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)
    elif conf_loader.attempt_load_param("lr_scheduler_name") == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
    elif conf_loader.attempt_load_param('lr_scheduler_name') == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_params)

    # grad scaler
    scaler = torch.cuda.amp.GradScaler()

    #train
    log_cols = ['fold', 'epoch', 'lr', 'loss_trn', 'loss_val', 'trn_score', 'val_score', 'elapsed_time']
    for epoch in range(1, conf_loader.attempt_load_param("num_epochs") + 1):
        train_loss = 0
        valid_loss = 0
        logger.info(f"start trainning epoch : {epoch}.")
        logger.info(f"lr: {[group['lr'] for group in optimizer.param_groups]}")
        model.train()
        model = model.to(device)
        train_epoch = tqdm(train_loader, total=int(len(train_loader)))
        # FI score container
        prob_all = []
        target_all = []
        for i, data in enumerate(train_epoch):
            inputs = data[0]
            targets = data[1]
            #logger.info(f"start training batch : {i}.")
            # print(data[0].shape, data[1].shape)
            if model_params["clf_head"]:
                y_clf = targets.to(device, torch.float32, non_blocking=True)
                if model_params["deep_supervision"]:
                    logits, logits_deeps, logits_clf = model(inputs.to(device, torch.float32, non_blocking=True))
                else:
                    logits, logits_clf = model(inputs.to(device, torch.float32, non_blocking=True))
            else:
                if model_params["deep_supervision"]:
                    logits, logits_deeps = model(inputs.to(device, torch.float32, non_blocking=True))
                else:
                    logits = model(inputs.to(device, torch.float32, non_blocking=True))
            y_true = targets.to(device, torch.float32, non_blocking=True)
            #logger.info(f"{logits.shape}, {y_true.shape}")
            logits = logits.squeeze(1)
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss
        #     target_all.extend(targets.flatten().tolist())
        #     logits[logits >= 0.5] = 1
        #     logits[logits <= 0.5] = 0
        #     prob_all.extend(logits.flatten().tolist())
        # logger.info(f"epoch {epoch}, train F1 score: {f1_score(target_all, prob_all)}")

        avg_train_loss = train_loss / len(train_dataset)
        logger.info(f"epoch {epoch}, train loss: {avg_train_loss}")
        #validation
        del data, loss, logits, y_true, inputs, targets
        model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        loss_val = 0
        y_preds = []
        y_trues = []
        val_epoch = tqdm(valid_loader, total=int(len(valid_loader)))
        for i, data in enumerate(val_epoch):
            inputs = data[0]
            targets = data[1]
            with torch.no_grad():
                batch, c, h, w = inputs.shape
                if model_params["clf_head"]:
                    y_clf = targets.to(device, torch.float32, non_blocking=True)
                    if model_params["deep_supervision"]:
                        logits, logits_deeps, logits_clf = model(
                            inputs.to(device, torch.float32, non_blocking=True))
                    else:
                        logits, logits_clf = model(inputs.to(device, torch.float32, non_blocking=True))
                else:
                    if model_params["deep_supervision"]:
                        logits, logits_deeps = model(inputs.to(device, torch.float32, non_blocking=True))
                    else:
                        logits = model(inputs.to(device, torch.float32, non_blocking=True))
                y_true = targets.to(device, torch.float32, non_blocking=True)
                loss_val += criterion(logits.squeeze(1), y_true).item() * batch
            # release GPU memory cache
            #del data, loss, logits, y_true, inputs, targets
            torch.cuda.empty_cache()
            gc.collect()
        avg_val_loss = loss_val / len(valid_dataset)
        scheduler.step(loss_val)
        logger.info(f"epoch {epoch}, val loss: {avg_train_loss}")
