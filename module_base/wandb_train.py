# python native
import os
import random
import time
import datetime

import numpy as np
from tqdm.auto import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import XRayDataset
from model import ModelSelector
from transform import TransformSelector
from loss import LossSelector
from scheduler import SchedulerSelector

import warnings
warnings.filterwarnings('ignore')

from omegaconf import OmegaConf
from argparse import ArgumentParser
from wandb_logger import WandbLogger

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def set_data(cfg):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=cfg.image_root)
        for root, _dirs, files in os.walk(cfg.image_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=cfg.label_root)
        for root, _dirs, files in os.walk(cfg.label_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }
    return np.array(sorted(pngs)), np.array(sorted(jsons))

def save_model(model, save_dir, file_name='best_model.pt'):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)
    return output_path

def validation(epoch, model, val_loader, criterion, model_type, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            if model_type == 'torchvision':
                outputs = model(images)['out']
            elif model_type == 'smp':
                outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            dice = dice_coef(outputs, masks)
            dices.append(dice.detach().cpu())
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(val_loader.dataset.classes, dices_per_class)
    ]
    dice_str = "\n".join(dice_str) 
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    val_loss = total_loss / len(val_loader.dataset)
    
    return avg_dice, val_loss

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg):
    print(f'Start training..')
    logger = WandbLogger(name=cfg.wandb_run_name)
    
    wandb_config = {
        "model_name": cfg.model_name,
        "model_backbone": cfg.encoder_name,
        "learning_rate": cfg.lr,
        "max_epoch": cfg.max_epoch,
        "optimizer": cfg.optim,
        "criterion": cfg.loss.name
    }
    best_dice = 0.
    scaler = torch.cuda.amp.GradScaler()
    model = model.cuda()
    logger.initialize(wandb_config)
    
    for epoch in range(cfg.max_epoch):
        epoch_start = time.time()
        epoch_loss = 0
        torch.cuda.empty_cache() # 학습 시작 전 캐시 삭제
        model.train()
        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            
            # Mixed Precision Training으로 loss 계산에서만 FP32 사용
            with torch.cuda.amp.autocast():
                if cfg.model_type == 'torchvision':
                    outputs = model(images)['out']
                elif cfg.model_type == 'smp':
                    outputs = model(images)
                loss = criterion(outputs, masks)
            # 스케일된 loss를 사용해 backward 및 optimizer step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 80 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{cfg.max_epoch}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
        scheduler.step()
        epoch_time = datetime.timedelta(seconds=time.time() - epoch_start)
        dataset_size = len(train_loader.dataset)
        epoch_loss = epoch_loss / dataset_size
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % cfg.val_every == 0:
            dice, val_loss = validation(epoch + 1, model, val_loader, criterion, cfg.model_type)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {cfg.save_dir}")
                best_dice = dice
                ckpt_path = save_model(model, cfg.save_dir)
            logger.log_model(ckpt_path, f'model-epoch-{epoch+1}')
            logger.log_epoch_metrics(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "dice": dice,
                    "val_loss": val_loss,
                    "epoch_time": epoch_time.total_seconds()
                }
            )
    logger.finish()

def main(cfg):
    set_seed(cfg.random_seed)
    fnames, labels = set_data(cfg)

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    
    train_trans = TransformSelector('albumentation')
    train_tf = train_trans.get_transform(True, cfg.size)
    val_trans = TransformSelector('albumentation')
    val_tf = val_trans.get_transform(False, cfg.size)

    train_dataset = XRayDataset(fnames, labels, cfg.image_root, cfg.label_root, cfg.kfold, train_tf, is_train=True)
    valid_dataset = XRayDataset(fnames, labels, cfg.image_root, cfg.label_root, cfg.kfold, val_tf, is_train=False)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.train_num_workers,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.valid_num_workers,
        drop_last=False
    )
    
    model_selector = ModelSelector(
        model_type=cfg.model_type,
        num_classes=len(valid_loader.dataset.classes),
        model_name=cfg.model_name,
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights
        # pretrained=cfg.pretrained
    )
    model = model_selector.get_model()
    
    if cfg.loss.params:
        loss = LossSelector(cfg.loss.name, **cfg.loss.params)
    else:
        loss = LossSelector(cfg.loss.name)
    criterion = loss.get_loss()

    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sched = SchedulerSelector(cfg.scheduler, optimizer, cfg.max_epoch)
    scheduler = sched.get_sched()
    
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler, cfg)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)