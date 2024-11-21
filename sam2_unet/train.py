import os
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from utils.loss import structure_loss
from utils.dataset import FullDataset
from utils.optimizer import OptimizerSelector
from utils.scheduler import SchedulerSelector
from utils.transform import TransformSelector
from trainer import Trainer
from SAM2UNet import SAM2UNet
from omegaconf import OmegaConf
from argparse import ArgumentParser

def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def main(cfg):    
    device = torch.device("cuda")
    model = SAM2UNet(cfg.hiera_path)
    
    transform_selector = TransformSelector(image_size=cfg.image_size)
    train_transform = transform_selector.get_transform(is_train=True)
    valid_transform = transform_selector.get_transform(is_train=False)

    train_dataset = FullDataset(
        image_root=cfg.train_image_path, 
        gt_root=cfg.train_mask_path, 
        is_train=True,
        transforms=train_transform,
        kfold=cfg.kfold,
        k=cfg.k)
    
    valid_dataset = FullDataset(
        image_root=cfg.train_image_path, 
        gt_root=cfg.train_mask_path, 
        is_train=False,
        transforms=valid_transform,
        kfold=cfg.kfold,
        k=cfg.k)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.train_batch_size, 
        num_workers=cfg.train_num_workers,
        shuffle=True)
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.valid_batch_size, 
        num_workers=cfg.valid_num_workers,
        shuffle=False)
    
    optimizer_selector = OptimizerSelector(model, cfg.lr)
    optimizer = optimizer_selector.get_optimizer(cfg.optimizer, **cfg.optimizer_parameters)

    scheduler_selector = SchedulerSelector(optimizer)
    scheduler = scheduler_selector.get_scheduler(cfg.scheduler, **cfg.scheduler_parameters)

    loss_fn = structure_loss

    os.makedirs("./checkpoints", exist_ok=True)
    now = datetime.now()
    time = now.strftime('%Y-%m-%d_%H:%M:%S')
    save_dir = os.path.join('./checkpoints', time)
    os.makedirs(save_dir, exist_ok=True)

    # Config 저장
    save_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(cfg, save_path)

    print(f"Config saved at {save_path}")

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=cfg.max_epoch,
        threshold=cfg.threshold,
        save_dir=save_dir,
        save_every=cfg.save_every,
        wandb_id=cfg.wandb_id,
        wandb_name=cfg.wandb_name,
        resume=cfg.resume,
        ckpt_path=cfg.ckpt_path,
        start_epoch=cfg.start_epoch
    )

    trainer.train()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    set_seed(cfg.random_seed)
    main(cfg)