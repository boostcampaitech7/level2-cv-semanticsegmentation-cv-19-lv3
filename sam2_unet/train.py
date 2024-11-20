import os
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.dataset import FullDataset
from SAM2UNet import SAM2UNet
from omegaconf import OmegaConf
from argparse import ArgumentParser


def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(cfg):    
    dataset = FullDataset(cfg.train_image_path, cfg.train_mask_path, 512, mode='train')
    dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda")
    model = SAM2UNet(cfg.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": cfg.lr}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optim, cfg.max_epoch, eta_min=1.0e-7)
    # scheduler = MultiStepLR(optim, milestones=[8, 16], gamma=0.1, last_epoch=cfg.epoch)
    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(cfg.max_epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == cfg.max_epoch:
            torch.save(model.state_dict(), os.path.join('./checkpoints', 'SAM2-UNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join('./checkpoints', 'SAM2-UNet-%d.pth'% (epoch + 1)))

if __name__ == "__main__":
    set_seed(2024)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    main(cfg)