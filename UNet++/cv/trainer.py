import os
import time
import wandb
import torch
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
from datetime import timedelta
from torch.utils.data import DataLoader


def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 threshold: float,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler,
                 criterion: torch.nn.modules.loss._Loss,
                 max_epoch: int,
                 save_dir: str,
                 val_interval: int):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_epoch = max_epoch
        self.save_dir = save_dir
        self.threshold = threshold
        self.val_interval = val_interval


    def save_model(self, epoch, dice_score, before_path):
        # checkpoint 저장 폴더 생성
        if not osp.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        if before_path != "" and osp.exists(before_path):
            os.remove(before_path)

        output_path = osp.join(self.save_dir, f"best_{epoch}epoch_{dice_score:.4f}.pt")
        torch.save(self.model, output_path)
        return output_path


    def train_epoch(self, epoch):
        train_start = time.time()
        self.model.train()
        total_loss = 0.0

        with tqdm(total=len(self.train_loader), desc=f"[Training Epoch {epoch}]", disable=False) as pbar:
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        train_end = time.time() - train_start 
        print("Epoch {}, Train Loss: {:.4f} || Elapsed time: {} || ETA: {}\n".format(
            epoch,
            total_loss / len(self.train_loader),
            timedelta(seconds=train_end),
            timedelta(seconds=train_end * (self.max_epoch - epoch))
        ))
        return total_loss / len(self.train_loader)
    

    def validation(self, epoch):
        val_start = time.time()
        self.model.eval()

        total_loss = 0
        dices = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'[Validation Epoch {epoch}]', disable=False) as pbar:
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)

                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)

                    # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    loss = self.criterion(outputs, masks)
                    total_loss += loss.item()

                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > self.threshold).detach().cpu()
                    masks = masks.detach().cpu()

                    dice = dice_coef(outputs, masks)
                    dices.append(dice)

                    pbar.update(1)
                    pbar.set_postfix(dice=torch.mean(dice).item(), loss=loss.item())

        val_end = time.time() - val_start
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)
        ]

        dice_str = "\n".join(dice_str)
        print(dice_str)
        
        avg_dice = torch.mean(dices_per_class).item()
        print("avg_dice: {:.4f}".format(avg_dice))
        print("Validation Loss: {:.4f} || Elapsed time: {}\n".format(
            total_loss / len(self.val_loader),
            timedelta(seconds=val_end),
        ))

        class_dice_dict = {f"{c}'s dice score" : d for c, d in zip(self.val_loader.dataset.class2ind, dices_per_class)}
        
        return avg_dice, class_dice_dict, total_loss / len(self.val_loader)
    


    def train(self):
        print(f'Start training..')

        best_dice = 0.
        before_path = ""
        
        for epoch in range(1, self.max_epoch + 1):

            train_loss = self.train_epoch(epoch)

            wandb.log({
                "Epoch" : epoch,
                "Train Loss" : train_loss,
                "Learning Rate": self.scheduler.get_last_lr()[0]
            }, step=epoch)

            # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
            if epoch % self.val_interval == 0:
                avg_dice, dices_per_class, val_loss = self.validation(epoch)
                wandb.log({
                    "Validation Loss" : val_loss,
                    "Avg Dice Score" : avg_dice,
                    **dices_per_class
                }, step=epoch)
                
                if best_dice < avg_dice:
                    print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {avg_dice:.4f}\n")
                    best_dice = avg_dice
                    before_path = self.save_model(epoch, best_dice, before_path)

            self.scheduler.step()