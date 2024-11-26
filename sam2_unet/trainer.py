import os
import time
import datetime
from tqdm.auto import tqdm
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.wandb_logger import WandbLogger

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        loss_fn,
        smooth_factor,
        epochs,
        threshold,
        save_dir,
        save_every,
        valid_every,
        wandb_id,
        wandb_name,
        resume=False,
        ckpt_path="",
        start_epoch=0
        ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.smooth_factor = smooth_factor
        self.epochs = epochs
        self.threshold = threshold
        self.save_dir = save_dir
        self.save_every = save_every
        self.valid_every = valid_every
        self.wandb_id = wandb_id
        self.wandb_name = wandb_name
        self.resume = resume
        self.ckpt_file = ckpt_path
        self.start_epoch = start_epoch
        self.scaler = torch.cuda.amp.GradScaler()
        self.classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]

    def save_checkpoint_epoch(self, epoch: int, valid_loss: float, valid_dice: float) -> None:
        ckpt_path = os.path.join(self.save_dir, f'epoch-{epoch + 1}-loss-{valid_loss:.4f}-dice-{valid_dice:.4f}.pth')
        save_checkpoint(self.model, self.optimizer, self.scheduler, epoch+1, valid_dice, ckpt_path)
        print(f"Checkpoint updated at epoch {epoch + 1} and saved as {ckpt_path}")
    
    def dice_coef(self, y_true, y_pred):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def train(self, train_loader: DataLoader):
        self.model.train()
        
        train_loss = 0.
        train_dices = []
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for images, masks in progress_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            # with torch.cuda.amp.autocast():
            pred0, pred1, pred2 = self.model(images)
            loss0 = self.loss_fn(pred0, masks, self.smooth_factor)
            loss1 = self.loss_fn(pred1, masks, self.smooth_factor)
            loss2 = self.loss_fn(pred2, masks, self.smooth_factor)
            loss = loss0 + loss1 + loss2

            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * masks.shape[0]

            outputs = torch.sigmoid(pred0)
            outputs = (outputs > self.threshold)
            dice = self.dice_coef(outputs, masks)
            train_dices.append(dice.detach().cpu())

            del outputs, masks, dice

            progress_bar.set_postfix(loss=loss.item())
        progress_bar.close()

        dices = torch.cat(train_dices, 0)
        dices_per_class = torch.mean(dices, 0)
        train_dice = torch.mean(dices_per_class).item()

        return train_loss, train_dice

    def validate(self, valid_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()

        valid_loss = 0.
        valid_dices = []

        progress_bar = tqdm(valid_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(progress_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs, _, _ = self.model(images)

                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
                loss = self.loss_fn(outputs, masks, self.smooth_factor)
                valid_loss += loss.item() * masks.shape[0]

                outputs = torch.sigmoid(outputs)
                outputs = (outputs > self.threshold)
                dice = self.dice_coef(outputs, masks)
                valid_dices.append(dice.detach().cpu())
                
                del outputs, masks, dice

                progress_bar.set_postfix(loss=loss.item())
        progress_bar.close()

        dices = torch.cat(valid_dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(self.classes, dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        print(dice_str)
        valid_dice = torch.mean(dices_per_class).item()

        return valid_loss, valid_dice

    def do_train(self) -> None:
        logger = WandbLogger(name=self.wandb_name)
        logger.initialize({
            "model_type": "SAM2-UNet",
            "model_name": "SAM2-UNet",
            "criterion": "BCEWithLogitLoss",
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "max_epoch": self.epochs,
            "optimizer": type(self.optimizer).__name__,
            "scheduler": type(self.scheduler).__name__
        })
        if self.resume:
            self.load_settings()
        print(f"training start")

        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            train_loss, train_dice = 0., 0.
            train_dice, valid_dice = 0., 0.
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_dice = self.train(self.train_loader)
            train_loss = train_loss / len(self.train_loader.dataset)

            torch.cuda.empty_cache()
            if epoch % self.valid_every == 0:
                valid_loss, valid_dice = self.validate(self.valid_loader)
                valid_loss = valid_loss / len(self.valid_loader.dataset)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            epoch_time = datetime.timedelta(seconds=time.time() - epoch_start)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.6f} | Vaild Loss: {valid_loss:.6f} | Vaild Dice: {valid_dice:.6f}\n")
            logger.log_epoch_metrics(
                {
                    "epoch": epoch,
                    "epoch_time": epoch_time.total_seconds(),
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "valid_loss": valid_loss,
                    "valid_dice": valid_dice,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
            )
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint_epoch(epoch, valid_loss, valid_dice)

        self.save_checkpoint_epoch(epoch, valid_loss, valid_dice)
        
    def load_settings(self) -> None:
        print("loading prev training setttings")
        try:
            self.epochs, self.model, self.optimizer, self.scheduler, self.loss = load_checkpoint(self.ckpt_path, self.model, self.optimizer, self.scheduler)
            print("loading successful")
        except:
            raise Exception('loading failed')