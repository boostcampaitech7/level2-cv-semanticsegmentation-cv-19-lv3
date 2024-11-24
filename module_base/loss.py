import torch
import torch.nn as nn
import torch.nn.functional as F

class BCE(nn.Module):
    def __init__(self, **kwargs):
        super(BCE, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds, targets):
        return self.loss(preds, targets)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = F.sigmoid(preds)
        preds_f = preds.flatten(2)
        targets_f = targets.flatten(2)
        intersection = torch.sum(preds_f * targets_f, -1)

        dice = (2. * intersection + self.eps) / (torch.sum(preds_f, -1) + torch.sum(targets_f, -1) + self.eps)
        loss = 1 - dice

        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 각 클래스에 대한 가중치 조정
        self.gamma = gamma # Easy sample에 대한 loss 조절 (커질수록 loss가 줄어듦)

    def forward(self, preds, targets):
        # CE() -> -log(pt), FL() -> -a*((1-pt)^r) * log(pt)
        BCE = F.binary_cross_entropy_with_logits(preds, targets) # CE
        BCE_EXP = torch.exp(-BCE)
        loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(DiceBCELoss, self).__init__()
        self.bceWithLogitLoss = nn.BCEWithLogitsLoss(**kwargs)
        self.diceLoss = DiceLoss()

    def forward(self, preds, targets):
        bce_loss = self.bceWithLogitLoss(preds, targets)
        dice_loss = self.diceLoss(preds, targets)
        dice_bce_loss = bce_loss + dice_loss

        return dice_bce_loss

class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.focalLoss = FocalLoss()
    
    def forward(self, preds, targets):
        dice_loss = self.diceLoss(preds, targets)
        focal_loss = self.focalLoss(preds, targets)
        dice_focal_loss = dice_loss + focal_loss

        return dice_focal_loss

class LossSelector:
    def __init__(self, loss, **kwargs):
        if loss == 'BCEWithLogitsLoss':
            self.loss = BCE(**kwargs)
        elif loss == 'DiceBCELoss':
            self.loss = DiceBCELoss(**kwargs)
        elif loss == 'DiceFocalLoss':
            self.loss = DiceFocalLoss()

    def get_loss(self):
        return self.loss