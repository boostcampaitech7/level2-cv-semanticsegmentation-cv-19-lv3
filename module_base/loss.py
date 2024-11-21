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

class IoULoss(nn.Module):
    def __init__(self, **kwargs):
        super(IoULoss, self).__init__()

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


class LossSelector:
    def __init__(self, loss, **kwargs):
        if loss == 'BCEWithLogitsLoss':
            self.loss = BCE(**kwargs)
        elif loss == 'DiceBCELoss':
            self.loss = DiceBCELoss(**kwargs)

    def get_loss(self):
        return self.loss