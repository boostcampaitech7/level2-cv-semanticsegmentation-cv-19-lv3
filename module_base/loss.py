import torch.nn as nn

class BCE(nn.Module):
    def __init__(self, **kwargs):
        super(BCE, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, preds, targets):
        return self.loss(preds, targets)

class LossSelector:
    def __init__(self, loss, **kwargs):
        if loss == 'BCEWithLogitsLoss':
            self.loss = BCE(**kwargs)

    def get_loss(self):
        return self.loss