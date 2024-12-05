import torch.nn as nn

class CustomBCEWithLogitsLoss(nn.Module):
    """
    Base Code에서 사용된 BCEWithLogitsLoss
    """
    def __init__(self, **kwargs):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)

