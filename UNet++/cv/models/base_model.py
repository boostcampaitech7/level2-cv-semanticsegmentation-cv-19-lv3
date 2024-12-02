import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetModel(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self,
                 **kwargs):
        super(UnetModel, self).__init__()
        self.model = smp.Unet(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)