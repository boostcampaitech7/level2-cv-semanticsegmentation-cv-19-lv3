import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

'''
model_selector = ModelSelector(
        model_type=model_type, 
        num_classes=num_classes,
        model_name=model_name,
        encoder_weights=encoder_weights,
        pretrained=True
    )
model = model_selector.get_model()
'''

class TorchvisionModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(TorchvisionModel, self).__init__()
        self.model = models.segmentation.__dict__[model_name](pretrained=pretrained)
        self.model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class SMP(nn.Module):
    """
    SMP에서 제공하는 사전 훈련된 모델을 사용하는 클래스
    """
    def __init__(self, model_name, encoder_weights, num_classes):
        super(SMP, self).__init__()
        self.model = smp.Unet(
            encoder_name=model_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        
        return self.model(x)

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스
    """
    def __init__(self, model_type, num_classes, **kwargs):
        
        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'smp':
            self.model = SMP(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")
        
    def get_model(self) -> nn.Module:
        
        # 생성된 모델 객체 반환
        return self.model