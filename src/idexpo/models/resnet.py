from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from torchvision.ops import FrozenBatchNorm2d

import idexpo.models.cifar_resnet as cifar_models

__all__ = ['get_resnet']


OFFICIAL_WEIGHTS = {
    'resnet18': torchvision_models.ResNet18_Weights.IMAGENET1K_V1,
    'resnet34': torchvision_models.ResNet34_Weights.IMAGENET1K_V1,
    'resnet50': torchvision_models.ResNet50_Weights.IMAGENET1K_V1,
    'resnet101': torchvision_models.ResNet101_Weights.IMAGENET1K_V1,
    'resnet152': torchvision_models.ResNet152_Weights.IMAGENET1K_V1,
}

def get_resnet(
    name: str = 'resnet18', n_classes: int = 10, pretrained: Union[bool, str] = True, 
    finetune_last_layers: bool = False, use_cifar: bool = True,
    frozen_bn: bool = False,
):
    models = cifar_models if use_cifar else torchvision_models
    if pretrained == True:
        assert use_cifar == False
        model = getattr(models, name)(
            weights=OFFICIAL_WEIGHTS[name],
            norm_layer=FrozenBatchNorm2d if frozen_bn else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif pretrained == 'imagenet':
        model = getattr(models, name)(
            weights=OFFICIAL_WEIGHTS[name],
            norm_layer=FrozenBatchNorm2d if frozen_bn else None)
    elif isinstance(pretrained, str):
        model = getattr(models, name)(
            norm_layer=FrozenBatchNorm2d if frozen_bn else None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    else:
        model = getattr(models, name)()
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    if finetune_last_layers:
        for param in model.parameters():
            param.requires_grad = False
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True

    return model
