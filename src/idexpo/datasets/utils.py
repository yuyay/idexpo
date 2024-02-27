import torch
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    

def cifar10_mean_std(normalize: bool = True):
    if normalize:
        mean = torch.tensor([125.3 / 255, 123.0 / 255, 113.9 / 255])
        std = torch.tensor([63.0 / 255, 62.1 / 255, 66.7 / 255])
    else:
        mean = torch.tensor([125.3, 123.0, 113.9])
        std = torch.tensor([63.0, 62.1, 66.7])
    return mean, std


def cifar100_mean_std(normalize: bool = True):
    if normalize:
        mean = torch.tensor([129.3 / 255, 124.1 / 255, 112.4 / 255])
        std = torch.tensor([68.2 / 255, 65.4 / 255, 70.4 / 255])
    else:
        mean = torch.tensor([129.3, 124.1, 112.4])
        std = torch.tensor([68.2, 65.4, 70.4])
    return mean, std


def imagenet_mean_std():
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return mean, std
    

def stl10_mean_std():
    mean = torch.tensor([0.43, 0.42, 0.39])
    std = torch.tensor([0.27, 0.26, 0.27])
    return mean, std


def cifar100_normalization():
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
        )

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
        std=[x / 255.0 for x in [68.2, 65.4, 70.4]],
    )
    return normalize
