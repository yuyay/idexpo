from argparse import ArgumentParser
from typing import Any, Callable, Optional, Sequence, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datamodules import CIFAR10DataModule
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, cifar100_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
# from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

from idexpo.datasets.utils import cifar100_normalization


if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import CIFAR100
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    CIFAR100 = None


class CIFAR100DataModule(CIFAR10DataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-100
    Specs:
        - 100 classes (1 per class)
        - Each image is (3 x 32 x 32)
    Standard CIFAR100, train, val, test splits and transforms
    Transforms::
        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
    Example::
        dm = CIFAR100DataModule(PATH)
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    Or you can set your own transforms
    Example::
        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    @property
    def num_samples(self) -> int:
        train_len, _ = self._get_splits(len_dataset=50_000)
        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 100

    def default_transforms(self) -> Callable:
        if self.normalize:
            cf100_transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar100_normalization()])
        else:
            cf100_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return cf100_transforms

