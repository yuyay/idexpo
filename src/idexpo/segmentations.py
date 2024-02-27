from typing import Optional, Union

import numpy as np
import torch


class GridSegmentation(object):
    def __init__(
        self, 
        cell_size: int = 4,
        tensor_order: str = 'hwc'  # 'hwc' for np.array, 'chw' for torch.tensor
    ):
        self.cell_size = cell_size
        self.tensor_order = tensor_order

    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.tensor_order == 'hwc':
            h, w, c = image.shape
        else:
            c, h, w = image.shape
        assert h % self.cell_size == 0 and w % self.cell_size == 0

        h_grid = h // self.cell_size
        w_grid = w // self.cell_size
        if isinstance(image, np.ndarray):
            S = np.arange(h_grid * w_grid).astype('int').reshape(h_grid, w_grid)
            S = np.repeat(np.repeat(S, self.cell_size, axis=0), self.cell_size, axis=1)
        elif isinstance(image, torch.Tensor):
            S = torch.arange(h_grid * w_grid).long().reshape(h_grid, w_grid)
            S = torch.repeat_interleave(S, self.cell_size, dim=0)
            S = torch.repeat_interleave(S, self.cell_size, dim=1)
        else:
            raise NotImplementedError

        return S
