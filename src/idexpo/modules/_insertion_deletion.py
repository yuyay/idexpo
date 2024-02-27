from typing import Union

import torch
from einops import repeat, rearrange

__all__ = [
    'batch_insertion_for_image', 'batch_deletion_for_image',
    # 'batch_insertion_for_tabular', 'batch_deletion_for_tabular',
]


def batch_insertion_for_image(
    x: torch.Tensor,  # batch of images (b c h w)
    expl: torch.Tensor, # batch of explanations (b h w)
    bg: torch.Tensor,  # channel-wise background values (c,)
    f: torch.nn.Module,  # classifier
    labels: torch.Tensor,  # label indices to evaluate
    K: Union[float, int] = 0.2,  # proportion or number of inserted pixels
    normalize: bool = False,  # normalize the values of f
    step: int = 1,  # number of pixels modified per one iteration.
    reduction: str = 'sum'  # 'sum' or 'mean'
):
    b, c, h, w = x.shape
    K_ = int(K * h * w) if isinstance(K, float) else K
    maxiter = K_ // step
    expl_flat = rearrange(expl, 'b h w -> b (h w)')
    bg_ = bg.reshape(c, 1)

    x_flat = rearrange(x, 'b c h w -> b c (h w)')
    x_batch = torch.empty((b, maxiter+1, c, h*w), dtype=x.dtype, device=x.device)
    values, indices = torch.sort(expl_flat, dim=-1, descending=True)
    for bi in range(b):
        for i in range(1, maxiter + 1):
            index = i * step
            x_batch[bi, i, :, indices[bi, :index]] = x_flat[bi, :, indices[bi, :index]]
            x_batch[bi, i, :, indices[bi, index:]] = bg_
    
    batch_labels = labels.detach().clone().repeat_interleave(maxiter+1)
    x_batch = rearrange(x_batch, 'b r c (h w) -> (b r) c h w', h=h)
    y_batch = torch.softmax(f(x_batch), dim=-1)[range(b * (maxiter+1)), batch_labels]
    y_batch = y_batch.reshape(b, maxiter+1)  # (b,r)  TODO: batch inference
    y_batch = y_batch / y_batch[:, 0].reshape(-1, 1) if normalize else y_batch

    if reduction == 'sum':
        scores = torch.sum(y_batch[:, 1:], dim=-1)
    elif reduction == 'mean':
        scores = torch.mean(y_batch[:, 1:], dim=-1)
        
    return scores


def batch_deletion_for_image(
    x: torch.Tensor,  # batch of images (b c h w)
    expl: torch.Tensor, # batch of explanations (b h w)
    bg: torch.Tensor,  # channel-wise background values (c,)
    f: torch.nn.Module,  # classifier
    labels: torch.Tensor,  # label indices to evaluate
    K: Union[float, int] = 0.2,  # proportion or number of removed pixels
    normalize: bool = False,  # normalize the values of f
    step: int = 1,  # number of pixels modified per one iteration.
    reduction: str = 'sum'  # 'sum' or 'mean'
):
    b, c, h, w = x.shape
    K_ = int(K * h * w) if isinstance(K, float) else K
    maxiter = K_ // step
    expl_flat = rearrange(expl, 'b h w -> b (h w)')
    bg_ = bg.reshape(c, 1)

    x_flat = rearrange(x, 'b c h w -> b c (h w)')
    x_batch = torch.empty((b, maxiter+1, c, h*w), dtype=x.dtype, device=x.device)
    values, indices = torch.sort(expl_flat, dim=-1, descending=True)
    for bi in range(b):
        for i in range(1, maxiter + 1):
            index = i * step
            x_batch[bi, i, :, indices[bi, :index]] = bg_
            x_batch[bi, i, :, indices[bi, index:]] = x_flat[bi, :, indices[bi, index:]]
    
    batch_labels = labels.detach().clone().repeat_interleave(maxiter+1)
    x_batch = rearrange(x_batch, 'b r c (h w) -> (b r) c h w', h=h)
    y_batch = torch.softmax(f(x_batch), dim=-1)[range(b * (maxiter+1)), batch_labels]
    y_batch = y_batch.reshape(b, maxiter+1)  # (b,r)  TODO: batch inference
    y_batch = y_batch / y_batch[:, 0].reshape(-1, 1) if normalize else y_batch

    if reduction == 'sum':
        scores = torch.sum(y_batch[:, 1:], dim=-1)
    elif reduction == 'mean':
        scores = torch.mean(y_batch[:, 1:], dim=-1)

    return scores