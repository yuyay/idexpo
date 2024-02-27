from typing import Union, Any, Optional, Tuple

import math
import numpy as np
import torch
from einops import repeat, rearrange

__all__ = [
    'batch_continuous_insertion_deletion_for_image',
    # 'batch_continuous_insertion_deletion_for_tabular',  
]


def batch_continuous_insertion_deletion_for_image(
    x: torch.Tensor,  # batch of images (b c h w)
    expl: torch.Tensor, # batch of explanations (b h w)
    bg: torch.Tensor,  # channel-wise background values (c,)
    f: Union[torch.nn.Module, Any],  # classifier
    labels: torch.Tensor,  # label indexes to evaluate
    n: int = 10,
    K: Union[float, int] = 1.0,  # proportion or number of removed pixels
    temperature: float = 1.0,
    weight_scale: Optional[float] = None,
    # normalize: bool = False,  # normalize the values of f
    with_softmax: bool = False,
    cdel_calculation: str = 'ratio'  # 'ratio' or 'direct' or 'pure'
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-2:] == expl.shape[-2:]
    assert cdel_calculation in ['ratio', 'direct', 'pure']
    B, c, h, w = x.shape

    step = h * w // n
    if with_softmax:
        expl_flat = torch.log_softmax(rearrange(expl, 'B h w -> B (h w)'), dim=-1)
    else:
        expl_flat = rearrange(expl, 'B h w -> B (h w)')
    norm_expl = rearrange(expl_flat, 'B (h w) -> B h w', h=h, w=w)
    K_ = int(K * h * w) if isinstance(K, float) else K
    maxiter = n if isinstance(K, float) and K == 1. else math.ceil(K_ / step)

    with torch.no_grad():
        sorted_expl, sorted_idxes = torch.sort(expl_flat, dim=-1, descending=True)
        diff = sorted_expl[:, :-1] - sorted_expl[:, 1:]
        sigmoid_scale = torch.mean(diff[diff.bool()], dim=-1)
        steps = torch.arange(1, maxiter + 1).long() * step
        ts = (sorted_expl[:, steps - 1] + sorted_expl[:, steps]) / 2

    if len(x.shape) == 4:
        x_batch_del = torch.empty((B, maxiter, c, h, w), dtype=x.dtype, device=x.device)
        x_batch_ins = torch.empty((B, maxiter, c, h, w), dtype=x.dtype, device=x.device)
    else:
        raise NotImplementedError
    
    bg_ = repeat(bg, 'c -> c h w', h=h, w=w)
    for j in range(B):
        alphas = torch.sigmoid(
            temperature / sigmoid_scale * (norm_expl[j].unsqueeze(0).repeat(maxiter,1,1) - ts[j].reshape(-1,1,1)))  # (n h w)

        alphas = repeat(alphas, 'n h w -> n c h w', c=c)
        x_batch_del[j] = alphas * bg_ + (1 - alphas) * x[j]
        x_batch_ins[j] = alphas * x[j] + (1 - alphas) * bg_
    
    BM = B * maxiter
    batch_labels = labels.detach().clone().repeat_interleave(maxiter)
    x_batch_del = rearrange(x_batch_del, 'B M c h w -> (B M) c h w')
    if cdel_calculation == 'direct':
        y_batch_del = rearrange(
            torch.softmax(f(x_batch_del), dim=-1)[range(BM), batch_labels], '(B M) -> B M', B=x.shape[0])
        cdel_scores = -torch.mean(torch.log(1 - y_batch_del), dim=1) 
    elif cdel_calculation == 'ratio':
        y_batch_del = rearrange(
            torch.log_softmax(f(x_batch_del), dim=-1)[range(BM), batch_labels], '(B M) -> B M', B=x.shape[0])
        y0 = torch.log_softmax(f(x), dim=-1)[range(B), labels].unsqueeze(1)  # (B, 1)
        cdel_scores = -torch.mean(y0 - y_batch_del, dim=1) 
    elif cdel_calculation == 'log':
        y_batch_del = rearrange(
            torch.log_softmax(f(x_batch_del), dim=-1)[range(BM), batch_labels], '(B M) -> B M', B=x.shape[0])
        cdel_scores = torch.mean(y_batch_del, dim=1) 
    else:
        raise NotImplementedError

    x_batch_ins = rearrange(x_batch_ins, 'B M c h w -> (B M) c h w')
    y_batch_ins = rearrange(
        torch.log_softmax(f(x_batch_ins), dim=-1)[range(BM), batch_labels], '(B M) -> B M', B=x.shape[0])
    cins_scores = -torch.mean(y_batch_ins, dim=1) 

    return cins_scores, cdel_scores
