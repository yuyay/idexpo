from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.functional import accuracy

from idexpo.modules import (
    batch_insertion_for_image, batch_deletion_for_image, 
    batch_continuous_insertion_deletion_for_image)
from idexpo.models import IDExpOBase, IDExpOFinetuneBase


class IDExpOGradCam(IDExpOBase):
    def forward(
        self,
        X: torch.Tensor,  # batch of images (b c h w)
        y: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        expls = self._forward_explanation(X, y, stage)
        preds = self._forward_prediction(X)
        return preds, expls

    def _forward_prediction(
        self,
        X: torch.Tensor  # batch of images (b c h w)
    ):
        return F.log_softmax(self.model(X), dim=-1)  # (b C)

    @torch.enable_grad() 
    @torch.inference_mode(False)
    def _forward_explanation(
        self,
        X: torch.Tensor,  # batch of images (b c h w)
        y: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        batchsize, c, h, w = X.shape
        device = X.device
        dtype = X.dtype

        # This one will be called during the backward
        def tensor_hook(grad):
            global grads 
            grads = grad.clone()

        def forward_hook(module, input, output):
            global activations
            activations = output
            output.register_hook(tensor_hook)

        target_layer = eval(f"self.model.{self.kwargs['layer_name']}")  
        handle_forward = target_layer.register_forward_hook(forward_hook)

        # self.model.zero_grad()
        grad_X = X.clone().requires_grad_()
        outputs = self.model(grad_X)
        if y == None:
            y = torch.argmax(outputs, dim=-1)

        params = []
        expls = []
        for i in range(batchsize):
            # torch.autograd.grad(outputs[i, y[i]], params.values(), create_graph=True)
            outputs[i, y[i]].backward(retain_graph=True)
            weights = torch.mean(grads[i], dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations[i], dim=0)
            cam = F.relu(cam)
            expls.append(cam)

        expls = torch.stack(expls)

        _, h_cam, w_cam = expls.shape
        expls = torch.repeat_interleave(expls, h // h_cam, dim=1)
        expls = torch.repeat_interleave(expls, w // w_cam, dim=2)

        handle_forward.remove()
        # handle_backward.remove()

        return expls

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        batchsize = x.shape[0]
        
        logits = self._forward_prediction(x) 
        ce_loss = F.nll_loss(logits, y)
        bg = self.bg.to(x.device)

        expls = self._forward_explanation(x, y, 'train')
        cins_scores, cdel_scores = batch_continuous_insertion_deletion_for_image(
            x, expls, bg,
            self.model, y, self.cdel_n, self.K, self.cdel_temperature, self.weight_scale,
            self.cdel_with_softmax)
        cins_mean = cins_scores.mean()
        cdel_mean = cdel_scores.mean()
        expl_loss = cins_mean + cdel_mean
        regul = torch.norm(expls)**2 / expls.numel()
        total_loss = ce_loss + self.lamb1*expl_loss + self.lamb2*regul

        acc = accuracy(logits, y, 'multiclass', num_classes=self.num_classes)
        self.log("train_loss", total_loss, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_ce", ce_loss, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_cins", cins_mean, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_cdel", cdel_mean, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_expl_loss", expl_loss, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_regul", regul, on_epoch=True, on_step=True, add_dataloader_idx=False)
        self.log("train_acc", acc, on_epoch=True, on_step=True, add_dataloader_idx=False)

        return total_loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        batchsize = x.shape[0]
        logits, expls = self.forward(x, None, stage)
        ce_loss = F.nll_loss(logits, y)
        
        bg = self.bg.to(x.device)
        del_scores = batch_deletion_for_image(
            x, expls, bg, self.model, y, K=self.K, step=self.del_step, reduction='mean')
        del_mean = del_scores.mean()
        ins_scores = batch_insertion_for_image(
            x, expls, bg, self.model, y, K=self.K, step=self.del_step, reduction='mean')
        ins_mean = ins_scores.mean()

        cins_scores, cdel_scores = batch_continuous_insertion_deletion_for_image(
            x, expls, bg,
            self.model, y, self.cdel_n, self.K, self.cdel_temperature, self.weight_scale,
            self.cdel_with_softmax)
        cins_mean = cins_scores.mean()
        cdel_mean = cdel_scores.mean()
        expl_loss = cins_mean + cdel_mean
        regul = torch.norm(expls)**2 / expls.numel()
        acc = accuracy(logits, y, 'multiclass', num_classes=self.num_classes)
        total_loss = ce_loss + self.lamb1*expl_loss + self.lamb2*regul
        
        if stage:
            self.log(f"{stage}_ce", ce_loss, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{stage}_del", del_mean, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{stage}_cdel", cdel_mean, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{stage}_ins", ins_mean, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{stage}_cins", cins_mean, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{stage}_expl_loss", expl_loss, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{stage}_regul", regul, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{stage}_loss", total_loss, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{stage}_acc", acc, prog_bar=True, add_dataloader_idx=False)


class IDExpOFinetuneGradCam(IDExpOGradCam, IDExpOFinetuneBase):
    def forward(
        self, 
        X: torch.Tensor,  # batch of images (b c h w)
        y: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        return IDExpOGradCam.forward(self, X, None, stage)

    def _forward_prediction(self, X: torch.Tensor):
        return IDExpOGradCam._forward_prediction(self, X)

    def _forward_explanation(
        self, 
        X: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ):
        return IDExpOGradCam._forward_explanation(self, X, y, stage)

