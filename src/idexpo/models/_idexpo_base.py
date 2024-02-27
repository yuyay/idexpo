from typing import Optional, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from idexpo.modules import (
    batch_continuous_insertion_deletion_for_image, 
    batch_insertion_for_image, batch_deletion_for_image)


__all__ = ['IDExpOBase', 'IDExpOFinetuneBase']


class IDExpOBase(LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        model_params: dict = {}, 
        num_classes: int = 10,
        lamb1: float = 1.0,  # hparam for ins/del regularizer
        lamb2: float = 1.0,  # hparam for l2 regularizer
        cdel_n: int = 10,  # hparam for n of cdel
        K: Union[float, int] = 1.0,  # hparam for K of cdel and del
        cdel_temperature: float = 10.,  # hparam for temp of cdel
        del_step: int = 100,
        lr: float = 0.05,
        seed: int = 0,
        bg: torch.Tensor = torch.Tensor([0.0, 0.0, 0.0]),  # channel-wise bg values
        pretrained: Optional[str] = None,
        switch_epoch: Optional[int] = 10,  # epoch to switch to finetuning mode. None means no switching.
        expl_loss_name: str = 'both',  # 'both' or 'cins' or 'cdel'
        weight_scale: Optional[float] = None,  # None or float
        cdel_with_softmax: bool = False,
        ckpt_acc_thresh: float = 0.8,
        cdel_calculation: str = 'direct',
        **kwargs,
    ):
        assert expl_loss_name in ['both', 'cins', 'cdel']

        super().__init__()
        self.model = model
        self.model_params = model_params
        self.num_classes = num_classes
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.cdel_n = cdel_n
        self.K = K
        self.cdel_temperature = cdel_temperature
        self.del_step = del_step
        self.lr = lr 
        self.seed = seed
        self.bg = bg
        self.pretrained = pretrained
        self.switch_epoch: int = switch_epoch if isinstance(switch_epoch, int) else self.trainer.max_epochs  # type: ignore
        self.expl_loss_name = expl_loss_name
        self.weight_scale = weight_scale
        self.cdel_with_softmax = cdel_with_softmax
        self.ckpt_acc_thresh = ckpt_acc_thresh
        self.cdel_calculation = cdel_calculation
        self.kwargs = kwargs
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor):
        preds = self._forward_prediction(x)
        expls = self._forward_explanation(x)
        return preds, expls

    def _forward_prediction(self, x: torch.Tensor):
        pass

    def _forward_explanation(self, x: torch.Tensor):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        batchsize = x.shape[0]
        logits, expls = self.forward(x)
        ce_loss = F.nll_loss(logits, y)
        bg = self.bg.to(x.device)

        cins_scores, cdel_scores = batch_continuous_insertion_deletion_for_image(
            x, expls[range(batchsize), y, ...], bg,
            self.model, y, self.cdel_n, self.K, self.cdel_temperature, self.weight_scale,
            self.cdel_with_softmax, self.cdel_calculation)
        cins_mean = cins_scores.mean()
        cdel_mean = cdel_scores.mean()

        expl_loss = cins_mean + cdel_mean
        regul = torch.norm(expls)**2 / expls.numel()
        total_loss = ce_loss + self.lamb1*expl_loss + self.lamb2*regul
        # total_loss = self.lamb1*expl_loss + self.lamb2*regul
        acc = accuracy(logits, y, 'multiclass', num_classes=self.num_classes)
        self.log("train_loss", total_loss, on_epoch=True, on_step=True)
        self.log("train_ce", ce_loss, on_epoch=True, on_step=True)
        self.log("train_cins", cins_mean, on_epoch=True, on_step=True)
        self.log("train_cdel", cdel_mean, on_epoch=True, on_step=True)
        self.log("train_expl_loss", expl_loss, on_epoch=True, on_step=True)
        self.log("train_regul", regul, on_epoch=True, on_step=True)
        # self.log("train_del", del_mean, on_epoch=True, on_step=True)
        # self.log("train_ins", ins_mean, on_epoch=True, on_step=True)
        self.log("train_acc", acc, on_epoch=True, on_step=True)

        return total_loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        batchsize = x.shape[0]

        logits, expls = self.forward(x)
        ce_loss = F.nll_loss(logits, y)
        bg = self.bg.to(x.device)
        del_scores = batch_deletion_for_image(
            x, expls[range(batchsize), y, ...], bg, self.model, y, K=self.K, step=self.del_step, reduction='mean')
        del_mean = del_scores.mean()
        ins_scores = batch_insertion_for_image(
            x, expls[range(batchsize), y, ...], bg, self.model, y, K=self.K, step=self.del_step, reduction='mean')
        ins_mean = ins_scores.mean()

        cins_scores, cdel_scores = batch_continuous_insertion_deletion_for_image(
            x, expls[range(batchsize), y, ...], bg,
            self.model, y, self.cdel_n, self.K, self.cdel_temperature, self.weight_scale,
            self.cdel_with_softmax, self.cdel_calculation)
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            self.evaluate(batch, "val")
        elif dataloader_idx == 1:
            self.evaluate(batch, "test")
        else:
            raise NotImplementedError

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def on_validation_epoch_start(self):
        # self.log('start_date', int(datetime.now().strftime('%Y%m%d%H%M%S'))) 
        pass

    def validation_epoch_end(self, outputs):
        metrics = self.trainer.callback_metrics
        ckptobj = -10**8
        if metrics['val_acc'] >= self.ckpt_acc_thresh:
            ckptobj = 2.0 * metrics['val_acc'] + metrics['val_ins'] + (1 - metrics['val_del'])
        self.log('val_ckptobj', ckptobj, add_dataloader_idx=False)
        self.log('global_step', self.trainer.global_step, add_dataloader_idx=False)
        # self.log('end_date', int(datetime.now().strftime('%Y%m%d%H%M%S'))) 
        return

    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        n_steps_per_epoch = len(self.trainer.train_dataloader)
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [opt], [sche]


class IDExpOFinetuneBase(IDExpOBase):
    def configure_optimizers(self):
        finet_opt = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        return finet_opt