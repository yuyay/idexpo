import os
from typing import Optional, Union

from pathlib import Path
import yaml
import fire
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from idexpo.models import IDExpOFinetuneGradCam
from idexpo.datasets.utils import cifar10_mean_std
from idexpo.config_utils import Config
from idexpo.models.resnet import get_resnet


def run(
    base_config: Config,
    devices: Union[int,list,None] = None,
    version: Union[int,str,None] = None,
):
    cfg = base_config.config
    seed_everything(cfg['seed'])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])
    cifar10_dm = CIFAR10DataModule(
        data_dir=cfg['datamodule']['data_root'],
        batch_size=cfg['datamodule']['batchsize'],
        num_workers=cfg['num_workers'],
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    
    model = get_resnet(
        cfg['model']['model_name'], 10, 
        cfg['model']['pretrained'], cfg['model']['finetune_last_layers'], 
        frozen_bn=cfg['model']['frozen_bn'] if 'frozen_bn' in cfg['model'] else False)

    method_name, explainer_name = cfg['method'].split('_')
    pl_model = IDExpOFinetuneGradCam(
        model=model,
        bg=cifar10_mean_std()[0],
        **cfg['model'],
    )

    name = os.path.basename(__file__).strip('.py')
    logger = CSVLogger(cfg['output_dir'], name=name, version=version)
    ckpt_cb = ModelCheckpoint(
        filename='best-ckptobj-{epoch}-{step}-{val_acc:.3f}-{val_ins:.3f}-{val_del:.3f}',
        monitor='val_ckptobj', mode='max')
    trainer = Trainer(
        logger=logger,
        callbacks=[
            ckpt_cb, 
            LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
        **cfg['trainer']
    )

    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    trainer.validate(
        pl_model, 
        dataloaders=[cifar10_dm.val_dataloader(), cifar10_dm.test_dataloader()])
    logdir = Path(trainer.logger.log_dir)
    with open(logdir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)
    trainer.fit(
        pl_model, 
        train_dataloaders=cifar10_dm.train_dataloader(), 
        val_dataloaders=[cifar10_dm.val_dataloader(), cifar10_dm.test_dataloader()])

    # Visualize evaluation
    log = pd.read_csv(logdir / 'metrics.csv')

    metric_names = ['loss', 'ce', 'expl_loss', 'regul', 'cins', 'ins', 'cdel', 'del', 'acc', 'expo']
    f, axs = plt.subplots(len(metric_names), 3, figsize=(14, 4*len(metric_names)), tight_layout=True)
    for i, metric_name in enumerate(metric_names):
        for j, subset in enumerate(['train', 'val', 'test']):
            if subset == 'train':
                metric = f'train_{metric_name}_epoch'
            elif subset == 'val':
                metric = f'val_{metric_name}'
            elif subset == 'test':
                metric = f'test_{metric_name}'
            if metric not in log.columns:
                continue

            vallog = log[[metric] + ['step']].dropna(how='all', subset=metric)
            sns.lineplot(
                vallog, x='step', y=metric, markers=True, dashes=False, ax=axs[i, j])
            axs[i,j].set_title(metric)
            axs[i,j].set_xlabel('Iterations')
            axs[i,j].set_ylabel(metric_names[i])
    plt.savefig(logdir / 'evaluations.pdf')

    return trainer.callback_metrics


def main(
    cfg_path: str = 'cifar10_gradcam.yaml',
    devices: Union[int,list,None] = None,
):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if devices:
        cfg['trainer']['devices'] = devices

    base_config = Config(cfg, replace_env_variables=True)
    run(base_config)


if __name__ == '__main__':
    fire.Fire(main)
