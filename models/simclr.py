## Standard libraries
import os
from copy import deepcopy
from PIL import Image
## tqdm for loading bars
from tqdm.notebook import tqdm
import typing as tp
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision and transforms
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms
import matplotlib.pyplot as plt

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pyrallis

import typing as tp



class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim: int, lr: float, temperature: float, weight_decay: float,
                    max_epochs: int = 500) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.convnet = torchvision.models.resnet18(num_classes=4 * hidden_dim)
        self.convnet.fc = nn.Sequential(
            self.convnet.fc, 
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def configure_optimizers(self) -> tp.Any:
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)

        return [optimizer], [lr_scheduler]

    def info_nce(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.vstack(imgs)

        feats = self.convnet(imgs)
        print(feats.shape)
        #similiarity = F.cosine_similarity()