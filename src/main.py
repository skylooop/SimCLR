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


from dataclasses import field, dataclass
# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import pyrallis
from dataclasses import dataclass, field
from transforms import ConstrastiveTrans
from .models.simclr import SimCLR


@dataclass
class SimCLR_cfg:
    dataset_path: str = field(default="./data")
    assets_path: str = field(default="./assets")
    batch_size: int = 128
    hidden_dim: int = 128
    lr: float = 5e-4
    temperature: float = 0.07
    weight_decay: float = 1e-4
    max_epochs:int = 500

@pyrallis.wrap()
def main(opts: SimCLR_cfg):
    print("Loading STL10 Datasets")

    transform_clr = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
    unlabeled_ds, train_data = load_datasets(transform_clr, opts)

    print(f"Saving random grid into {opts.assets_path}")
    save_grid(unlabeled_ds, opts)

    print("Loading Model")
    
    model = SimCLR() #add arguments

def load_datasets(transform_clr: tp.List[tp.Any], opts: SimCLR_cfg) -> tp.List[STL10]:
    unlabeled_dataset = STL10(
        root = opts.dataset_path, download=True, split="unlabeled", transform=ConstrastiveTrans(transform_clr)
    )
    train_data_contrastive = STL10(
        root = opts.dataset_path, download=True, split="train", transform=ConstrastiveTrans(transform_clr)
    )
    return unlabeled_dataset, train_data_contrastive

def save_grid(dataset, opts: SimCLR_cfg):
    imgs = torch.stack([img for idx in range(6) for img in dataset[idx][0]], dim = 0)
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9).detach().cpu()
    torchvision.transforms.ToPILImage()(imgs_grid).save(f"{opts.assets_path}/grid.jpg")

if __name__ == "__main__":
    main()
