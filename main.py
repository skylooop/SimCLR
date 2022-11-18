## Standard libraries
import os
from copy import deepcopy

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
import albumentations as A
from torchvision.datasets import STL10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dotenv import load_dotenv
load_dotenv()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_data() -> None:
    unlabeled_data = STL10(root=os.getenv(DATASET_PATH), split="unlabeled", download=True,
                           transform=)

def main():
    augmentations: tp.Dict[str, tp.Any] = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(96, 96),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.8),
        A.ToGray(p=0.2),
        A.ToTensorV2(),
        A.Normalize()
    ])

    load_data(augmentations)
if __name__ == "__main__":
    NUM_WORKERS = os.cpu_count()
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)
    
    main()