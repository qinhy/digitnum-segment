# Standard libraries
import os
import time
import random
import ssl
from typing import Optional, Sequence

# Image handling and visualization
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# TorchVision
from torchvision import datasets, transforms as T
from torchvision.datasets import CIFAR10

# Albumentations for augmentation
import albumentations as A

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Segmentation Models
import segmentation_models_pytorch as smp


torch.set_float32_matmul_precision('medium')
ssl._create_default_https_context = ssl._create_unverified_context


# --- Configuration ---
IMAGE_SIZE = 224

# --- Dataset ---
class AugDataset(Dataset):
    class ToUint8(A.ImageOnlyTransform):
        def __init__(self, max_value=255.0, p=1.0):
            super().__init__(p=p)
            self.max_value = max_value

        def apply(self, img, **params):
            img = img * self.max_value
            img = img.clip(0, 255).astype(np.uint8)
            return img
        
    def __init__(self, train, mixup, cutmix):
        self.train=train
        self.mixup = mixup
        self.cutmix = cutmix
        self.aug = A.Compose([
            A.Affine(
                rotate=(-25, 25),             # Arbitrary rotation range
                translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},  # Up to ±20% shift
                scale=(0.8, 1.2),             # Slight zoom in/out
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=(0.3, 0.3), p=0.7),

            # A.CoarseDropout(
            #     num_holes_range=(1, 10),
            #     hole_height_range=(IMAGE_SIZE//10, IMAGE_SIZE//3),
            #     hole_width_range=(IMAGE_SIZE//10, IMAGE_SIZE//3),
            #     fill=0,
            #     p=0.5
            # ),

            A.CoarseDropout(
                num_holes_range=(100, 200),
                hole_height_range=(IMAGE_SIZE//100, IMAGE_SIZE//50),
                hole_width_range=(IMAGE_SIZE//100, IMAGE_SIZE//50),
                fill=0,
                p=0.9
            ),

            A.GaussianBlur(p=0.5),
            # A.RandomBrightnessContrast(
            #     brightness_limit=(-0.05, 0.05),
            #     contrast_limit=(-0.05, 0.05),
            #     brightness_by_max=True,
            #     ensure_safe_range=True,
            #     p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.5),

            # A.Normalize(mean=0.0, std=1.0),
            # AugDataset.ToUint8(),
        ])

        self.no_aug = A.Compose([
            # A.Normalize(mean=0.0, std=1.0),
            # AugDataset.ToUint8(),
        ])

    def _build_sample(self, idx):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        def augment(input_tensor, mask_tensor):
            augmented = self.aug(image=input_tensor, mask=mask_tensor)
            input_tensor = augmented['image']           
            mask_tensor = augmented['mask']
            return input_tensor, mask_tensor

        if self.train:        
            input_tensor, mask_tensor = augment(*self._build_sample(idx))
            H, W = input_tensor.shape[:2]
            M = min(H,W)

            # # Apply Mixup
            # if self.mixup and random.random() < 0.5:
            #     idx2 = random.randint(0, len(self) - 1)
            #     input2, mask2 = augment(*self._build_sample(idx2))

            #     lam = np.random.beta(0.4, 0.4)
            #     input_tensor = lam * input_tensor + (1 - lam) * input2
            #     mask_tensor = lam * mask_tensor + (1 - lam) * mask2

            # # Apply CutMix
            # if self.cutmix and random.random() < 0.5:
            #     idx2 = random.randint(0, len(self) - 1)
            #     input2, mask2 = augment(*self._build_sample(idx2))

            #     cut_h, cut_w = random.randint(M//10, M//5), random.randint(M//10, M//5)
            #     cy, cx = random.randint(0, H - cut_h), random.randint(0, W - cut_w)

            #     input_tensor[cy:cy+cut_h, cx:cx+cut_w] = input2[cy:cy+cut_h, cx:cx+cut_w]
            #     mask_tensor[cy:cy+cut_h, cx:cx+cut_w] = mask2[cy:cy+cut_h, cx:cx+cut_w]

        else:
            input_tensor, mask_tensor=self._build_sample(idx)
            augmented = self.no_aug(image=input_tensor, mask=mask_tensor)
            input_tensor = augmented['image']           
            mask_tensor = augmented['mask']
            
        input_tensor, mask_tensor = input_tensor.astype(np.uint8), mask_tensor.astype(np.uint8)
        input_tensor = torch.tensor(input_tensor).float().div(255.0).unsqueeze(0)
        mask_tensor = torch.tensor(mask_tensor).float().div(255.0).unsqueeze(0)
        return input_tensor, mask_tensor
    
class MNISTOnCIFARSegmentation(AugDataset):
    def __init__(self, train=True, mixup=True, cutmix=True):
        super().__init__(train, mixup, cutmix)
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.cifar = CIFAR10(root='./data', train=train, download=True)
        self.train = train

    def __len__(self):
        return len(self.mnist)
    
    def _build_sample(self, idx):
        digit_img, _ = self.mnist[idx]

        # Create mask
        bg_idx = random.randint(0, len(self.cifar) - 1)
        bg_img, _ = self.cifar[bg_idx]
        bg_img:Image = bg_img.convert('L')
        bg_img = bg_img.resize((IMAGE_SIZE,IMAGE_SIZE))
        digit_img = digit_img.resize((IMAGE_SIZE,IMAGE_SIZE))

        # Convert to float32 numpy
        digit_np = np.array(digit_img, dtype=np.float32) / 255.0
        bg_np = np.array(bg_img, dtype=np.float32) / 255.0
        mask_np = digit_np > 0.5

        combined = bg_np * (1 - mask_np) + digit_np * mask_np
        combined = (combined * 255).astype(np.uint8)

        return combined, (mask_np * 255).astype(np.uint8)
    
    # --- Dataloaders ---
    @staticmethod
    def get_dataloaders(batch_size=32):
        train_dataset = MNISTOnCIFARSegmentation(train=True)
        val_dataset = MNISTOnCIFARSegmentation(train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
  
class TwoImagesSegmentationDataset(AugDataset):
    def __init__(
        self,
        image_dir: str,
        input_prename: str,
        mask_prename: str,
        train=False,
        exts: Sequence[str] = ('.png', '.jpg', '.jpeg', '.tif'),
        image_size: int = IMAGE_SIZE,
        mixup=True, cutmix=True, in_memory=True):

        super().__init__(train, mixup, cutmix)
        self.image_dir = image_dir
        self.input_prename = input_prename
        self.mask_prename = mask_prename
        self.exts = tuple(e.lower() for e in exts)
        self.image_size = image_size

        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(self.exts)]

        # Build maps from "key" -> filename
        inputnames = {f.replace(input_prename, ''): f
                      for f in all_files if input_prename in f}
        masknames  = {f.replace(mask_prename, ''): f
                      for f in all_files if mask_prename in f}

        # Common keys only
        self.keys = sorted(set((inputnames.keys())) & set(masknames.keys()))
        if len(self.keys) == 0:
            raise RuntimeError("No matching (input, mask) pairs found.")

        # Store the resolved pairs
        self.inputnames = {k: inputnames[k] for k in self.keys}
        self.masknames  = {k: masknames[k] for k in self.keys}
        self.in_memory = in_memory
        if in_memory:
            self.input_imgs = [Image.open(os.path.join(self.image_dir, 
                                    self.inputnames[k])
                                    ).convert('L').resize((IMAGE_SIZE,IMAGE_SIZE)) for k in self.keys]
            self.mask_imgs  = [Image.open(os.path.join(self.image_dir, 
                                    self.masknames[k])
                                    ).convert('L').resize((IMAGE_SIZE,IMAGE_SIZE)) for k in self.keys]
            self.input_imgs = [np.asarray(i).copy().astype(np.uint8) for i in self.input_imgs]
            self.mask_imgs = [np.asarray(i).copy().astype(np.uint8) for i in self.mask_imgs]

    def __len__(self):
        return len(self.keys)

    def _build_sample(self, idx):
        k = self.keys[idx]
        # Load PIL
        if self.in_memory:
            input_img = self.input_imgs[idx]
            mask_img  = self.mask_imgs[idx]
        else:
            input_img = Image.open(os.path.join(self.image_dir, self.inputnames[k])).convert('L')
            mask_img  = Image.open(os.path.join(self.image_dir, self.masknames[k])).convert('L')
            input_img = input_img.resize((IMAGE_SIZE,IMAGE_SIZE))
            mask_img = mask_img.resize((IMAGE_SIZE,IMAGE_SIZE))
            input_img = np.asarray(input_img).copy().astype(np.uint8)
            mask_img = np.asarray(mask_img).copy().astype(np.uint8)

        # mask_img = ((mask_img-mask_img.min())/(mask_img.max()-mask_img.min())*255.0).astype(np.uint8)
        mask_img  = ((mask_img > 0)*255.0).astype(np.uint8)
        return input_img,mask_img
    
    @staticmethod
    def get_dataloaders(
        train_dir: str,
        val_dir: str,
        input_prename: str,
        mask_prename: str,
        batch_size: int = 32,
        num_workers: int = 1,
        **ds_kwargs
    ):
        train_dataset = TwoImagesSegmentationDataset(
            train_dir, input_prename, mask_prename,  train=True, **ds_kwargs
        )
        val_dataset = TwoImagesSegmentationDataset(
            val_dir, input_prename, mask_prename, train=False, **ds_kwargs
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_loader

class SideBySideSegmentationDataset(AugDataset):
    def __init__(self, image_dir, 
        train=False,
        exts: Sequence[str] = ('.png', '.jpg', '.jpeg', '.tif'),
        image_size: int = IMAGE_SIZE,
        color: str = 'L',
        mixup=False, cutmix=False):
        super().__init__(train, mixup, cutmix)

        self.image_dir = image_dir
        self.image_size = image_size
        self.color = color
        self.exts = tuple(e.lower() for e in exts)        
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(self.exts)]

    def __len__(self):
        return len(self.filenames)
    
    def _build_sample(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        full_img = Image.open(image_path).convert(self.color)
        w,h = full_img.size

        # Crop left and right halves
        input_img = full_img.crop((0, 0, w//2, h))
        mask_img  = full_img.crop((w//2, 0, w, h))

        input_img = input_img.resize((self.image_size,self.image_size))
        mask_img  = mask_img.resize((self.image_size,self.image_size))        
        
        input_img = np.asarray(input_img).copy().astype(np.uint8)
        mask_img  = np.asarray(mask_img).copy().astype(np.uint8)

        # Normalize mask to binary
        # mask_img  = ((mask_img > 0)*255.0).astype(np.uint8)
        return input_img,mask_img

    @staticmethod
    def get_dataloader(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        **ds_kwargs
    ):
        the_dataset = SideBySideSegmentationDataset(
            image_dir=data_dir,  **ds_kwargs
        )
        data_loader = DataLoader(
            the_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        return data_loader
    
class SegmentationModel(pl.LightningModule):

    class BCEDiceLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, logits, targets):
            bce_loss = self.bce(logits, targets)
            preds = torch.sigmoid(logits)
            smooth = 1e-6
            intersection = (preds * targets).sum()
            union = preds.sum() + targets.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            return bce_loss + (1 - dice)
        
    def __init__(self, arch_name='Segformer', encoder_name='efficientnet-b7', lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.__dict__[arch_name](
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=1,
            classes=1,
        )

        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = SegmentationModel.BCEDiceLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.sigmoid(logits)
        preds_bin = (preds > 0.5).float()
        iou = self._iou_score(preds_bin, masks)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_iou", iou, on_epoch=True, prog_bar=True)
        print()
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _iou_score(self, preds, targets, eps=1e-6):
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
        iou = (intersection + eps) / (union + eps)
        return iou.mean()

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, input_prename, mask_prename, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.input_prename = input_prename
        self.mask_prename = mask_prename
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SideBySideSegmentationDataset(
            image_dir=self.train_dir,
            train=True
        )
        self.val_dataset = SideBySideSegmentationDataset(
            image_dir=self.train_dir,
            train=False
        )
        # self.train_dataset = TwoImagesSegmentationDataset(
        #     image_dir=self.train_dir,
        #     input_prename=self.input_prename,
        #     mask_prename=self.mask_prename,
        #     train=True,
        # )
        # self.val_dataset = TwoImagesSegmentationDataset(
        #     image_dir=self.val_dir,
        #     input_prename=self.input_prename,
        #     mask_prename=self.mask_prename,
        #     train=False,
        # )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,persistent_workers=True)
    
    def show_samples(self, split='val', num_samples=5):
        if split == 'train':
            loader = self.train_dataloader()
        elif split == 'val':
            loader = self.val_dataloader()
        else:
            raise ValueError("split must be either 'train' or 'val'")

        batch = next(iter(loader))
        inputs, gts = batch[:2]  # Assuming dataset returns (input, gt, ...)

        plot(inputs[:num_samples], gts[:num_samples])

def plot(inputs, gts, preds=None):
    for i in range(len(inputs)):
        input_img = inputs[i].squeeze().cpu().mul(255.0).numpy().astype(np.uint8)
        gt_mask   = gts[i].squeeze().cpu().mul(255.0).numpy().astype(np.uint8)
        if preds is not None:
            # make sure we pass probabilities or thresholded masks
            pred_mask = preds[i].squeeze().cpu().mul(255.0).numpy().astype(np.uint8)

        cols = 3 if preds is not None else 2
        plt.figure(figsize=(3 * cols, 3))

        plt.subplot(1, cols, 1); plt.imshow(input_img, cmap='gray'); plt.title('Input'); plt.axis('off')
        plt.subplot(1, cols, 2); plt.imshow(gt_mask,   cmap='gray'); plt.title('GT');    plt.axis('off')
        if preds is not None:
            plt.subplot(1, cols, 3); plt.imshow(pred_mask, cmap='gray'); plt.title('Pred'); plt.axis('off')
        plt.tight_layout(); plt.show()

def show_samples(data):
    data.setup()
    data.show_samples('train',10)

def train(data,max_epochs=100):
    arch_name='Segformer'
    encoder_name="efficientnet-b7"

    model = SegmentationModel(arch_name=arch_name,encoder_name=encoder_name, lr=1e-4)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=CSVLogger(save_dir=f"{arch_name}-{encoder_name}-logs/"),
        callbacks=[ModelCheckpoint(monitor="val_iou", mode="max")]
    )

    trainer.fit(model, datamodule=data)

def infer(ckpt): 
    batch_size=32

    real_loader = SideBySideSegmentationDataset.get_dataloader('./tmp/real',batch_size=batch_size)
    model = SegmentationModel.load_from_checkpoint(ckpt,map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        test_inputs, test_masks = next(iter(real_loader))
        test_inputs = test_inputs.to(model.device)
        pred_logits = model(test_inputs)
        pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
        plot(test_inputs, test_masks, pred_masks)

if __name__ == "__main__":
    batch_size=32
    data = SegmentationDataModule(
        train_dir='./tmp/sim2',
        val_dir='./tmp/sim2',
        input_prename='viewport_2',
        mask_prename='viewport_1',
        batch_size=batch_size,
    )
    train(data,1)
    # show_samples(data)
    # infer('./logs/lightning_logs/version_1/checkpoints/epoch=12-step=52.ckpt')
