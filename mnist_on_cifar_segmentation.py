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

import math, torch, torch.nn as nn, torch.nn.functional as F
from torchmetrics import JaccardIndex
from huggingface_hub import hf_hub_download

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
        self.aug_lite = A.Compose([
            A.Affine(
                rotate=(-20, 20),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                scale=(0.9, 1.1),
                shear=(-8, 8),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.8
            ),

            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            ], p=0.6),

            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5),
                                       contrast_limit=(-0.5, 0.5), p=0.6),

            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),

            A.CoarseDropout(max_holes=4, max_height=0.08, max_width=0.08,
                            min_holes=1, min_height=0.03, min_width=0.03,
                            fill_value=0, p=0.35),

            # Rare salt-or-pepper
            A.OneOf([
                A.PixelDropout(dropout_prob=0.02, drop_value=0, p=1.0),
                A.PixelDropout(dropout_prob=0.02, drop_value=255, p=1.0),
            ], p=0.15),
        ])

        self.aug_strong = A.Compose([
            A.Affine(
                rotate=(-25, 25),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                scale=(0.85, 1.15),
                shear=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.9
            ),

            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.ElasticTransform(alpha=1.2, sigma=30, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.07, p=1.0),
                A.OpticalDistortion(distort_limit=0.12, shift_limit=0.05, p=1.0),
            ], p=0.6),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
            ], p=0.7),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.CLAHE(clip_limit=(2, 4), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.6),

            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=15, p=0.5),

            A.CoarseDropout(max_holes=6, max_height=0.1, max_width=0.1,
                            min_holes=1, min_height=0.04, min_width=0.04,
                            fill_value=0, p=0.5),

            A.ChannelShuffle(p=0.05),
            A.ToGray(p=0.05),
        ])

        self.aug_very_strong = A.Compose([
            A.Affine(
                rotate=(-30, 30),
                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                scale=(0.8, 1.2),
                shear=(-12, 12),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.9
            ),

            A.OneOf([
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.75),

            
            A.ElasticTransform(alpha=1.4, sigma=32, interpolation=cv2.INTER_LINEAR, 
               p=0.7),
            A.GridDistortion(num_steps=7, distort_limit=0.1, 
               p=0.7),
            A.OpticalDistortion(distort_limit=0.15, shift_limit=0.07, 
               p=0.7),
            A.Perspective(scale=(0.04, 0.08), 
               p=0.7),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=11, p=1.0),
                A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0),
                A.GaussNoise(var_limit=(15.0, 60.0), p=1.0),
            ], p=0.75),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5),
                                        contrast_limit=(-0.5, 0.5), p=1.0),
                A.CLAHE(clip_limit=(2, 4), p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=0.7),

            A.CoarseDropout(max_holes=10, max_height=0.15, max_width=0.15,
                            min_holes=2, min_height=0.05, min_width=0.05,
                            fill_value=0, 
               p=0.7),

            A.OneOf([
                A.PixelDropout(dropout_prob=0.03, drop_value=0, p=1.0),
                A.PixelDropout(dropout_prob=0.03, drop_value=255, p=1.0),
            ], p=0.75),
        ])

        self.aug = self.aug_very_strong

        self.no_aug = A.Compose([
            A.PixelDropout(dropout_prob=0.025, per_channel=False, drop_value=0, p=1.0),
            A.PixelDropout(dropout_prob=0.025, per_channel=False, drop_value=255, p=1.0),
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
            # if self.mixup and random.random() < 0.2:
            #     idx2 = random.randint(0, len(self) - 1)
            #     input2, mask2 = augment(*self._build_sample(idx2))

            #     lam = 0.5#np.random.beta(0.4, 0.4)
            #     input_tensor = lam * input_tensor + (1 - lam) * input2
            #     mask_tensor *= 0.0
            #     # mask_tensor = lam * mask_tensor + (1 - lam) * mask2

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

        if len(input_tensor.shape)==3:
            input_tensor = torch.from_numpy(input_tensor).float().div(255.0).permute(2, 0, 1)
        elif len(input_tensor.shape)==2:
            input_tensor = torch.from_numpy(input_tensor).float().div(255.0).unsqueeze(0)
            
        if len(mask_tensor.shape)==3:
            mask_tensor = torch.from_numpy(mask_tensor).float().div(255.0).permute(2, 0, 1)
        elif len(mask_tensor.shape)==2:
            mask_tensor = torch.from_numpy(mask_tensor).float().div(255.0).unsqueeze(0)

        return input_tensor, mask_tensor
    
class MNISTOnCIFARSegmentationDataset(AugDataset):
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
        train_dataset = MNISTOnCIFARSegmentationDataset(train=True)
        val_dataset = MNISTOnCIFARSegmentationDataset(train=False)

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
        mixup=False, cutmix=False, data_step=1):
        super().__init__(train, mixup, cutmix)

        self.image_dir = image_dir
        self.image_size = image_size
        self.color = color
        self.exts = tuple(e.lower() for e in exts)        
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(self.exts)]
        self.filenames = [self.filenames[i] for i in range(0,len(self.filenames),data_step)]

    def __len__(self):
        return len(self.filenames)
    
    def _build_sample(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        full_img = Image.open(image_path)
        w,h = full_img.size

        # Crop left and right halves
        input_img = full_img.crop((0, 0, w//2, h)).convert(self.color)
        mask_img  = full_img.crop((w//2, 0, w, h)).convert('L')

        input_img = input_img.resize((self.image_size,self.image_size))
        mask_img  = mask_img.resize((self.image_size,self.image_size))        
        
        input_img = np.asarray(input_img).copy().astype(np.uint8)
        mask_img  = np.asarray(mask_img).copy().astype(np.uint8)

        # Normalize mask to binary
        mask_img  = ((mask_img > 0)*255.0).astype(np.uint8)
        # Remove small noise using morphological opening
        kernel = np.ones((3,3), np.uint8)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)        
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

    class DINOv3Seg(nn.Module):

        def __init__(self, variant="vitb16", hub_repo_dir:str=None, hub_weights:str=None,
                    in_channels:int=1, out_channels:int=1,
                    freeze_backbone: bool = True,
                    adapter_mode = "repeat"):
            super().__init__()
            assert hub_repo_dir and hub_weights, "Must provide hub_repo_dir and hub_weights."
            builder = f"dinov3_{variant}"  # e.g., dinov3_vitb16
            # --- your exact local hub load ---
            self.backbone = torch.hub.load(hub_repo_dir, builder, source='local', weights=hub_weights)

            if in_channels == 3:
                self.input_adapter = nn.Identity()
            else:
                if adapter_mode == "repeat":
                    class RepeatGray3(nn.Module):
                        def forward(self, x):        # x: [B,1,H,W] float in [0,1] or [0,255]
                            if x.shape[1] == 1:
                                return x.repeat(1, 3, 1, 1)
                            return x
                    self.input_adapter = RepeatGray3()
                elif adapter_mode == "fixed":
                    def fixed_gray_to_rgb_conv():
                        conv = nn.Conv2d(1, 3, kernel_size=1, bias=True)
                        with torch.no_grad():
                            conv.weight.fill_(1.0)   # each RGB = gray * 1
                            conv.bias.zero_()
                        for p in conv.parameters():
                            p.requires_grad = False
                        return conv
                    self.input_adapter = fixed_gray_to_rgb_conv()
                elif adapter_mode == "conv":  # trainable
                    self.input_adapter = nn.Conv2d(in_channels, 3, 1)
                else:
                    raise ValueError(f"Unknown adapter_mode: {adapter_mode}")

            self.embed_dim = 768 if "vitb16" in variant.lower() else (1024 if "vitl16" in variant.lower() else 1280)
            hidden= 256 # 256 384 512
            self.proj = nn.Conv2d(self.embed_dim, hidden, 1)
            
            # --- depthwise-separable conv block in nn.Sequential ---
            def sepconv_block(ch: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),  # depthwise
                    nn.Conv2d(ch, ch, kernel_size=1, bias=False),                        # pointwise
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            
            self.dec = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                
                sepconv_block(hidden),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                
                sepconv_block(hidden),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                
                sepconv_block(hidden),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                
                sepconv_block(hidden),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

                nn.Conv2d(hidden, out_channels, 1),
            )

            # freeze control
            self.is_backbone_frozen = False
            self._auto_unfreeze_epoch = None
            if freeze_backbone:
                self.freeze_backbone()

        # ---------- freeze/unfreeze API (internal to the class) ----------
        def freeze_backbone(self):
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.is_backbone_frozen = True
            # keep decoder/head trainable
            self.backbone.eval()
            return self

        def unfreeze_backbone(self):
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.is_backbone_frozen = False
            if self.training:
                self.backbone.train()
            return self

        def schedule_unfreeze(self, epoch:int):
            """Set an epoch at which you'll call maybe_auto_unfreeze(current_epoch)."""
            self._auto_unfreeze_epoch = int(epoch)

        def maybe_auto_unfreeze(self, current_epoch:int):
            if self._auto_unfreeze_epoch is not None and current_epoch >= self._auto_unfreeze_epoch:
                if self.is_backbone_frozen:
                    self.unfreeze_backbone()

        # keep backbone in eval() whenever it's frozen (even if module is set to train)
        def train(self, mode: bool = True):
            super().train(mode)
            if self.is_backbone_frozen:
                self.backbone.eval()
            return self

        # handy param groups for 2-LR optimizers
        def param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float = 0.0):
            head = list(self.input_adapter.parameters()) + list(self.proj.parameters()) + list(self.dec.parameters())
            bb   = [p for p in self.backbone.parameters() if p.requires_grad]
            groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
            if bb:
                groups.append({"params": bb, "lr": lr_backbone, "weight_decay": weight_decay})
            return groups

        # ---------- forward utils ----------
        @staticmethod
        def _pad_to_multiple(x, multiple=16):
            H, W = x.shape[-2:]
            ph = (multiple - H % multiple) % multiple
            pw = (multiple - W % multiple) % multiple
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph))
            return x, (ph, pw)

        @staticmethod
        def _unpad(x, pad_hw):
            ph, pw = pad_hw
            return x[..., : x.shape[-2] - ph if ph else x.shape[-2], : x.shape[-1] - pw if pw else x.shape[-1]]

        def _tokens_to_grid(self, toks, H, W, patch=16):
            if toks.dim() == 3 and toks.size(1) == (H // patch) * (W // patch) + 1:
                toks = toks[:, 1:, :]
            B, N, C = toks.shape
            h, w = H // patch, W // patch
            return toks.transpose(1, 2).reshape(B, C, h, w)

        def _forward_tokens(self, x):
            out = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
            if isinstance(out, dict):
                for k in ("x", "x_norm_patchtokens", "x_prenorm", "tokens"):
                    if k in out:
                        return out[k]
                raise RuntimeError("Backbone dict lacks token tensor.")
            if torch.is_tensor(out) and out.dim() == 3:
                return out
            raise RuntimeError("Unexpected backbone output (expected [B,N,C] tokens or dict).")

        def forward(self, x):
            x = self.input_adapter(x)
            x, pad_hw = self._pad_to_multiple(x, 16)
            H, W = x.shape[-2:]
            toks = self._forward_tokens(x)                   # [B, N(+1), C]
            grid = self._tokens_to_grid(toks, H, W, 16)      # [B, C, H/16, W/16]
            logits = self.dec(self.proj(grid))            # [B, out_ch, H/16, W/16]
            oB,oC,oH,oW = logits.shape
            if oH!=H and W!=oW:
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            return self._unpad(logits, pad_hw)
        
    class BCEDiceLoss(nn.Module):
        def __init__(self, bce_weight=0.5, dice_weight=0.5):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight

        def forward(self, logits, targets, smooth = 1e-6):
            bce_loss = self.bce(logits, targets)
            preds = torch.clamp(torch.sigmoid(logits), min=1e-7, max=1 - 1e-7)
            intersection = (preds * targets).sum(dim=(1,2,3))
            union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_loss = 1 - dice
            return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()
        
    class FocalDiceLoss(nn.Module):
        def __init__(self, alpha=0.8, gamma=2.0, dice_weight=0.5):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.dice_weight = dice_weight

        def focal_loss(self, logits, targets):
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
            pt = probs * targets + (1 - probs) * (1 - targets)
            w = self.alpha * (1 - pt).pow(self.gamma)
            return -(w * pt.log()).mean()

        def dice_loss(self, probs, targets, smooth=1e-6):
            dims = tuple(range(1, probs.dim()))
            intersection = (probs * targets).sum(dim=dims)
            union = probs.sum(dim=dims) + targets.sum(dim=dims)
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - dice.mean()

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            fl = self.focal_loss(logits, targets)
            dl = self.dice_loss(probs, targets)
            return (1 - self.dice_weight) * fl + self.dice_weight * dl        

    class TverskyDiceLoss(nn.Module):
        def __init__(self, alpha=0.3, beta=0.7, tversky_weight=0.5, smooth=1e-6):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.tversky_weight = tversky_weight
            self.smooth = smooth

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            dims = tuple(range(1, logits.dim()))
            tp = (probs * targets).sum(dim=dims)
            fp = ((1 - targets) * probs).sum(dim=dims)
            fn = (targets * (1 - probs)).sum(dim=dims)
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_loss = 1 - tversky.mean()

            intersection = (probs * targets).sum(dim=dims)
            union = probs.sum(dim=dims) + targets.sum(dim=dims)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice.mean()

            return self.tversky_weight * tversky_loss + (1 - self.tversky_weight) * dice_loss

    def __init__(self, arch_name='DeepLabV3Plus', encoder_name='efficientnet-b7',
                 encoder_weights='imagenet', in_channels=1, lr=1e-4):
        super().__init__()
        self.loss_fn = SegmentationModel.BCEDiceLoss()
        self.arch_name = arch_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.lr = lr
        self.in_channels = in_channels

        self.save_hyperparameters()


        if "dinov3" in str(arch_name).lower():
            # ---- DINOv3 branch (LOCAL hub) ----
            self.model = SegmentationModel.DINOv3Seg(
                variant=encoder_name,
                hub_repo_dir=arch_name,
                hub_weights=encoder_weights,
                in_channels=in_channels,
            )
            # Use ImageNet normalization by default (DINO pretraining uses RGB stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            if in_channels == 1:
                mean = mean.mean(1, keepdim=True)
                std = std.mean(1, keepdim=True)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        else:
            # --- your original SMP path ---
            self.model = smp.create_model(
                self.arch_name,
                self.encoder_name,
                self.encoder_weights,
                in_channels=in_channels,
                classes=1,
            )
            # preprocessing parameters for SMP encoders
            params = smp.encoders.get_preprocessing_params(self.encoder_name)
            std = torch.tensor(params["std"]).view(1, 3, 1, 1)
            mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)
            if in_channels == 1:
                std = std.mean(1, keepdim=True)
                mean = mean.mean(1, keepdim=True)
            self.register_buffer("std", std)
            self.register_buffer("mean", mean)

    def forward(self, x: torch.Tensor):
        # normalize image here (works for both SMP and DINOv3 branches)
        x = (x - self.mean) / self.std
        return self.model(x)

    # ---- training / eval loop stays identical ----
    def _shared_step(self, batch, stage):
        images, masks = batch                       # masks shape: [B,1,H,W] with {0,1}
        logits = self(images)                      # logits: [B,1,H,W]
        loss = self.loss_fn(logits, masks)
        preds_bin = (torch.sigmoid(logits) > 0.5).float()
        iou = self._iou_score(preds_bin, masks)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_iou", iou, on_epoch=True, prog_bar=True)
        print()
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)

    def _iou_score(self, preds, targets, eps=1e-6):
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
        iou = (intersection + eps) / (union + eps)
        return iou.mean()

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, input_prename=None, mask_prename=None,
                 batch_size=32, num_workers=4,
                 data_step=1):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.input_prename = input_prename
        self.mask_prename = mask_prename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_step = data_step

    def setup(self, stage=None):
        self.train_dataset = SideBySideSegmentationDataset(
            image_dir=self.train_dir,
            train=True,
            data_step = self.data_step,
        )
        self.val_dataset = SideBySideSegmentationDataset(
            image_dir=self.train_dir,
            train=False,
            data_step = self.data_step,
        )
        return self
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
        # transform = transforms.Compose(transforms.ToTensor()])
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

        inputs, gts = next(iter(loader))

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

def show_samples(data:SegmentationDataModule, num_samples=10, what='train'):
    data.setup()
    data.show_samples(what,num_samples)

def train(data,encoder_name,encoder_weights,
          max_epochs=100,in_channels=1,lr=1e-4, arch_name='DeepLabV3Plus'):

    model = SegmentationModel(arch_name,encoder_name,encoder_weights,in_channels,lr)
    if os.path.exists(arch_name):
        arch_name = os.path.basename(arch_name)
    get_trainer = lambda max_epochs:Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        logger=CSVLogger(save_dir=f"{arch_name}-{encoder_name}-logs/"),
        callbacks=[ModelCheckpoint(monitor="val_iou", mode="max")]
    )
    
    if isinstance(data,list):
        t = get_trainer(max_epochs)
        for d in data:
            t.fit(model, datamodule=d)
            t.fit_loop.max_epochs+=max_epochs
    else:
        get_trainer(max_epochs).fit(model, datamodule=data)

def infer(ckpt,data=None): 
    batch_size=32
    if data is None:
        data = real_loader = SideBySideSegmentationDataset.get_dataloader('./tmp/real',batch_size=batch_size)

    model = SegmentationModel.load_from_checkpoint(ckpt,map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        test_inputs, test_masks = next(iter(data))
        test_inputs = test_inputs.to(model.device)
        pred_logits = model(test_inputs)
        pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
        plot(test_inputs, test_masks, pred_masks)

if __name__ == "__main__":
    # NO_ALBUMENTATIONS_UPDATE=1
    # CUDA_VISIBLE_DEVICES=0
    get_data = lambda batch_size,dir='./tmp/sim2':SegmentationDataModule(
        train_dir=dir,val_dir=dir,batch_size=batch_size,)
    # show_samples(get_data(32),30,'train')
    # show_samples(get_data(32),30,'val')
    
    arch_name,encoder_name,encoder_weights='DeepLabV3Plus',"timm-efficientnet-b8","imagenet"
    arch_name,encoder_name,encoder_weights='../dinov3',"vitb16","./tmp/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    # train(get_data(128,'./tmp/sim2'),encoder_name,encoder_weights,arch_name=arch_name,
    #       max_epochs=1000,lr=0.001)
    # encoder_name,encoder_weights="timm-efficientnet-b8","imagenet"
    # train(get_data(16),encoder_name,encoder_weights,500)

    # encoder_name,encoder_weights="timm-efficientnet-l2","noisy-student"
    # train(get_data(16),encoder_name,encoder_weights,100)
    # encoder_name,encoder_weights="timm-efficientnet-l2","noisy-student"
    # train(get_data(16),encoder_name,encoder_weights,500)
    # infer('Segformer-timm-efficientnet-b8-logs/lightning_logs/version_best_C=1/checkpoints/epoch=183-step=58144.ckpt')
    infer('dinov3-vitb16-logs/lightning_logs/version_5/checkpoints/epoch=3-step=200.ckpt')

    