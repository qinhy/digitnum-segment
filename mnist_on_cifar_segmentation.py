from typing import Optional, Sequence
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# --- Configuration ---
IMAGE_SIZE = 32

# --- Dataset ---
class AugDataset(Dataset):
    def __init__(self, train, mixup, cutmix):
        self.train=train
        self.mixup = mixup
        self.cutmix = cutmix
        self.aug = A.Compose([
            # A.CoarseDropout(
            #     num_holes_range=(1, 3),
            #     hole_height_range=(8, 16),
            #     hole_width_range=(8, 16),
            #     fill=0,
            #     fill_mask=0,
            #     p=0.5
            # ),
            # A.Affine(
            #     rotate=(-25, 25),             # Arbitrary rotation range
            #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Up to ±20% shift
            #     scale=(0.9, 1.1),             # Slight zoom in/out
            #     shear=(-10, 10),              # Optional: slight shearing
            #     p=0.7
            # ),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.5),
            # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            # A.GaussianBlur(p=0.2),
            # A.RandomBrightnessContrast(p=0.2),

            # A.GaussNoise(p=0.2),
            # A.Normalize(mean=0.0, std=1.0),
        ])

        self.no_aug = A.Compose([
            # A.Normalize(mean=0.0, std=1.0),
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

            # Apply Mixup
            # if self.mixup and random.random() < 0.5:
            #     idx2 = random.randint(0, len(self) - 1)
            #     input2, mask2 = augment(*self._build_sample(idx2))

            #     lam = np.random.beta(0.4, 0.4)
            #     input_tensor = lam * input_tensor + (1 - lam) * input2
            #     mask_tensor = lam * mask_tensor + (1 - lam) * mask2

            # # Apply CutMix
            # elif self.cutmix and random.random() < 0.5:
            #     idx2 = random.randint(0, len(self) - 1)
            #     input2, mask2 = augment(*self._build_sample(idx2))

            #     cut_h, cut_w = random.randint(8, 16), random.randint(8, 16)
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

        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        self.to_tensor = transforms.ToTensor()

        self.train = train

    def __len__(self):
        return len(self.mnist)
    
    def _build_sample(self, idx):
        digit_img, _ = self.mnist[idx]
        digit_img = self.resize(digit_img)

        # Create mask
        bg_idx = random.randint(0, len(self.cifar) - 1)
        bg_img, _ = self.cifar[bg_idx]
        bg_img = self.grayscale(self.resize(bg_img))

        # Convert to float32 numpy
        digit_np = np.array(digit_img, dtype=np.float32) / 255.0
        bg_np = np.array(bg_img, dtype=np.float32) / 255.0
        mask_np = digit_np > 0.0

        combined = bg_np * (1 - mask_np) + digit_np * mask_np
        combined = (combined * 255).astype(np.uint8)

        return combined, digit_np
    
    # --- Dataloaders ---
    @staticmethod
    def get_dataloaders(batch_size=32):
        train_dataset = MNISTOnCIFARSegmentation(train=True)
        val_dataset = MNISTOnCIFARSegmentation(train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

class SimpleUNet(nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self):
        super().__init__()
        self.enc1 = SimpleUNet.DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = SimpleUNet.DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = SimpleUNet.DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = SimpleUNet.DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = SimpleUNet.DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)  # raw logits
    
    def infer(self, imgs, conf: float | None = 0.5):
        self.eval()
        with torch.no_grad():
            x = torch.stack(list(imgs)).to(next(self.parameters()).device)
            probs = torch.sigmoid(self(x))
            if conf is None:
                return probs.cpu()
            return (probs > conf).float().cpu()
    
    def save_model(self, path="mnist_on_cifar_unet.pth"):
        torch.save(self.state_dict(), path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="mnist_on_cifar_unet.pth", device='cuda'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
        print(f"📥 Model loaded from {path}")
        return self

    def train_data(self, train_loader, val_loader=None, epochs=5, device='cuda',
                   criterion = nn.BCELoss()):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                if isinstance(criterion,nn.BCELoss):
                    masks = (masks>0.5).float()

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

            if val_loader:
                self.validate_data(val_loader, device, criterion=criterion)
    
    def validate_data(self, val_loader, device, criterion):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                if isinstance(criterion,nn.BCELoss):
                    masks = (masks>0.5).float()
                outputs = self(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
            probs = torch.sigmoid(outputs)
            bin_preds = (probs > 0.3).float()
            batch_iou = ((bin_preds * masks).sum((1,2,3)) /
                        (bin_preds + masks - bin_preds * masks + 1e-6).sum((1,2,3))).mean()
            
        avg_val_loss = total_loss / len(val_loader)
        print(f"  🔍 Validation Loss: {avg_val_loss:.4f}, batch_iou: {batch_iou:.4f}")

def plot(inputs, gts, preds=None):
    for i in range(len(inputs)):
        input_img = inputs[i].squeeze().cpu().numpy()
        gt_mask   = gts[i].squeeze().cpu().numpy()
        if preds is not None:
            # make sure we pass probabilities or thresholded masks
            pred_mask = preds[i].squeeze().cpu().numpy()

        cols = 3 if preds is not None else 2
        plt.figure(figsize=(3 * cols, 3))

        plt.subplot(1, cols, 1); plt.imshow(input_img, cmap='gray'); plt.title('Input'); plt.axis('off')
        plt.subplot(1, cols, 2); plt.imshow(gt_mask,   cmap='gray'); plt.title('GT');    plt.axis('off')
        if preds is not None:
            plt.subplot(1, cols, 3); plt.imshow(pred_mask, cmap='gray'); plt.title('Pred'); plt.axis('off')
        plt.tight_layout(); plt.show()
        
class SideBySideSegmentationDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

        self.to_tensor = transforms.ToTensor()
        self.resize_input = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.resize_mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        full_img = Image.open(image_path).convert('RGB')
        w,h = full_img.size()

        # Crop left and right halves
        input_img = full_img.crop((0, 0, h, h))
        mask_img  = full_img.crop((h, 0, w//2, h))
        
        input_img = np.asarray(input_img)[:,:,1]
        mask_img  = np.asarray(mask_img)[:,:,1]

        # Resize to match training shape (32x32)
        input_img = self.resize_input(input_img)
        mask_img = self.resize_mask(mask_img)

        # Convert to tensor
        input_tensor = self.to_tensor(input_img)
        mask_tensor  = self.to_tensor(mask_img)

        # Normalize mask to binary
        mask_tensor = (mask_tensor > 0).float()

        return input_tensor, mask_tensor

class TwoImagesSegmentationDataset(AugDataset):
    def __init__(
        self,
        image_dir: str,
        input_prename: str,
        mask_prename: str,
        train=True,
        exts: Sequence[str] = ('.png', '.jpg', '.jpeg', '.tif'),
        image_size: int = IMAGE_SIZE,
        mixup=True, cutmix=True):

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
        self.keys = sorted(set(inputnames.keys()) & set(masknames.keys()))
        if len(self.keys) == 0:
            raise RuntimeError("No matching (input, mask) pairs found.")

        # Store the resolved pairs
        self.inputnames = {k: inputnames[k] for k in self.keys}
        self.masknames  = {k: masknames[k] for k in self.keys}

    def __len__(self):
        return len(self.keys)

    def _build_sample(self, idx):
        k = self.keys[idx]
        # Load PIL
        input_img = Image.open(os.path.join(self.image_dir, self.inputnames[k])).convert('L')
        mask_img  = Image.open(os.path.join(self.image_dir, self.masknames[k])).convert('L')
        input_img = input_img.resize((IMAGE_SIZE,IMAGE_SIZE))
        mask_img = mask_img.resize((IMAGE_SIZE,IMAGE_SIZE))
        input_img = np.asarray(input_img).copy()
        mask_img = np.asarray(mask_img).copy()
        input_img = input_img.astype(np.uint8)
        mask_img = mask_img.astype(np.uint8)
        mask_img  = (mask_img > 0).astype(np.uint8)  # <— binarize here
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
    
def dice_loss(pred_probs, target, smooth=1e-6):
    pred = pred_probs.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def combo_loss(logits, target, bce_logits = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0], device='cuda') )):
    probs = torch.sigmoid(logits)
    return dice_loss(probs, target) + bce_logits(logits, target)

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities if needed
    pred = (pred > threshold).float()  # Binarize predictions

    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

if __name__ == "__main__":
    train_loader, val_loader = MNISTOnCIFARSegmentation.get_dataloaders(batch_size=1024)

    # train_loader, val_loader = TwoImagesSegmentationDataset.get_dataloaders(
    #     val_dir='./tmp/sim',train_dir='./tmp/sim',input_prename='viewport_2 ',mask_prename='viewport_1 ',
    #     batch_size=32)
        
    # images, masks = next(iter(train_loader))
    # plot(images[:5], masks[:5])
    # images, masks = next(iter(val_loader))
    # plot(images[:5], masks[:5])

    model = SimpleUNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    model.train_data(train_loader, val_loader, epochs=10, device=device, #)
                     criterion=combo_loss)
                    #  criterion=nn.BCEWithLogitsLoss())
                    #  criterion=nn.MSELoss())
                    #  criterion=nn.L1Loss())
                    #  criterion=nn.SmoothL1Loss())
    model.save_model()

    # # Test and visualize
    test_inputs, test_masks = next(iter(val_loader))
    preds = model.infer(test_inputs[:5])  # Inference on first 5 samples
    print(preds.min().item(), preds.max().item(), preds.mean().item())
    print(iou_score(preds,test_masks[:5]))
    plot(test_inputs[:5], test_masks[:5], preds)

# --- Example Visualization ---
# if __name__ == '__main__':
#     train_loader, _ = get_dataloaders(batch_size=4)
#     images, masks = next(iter(train_loader))

#     # train_loader, val_loader = get_dataloaders(batch_size=32)
#     # model = SimpleUNet()
#     # train(model, train_loader, val_loader, epochs=10)

#     plot(images, masks)