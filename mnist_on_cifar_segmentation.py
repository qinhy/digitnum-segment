from typing import Optional, Sequence
import cv2
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
                scale=(0.5, 2.0),             # Slight zoom in/out
                shear=(-10, 10),              # Optional: slight shearing
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=(0.3, 0.3), p=0.7),

            A.CoarseDropout(
                num_holes_range=(1, 10),
                hole_height_range=(IMAGE_SIZE//10, IMAGE_SIZE//3),
                hole_width_range=(IMAGE_SIZE//10, IMAGE_SIZE//3),
                fill=0,
                p=0.5
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
            # A.Affine(
            #     rotate=(-25, 25),             # Arbitrary rotation range
            #     translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},  # Up to ±20% shift
            #     scale=(0.5, 2.0),             # Slight zoom in/out
            #     shear=(-10, 10),              # Optional: slight shearing
            #     p=0.7
            # ),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.5),
            # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
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
        self.keys = sorted(set((inputnames.keys())) & set(masknames.keys()))
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
        input_img = np.asarray(input_img).copy().astype(np.uint8)
        mask_img = np.asarray(mask_img).copy().astype(np.uint8)

        # mask_img = ((mask_img-mask_img.min())/(mask_img.max()-mask_img.min())*255.0).astype(np.uint8)
        mask_img  = ((mask_img > 0)*255.0).astype(np.uint8)
        mask_img = cv2.erode(mask_img, (5, 5), iterations=3)
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), sigmaX=0)
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), sigmaX=0)
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
        mixup=False, cutmix=False):
        super().__init__(train, mixup, cutmix)
        self.image_dir = image_dir
        self.exts = tuple(e.lower() for e in exts)        
        self.filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(self.exts)]

    def __len__(self):
        return len(self.filenames)
    
    def _build_sample(self, idx):
        image_path = os.path.join(self.image_dir, self.filenames[idx])
        full_img = Image.open(image_path).convert('RGB')
        w,h = full_img.size

        # Crop left and right halves
        input_img = full_img.crop((0, 0, w//2, h))
        mask_img  = full_img.crop((w//2, 0, w, h))

        input_img = input_img.resize((IMAGE_SIZE,IMAGE_SIZE))
        mask_img  = mask_img.resize((IMAGE_SIZE,IMAGE_SIZE))        
        
        input_img = np.asarray(input_img)[:,:,1].astype(np.uint8)
        mask_img  = np.asarray(mask_img)[:,:,1].astype(np.uint8)

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
    
class SimpleUNet(nn.Module):
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        """
        Args:
            in_channels (int): number of input image channels
            out_channels (int): number of output segmentation channels
            features (list): number of feature maps at each encoder level
        """
        super().__init__()
        assert len(features) >= 2, "At least 2 levels (encoder + bottleneck) are required."

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.enc_blocks.append(SimpleUNet.DoubleConv(prev_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = SimpleUNet.DoubleConv(prev_channels, prev_channels * 2)

        # Decoder (reverse order)
        reversed_features = features[::-1]
        curr_channels = prev_channels * 2
        for feature in reversed_features:
            self.dec_ups.append(nn.ConvTranspose2d(curr_channels, feature, kernel_size=2, stride=2))
            self.dec_blocks.append(SimpleUNet.DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Output layer
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        # Encoder
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec, skip in zip(self.dec_ups, self.dec_blocks, reversed(enc_features)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final_conv(x)  # Raw logits
    
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
                optimizer.zero_grad()
                
                images, masks = images.to(device), masks.to(device)
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
                outputs = self(images)
                loss = criterion(outputs, masks)

                total_loss += loss.item()
                
            probs = torch.sigmoid(outputs)
            bin_preds = (probs >0.5).float()
            batch_iou = ((bin_preds * masks).sum((1,2,3)) /
                        (bin_preds + masks - bin_preds * masks + 1e-6).sum((1,2,3))).mean()
            
        avg_val_loss = total_loss / len(val_loader)
        print(f"  🔍 Validation Loss: {avg_val_loss:.4f}, batch_iou: {batch_iou:.4f}")

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
    encoder_name='timm-efficientnet-b0'
    epochs=500
    batch_size=32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_loader = SideBySideSegmentationDataset.get_dataloader('./tmp/real',batch_size=batch_size)

    train_loader, val_loader = TwoImagesSegmentationDataset.get_dataloaders(
        val_dir='./tmp/sim',train_dir='./tmp/sim',input_prename='viewport_2',mask_prename='viewport_1',
        batch_size=batch_size)
        
    # images, masks = next(iter(real_loader))
    # plot(images[:5], masks[:5])
    images, masks = next(iter(train_loader))
    plot(images[:5], masks[:5])
    # images, masks = next(iter(val_loader))
    # plot(images[:5], masks[:5])
    
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        # encoder_name="resnet34",        # choose from resnet34, efficientnet-b0, etc.
        # encoder_weights="imagenet",     # use ImageNet pre-trained weights
        # in_channels=1,                  # input channels (e.g. 3 for RGB)
        # classes=1,                      # output channels (e.g. 1 for binary segmentation)        
        encoder_name='timm-efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
    )
    # model.load_state_dict(torch.load(f"unet_{encoder_name}_ep{epochs}.pth",weights_only=True))
    # model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print(f"save unet_{encoder_name}_ep{epochs}.pth")
    torch.save(model.state_dict(), f"unet_{encoder_name}_ep{epochs}.pth")

    print(f"load unet_{encoder_name}_ep{epochs}.pth")
    model.load_state_dict(torch.load(f"unet_{encoder_name}_ep{epochs}.pth",weights_only=True))
    model.to(device)

    # # # Test and visualize
    test_inputs, test_masks = next(iter(real_loader))
    test_inputs, test_masks = test_inputs.to(device), test_masks.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(test_inputs)
        pred_binary = (torch.sigmoid(pred > 0.99)).float()
        plot(test_inputs, test_masks, pred_binary)

# if __name__ == "__main__":
#     epochs=10
#     batch_size=64

#     # train_loader, val_loader = MNISTOnCIFARSegmentation.get_dataloaders(batch_size=batch_size)

#     train_loader, val_loader = TwoImagesSegmentationDataset.get_dataloaders(
#         val_dir='./tmp/sim',train_dir='./tmp/sim',input_prename='viewport_2 ',mask_prename='viewport_1 ',
#         batch_size=batch_size)
        
#     images, masks = next(iter(train_loader))
#     plot(images[:5], masks[:5])
#     images, masks = next(iter(val_loader))
#     plot(images[:5], masks[:5])

#     lite   = SimpleUNet(features=[16, 32])       # ~lite model
#     middle = SimpleUNet(features=[32, 64, 128]) # default / middle model
#     heavy  = SimpleUNet(features=[64, 128, 256, 512]) # deeper, heavy model

#     model = lite #SimpleUNet()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.train_data(train_loader, val_loader, epochs=epochs, device=device, #)
#                      criterion=combo_loss)
#                     #  criterion=nn.BCEWithLogitsLoss())
#                     #  criterion=nn.MSELoss())
#                     #  criterion=nn.L1Loss())
#                     #  criterion=nn.SmoothL1Loss())
#     model.save_model()

#     # # Test and visualize
#     test_inputs, test_masks = next(iter(val_loader))
#     preds = model.infer(test_inputs[:5])  # Inference on first 5 samples
#     print(preds.min().item(), preds.max().item(), preds.mean().item())
#     print(iou_score(preds,test_masks[:5]))
#     plot(test_inputs[:5], test_masks[:5], preds)

# --- Example Visualization ---
# if __name__ == '__main__':
#     train_loader, _ = get_dataloaders(batch_size=4)
#     images, masks = next(iter(train_loader))

#     # train_loader, val_loader = get_dataloaders(batch_size=32)
#     # model = SimpleUNet()
#     # train(model, train_loader, val_loader, epochs=10)

#     plot(images, masks)
