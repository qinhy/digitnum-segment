import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
class MNISTOnCIFARSegmentation(Dataset):
    def __init__(self, train=True, mixup=False, cutmix=False):
        self.mixup = mixup
        self.cutmix = cutmix
        self.mnist = datasets.MNIST(root='./data', train=train, download=True)
        self.cifar = CIFAR10(root='./data', train=train, download=True)

        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.grayscale = transforms.Grayscale(num_output_channels=1)

        self.train = train
        self.aug = A.Compose([
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(8, 16),
                hole_width_range=(8, 16),
                fill=0,
                fill_mask=0,
                p=0.5
            ),
            A.Affine(
                rotate=(-25, 25),             # Arbitrary rotation range
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Up to ±20% shift
                scale=(0.9, 1.1),             # Slight zoom in/out
                shear=(-10, 10),              # Optional: slight shearing
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])

        self.no_aug = A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()
        ])

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
        digit_np = np.array(digit_img, dtype=np.float32) / 255.0 * random.random()
        bg_np = np.array(bg_img, dtype=np.float32) / 255.0
        mask_np = digit_np > 0.0

        combined = bg_np * (1 - mask_np) + digit_np * mask_np
        combined = (combined * 255).astype(np.uint8)
        mask_np  = (mask_np * 255).astype(np.uint8)

        if self.train:
            augmented = self.aug(image=combined, mask=mask_np)
        else:
            augmented = self.no_aug(image=combined, mask=mask_np)

        input_tensor = augmented['image']            # [1, 32, 32]
        mask_tensor = augmented['mask'].unsqueeze(0) # [1, 32, 32]

        return input_tensor.float(), mask_tensor.float() / 255.0
    
    def __getitem__(self, idx):
        input_tensor, mask_tensor = self._build_sample(idx)

        # Apply Mixup
        if self.mixup and random.random() < 0.5:
            idx2 = random.randint(0, len(self) - 1)
            input2, mask2 = self._build_sample(idx2)
            lam = np.random.beta(0.4, 0.4)
            input_tensor = lam * input_tensor + (1 - lam) * input2
            mask_tensor = lam * mask_tensor + (1 - lam) * mask2

        # Apply CutMix
        elif self.cutmix and random.random() < 0.5:
            idx2 = random.randint(0, len(self) - 1)
            input2, mask2 = self._build_sample(idx2)

            _, H, W = input_tensor.shape
            cut_h, cut_w = random.randint(8, 16), random.randint(8, 16)
            cy, cx = random.randint(0, H - cut_h), random.randint(0, W - cut_w)

            input_tensor[:, cy:cy+cut_h, cx:cx+cut_w] = input2[:, cy:cy+cut_h, cx:cx+cut_w]
            mask_tensor[:, cy:cy+cut_h, cx:cx+cut_w] = mask2[:, cy:cy+cut_h, cx:cx+cut_w]

        return input_tensor.float(), mask_tensor.float()

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

        return torch.sigmoid(self.out(d1))
    
    def inference(self,imgs, conf=0.5):
        with torch.no_grad():
            inputs = torch.stack([img for img in imgs]).to(self.out.bias.device)
            outputs = self(inputs)
            preds = (outputs > conf).float()  # Thresholding to get binary masks
        return preds.cpu()
    
    def save_model(self, path="mnist_on_cifar_unet.pth"):
        torch.save(self.state_dict(), path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path="mnist_on_cifar_unet.pth", device='cuda'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()
        print(f"📥 Model loaded from {path}")
        return self

    def train_data(self, train_loader, val_loader=None, epochs=5, device='cuda'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

            if val_loader:
                self.validate_data(val_loader, device)
    
    def validate_data(self, val_loader, device):
        self.eval()
        total_loss = 0.0
        criterion = nn.BCELoss()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = self(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)
        print(f"  🔍 Validation Loss: {avg_val_loss:.4f}")

def plot(inputs,gts,preds=[]):
    for i in range(len(inputs)):
        input_img = inputs[i].squeeze().numpy()
        gt_mask = gts[i].squeeze().numpy()
        cols = 2
        if len(preds):
            pred_mask = preds[i].squeeze().numpy()
            cols += 1

        plt.figure(figsize=(9, cols))
        
        plt.subplot(1, cols, 1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, cols, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        if len(preds):
            plt.subplot(1, cols, 3)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

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
        full_img = Image.open(image_path).convert('RGB')  # Grayscale
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

if __name__ == "__main__":
    train_loader, val_loader = MNISTOnCIFARSegmentation.get_dataloaders(batch_size=32)
    images, masks = next(iter(train_loader))
    plot(images, masks)

    model = SimpleUNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train_data(train_loader, val_loader, epochs=5, device=device)
    model.save_model()

    # Test and visualize
    test_inputs, test_masks = next(iter(val_loader))
    preds = model.inference(test_inputs[:5])  # Inference on first 5 samples
    plot(test_inputs[:5], test_masks[:5], preds)

# --- Example Visualization ---
# if __name__ == '__main__':
#     train_loader, _ = get_dataloaders(batch_size=4)
#     images, masks = next(iter(train_loader))

#     # train_loader, val_loader = get_dataloaders(batch_size=32)
#     # model = SimpleUNet()
#     # train(model, train_loader, val_loader, epochs=10)

#     plot(images, masks)