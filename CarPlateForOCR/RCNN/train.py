import os
import re
import math
import random
from PIL import Image

# Enable synchronous CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

# ==== GLOBAL CONFIGURATION VARIABLES ====
TRAIN_IMG_DIR = r'C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\RCNN\Dataset\train'
TEST_IMG_DIR = r'C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\RCNN\Dataset\test'
OUTPUT_DIR = './checkpoints'

# Model & training hyperparameters
NUM_CLASSES = 37               # 36 char classes + background
NUM_EPOCHS = 100               # number of epochs
LEARNING_RATE = 5e-4           # initial learning rate
MOMENTUM = 0.9                 # for SGD
WEIGHT_DECAY = 5e-4

# Optimizer choice: 'sgd' or 'adamw'
OPTIMIZER = 'sgd'

# LR scheduler parameters
LR_SCHEDULER = 'cosine'       # options: 'step', 'cosine'
STEP_SIZE = 10
GAMMA = 0.5

# Data & augmentation
BATCH_SIZE = 4                 # default batch size
NUM_WORKERS = 0                # set >0 for performance once stable
ANCHOR_SIZES = ((4, 8, 16),)
ROTATION_ANGLE = 15.0
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.7, 1.3)
HUE_RANGE = (-0.05, 0.05)


def collate_fn(batch):
    return tuple(zip(*batch))


class PlateCharDataset(Dataset):
    """Splits each plate into per-char boxes by filename."""
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        raw = os.path.splitext(fname)[0]
        text = re.sub(r'[^0-9A-Za-z]', '', raw)
        assert text, f"Invalid filename {fname}"
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        w, h = img.size

        box_w = w / len(text)
        boxes, labels = [], []
        for i, ch in enumerate(text):
            xmin, xmax = i * box_w, (i + 1) * box_w
            boxes.append([xmin, 0.0, xmax, float(h)])
            lbl = int(ch) + 1 if ch.isdigit() else ord(ch.upper()) - ord('A') + 11
            labels.append(lbl)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return F.to_tensor(img), target


def rotate_boxes(boxes, angle, img_size):
    w, h = img_size; cx, cy = w/2, h/2
    theta = math.radians(angle)
    cos_v, sin_v = math.cos(theta), math.sin(theta)
    new = []
    for b in boxes:
        corners = torch.tensor([[b[0], b[1]], [b[0], b[3]], [b[2], b[1]], [b[2], b[3]]])
        corners -= torch.tensor([cx, cy])
        rot = torch.zeros_like(corners)
        rot[:,0] = corners[:,0] * cos_v - corners[:,1] * sin_v
        rot[:,1] = corners[:,0] * sin_v + corners[:,1] * cos_v
        rot += torch.tensor([cx, cy])
        xminn, yminn = rot[:,0].min(), rot[:,1].min()
        xmaxx, ymaxy = rot[:,0].max(), rot[:,1].max()
        new.append([xminn.clamp(0, w), yminn.clamp(0, h), xmaxx.clamp(0, w), ymaxy.clamp(0, h)])
    return torch.stack([torch.tensor(nb) for nb in new])


class Transform:
    def __init__(self, train: bool):
        self.train = train

    def __call__(self, img, target):
        if self.train:
            if random.random() < 0.5:
                img = F.hflip(img); w, _ = img.size
                target['boxes'][:, [0, 2]] = w - target['boxes'][:, [2, 0]]
            if random.random() < 0.5:
                img = F.adjust_brightness(img, random.uniform(*BRIGHTNESS_RANGE))
                img = F.adjust_contrast(img, random.uniform(*CONTRAST_RANGE))
                img = F.adjust_saturation(img, random.uniform(*SATURATION_RANGE))
                img = F.adjust_hue(img, random.uniform(*HUE_RANGE))
            if random.random() < 0.5:
                ang = random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
                img = F.rotate(img, ang)
                target['boxes'] = rotate_boxes(target['boxes'], ang, img.size)
        return img, target


def build_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    model.rpn.anchor_generator.sizes = ANCHOR_SIZES
    return model


def train_one_epoch(model, opt, loader, dev, epoch):
    model.train(); total = 0.0
    for imgs, targs in loader:
        imgs = [i.to(dev) for i in imgs]
        targs = [{k: v.to(dev) for k, v in tar.items()} for tar in targs]
        loss_dict = model(imgs, targs)
        loss = sum(loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
    avg = total / len(loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} Loss: {avg:.4f}")
    return avg


def evaluate(model, loader, dev):
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            _ = model([i.to(dev) for i in imgs])
    print("Final evaluation done")


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-adjust batch size for 8GB GPU
    batch_size = BATCH_SIZE
    if dev.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        if total_mem < 10:
            batch_size = max(1, BATCH_SIZE // 2)
            print(f"Reduced batch size to {batch_size} for {total_mem:.1f} GB GPU memory")

    # Data loaders
    train_ds = PlateCharDataset(TRAIN_IMG_DIR, transforms=Transform(True))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    test_loader = None
    if TEST_IMG_DIR and os.path.isdir(TEST_IMG_DIR):
        test_ds = PlateCharDataset(TEST_IMG_DIR, transforms=Transform(False))
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=collate_fn
        )

    # Model and optimizer
    model = build_model().to(dev)
    if OPTIMIZER.lower() == 'adamw':
        opt = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    else:
        opt = torch.optim.SGD(
            model.parameters(), lr=LEARNING_RATE,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )

    # LR scheduler
    if LR_SCHEDULER == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    else:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_losses = []
    for e in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, opt, train_loader, dev, e)
        train_losses.append(loss)
        sched.step()

    # Final evaluation
    if test_loader:
        print("Running final evaluation...")
        evaluate(model, test_loader, dev)

    # Save final model
    state_path = os.path.join(OUTPUT_DIR, 'model_final.pth')
    model_path = os.path.join(OUTPUT_DIR, 'model_final.pt')
    torch.save(model.state_dict(), state_path)
    torch.save(model, model_path)
    print(f"Saved final state to {state_path} and model to {model_path}")

    # Plot training loss curve
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    loss_plot = os.path.join(OUTPUT_DIR, 'loss_curve.png')
    plt.savefig(loss_plot)
    print(f"Saved loss curve to {loss_plot}")

if __name__ == '__main__':
    main()