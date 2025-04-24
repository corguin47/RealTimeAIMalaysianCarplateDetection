import os
from PIL import Image
import math
import random

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# ==== GLOBAL CONFIGURATION VARIABLES ====
TRAIN_IMG_DIR = 'data/train/images'
TEST_IMG_DIR = 'data/test/images'  # set to your test folder, or leave None
OUTPUT_DIR = './checkpoints'

# Model & training hyperparameters
NUM_CLASSES = 37            # 36 char classes + background
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
STEP_SIZE = 5               # LR scheduler step size (epochs)
GAMMA = 0.1                 # LR scheduler decay factor
NUM_WORKERS = 4
ANCHOR_SIZES = ((4, 8, 16),)  # for small characters
ROTATION_ANGLE = 15.0         # max rotation angle
BRIGHTNESS_RANGE = (0.5, 1.5)


def collate_fn(batch):
    return tuple(zip(*batch))


class PlateCharDataset(Dataset):
    """
    Dataset that infers per-character boxes by splitting the full image width
    evenly based on filename (plate text).
    """
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        # collect all image files
        self.image_files = [f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        plate_text = os.path.splitext(fname)[0]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # compute boxes by equal splitting
        char_count = len(plate_text)
        box_w = w / char_count
        boxes = []
        labels = []
        for i, ch in enumerate(plate_text):
            xmin = i * box_w
            xmax = (i + 1) * box_w
            boxes.append([xmin, 0.0, xmax, float(h)])
            # map char to label: digits 0-9, A-Z -> 10-35
            if ch.isdigit():
                label = int(ch)
            else:
                label = ord(ch.upper()) - ord('A') + 10
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'iscrowd': iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return F.to_tensor(img), target


def rotate_boxes(boxes, angle, img_size):
    w, h = img_size
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(angle)
    cos_val, sin_val = math.cos(theta), math.sin(theta)
    new_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        corners = torch.tensor([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
        corners -= torch.tensor([cx, cy])
        rotated = torch.zeros_like(corners)
        rotated[:, 0] = corners[:, 0] * cos_val - corners[:, 1] * sin_val
        rotated[:, 1] = corners[:, 0] * sin_val + corners[:, 1] * cos_val
        rotated += torch.tensor([cx, cy])
        xminn = float(rotated[:, 0].min())
        yminn = float(rotated[:, 1].min())
        xmaxx = float(rotated[:, 0].max())
        ymaxy = float(rotated[:, 1].max())
        new_boxes.append([xminn, yminn, xmaxx, ymaxy])
    return torch.tensor(new_boxes, dtype=torch.float32)


def get_transform(train):
    def transform(img, target):
        if train:
            if random.random() < 0.5:
                img = F.hflip(img)
                w, _ = img.size
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
            if random.random() < 0.5:
                factor = random.uniform(*BRIGHTNESS_RANGE)
                img = F.adjust_brightness(img, factor)
            if random.random() < 0.5:
                angle = random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
                img = F.rotate(img, angle)
                target['boxes'] = rotate_boxes(target['boxes'], angle, img.size)
        return img, target
    return transform


def build_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.rpn.anchor_generator.sizes = ANCHOR_SIZES
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            results.extend(outputs)
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training data
    train_ds = PlateCharDataset(TRAIN_IMG_DIR, transforms=get_transform(train=True))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)

    # Optional test data
    test_loader = None
    if TEST_IMG_DIR and os.path.isdir(TEST_IMG_DIR):
        test_ds = PlateCharDataset(TEST_IMG_DIR, transforms=get_transform(train=False))
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 collate_fn=collate_fn)

    # Model + optimizer + scheduler
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=STEP_SIZE,
                                                   gamma=GAMMA)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_epoch_{epoch}.pth'))
        torch.save(model, os.path.join(OUTPUT_DIR, f'model_epoch_{epoch}.pt'))
        print(f"Saved epoch {epoch} checkpoint(s)")

        # Run evaluation if test set is available
        if test_loader:
            results = evaluate(model, test_loader, device)
            print(f"Epoch {epoch} evaluation completed: {len(results)} batches processed")

if __name__ == '__main__':
    main()
