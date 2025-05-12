import os
import random
from PIL import Image
from tqdm import tqdm
import pycocotools.coco as coco
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt

# === Constants ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_ANNOTATION_FILE = r'D:/RealTimeAIMalaysianCarplateDetection/ObjectDetection/Retina/final-dataset.v6i.coco/train/_annotations.coco.json'
TEST_ANNOTATION_FILE = r'D:/RealTimeAIMalaysianCarplateDetection/ObjectDetection/Retina/final-dataset.v6i.coco/test/_annotations.coco.json'
IMAGE_DIR = r'D:/RealTimeAIMalaysianCarplateDetection/ObjectDetection/Retina/final-dataset.v6i.coco'
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 5  # early stopping patience

class CustomCocoDataset(Dataset):
    def __init__(self, coco_annotations_file, image_dir, transform=None):
        self.coco = coco.COCO(coco_annotations_file)
        self.image_dir = image_dir
        self.transforms = transform if transform is not None else transforms.ToTensor()
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))

        boxes = []
        labels = []
        for ann in annotations:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        labels = torch.tensor(labels, dtype=torch.int64)[keep]

        if self.transforms:
            image = self.transforms(image)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return image, target

    def __len__(self):
        return len(self.image_ids)

# === Load Datasets ===
train_dataset = CustomCocoDataset(TRAIN_ANNOTATION_FILE, os.path.join(IMAGE_DIR, "train"))
test_dataset = CustomCocoDataset(TEST_ANNOTATION_FILE, os.path.join(IMAGE_DIR, "test"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# === Model ===
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
num_classes = 3  # background + car + plate
in_channels = model.head.classification_head.cls_logits.in_channels
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
model.head.classification_head.num_classes = num_classes
model.to(DEVICE)

# === Optimizer & Scheduler ===
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# === Metrics ===
metric = MeanAveragePrecision()
train_losses = []
val_maps = []
best_map = 0.0
best_epoch = 0
patience_counter = 0

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for images, targets in tqdm(train_loader, desc=f"Training", unit="batch"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # === Validation mAP ===
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Validating", unit="batch"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)

    val_metrics = metric.compute()
    val_map = val_metrics['map'].item()
    val_maps.append(val_map)
    print(f"Train Loss: {train_loss:.4f} | Validation mAP: {val_map:.4f}")

    # === Save best model ===
    if val_map > best_map:
        best_map = val_map
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'retinanet_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    scheduler.step()

# === Final Save ===
torch.save(model.state_dict(), 'retinanet_last.pth')

# === Plot and Save Training Curves ===
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('retinanet_train_loss.png')

plt.figure()
plt.plot(range(1, len(val_maps) + 1), val_maps, label='Validation mAP')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Validation mAP over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('retinanet_val_map.png')

print(f"Best Validation mAP: {best_map:.4f} at epoch {best_epoch}")
