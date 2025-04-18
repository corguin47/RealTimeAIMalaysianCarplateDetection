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
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn as nn

# === Constants ===
# Check DEVICE (CUDA or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANNOTATION_FILE = 'C:/Users/austi/Downloads/retina/_annotations.coco.json'
IMAGE_DIR = 'C:/Users/austi/Downloads/retina/images'
TOTAL_DATASET_NUMBER = 2000
VALIDATION_RATIO = 0.2
BATCH_SIZE = 8
EPOCHS = 5

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
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y + height

        # Optional: Filter invalid boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        labels = torch.tensor(labels, dtype=torch.int64)[keep]

        if self.transforms:
            image = self.transforms(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target

    def __len__(self):
        return len(self.image_ids)

# === Validation Function ===
def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            model.eval()

            val_loss += sum(loss for loss in loss_dict.values()).item()

    return val_loss / len(val_loader)

# === Load Pre-trained Model ===
# Load pretrained RetinaNet
model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)

# Modify only the LAST conv layer of classification head
num_classes = 3  # (background + car + plate)

cls_logits = model.head.classification_head.cls_logits

in_channels = cls_logits.in_channels
num_anchors = model.head.classification_head.num_anchors

new_cls_logits = nn.Conv2d(
    in_channels, 
    num_anchors * num_classes, 
    kernel_size=3, 
    stride=1, 
    padding=1
)

# Replace the cls_logits
model.head.classification_head.cls_logits = new_cls_logits

# Important!! Also update the num_classes attribute!!
model.head.classification_head.num_classes = num_classes

# Move to GPU
model = model.to(DEVICE)


# === Load Dataset ===
# Load full dataset
full_dataset = CustomCocoDataset(ANNOTATION_FILE, IMAGE_DIR)

# Subsample only N training images
subsample_dataset = Subset(full_dataset, random.sample(range(len(full_dataset)), TOTAL_DATASET_NUMBER))

# Split into train + val
val_size = int(len(subsample_dataset) * VALIDATION_RATIO)
train_size = len(subsample_dataset) - val_size
train_dataset, val_dataset = random_split(subsample_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)


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
    val_loss = validate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'retinanet_finetuned.pth')