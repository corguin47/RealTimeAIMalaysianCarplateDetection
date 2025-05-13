import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torch.nn.utils.rnn import pack_padded_sequence
from crnn import CRNN
import csv
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter
from pyctcdecode import build_ctcdecoder
import random


# === CONFIG ===
TRAIN_IMG_DIR = r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\train'
TEST_IMG_DIR = r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\test'
OUTPUT_DIR = './CarPlateForOCR/SecondApproach/CRNN/Models/test'
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NH = 128  #  hidden dim (CRNN capacity to model character dependencies) + use small because dataset is small
OPTIMIZER_TYPE = 'adamw'  # 'adam', 'adamw', or 'sgd'
IMG_HEIGHT = 32
IMG_WIDTH = 320
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BLANK_LABEL = 0  # For CTC blank
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
IDX2CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
IDX2CHAR[BLANK_LABEL] = ''
USE_BEAM = True  # Use beam search decoding for CTC
USE_WEIGHTED_SAMPLER = False
CTC_VOCAB_STRING = " " + CHARACTERS  # '' for blank (index 0)
beam_decoder = build_ctcdecoder(
    labels=[' '] + list(CHARACTERS[:-1]) 
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === DATASET ===
class PlateSequenceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            
            
            # === (too excessive) ===
            # transforms.RandomRotation(degrees=5, fill=(255,)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=5),
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label_str = os.path.splitext(fname)[0]
        label_str = ''.join(filter(str.isalnum, label_str)).upper()
        label = torch.tensor([CHAR2IDX[ch] for ch in label_str], dtype=torch.long)
        return img, label, len(label), label_str

# === WEIGHTED SAMPLER OVER ALL CHARACTERS ===
def get_weighted_sampler(dataset):
    char_freq = Counter()
    for _, _, _, label_str in dataset:
        char_freq.update(label_str)
    total_chars = sum(char_freq.values())
    char_weights = {ch: total_chars / char_freq[ch] for ch in char_freq}
    sample_weights = []
    for _, _, _, label_str in dataset:
        weight = sum(char_weights.get(ch, 0) for ch in label_str) / len(label_str)
        sample_weights.append(weight)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# === DECODE FUNCTION ===
def decode_preds(preds):
    preds = preds.argmax(2).permute(1, 0)  # [B, T]
    decoded = []
    for pred in preds:
        prev = -1
        chars = []
        for p in pred:
            p = p.item()
            if p != prev and p != BLANK_LABEL:
                chars.append(IDX2CHAR[p])
            prev = p
        decoded.append(''.join(chars))
    return decoded

def decode_preds_beam(log_probs):
    log_probs = log_probs.permute(1, 0, 2)  # [B, T, C]
    decoded = []
    for lp in log_probs:
        beam_result = beam_decoder.decode(lp.cpu().numpy())
        decoded.append(beam_result)
    return decoded


def evaluate_model(model, test_loader, device, use_beam=False):
    model.eval()
    all_preds, all_labels = [], []
    
    def cer(s1, s2):
        import editdistance
        return editdistance.eval(s1, s2) / max(1, len(s2))

    with torch.no_grad():
        for imgs, _, _, label_strs in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if use_beam:
                preds = decode_preds_beam(outputs.cpu().log_softmax(2))  # for CTC decoder
            else:
                preds = decode_preds(outputs.cpu())
            all_preds.extend(preds)
            all_labels.extend(label_strs)

    cer_scores = [cer(p, t) for p, t in zip(all_preds, all_labels)]
    avg_cer = sum(cer_scores) / len(cer_scores)
    print(f"Test CER: {avg_cer:.4f}")

    acc = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_preds)
    print(f"Test Exact Match Accuracy: {acc:.4f}")

    total_chars = sum(len(t) for t in all_labels)
    correct_chars = sum(c1 == c2 for pred, gt in zip(all_preds, all_labels) for c1, c2 in zip(pred, gt))
    char_acc = correct_chars / total_chars
    print(f"Character-Level Accuracy: {char_acc:.4f}")

    worst = sorted(zip(all_labels, all_preds, cer_scores), key=lambda x: -x[2])[:10]
    with open(os.path.join(OUTPUT_DIR, 'worst_errors.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ground Truth', 'Prediction', 'CER'])
        writer.writerows(worst)

    true_first = [t[0] if t else '?' for t in all_labels]
    pred_first = [p[0] if p else '?' for p in all_preds]
    cm = confusion_matrix(true_first, pred_first, labels=list(CHARACTERS))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(CHARACTERS), yticklabels=list(CHARACTERS))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix (First Char Only)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png'))
    plt.close()

    with open(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ground Truth', 'Prediction'])
        writer.writerows(zip(all_labels, all_preds))

    # Classification report (character-level)
    y_true_chars, y_pred_chars = [], []
    for gt, pred in zip(all_labels, all_preds):
        min_len = min(len(gt), len(pred))
        y_true_chars.extend(gt[:min_len])
        y_pred_chars.extend(pred[:min_len])
    report = classification_report(y_true_chars, y_pred_chars, labels=list(CHARACTERS), digits=4)
    print("Character Classification Report:")
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'char_classification_report.txt'), 'w') as f:
        f.write(report)

def save_training_config(final_loss, final_cer):
    config_path = os.path.join(OUTPUT_DIR, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"NUM_EPOCHS = {NUM_EPOCHS}\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"CRNN MODEL NH = {NH}\n")
        f.write(f"OPTIMIZER_TYPE = {OPTIMIZER_TYPE}\n")
        f.write(f"IMG_HEIGHT = {IMG_HEIGHT}\n")
        f.write(f"IMG_WIDTH = {IMG_WIDTH}\n")
        f.write(f"CHARACTERS = {CHARACTERS}\n")
        f.write(f"FINAL_EPOCH = {NUM_EPOCHS}\n")
        f.write(f"FINAL_LOSS = {final_loss:.4f}\n")
        f.write(f"FINAL_CER = {final_cer:.4f}\n")
        f.write(f"USE_BEAM = {USE_BEAM}\n")

# === TRAIN ===
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = PlateSequenceDataset(TRAIN_IMG_DIR)
    
    if USE_WEIGHTED_SAMPLER:
        sampler = get_weighted_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN(IMG_HEIGHT, 1, len(CHARACTERS) + 1, NH).to(device)
    criterion = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)

    if OPTIMIZER_TYPE == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    elif OPTIMIZER_TYPE == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


    elif OPTIMIZER_TYPE == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    else:
        raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")

    all_losses = []
    all_cers = []

    def cer(s1, s2):
        import editdistance
        return editdistance.eval(s1, s2) / max(1, len(s2))

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        total_cer = 0
        num_samples = 0

        for imgs, labels, label_lens, label_strs in train_loader:
            imgs = imgs.to(device)
            labels_flatten = torch.cat(labels).to(device)
            label_lens_tensor = torch.tensor(label_lens, dtype=torch.long)

            outputs = model(imgs)  # [T, B, C]
            T, B, C = outputs.size()
            input_lens = torch.full(size=(B,), fill_value=T, dtype=torch.long)

            loss = criterion(outputs, labels_flatten, input_lens, label_lens_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                preds = decode_preds(outputs.cpu())
                batch_cer = sum(cer(p, t) for p, t in zip(preds, label_strs))
                total_cer += batch_cer
                num_samples += len(label_strs)

        epoch_loss = total_loss / len(train_loader)
        epoch_cer = total_cer / num_samples
        all_losses.append(epoch_loss)
        all_cers.append(epoch_cer)
        scheduler.step()

        print(f"Epoch {epoch}/{NUM_EPOCHS} Loss: {epoch_loss:.4f} CER: {epoch_cer:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_final.pth'))
    
    save_training_config(epoch_loss, epoch_cer)

    # Plot training curves
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), all_losses, marker='o', label='Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), all_cers, marker='x', label='CER')
    plt.title('Training Loss & CER Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss_cer.png'))
    plt.close()

# === Collate ===
def collate_fn(batch):
    imgs, labels, label_lens, label_strs = zip(*batch)
    return torch.stack(imgs), labels, label_lens, label_strs

if __name__ == '__main__':
    train()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(IMG_HEIGHT, 1, len(CHARACTERS) + 1, NH).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'model_final.pth'), map_location=device))

    test_ds = PlateSequenceDataset(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    evaluate_model(model, test_loader, device, USE_BEAM)