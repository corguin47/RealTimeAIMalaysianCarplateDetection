import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
import numpy as np
import random
import csv
import editdistance
from datetime import datetime

# === CONFIG ===
MODEL_NAME = "DunnBC22/trocr-base-printed_license_plates_ocr"
BASE_MODEL = "microsoft/trocr-base-printed"
TRAIN_DIR = r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\train"
TEST_DIR = r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\test"
OUTPUT_DIR = "./CarPlateForOCR/SecondApproach/TrOCR/Models"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 32
IMG_WIDTH = 384

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# === SEED ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# === AUGMENTATION ===
augment = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(degrees=1, fill=(255, 255, 255)),
])

# === LOAD PROCESSORS ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# === DATASET ===
class PlateOCRDataset(Dataset):
    def __init__(self, image_dir):
        self.data = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                label = os.path.splitext(fname)[0].upper()
                if label.strip() and label.isalnum():
                    self.data.append({
                        "image_path": os.path.join(image_dir, fname),
                        "text": label
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = augment(image)
        encoding = image_processor(images=image, return_tensors="pt")
        input_ids = tokenizer(item["text"], return_tensors="pt", padding="max_length", truncation=True)["input_ids"]
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": input_ids.squeeze(0)
        }

# === METRICS ===
def evaluate_cer_accuracy(pred, write_errors=True, max_samples=50):
    def compute_cer(true_text, pred_text):
        return editdistance.eval(true_text, pred_text) / max(len(true_text), 1)

    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids = np.where(label_ids != -100, label_ids, model.config.pad_token_id)

    pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_strs = [''.join(filter(str.isalnum, p.upper())) for p in pred_strs]
    label_strs = [''.join(filter(str.isalnum, l.upper())) for l in label_strs]

    cer_scores = [compute_cer(t, p) for t, p in zip(label_strs, pred_strs)]
    exact_matches = [p == t for p, t in zip(pred_strs, label_strs)]

    if write_errors:
        os.makedirs("./logs/errors", exist_ok=True)
        log_path = f"./logs/errors/worst_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Ground Truth", "Prediction", "CER"])
            for gt, pr, cer in sorted(zip(label_strs, pred_strs, cer_scores), key=lambda x: -x[2])[:max_samples]:
                writer.writerow([gt, pr, f"{cer:.4f}"])

        best_path = f"./logs/errors/best_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(best_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Ground Truth", "Prediction", "CER"])
            for gt, pr, cer in sorted(zip(label_strs, pred_strs, cer_scores), key=lambda x: x[2])[:max_samples]:
                writer.writerow([gt, pr, f"{cer:.4f}"])

        print(f"[Saved worst predictions to {log_path}]")
        print(f"[Saved best predictions to {best_path}]")

    return {
        "cer": float(np.mean(cer_scores)),
        "exact_match": float(np.mean(exact_matches))
    }

# === COLLATE ===
def trocr_collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    padded = tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt")
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": padded["input_ids"]
    }

# === CUSTOM TRAINER ===
from transformers import Seq2SeqTrainer
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        label_counts = {}
        for item in self.train_dataset:
            label = item["labels"].tolist()
            label_str = ''.join(map(str, label))
            label_counts[label_str] = label_counts.get(label_str, 0) + 1

        total = len(self.train_dataset)
        weights = []
        for item in self.train_dataset:
            label = item["labels"].tolist()
            label_str = ''.join(map(str, label))
            weights.append(1.0 / label_counts[label_str])

        sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers
        )

# === LOAD DATASETS ===
train_dataset = PlateOCRDataset(TRAIN_DIR)
test_dataset = PlateOCRDataset(TEST_DIR)

# === TRAINING ARGS ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    logging_steps=200,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    lr_scheduler_type="linear",
    logging_dir=f"{OUTPUT_DIR}/logs",
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# === TRAIN ===
trainer = WeightedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=trocr_collate_fn,
    compute_metrics=evaluate_cer_accuracy,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
