import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
import numpy as np
import random
import evaluate
from jiwer import cer as compute_cer
import editdistance

# === CONFIG ===
MODEL_NAME = "microsoft/trocr-small-handwritten"
TRAIN_DIR = r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\train"
TEST_DIR   = r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\test"
OUTPUT_DIR = "./CarPlateForOCR/SecondApproach/TrOCR/Models"
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 32
IMG_WIDTH = 384
# Img size is 384x32 for TrOCR small model (need in resolution 16x16, where both values are divisble by 16).
# https://huggingface.co/microsoft/trocr-small-handwritten#:~:text=Images%20are%20presented%20to%20the,)%2C%20which%20are%20linearly%20embedded.

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

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
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    
    # No normalisation for TrOCR, as it expects raw pixel values
    # transforms.Normalize((0.5,), (0.5,)),
    
    # No need to convert to grayscale for TrOCR, as it works with RGB images
    # transforms.Grayscale(),
])

# === DATASET ===
class PlateOCRDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.processor = processor
        self.data = []
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                label = os.path.splitext(fname)[0].upper()
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
        encoding = self.processor(images=image, text=item["text"], return_tensors="pt", padding=False, do_rescale=False)
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"].squeeze(0)
        }

# === LOAD MODEL AND PROCESSOR ===
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

def evaluate_cer_accuracy(pred, write_errors=True, max_samples=50):
    import numpy as np
    import os
    import csv
    import editdistance
    from datetime import datetime

    def compute_cer(true_text, pred_text):
        return editdistance.eval(true_text, pred_text) / max(len(true_text), 1)

    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids

    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Clean both: keep only alphanumeric and uppercase
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

        print(f"[Saved worst predictions to {log_path}]")

    return {
        "cer": float(np.mean(cer_scores)),
        "exact_match": float(np.mean(exact_matches))
    }


# === LOAD DATASETS ===
train_dataset = PlateOCRDataset(TRAIN_DIR, processor)
test_dataset = PlateOCRDataset(TEST_DIR, processor)

# === TRAINING ARGS ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    logging_steps=10,
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
    metric_for_best_model="loss",
    greater_is_better=False,
)

# === COLLATE FUNCTION ===
def trocr_collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad labels to same length
    padded = processor.tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt"
    )

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": padded["input_ids"]
    }



# === TRAINER ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=evaluate_cer_accuracy,
    eval_dataset=test_dataset,
    tokenizer=processor.tokenizer,
    data_collator=trocr_collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

# === START TRAINING ===
if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)