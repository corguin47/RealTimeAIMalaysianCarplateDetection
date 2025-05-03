import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# === CONFIG ===
MODEL_NAME = "microsoft/trocr-base-stage1"
TRAIN_DIR = r"C:\Users\NITRO 5\OneDrive - Swinburne Sarawak\General - COS30018 INTELLIGENT SYSTEMS\Dataset\OCR\Combined\train"
VAL_DIR   = r"C:\Users\NITRO 5\OneDrive - Swinburne Sarawak\General - COS30018 INTELLIGENT SYSTEMS\Dataset\OCR\Combined\test"
OUTPUT_DIR = "./trocr_plate_model"
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

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
        encoding = self.processor(images=image, text=item["text"], return_tensors="pt", padding="max_length", truncation=True)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

# === LOAD MODEL AND PROCESSOR ===
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# === LOAD DATASETS ===
train_dataset = PlateOCRDataset(TRAIN_DIR, processor)
val_dataset = PlateOCRDataset(VAL_DIR, processor)

# === TRAINING ARGS ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    fp16=torch.cuda.is_available(),  
)

# === TRAINER ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    data_collator=default_data_collator, 
)

# === START TRAINING ===
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
