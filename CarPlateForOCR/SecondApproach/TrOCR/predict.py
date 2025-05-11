import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import editdistance

# === CONFIG ===
MODEL_DIR = "./CarPlateForOCR/SecondApproach/TrOCR/Models"
IMAGE_DIR = r"C:\Users\User\Downloads\test"
OUTPUT_FILE = "./predictions.txt"
SHOW_IMAGES = False  # Toggle this to True if you want image preview
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL AND PROCESSOR ===
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# === METRIC ===
def cer(gt, pred):
    return editdistance.eval(gt, pred) / max(len(gt), 1)

# === PREDICT FUNCTION ===
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return prediction.strip()

# === RUN PREDICTION ON FOLDER ===
results = []
for fname in sorted(os.listdir(IMAGE_DIR)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        gt_text = os.path.splitext(fname)[0].upper()
        img_path = os.path.join(IMAGE_DIR, fname)
        pred_text = predict_image(img_path)

        cer_score = cer(gt_text, pred_text)
        exact = gt_text == pred_text

        results.append((fname, gt_text, pred_text, cer_score, exact))

        if SHOW_IMAGES:
            img = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(6, 2))
            plt.imshow(img)
            plt.title(f"{fname}\nGT: {gt_text} → Predicted: {pred_text}")
            plt.axis("off")
            plt.show()

# === WRITE TO FILE ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("Filename\tGroundTruth\tPrediction\tCER\tExactMatch\n")
    for fname, gt, pred, cer_score, exact in results:
        f.write(f"{fname}\t{gt}\t{pred}\t{cer_score:.4f}\t{int(exact)}\n")

# === METRIC SUMMARY ===
avg_cer = sum(r[3] for r in results) / len(results)
exact_acc = sum(r[4] for r in results) / len(results)
print(f"\n[Saved to {OUTPUT_FILE}]")
print(f"Average CER: {avg_cer:.4f}")
print(f"Exact Match Accuracy: {exact_acc:.4f}")
