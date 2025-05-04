import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_DIR = "./CarPlateForOCR/SecondApproach/TrOCR/Models"  # your saved model path
IMAGE_DIR = r"D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\test"  # folder with new images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL AND PROCESSOR ===
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# === PREDICT FUNCTION ===
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return prediction.strip()

# === RUN PREDICTION ON FOLDER ===
for fname in os.listdir(IMAGE_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, fname)
        pred_text = predict_image(img_path)

        # === Display ===
        img = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(6, 2))
        plt.imshow(img)
        plt.title(f"{fname} → Predicted: {pred_text}")
        plt.axis("off")
        plt.show()
