import os
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import retinanet_resnet50_fpn


# Check DEVICE (CUDA or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test model
# Choose your model and image folder here
MODEL_PATH = 'retinanet_finetuned.pth'  # << your trained model path
IMAGE_FOLDER = 'images/'  # << your folder of images
CLASS_NAMES = {1: "Car", 2: "CarPlate"}  # << depends on your label IDs

# ==== Inference Function ====
def inference_single_image(model, image_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    keep = scores > threshold
    return boxes[keep], labels[keep], scores[keep], image

# ==== Draw Results ====
def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        pass  # if font not found, skip nice font

    for box, label, score in zip(boxes, labels, scores):
        box = box.tolist()
        label_name = CLASS_NAMES.get(label.item(), f"Class {label.item()}")
        draw.rectangle(box, outline="red", width=3)
        text = f"{label_name}: {score:.2f}"
        text_pos = (box[0], box[1] - 10)
        draw.text(text_pos, text, fill="red", font=font)

    return image

# # ==== Load model ====
print("Loading model...")
model = retinanet_resnet50_fpn(weights=None, num_classes=len(CLASS_NAMES)+1)  # Important: num_classes=3
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ==== Run Detection on Folder ====
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    print(f"Processing {img_file}...")

    boxes, labels, scores, orig_image = inference_single_image(model, img_path)
    result_img = draw_boxes(orig_image, boxes, labels, scores)

    # Show result
    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    plt.title(img_file)
    plt.axis('off')
    plt.show()