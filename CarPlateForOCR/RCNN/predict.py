import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

from train import build_model, NUM_CLASSES

# === SET INPUT HERE ===
IMAGE_PATH = r'C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\AKL9971.png'  # <- replace with your test image path
WEIGHTS_PATH = r'C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\checkpoints_adam_opt\model_final.pth'  # <- leave None to use untrained model
DEVICE = 'cuda'  # or 'cpu'
SCORE_THRESHOLD = 0.5


def decode_label(label_id):
    if 1 <= label_id <= 10:
        return str(label_id - 1)
    elif 11 <= label_id <= 36:
        return chr(label_id - 11 + ord('A'))
    else:
        return '?'  # background or unknown


def load_model(weights_path=None, device='cuda'):
    model = build_model()
    model.to(device)
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from {weights_path}")
    else:
        print("⚠️ No weights loaded, using untrained model.")
    model.eval()
    return model


def predict_on_image(model, image_path, device='cuda', score_thresh=0.5):
    image = Image.open(image_path).convert('RGB')
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs['boxes'].cpu()
    scores = outputs['scores'].cpu()
    labels = outputs['labels'].cpu()

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Sort by x1 to order left to right
    left_to_right = boxes[:, 0].argsort()
    labels = labels[left_to_right]

    prediction = ''.join(decode_label(label.item()) for label in labels)
    print(f"🔤 Predicted Plate Characters: {prediction}")


if __name__ == '__main__':
    model = load_model(WEIGHTS_PATH, device=DEVICE)
    predict_on_image(model, IMAGE_PATH, device=DEVICE, score_thresh=SCORE_THRESHOLD)
