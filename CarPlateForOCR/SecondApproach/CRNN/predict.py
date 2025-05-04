import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from crnn import CRNN
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\SecondApproach\CRNN\Models\model_final.pth'
IMAGE_PATH = r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\Dataset\test\RM2285.png'
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BLANK_LABEL = 0
IDX2CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
IDX2CHAR[BLANK_LABEL] = ''

# adjust based on training
NH = 128
IMG_HEIGHT = 32
IMG_WIDTH = 320

# === Utilities ===
def decode_prediction(preds):
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
    return decoded[0]  # since batch size is 1


# === Predict Function ===
def predict(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # [B=1, C, H, W]
    
    # Visualize the transformed image before prediction
    debug_image = image.clone().squeeze(0).cpu()  # [1, C, H, W] → [C, H, W]
    debug_image = debug_image * 0.5 + 0.5  # Undo normalization (from [-1,1] back to [0,1])
    debug_image = debug_image.permute(1, 2, 0).numpy()  # [H, W, C]

    if debug_image.shape[2] == 1:
        debug_image = debug_image.squeeze(2)  # convert [H, W, 1] → [H, W] for grayscale

    plt.imshow(debug_image, cmap='gray')
    plt.title("Transformed Image Before Inference")
    plt.axis('off')
    plt.show()

    model = CRNN(IMG_HEIGHT, 1, len(CHARACTERS) + 1, NH)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(image)  # [T, B, C] with log_softmax
        pred_text = decode_prediction(preds.cpu())

    print(f"Predicted Text: {pred_text}")


if __name__ == '__main__':
    predict(IMAGE_PATH, MODEL_PATH)