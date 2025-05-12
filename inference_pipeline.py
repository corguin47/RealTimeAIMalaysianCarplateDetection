import os
import string
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import torch
import numpy as np
import mss
from ultralytics import YOLO
import easyocr
from collections import deque, Counter, defaultdict
import re

# ==== CONFIGURATION ====
USE_GPU = torch.cuda.is_available()
YOLO_MODEL_PATH = r"D:\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\Models\yolo11n_finetuned.pt"
DETECTION_CONF = 0.25

# Toggles
SAVE_CROPS = False
USE_CAR_FILTER = False
USE_PREPROCESS = True
DEBUG_OCR_CROP = True
USE_LOW_LIGHT_ENHANCE = False

# Globals
gamma_value = 1.2
ocr_buffers = defaultdict(lambda: deque(maxlen=7))

# ==== LOW LIGHT ENHANCEMENT ====
def retinex_enhance(img, sigma=30):
    img = img.astype(np.float32) + 1.0
    log_img = np.log(img)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    log_blur = np.log(blur + 1.0)
    retinex = log_img - log_blur
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8)):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_low_light_image(img):
    img = retinex_enhance(img, sigma=30)
    img = clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8))
    return gamma_correction(img, gamma=1.5)

# ==== EASYOCR INIT ====
easyocr_reader = easyocr.Reader(
    ['en'],
    recog_network='ocr_ft',
    gpu=USE_GPU,
    model_storage_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\saved_models\en_filtered',
    user_network_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\user_network'
)

# ==== MONITOR & MODEL ====
display_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss.mss()
yolo = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo.names
PLATE_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'plate' in nm.lower()]
CAR_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'car' in nm.lower() and cid not in PLATE_CLASS_IDS]

# === Helpers ===
def is_inside(plate_box, car_box):
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cx1, cy1, cx2, cy2 = car_box
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2

def is_blurry(image, threshold=150.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def preprocess_plate(crop_bgr):
    h, w = crop_bgr.shape[:2]
    scale = 200 / max(h, 1)
    resized = cv2.resize(crop_bgr, (int(w * scale), 200), interpolation=cv2.INTER_CUBIC)
    if not USE_PREPROCESS:
        return resized
    return enhance_low_light_image(resized) if USE_LOW_LIGHT_ENHANCE else resized

def ocr_plate(crop_bgr):
    prep = preprocess_plate(crop_bgr)
    if DEBUG_OCR_CROP:
        cv2.imshow('OCR Crop', prep)
        cv2.waitKey(1)
    text_list = easyocr_reader.readtext(prep, detail=0)
    raw = ''.join(text_list).strip().upper()
    return ''.join(ch for ch in raw if ch in string.digits + string.ascii_uppercase)

def clean_plate_text(text):
    text = ''.join(ch for ch in text if ch in string.ascii_uppercase + string.digits)
    if text and text[0] in string.digits:
        return ""
    match = re.match(r'^([A-Z]+)?(\d{1,4})([A-Z]+)?$', text)
    if not match:
        return ""
    prefix, digits, suffix = match.groups()
    return (prefix or "") + digits + (suffix or "") if (prefix or suffix) else ""

# ==== MAIN LOOP ====
if __name__ == '__main__':
    print(f"OCR using fine-tuned EasyOCR on {'GPU' if USE_GPU else 'CPU'} | Classes={CLASS_NAMES}")
    cv2.namedWindow('Pipeline', cv2.WINDOW_NORMAL)
    frame_idx = 0

    while True:
        img = np.array(sct.grab(display_monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = yolo(frame, stream=True, conf=DETECTION_CONF)
        detections = [
            {'cls': int(b.cls[0]), 'label': CLASS_NAMES[int(b.cls[0])], 'conf': float(b.conf[0]), 'box': tuple(map(int, b.xyxy[0]))}
            for r in results for b in r.boxes
        ]

        car_boxes = [d['box'] for d in detections if d['cls'] in CAR_CLASS_IDS]
        plates = [d for d in detections if d['cls'] in PLATE_CLASS_IDS]
        valid = [p for p in plates if not USE_CAR_FILTER or any(is_inside(p['box'], cb) for cb in car_boxes)]

        for idx, p in enumerate(valid):
            x1, y1, x2, y2 = p['box']
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                continue
            dirty_txt = ocr_plate(crop)
            txt = clean_plate_text(dirty_txt)
            if not txt:
                continue
            box_id = (x1, y1, x2, y2)
            ocr_buffers[box_id].append(txt)
            most_common, count = Counter(ocr_buffers[box_id]).most_common(1)[0]
            final_text = most_common if count >= 5 else ""

            print(f"[Frame {frame_idx}] Text: {txt} | Blur: {cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var():.1f}")
            if SAVE_CROPS:
                cv2.imwrite(os.path.join('plate_crops', f"crop_{frame_idx:05d}_{idx}.png"), crop)
            cv2.putText(frame, final_text, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        frame_idx += 1
        cv2.imshow('Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()