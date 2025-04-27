import os
import string
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import torch
import numpy as np
import mss
from ultralytics import YOLO
import easyocr

# ==== CONFIGURATION ====
USE_GPU = torch.cuda.is_available()
YOLO_MODEL_PATH = r"C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\Models\YoloV11n_trained.pt"
DETECTION_CONF = 0.25
# Toggles
SAVE_CROPS = False       # Save raw plate crops?
USE_CAR_FILTER = False    # Require plate to be inside a car?
USE_PREPROCESS = True    # Apply color preprocessing before OCR
DEBUG_OCR_CROP = True   # Show the final preprocessed crops
# Gamma correction parameter
gamma_value = 1.2       # >1 brightens mid-tones, <1 darkens

# Initialize EasyOCR reader
easyocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)
# Screen capture region
display_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss.mss()

# Load YOLOv11 model
yolo = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo.names
# Class ID lists
PLATE_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'plate' in nm.lower()]
CAR_CLASS_IDS   = [cid for cid, nm in CLASS_NAMES.items() if 'car' in nm.lower() and cid not in PLATE_CLASS_IDS]

# Prepare folder for raw crops
CROP_FOLDER = 'plate_crops'
if SAVE_CROPS:
    os.makedirs(CROP_FOLDER, exist_ok=True)
frame_idx = 0

# Helper: check if center of plate is inside car box
def is_inside(plate_box, car_box):
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cx1, cy1, cx2, cy2 = car_box
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2

# Preprocess plate crop: color enhancement + normalization + gamma
def preprocess_plate(crop_bgr):
    # Resize to fixed height 200px, maintain aspect ratio of original cropped image
    h, w = crop_bgr.shape[:2]
    target_h = 200
    scale = target_h / max(h, 1)
    new_w = int(w * scale)
    resized = cv2.resize(crop_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    if not USE_PREPROCESS:
        return resized
    # Convert to LAB (lightness, green-red channel, blue-yellow channel) and apply CLAHE (Constrast-Limited Adaptive Historgram Equalisation) on L-channel (boost constract wihout losing colour) 
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    # Unsharp mask for sharpening
    blur = cv2.GaussianBlur(enhanced, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    # Contrast stretching normalization
    normed = cv2.normalize(sharp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Gamma correction
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype('uint8')
    gamma_corrected = cv2.LUT(normed, table)
    return gamma_corrected

# OCR function that filters to only digits and uppercase letters
def ocr_plate(crop_bgr):
    prep = preprocess_plate(crop_bgr)
    if DEBUG_OCR_CROP:
        cv2.imshow('OCR Crop', prep)
        cv2.waitKey(1)
    text_list = easyocr_reader.readtext(prep, detail=0)
    raw = ''.join(text_list).strip().upper()
    filtered = ''.join(ch for ch in raw if ch in string.digits + string.ascii_uppercase)
    return filtered

if __name__ == '__main__':
    print(f"OCR on {'GPU' if USE_GPU else 'CPU'} | Classes={CLASS_NAMES}")
    cv2.namedWindow('Pipeline', cv2.WINDOW_NORMAL)
    while True:
        # 1) Capture screen
        img = np.array(sct.grab(display_monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # 2) YOLO inference
        results = yolo(frame, stream=True, conf=DETECTION_CONF)
        detections = []
        for r in results:
            for b in r.boxes:
                cid = int(b.cls[0]); cf = float(b.conf[0])
                lbl = CLASS_NAMES[cid]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append({'cls': cid, 'label': lbl, 'conf': cf, 'box': (x1, y1, x2, y2)})
        # Draw detections
        for d in detections:
            x1, y1, x2, y2 = d['box']
            color = (0,255,0) if d['cls'] not in PLATE_CLASS_IDS else (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # 3) Filter plates in cars and OCR
        car_boxes = [d['box'] for d in detections if d['cls'] in CAR_CLASS_IDS]
        plates = [d for d in detections if d['cls'] in PLATE_CLASS_IDS]
        valid = [p for p in plates if (not USE_CAR_FILTER) or any(is_inside(p['box'], cb) for cb in car_boxes)]
        for idx, p in enumerate(valid):
            x1, y1, x2, y2 = p['box']
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            txt = ocr_plate(crop)
            if SAVE_CROPS:
                cv2.imwrite(os.path.join(CROP_FOLDER, f"crop_{frame_idx:05d}_{idx}.png"), crop)
            cv2.putText(frame, txt, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        frame_idx += 1
        # Display
        cv2.imshow('Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
