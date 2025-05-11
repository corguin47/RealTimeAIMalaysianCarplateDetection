import os
import string
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import torch
import numpy as np
import mss
from ultralytics import YOLO
import easyocr
from collections import deque, Counter
from collections import defaultdict
import re
import string

# ==== CONFIGURATION ====
USE_GPU = torch.cuda.is_available()
YOLO_MODEL_PATH = r"D:\RealTimeAIMalaysianCarplateDetection\yolo11n.yaml.pt"
DETECTION_CONF = 0.25

# Toggles
SAVE_CROPS = False       # Save raw plate crops?
USE_CAR_FILTER = False   # Require plate to be inside a car?
USE_PREPROCESS = True    # Apply color preprocessing before OCR
DEBUG_OCR_CROP = True    # Show the final preprocessed crops
gamma_value = 1.2        # Gamma correction
ocr_buffers = defaultdict(lambda: deque(maxlen=7))

# Initialize EasyOCR with fine-tuned model
easyocr_reader = easyocr.Reader(
    ['en'],
    recog_network='ocr_ft',
    gpu=USE_GPU,
    model_storage_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\saved_models\en_filtered',
    user_network_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\user_network'
)

# Screen capture region
display_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
sct = mss.mss()

# Load YOLOv11 model
yolo = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo.names
PLATE_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'plate' in nm.lower()]
CAR_CLASS_IDS   = [cid for cid, nm in CLASS_NAMES.items() if 'car' in nm.lower() and cid not in PLATE_CLASS_IDS]

# Save crops folder
CROP_FOLDER = 'plate_crops'
if SAVE_CROPS:
    os.makedirs(CROP_FOLDER, exist_ok=True)
frame_idx = 0


# Check if plate is inside car box
def is_inside(plate_box, car_box):
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cx1, cy1, cx2, cy2 = car_box
    return cx1 <= cx <= cx2 and cy1 <= cy <= cy2

def is_blurry(image, threshold=150.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

# Preprocess plate crop
def preprocess_plate(crop_bgr):
    h, w = crop_bgr.shape[:2]
    target_h = 200
    scale = target_h / max(h, 1)
    new_w = int(w * scale)
    resized = cv2.resize(crop_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    if not USE_PREPROCESS:
        return resized

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    normed = cv2.normalize(sharp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype('uint8')
    gamma_corrected = cv2.LUT(normed, table)

    return gamma_corrected

# OCR using fine-tuned EasyOCR
def ocr_plate(crop_bgr):
    prep = preprocess_plate(crop_bgr)
    if DEBUG_OCR_CROP:
        cv2.imshow('OCR Crop', prep)
        cv2.waitKey(1)

    text_list = easyocr_reader.readtext(prep, detail=0)
    raw = ''.join(text_list).strip().upper()
    filtered = ''.join(ch for ch in raw if ch in string.digits + string.ascii_uppercase)
    return filtered

# Only one digit block (1–4 digits)
# Must have at least one letter (prefix or suffix)
# Cannot start with a digit
# Technically not law, but common format in Malaysia
def clean_plate_text(text):
    # Strip unwanted characters
    text = ''.join(ch for ch in text if ch in string.ascii_uppercase + string.digits)

    # Reject if first character is a digit
    if text and text[0] in string.digits:
        return ""

    # Match: optional letters + 1 block of digits + optional letters
    match = re.match(r'^([A-Z]+)?(\d{1,4})([A-Z]+)?$', text)
    if not match:
        return ""

    prefix, digits, suffix = match.groups()

    # Require at least one letter (prefix or suffix)
    if prefix or suffix:
        return (prefix or "") + digits + (suffix or "")

    return ""

# Main loop
if __name__ == '__main__':
    print(f"OCR using fine-tuned EasyOCR on {'GPU' if USE_GPU else 'CPU'} | Classes={CLASS_NAMES}")
    cv2.namedWindow('Pipeline', cv2.WINDOW_NORMAL)

    while True:
        img = np.array(sct.grab(display_monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = yolo(frame, stream=True, conf=DETECTION_CONF)
        detections = []
        for r in results:
            for b in r.boxes:
                cid = int(b.cls[0]); cf = float(b.conf[0])
                lbl = CLASS_NAMES[cid]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append({'cls': cid, 'label': lbl, 'conf': cf, 'box': (x1, y1, x2, y2)})

        # Draw boxes
        for d in detections:
            x1, y1, x2, y2 = d['box']
            color = (0,255,0) if d['cls'] not in PLATE_CLASS_IDS else (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        car_boxes = [d['box'] for d in detections if d['cls'] in CAR_CLASS_IDS]
        plates = [d for d in detections if d['cls'] in PLATE_CLASS_IDS]
        valid = [p for p in plates if (not USE_CAR_FILTER) or any(is_inside(p['box'], cb) for cb in car_boxes)]

        for idx, p in enumerate(valid):
            x1, y1, x2, y2 = p['box']
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                continue
            dirty_txt = ocr_plate(crop)
            txt = clean_plate_text(dirty_txt)
            
            if not txt:
                continue  # Skip invalid format
            
            box_id = (x1, y1, x2, y2)  # crude box identifier
            ocr_buffers[box_id].append(txt)
            
            # Majority vote
            most_common, count = Counter(ocr_buffers[box_id]).most_common(1)[0]
            if count >= 5:
                final_text = most_common
            else:
                final_text = ""  # or keep last stable result
                
            blur_level = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            print(f"[Frame {frame_idx}] Text: {txt} | Blur: {blur_level:.1f}")
            if SAVE_CROPS:
                cv2.imwrite(os.path.join(CROP_FOLDER, f"crop_{frame_idx:05d}_{idx}.png"), crop)
            cv2.putText(frame, txt, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        frame_idx += 1
        cv2.imshow('Pipeline', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
