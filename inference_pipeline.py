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
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import csv

# ==== CONFIGURATION ====
USE_GPU = torch.cuda.is_available()
YOLO_MODEL_PATH = r"D:\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\Models\yolo11n_finetuned.pt"
DETECTION_CONF = 0.25
SAVE_CROPS = False
USE_CAR_FILTER = False
USE_PREPROCESS = True
DEBUG_OCR_CROP = True
USE_LOW_LIGHT_ENHANCE = True

# Globals
gamma_value = 1.2
ocr_buffers = defaultdict(lambda: deque(maxlen=3))
stable_predictions = {}

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

# OCR setup
easyocr_reader = easyocr.Reader(
    ['en'],
    recog_network='ocr_ft',
    gpu=USE_GPU,
    model_storage_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\saved_models\en_filtered',
    user_network_directory=r'D:\RealTimeAIMalaysianCarplateDetection\CarPlateForOCR\FirstApproach\user_network'
)

# Load YOLO model
yolo = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = yolo.names
PLATE_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'plate' in nm.lower()]
CAR_CLASS_IDS = [cid for cid, nm in CLASS_NAMES.items() if 'car' in nm.lower() and cid not in PLATE_CLASS_IDS]

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

def process_frame(frame, frame_idx=0, use_buffer=True):
    detections = yolo(frame, stream=True, conf=DETECTION_CONF)
    results = [
        {'cls': int(b.cls[0]), 'label': CLASS_NAMES[int(b.cls[0])], 'conf': float(b.conf[0]), 'box': tuple(map(int, b.xyxy[0]))}
        for r in detections for b in r.boxes
    ]
    car_boxes = [d['box'] for d in results if d['cls'] in CAR_CLASS_IDS]
    plates = [d for d in results if d['cls'] in PLATE_CLASS_IDS]
    valid = [p for p in plates if not USE_CAR_FILTER or any(is_inside(p['box'], cb) for cb in car_boxes)]

    predictions = []
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
        if use_buffer:
            ocr_buffers[box_id].append(txt)
            most_common, count = Counter(ocr_buffers[box_id]).most_common(1)[0]
            if count >= 3:
                stable_predictions[box_id] = most_common
            final_text = stable_predictions.get(box_id, txt)
        else:
            final_text = txt
        predictions.append((f"{x1},{y1},{x2},{y2}", final_text))
        cv2.putText(frame, final_text, (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
    return frame, predictions

def process_folder():
    folder = filedialog.askdirectory(title="Select Folder of Images")
    if not folder:
        return
    output_path = os.path.join(folder, "predictions.txt")
    with open(output_path, 'w') as f:
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)
                _, predictions = process_frame(img, use_buffer=False)
                for box_str, pred in predictions:
                    f.write(f"{fname},{box_str},{pred}\n")
    print(f"Saved predictions to: {output_path}")

def process_image():
    path = filedialog.askopenfilename(title="Select an image")
    if not path:
        return
    img = cv2.imread(path)
    result, predictions = process_frame(img, use_buffer=False)
    cv2.imshow("Image Result", result)
    print("Predictions:", predictions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video():
    path = filedialog.askopenfilename(title="Select a video")
    if not path:
        return
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result, _ = process_frame(frame)
        resized_result = cv2.resize(result, (960, 540))
        cv2.imshow("Video Result", resized_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_realtime():
    display_monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
    sct = mss.mss()
    cv2.namedWindow('Pipeline', cv2.WINDOW_NORMAL)
    frame_idx = 0
    while True:
        img = np.array(sct.grab(display_monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        result, _ = process_frame(frame, frame_idx)
        frame_idx += 1
        cv2.imshow('Pipeline', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def process_folder_with_verification():
    folder = filedialog.askdirectory(title="Select Folder of Images")
    if not folder:
        return
    predictions = []
    
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            display_img, preds = process_frame(img, use_buffer=False)

            for (box_str, pred_text) in preds:
                x1, y1, x2, y2 = map(int, box_str.split(','))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, pred_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Prediction Review', display_img)
            print(f"Image: {fname} Prediction: {pred_text}")
            key = cv2.waitKey(0)

            if key == ord('y'):
                correctness = 'Correct'
            elif key == ord('n'):
                correctness = 'Incorrect'
            else:
                correctness = 'Skipped'

            for (_, pred_text) in preds:
                predictions.append([fname, pred_text, correctness])
    cv2.destroyAllWindows()

    output_csv = os.path.join(folder, "manual_verification_results.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prediction", "Correctness"])
        writer.writerows(predictions)

    total = len([row for row in predictions if row[2] != 'Skipped'])
    correct = len([row for row in predictions if row[2] == 'Correct'])
    acc = correct / total if total > 0 else 0.0
    print(f"Manual Accuracy: {acc:.2%} ({correct}/{total})")

# === Updated GUI Entry ===
def main_gui():
    def on_select(event=None):
        mode = combo.get()
        if mode.lower() == 'image':
            process_image()
        elif mode.lower() == 'video':
            process_video()
        elif mode.lower() == 'realtime':
            process_realtime()
        elif mode.lower() == 'folder':
            process_folder_with_verification()
        combo.set('Select mode')

    def on_quit():
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title("Detection Mode Selector")

    label = tk.Label(root, text="Select detection mode:")
    label.pack(pady=10)

    combo = ttk.Combobox(root, values=["Image", "Video", "Realtime", "Folder"], state="readonly")
    combo.pack(padx=20, pady=5)
    combo.bind("<<ComboboxSelected>>", on_select)

    quit_button = tk.Button(root, text="Quit", command=on_quit)
    quit_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    while True:
        main_gui()
        break

