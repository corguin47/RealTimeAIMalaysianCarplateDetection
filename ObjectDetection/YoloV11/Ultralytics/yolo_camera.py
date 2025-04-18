from ultralytics import YOLO
import cv2
import numpy as np
import mss

# Load your trained model
model = YOLO(r"C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\Models\YoloV11n_trained.pt")  # adjust path if needed

# Define the screen capture area (you can adjust this to your monitor size or region)
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

sct = mss.mss()

while True:
    # Capture screen
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)

    # Convert BGRA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Run YOLO detection
    results = model(frame, stream=True, conf=0.25)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Screen Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
