import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import easyocr
import matplotlib.pyplot as plt


# 🧠 Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(current_dir, "crop_00017_0.png")
image_path = r"C:\Users\User\Downloads\test\Plate_Number_(QAB_4838_E).jpg"

# 🔍 OCR Prediction Function
def easyocr_predict_single(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, None

    result = reader.readtext(image, detail=0)
    predicted_text = ''.join(result).strip().replace(" ", "").upper()
    return predicted_text, image

# 🚀 Predict
predicted_text, image = easyocr_predict_single(image_path)

if predicted_text is not None:
    print(f"🔎 Predicted Text: {predicted_text}")

    # 🖼️ Show Image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 4))
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {predicted_text}", fontsize=12)
    plt.axis('off')
    plt.show()
else:
    print("No prediction could be made.")
