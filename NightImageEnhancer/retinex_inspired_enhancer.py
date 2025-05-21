import cv2
import numpy as np

# ===============================
# Retinex Enhancement (Simple SSR Version)
# ===============================
def retinex_enhance(img, sigma=30):
    img = img.astype(np.float32) + 1.0  # Avoid log(0)
    log_img = np.log(img)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    log_blur = np.log(blur + 1.0)
    retinex = log_img - log_blur
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(retinex)

# ===============================
# CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)
# ===============================
def clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8)):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

# ===============================
# Gamma Correction (for Brightening)
# ===============================
def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ===============================
# Combo Enhancement
# ===============================
def enhance_low_light_image(img):
    img = retinex_enhance(img, sigma=30)
    img = clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8))
    img = gamma_correction(img, gamma=1.5)
    return img

# ===============================
# Brightness Metrics
# ===============================
def compute_brightness_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    max_brightness = np.max(gray)
    min_brightness = np.min(gray)
    return mean_brightness, std_brightness, max_brightness, min_brightness

# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    input_path = r"CarPlateForOCR\Dataset\test\SYD9391.png"

    img = cv2.imread(input_path)
    enhanced_img = enhance_low_light_image(img)

    mean_orig, std_orig, max_orig, min_orig = compute_brightness_metrics(img)
    mean_enh, std_enh, max_enh, min_enh = compute_brightness_metrics(enhanced_img)

    print("--- Original Image Brightness ---")
    print(f"Mean: {mean_orig:.2f}, Std: {std_orig:.2f}, Max: {max_orig}, Min: {min_orig}")

    print("\n--- Enhanced Image Brightness ---")
    print(f"Mean: {mean_enh:.2f}, Std: {std_enh:.2f}, Max: {max_enh}, Min: {min_enh}")

    cv2.imshow("Original", img)
    cv2.imshow("Enhanced", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
