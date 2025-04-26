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

    # Normalize to 0-255
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    retinex = np.uint8(retinex)
    return retinex

# ===============================
# CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)
# ===============================
def clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8)):
    if len(img.shape) == 3:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe.apply(img)
    return enhanced_img

# ===============================
# Gamma Correction (for Brightening)
# ===============================
def gamma_correction(img, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ===============================
# Combo Enhancement (Retinex + CLAHE + Gamma)
# ===============================

def enhance_low_light_image(img):
    # Step 1: Retinex Enhancement
    img = retinex_enhance(img, sigma=30)
    
    # Step 2: CLAHE
    img = clahe_enhance(img, clip_limit=2.0, tile_grid_size=(8,8))
    
    # Step 3: Gamma Correction
    img = gamma_correction(img, gamma=1.5)
    
    return img

# ===============================
# Example Usage
# ===============================
if __name__ == "__main__":
    input_path = "AJE631.jpg"
    output_path = "enhanced_output1.jpg"

    img = cv2.imread(input_path)
    enhanced_img = enhance_low_light_image(img)

    cv2.imwrite(output_path, enhanced_img)
    cv2.imshow("Enhanced Image", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
