import cv2
import os
import tensorflow.compat.v1 as tf
import numpy as np

from retinex_enhancer import RetinexEnhancer

tf.disable_v2_behavior()

def main():
    # === Set the input and output paths ===
    input_image_path = 'ADD6379.jpg'  
    output_image_path = 'enhanced_ADD6379.jpg'  
    checkpoint_dir = './checkpoint'  

    # === Load image ===
    if not os.path.exists(input_image_path):
        print(f"[❌] File not found: {input_image_path}")
        return

    img = cv2.imread(input_image_path)
    h, w = img.shape[:2]

    # === Initialize Retinex Enhancer ===
    enhancer = RetinexEnhancer(checkpoint_dir=checkpoint_dir, use_gpu=False)

    # === Upscale image (2x) to prevent distortion ===
    upscale_factor = 2
    img_upscaled = cv2.resize(img, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)

    # === Enhance ===
    enhanced_upscaled = enhancer.enhance(img_upscaled)

    # === Downscale back to original size ===
    enhanced_final = cv2.resize(enhanced_upscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    # === Save enhanced image ===
    cv2.imwrite(output_image_path, enhanced_final)
    print(f"✅ Enhanced image saved at: {output_image_path}")

    # === Close TensorFlow session ===
    enhancer.close()

if __name__ == "__main__":
    main()