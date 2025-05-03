import os
import cv2
import tensorflow.compat.v1 as tf
import numpy as np

from NightImageEnhancer.retinex_enhancer import RetinexEnhancer

tf.disable_v2_behavior()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image_path = os.path.join(script_dir, 'AJE631.jpg')
    output_image_path = os.path.join(script_dir, 'enhanced_test2.jpg')
    checkpoint_dir = os.path.join(script_dir, 'checkpoint')

    if not os.path.exists(input_image_path):
        print(f"[❌] File not found: {input_image_path}")
        return

    img = cv2.imread(input_image_path)
    h, w = img.shape[:2]

    enhancer = RetinexEnhancer(checkpoint_dir=checkpoint_dir, use_gpu=False)

    upscale_factor = 2
    img_upscaled = cv2.resize(img, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)
    enhanced_upscaled = enhancer.enhance(img_upscaled)
    enhanced_final = cv2.resize(enhanced_upscaled, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_image_path, enhanced_final)
    print(f"✅ Enhanced image saved at: {output_image_path}")

    enhancer.close()

if __name__ == "__main__":
    main()
