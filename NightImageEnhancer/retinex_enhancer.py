import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import os

from model import lowlight_enhance   # <-- Import CLASS directly from model.py (this is correct)
from utils import load_images

tf.disable_v2_behavior()

class RetinexEnhancer:
    def __init__(self, checkpoint_dir='./checkpoint', use_gpu=False):
        self.checkpoint_dir = checkpoint_dir
        self.use_gpu = use_gpu
        self._build_session()

    def _build_session(self):
        if self.use_gpu:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()

        # === Initialize model
        self.model = lowlight_enhance(self.sess)   # <-- Correct: create model object

        # === Load DecomNet weights
        saver_decom = tf.train.Saver(var_list=[var for var in tf.global_variables() if 'DecomNet' in var.name])
        decom_checkpoint = os.path.join(self.checkpoint_dir, 'Decom', 'RetinexNet-tensorflow')
        saver_decom.restore(self.sess, decom_checkpoint)

        # === Load RelightNet weights
        saver_relight = tf.train.Saver(var_list=[var for var in tf.global_variables() if 'RelightNet' in var.name])
        relight_checkpoint = os.path.join(self.checkpoint_dir, 'Relight', 'RetinexNet-tensorflow')
        saver_relight.restore(self.sess, relight_checkpoint)

        print("✅ RetinexNet model loaded successfully.")

    def enhance(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        output = self.model.inference(img)
        output = np.squeeze(output)
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def close(self):
        self.sess.close()
