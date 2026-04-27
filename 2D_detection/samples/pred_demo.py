'''
Example of model prediction;
refer to the predict_eval folder for the specific implementation.
'''

import os
import sys
import numpy as np
import skimage.io
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config


class PlaquesConfig(Config):
    """Configuration consistent with the training"""
    NAME = "plaques"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE = "resnet101"
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([127.5])
    USE_MASK = False
    USE_MINI_MASK = False
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    RPN_ANCHOR_SCALES = (6, 12, 24, 48, 96)

    RPN_NMS_THRESHOLD = 0.7
    TOP_DOWN_PYRAMID_SIZE = 256

    DETECTION_MAX_INSTANCES = 300

    PRE_NMS_LIMIT = 1000
    POST_NMS_ROIS_INFERENCE = 300



class InferenceConfig(PlaquesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1


MODEL_DIR = "../"
MODEL_PATH = os.path.join(MODEL_DIR, "2D_plaque_detection.h5")
IMAGE_DIR = "../image/val"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "evaluation")


def load_image_as_grayscale(image_path):
    try:
        image = skimage.io.imread(image_path)

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = skimage.color.rgb2gray(image[:, :, :3])
            image = (image * 255).astype(np.uint8)

        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = skimage.color.rgb2gray(image)
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3:
            image = image[:, :, 0]

        # [height, width] -> [height, width, 1]
        image = np.expand_dims(image, axis=-1)

        return image

    except Exception as e:
        print(f"  Error loading image: {e}")
        return None


def main():
    print("=" * 70)
    print("Mask R-CNN Evaluation")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} images")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = InferenceConfig()
    config.display()

    print("\nLoading model...")
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)
    print("Model loaded successfully!")

    success_count = 0
    for idx, file_name in enumerate(image_files):
        print(f"\n[{idx + 1}/{len(image_files)}] {file_name}")

        try:
            image_path = os.path.join(IMAGE_DIR, file_name)
            image = load_image_as_grayscale(image_path)

            if image is None:
                continue

            print(f"  Image shape: {image.shape}")  # (512, 512, 1)

            results = model.detect([image], verbose=0)
            r = results[0]

            print(f"  Detected {len(r['class_ids'])} objects")

            image_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(OUTPUT_DIR, f'{image_name}.txt')

            if len(r['class_ids']) > 0:
                output_data = []
                for j in range(len(r['class_ids'])):
                    y1, x1, y2, x2 = r['rois'][j]
                    output_data.append({
                        "num": j, "start": 1,
                        "x_min": int(x1), "y_min": int(y1),
                        "x_max": int(x2), "y_max": int(y2),
                        "class_id": int(r['class_ids'][j]),
                        "score": float(r['scores'][j]), "end": -1
                    })
                pd.DataFrame(output_data).to_csv(output_path, sep=' ', index=False, header=False)
                print(f"  Saved {len(output_data)} detections")
            else:
                pd.DataFrame().to_csv(output_path, sep=' ', index=False, header=False)
                print(f"  No objects detected")

            success_count += 1

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompleted: {success_count}/{len(image_files)} images")


if __name__ == "__main__":
    main()