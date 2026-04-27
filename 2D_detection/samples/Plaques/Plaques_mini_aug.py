'''
Training on a small dataset.

The original intention was to compare the training performance of the mini-batch
using the original data versus the augmented data.
'''

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import pandas as pd
import tensorflow as tf
from datetime import datetime
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize_old2
from mrcnn.visualize_old2 import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "2D_method_plaques.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULT_DIR = DEFAULT_LOGS_DIR
loss_all = []
EPOCH_SET = 100
# EPOCH_INITIAL = 0
MODEL_SET = "none"
MODE_SET = "train"

dataset_dir = "../aug/image/"
# model_test_dir = os.path.join(ROOT_DIR, "2D_method_plaques_pre.h5")

class PlaquesConfig(Config):
    """Optimised configuration for small data sets"""

    NAME = "plaques"

    USE_PER_LEVEL_ENHANCEMENT = True
    USE_TOP_LEVEL_FUSION = True
    USE_MASK = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Training each image separately on a small dataset

    # Training step count settings (significantly reduced)
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20

    BACKBONE = "resnet50"  # ResNet-50 is faster; ResNet-101 is not required for small datasets

    RPN_ANCHOR_SCALES = (6, 12, 24, 48, 96)

    # Learning rate configuration (decrease)
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.001
    TRAIN_BN = False  # Do not train the BN layer with small batches

    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128
    PRE_NMS_LIMIT = 3000
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 300

    TRAIN_ROIS_PER_IMAGE = 100
    ROI_POSITIVE_RATIO = 0.5

    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.4

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 1

    GRADIENT_CLIP_NORM = 2.0

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.5,
        "mrcnn_mask_loss": 0.
    }

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class PlaquesDataset_old(utils.Dataset):
    def __init__(self):
        super().__init__()
        self._image_cache = {}
        self._mask_cache = {}
        self._box_cache = {}

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def load_plaques(self, dataset_dir, subset):
        """Load a subset of the dataset."""
        # Add class
        self.add_class("plaques", 1, "object")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        json_path = os.path.join(dataset_dir, "instances.json")
        print(f"\nLoading {subset} dataset from: {json_path}")


        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get("images", [])

        images_added = 0
        images_with_bbox = 0
        total_bboxes = 0

        valid_images = []
        for img_data in images:
            file_name = img_data.get("file_name")

            if not file_name:
                continue

            image_path = os.path.join(dataset_dir, file_name)

            if not os.path.exists(image_path):
                continue

            annotations = img_data.get("annotations", [])

            polygons = []
            original_class_ids = []

            for ann in annotations:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        polygons.append(bbox)
                        original_class_ids.append(ann.get("category_id", 1))

            if len(polygons) > 0:
                images_with_bbox += 1
                total_bboxes += len(polygons)

            width = int(img_data.get("width", 512))
            height = int(img_data.get("height", 512))

            valid_images.append({
                'file_name': file_name,
                'image_path': image_path,
                'width': width,
                'height': height,
                'polygons': polygons,
                'original_class_ids': original_class_ids
            })

        for img_info in valid_images:
            self.add_image(
                "plaques",
                image_id=img_info['file_name'],
                path=img_info['image_path'],
                original_class_ids=img_info['original_class_ids'],
                width=img_info['width'],
                height=img_info['height'],
                polygons=img_info['polygons']
            )
            images_added += 1

        print(f"  Successfully added {images_added} images")
        print(f"  Images with bounding boxes: {images_with_bbox}")
        print(f"  Total bounding boxes: {total_bboxes}")
        self._image_cache.clear()
        self._mask_cache.clear()
        self._box_cache.clear()

    def load_image(self, image_id):
        if image_id in self._image_cache:
            return self._image_cache[image_id]
        image = super().load_image(image_id)
        self._image_cache[image_id] = image
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        if image_id in self._mask_cache:
            return self._mask_cache[image_id]

        image_info = self.image_info[image_id]

        if image_info["source"] != "plaques":
            result = super(self.__class__, self).load_mask(image_id)
            self._mask_cache[image_id] = result
            return result

        if not config.USE_MASK:
            num_instances = len(image_info["polygons"])
            mask = np.zeros([image_info["height"], image_info["width"], 0], dtype=np.uint8)
            class_ids = np.array([], dtype=np.int32)
            result = (mask.astype(np.bool), class_ids)
            self._mask_cache[image_id] = result
            return result

        if "original_class_ids" in image_info:
            class_ids = np.array(image_info["original_class_ids"], dtype=np.int32)
        else:
            class_ids = np.ones([len(image_info["polygons"])], dtype=np.int32)

        info = self.image_info[image_id]
        num_polygons = len(info["polygons"])

        if num_polygons == 0:
            result = (np.zeros([info["height"], info["width"], 0], dtype=np.bool),
                      np.array([], dtype=np.int32))
            self._mask_cache[image_id] = result
            return result

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # p is [x, y, width, height]
            x, y, w, h = p

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(info["width"], x1 + int(w))
            y2 = min(info["height"], y1 + int(h))

            if x2 > x1 and y2 > y1:
                rr, cc = skimage.draw.rectangle((y1, x1), extent=(y2 - y1, x2 - x1))
                rr = np.clip(rr, 0, info["height"] - 1)
                cc = np.clip(cc, 0, info["width"] - 1)
                mask[rr, cc, i] = 1

        result = (mask.astype(np.bool), class_ids)
        self._mask_cache[image_id] = result
        return result

    def load_box(self, image_id):
        """Generate bounding boxes for an image."""
        if image_id in self._box_cache:
            return self._box_cache[image_id]

        image_info = self.image_info[image_id]
        if image_info["source"] != "plaques":
            result = super(self.__class__, self).load_mask(image_id)
            self._box_cache[image_id] = result
            return result

        num_instances = len(image_info["polygons"])
        if num_instances == 0:
            result = (np.zeros([4, 0], dtype=np.int32), np.array([], dtype=np.int32))
            self._box_cache[image_id] = result
            return result

        class_ids = np.ones([num_instances], dtype=np.int32)
        info = self.image_info[image_id]
        box = np.zeros([4, num_instances], dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            x, y, w, h = p

            x1 = max(1, int(x))
            y1 = max(1, int(y))
            x2 = min(info["width"] - 1, x1 + int(w))
            y2 = min(info["height"] - 1, y1 + int(h))

            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1

            box[:, i] = [y1, x1, y2, x2]

        result = (box, class_ids)
        self._box_cache[image_id] = result
        return result

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "plaques":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

    def clear_cache(self):
        self._image_cache.clear()
        self._mask_cache.clear()
        self._box_cache.clear()


class PlaquesDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self._image_cache = {}
        self._mask_cache = {}
        self._box_cache = {}

        self._preloaded_boxes = []
        self._preloaded_class_ids = []
        self._preloaded_image_shapes = []
        self._preloaded_polygons_count = []

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def load_plaques(self, dataset_dir, subset):
        """Load a subset of the dataset."""
        # Add class
        self.add_class("plaques", 1, "object")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        json_path = os.path.join(dataset_dir, "instances.json")
        print(f"\nLoading {subset} dataset from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get("images", [])

        images_added = 0
        images_with_bbox = 0
        total_bboxes = 0

        valid_images = []
        for img_data in images:
            file_name = img_data.get("file_name")
            if not file_name:
                continue

            image_path = os.path.join(dataset_dir, file_name)
            if not os.path.exists(image_path):
                continue

            annotations = img_data.get("annotations", [])

            polygons = []
            original_class_ids = []

            for ann in annotations:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        polygons.append(bbox)
                        original_class_ids.append(ann.get("category_id", 1))

            if len(polygons) > 0:
                images_with_bbox += 1
                total_bboxes += len(polygons)

            width = int(img_data.get("width", 512))
            height = int(img_data.get("height", 512))

            valid_images.append({
                'file_name': file_name,
                'image_path': image_path,
                'width': width,
                'height': height,
                'polygons': polygons,
                'original_class_ids': original_class_ids
            })

        for idx, img_info in enumerate(valid_images):
            self.add_image(
                "plaques",
                image_id=img_info['file_name'],
                path=img_info['image_path'],
                original_class_ids=img_info['original_class_ids'],
                width=img_info['width'],
                height=img_info['height'],
                polygons=img_info['polygons']
            )
            images_added += 1

            polygons = img_info['polygons']
            num_instances = len(polygons)

            if num_instances == 0:
                self._preloaded_boxes.append(np.zeros((0, 4), dtype=np.int32))
                self._preloaded_class_ids.append(np.array([], dtype=np.int32))
                self._preloaded_polygons_count.append(0)
            else:
                boxes = np.zeros((num_instances, 4), dtype=np.int32)
                class_ids = np.ones([num_instances], dtype=np.int32)

                for i, p in enumerate(polygons):
                    x, y, w, h = p
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(img_info['width'], x1 + int(w))
                    y2 = min(img_info['height'], y1 + int(h))

                    if x2 <= x1:
                        x2 = x1 + 1
                    if y2 <= y1:
                        y2 = y1 + 1

                    boxes[i] = [y1, x1, y2, x2]

                self._preloaded_boxes.append(boxes)
                self._preloaded_class_ids.append(class_ids)
                self._preloaded_polygons_count.append(num_instances)

            self._preloaded_image_shapes.append((img_info['height'], img_info['width']))

        print(f"  Successfully added {images_added} images")
        print(f"  Images with bounding boxes: {images_with_bbox}")
        print(f"  Total bounding boxes: {total_bboxes}")
        print(f"  Preloaded {len(self._preloaded_boxes)} sets of bounding boxes")

        self._image_cache.clear()
        self._mask_cache.clear()
        self._box_cache.clear()

    def load_image(self, image_id):
        if image_id in self._image_cache:
            return self._image_cache[image_id]

        image = super().load_image(image_id)

        self._image_cache[image_id] = image
        return image

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        if image_id in self._mask_cache:
            return self._mask_cache[image_id]

        image_info = self.image_info[image_id]

        if image_info["source"] != "plaques":
            result = super(self.__class__, self).load_mask(image_id)
            self._mask_cache[image_id] = result
            return result

        result = (np.zeros([image_info["height"], image_info["width"], 0], dtype=np.bool),
                  np.array([], dtype=np.int32))
        self._mask_cache[image_id] = result
        return result

    def load_box(self, image_id):
        """Generate bounding boxes for an image"""
        if image_id in self._box_cache:
            return self._box_cache[image_id]

        image_info = self.image_info[image_id]
        if image_info["source"] != "plaques":
            result = super(self.__class__, self).load_mask(image_id)
            self._box_cache[image_id] = result
            return result

        boxes = self._preloaded_boxes[image_id].copy()
        class_ids = self._preloaded_class_ids[image_id].copy()

        if boxes.shape[0] > 0:
            boxes_transposed = boxes.T
            result = (boxes_transposed, class_ids)
        else:
            result = (np.zeros((4, 0), dtype=np.int32), class_ids)

        self._box_cache[image_id] = result
        return result

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "plaques":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

    def clear_cache(self):
        self._image_cache.clear()
        self._mask_cache.clear()
        self._box_cache.clear()

    def get_preloaded_boxes(self, image_id):
        return self._preloaded_boxes[image_id].copy()

    def get_preloaded_class_ids(self, image_id):
        return self._preloaded_class_ids[image_id].copy()

    def has_boxes(self, image_id):
        return self._preloaded_boxes[image_id].shape[0] > 0



def calculate_precision_recall_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))

    fp = np.sum((y_pred == 1) & (y_true == 0))

    fn = np.sum((y_pred == 0) & (y_true == 1))

    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return precision, recall, f1, accuracy


class ModelCheckpointManager:
    def __init__(self, model, save_dir, eval_interval=50, dataset_dir=None):
        self.model = model
        self.save_dir = save_dir
        self.eval_interval = eval_interval
        self.dataset_dir = dataset_dir
        self.best_loss_dir = os.path.join(save_dir, 'best_by_loss')
        self.best_f1_dir = os.path.join(save_dir, 'best_by_f1')
        self.checkpoints_dir = os.path.join(save_dir, 'checkpoints')

        for d in [self.best_loss_dir, self.best_f1_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)

        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.best_epoch_loss = 0
        self.best_epoch_f1 = 0

        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }

        self.dataset_val = PlaquesDataset()
        if self.dataset_dir:
            self.dataset_val.load_plaques(self.dataset_dir, "val")
            self.dataset_val.prepare()

        self.epoch_losses = []

    def update_loss(self, epoch, train_loss, val_loss):
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch_loss = epoch
            self.save_best_loss_model(epoch, val_loss)

    def compute_validation_metrics(self):
        all_preds = []
        all_labels = []

        val_size = len(self.dataset_val.image_info)
        print(f"  Evaluating on {val_size} validation images...")

        for i in range(val_size):
            image = self.dataset_val.load_image(i)

            image_info = self.dataset_val.image_info[i]
            original_class_ids = image_info.get('original_class_ids', [])
            has_gt = 1 if len(original_class_ids) > 0 else 0

            results = self.model.detect([image], verbose=0)[0]
            has_pred = 1 if len(results['class_ids']) > 0 else 0

            all_labels.append(has_gt)
            all_preds.append(has_pred)

        precision, recall, f1, accuracy = calculate_precision_recall_f1(all_labels, all_preds)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def save_best_loss_model(self, epoch, val_loss):
        model_path = os.path.join(
            self.best_loss_dir,
            f'best_loss_model_epoch{epoch:04d}_loss{val_loss:.4f}.h5'
        )
        self.model.keras_model.save_weights(model_path)

        latest_path = os.path.join(self.best_loss_dir, 'latest_best_loss.h5')
        shutil.copy(model_path, latest_path)

        info = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            'model_path': model_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        info_path = os.path.join(self.best_loss_dir, 'best_loss_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Best loss model updated!")
        print(f"  Epoch: {epoch}, Val Loss: {val_loss:.4f}")
        print(f"  Saved to: {model_path}")

    def save_history(self):
        history_df = pd.DataFrame(self.metrics_history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)

        summary = {
            'best_loss_model': {
                'epoch': self.best_epoch_loss,
                'val_loss': self.best_loss
            },
            'best_f1_model': {
                'epoch': self.best_epoch_f1,
                'f1_score': self.best_f1
            }
        }

        with open(os.path.join(self.save_dir, 'best_models_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def plot_history(self):
        if len(self.metrics_history['epoch']) < 2:
            return

        plt.style.use('ggplot')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss
        axes[0, 0].plot(self.metrics_history['epoch'],
                        self.metrics_history['train_loss'],
                        'r-', label='Train Loss', alpha=0.7)
        axes[0, 0].plot(self.metrics_history['epoch'],
                        self.metrics_history['val_loss'],
                        'b-', label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        if self.best_epoch_loss > 0:
            axes[0, 0].plot(self.best_epoch_loss, self.best_loss,
                            'ro', markersize=10, label='Best Loss')

        # 2. Precision & Recall
        valid_indices = [i for i, v in enumerate(self.metrics_history['precision']) if v is not None]
        if valid_indices:
            epochs_valid = [self.metrics_history['epoch'][i] for i in valid_indices]
            precision_valid = [self.metrics_history['precision'][i] for i in valid_indices]
            recall_valid = [self.metrics_history['recall'][i] for i in valid_indices]

            axes[0, 1].plot(epochs_valid, precision_valid, 'g-', label='Precision', marker='o')
            axes[0, 1].plot(epochs_valid, recall_valid, 'm-', label='Recall', marker='s')
            axes[0, 1].set_title('Precision & Recall')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. F1 Score
        if valid_indices:
            f1_valid = [self.metrics_history['f1_score'][i] for i in valid_indices]
            axes[1, 0].plot(epochs_valid, f1_valid, 'c-', label='F1 Score', linewidth=2, marker='o')
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            if self.best_epoch_f1 > 0 and self.best_f1 > 0:
                axes[1, 0].plot(self.best_epoch_f1, self.best_f1,
                                'ro', markersize=10, label='Best F1')

        # 4. Accuracy
        if valid_indices:
            acc_valid = [self.metrics_history['accuracy'][i] for i in valid_indices]
            axes[1, 1].plot(epochs_valid, acc_valid, 'orange', label='Accuracy', marker='o')
            axes[1, 1].set_title('Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=150)
        plt.close()

    def get_summary(self):
        summary = f"""
{'=' * 70}
Training Summary
{'=' * 70}
Best Model by Loss:
  - Epoch: {self.best_epoch_loss}
  - Validation Loss: {self.best_loss:.4f}
  - Model Path: {os.path.join(self.best_loss_dir, 'latest_best_loss.h5')}

Best Model by F1 Score:
  - Epoch: {self.best_epoch_f1}
  - F1 Score: {self.best_f1:.4f}
  - Model Path: {os.path.join(self.best_f1_dir, 'latest_best_f1.h5')}

All checkpoints saved to: {self.checkpoints_dir}
{'=' * 70}
"""
        return summary


def train_with_dual_saving(model):

    dataset_train = PlaquesDataset()
    dataset_train.load_plaques(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = PlaquesDataset()
    dataset_val.load_plaques(args.dataset, "val")
    dataset_val.prepare()

    print("\n" + "=" * 70)
    print("Preloading bounding boxes to memory...")
    print("=" * 70)

    for i in range(len(dataset_train.image_ids)):
        dataset_train.load_box(i)
        if (i + 1) % 1000 == 0:
            print(f"  Preloaded {i + 1}/{len(dataset_train.image_ids)} training images")

    for i in range(len(dataset_val.image_ids)):
        dataset_val.load_box(i)
        if (i + 1) % 1000 == 0:
            print(f"  Preloaded {i + 1}/{len(dataset_val.image_ids)} validation images")

    print("Preloading completed!")
    print("=" * 70)

    model.config.CLASS_NAMES = ['BG', 'object']

    checkpoint_manager = ModelCheckpointManager(
        model=model,
        save_dir=model.model_dir,
        eval_interval=30,
        dataset_dir=args.dataset
    )

    print("=" * 70)
    print("Training with Dual Model Saving Strategy")
    print(f"  Model save directory: {model.model_dir}")
    print(f"  F1 evaluation interval: {checkpoint_manager.eval_interval} epochs")
    print(f"  Best loss model will be saved when loss improves")
    print(f"  Best F1 model will be saved when F1 improves")
    print("=" * 70)

    print("\nStarting training...")
    print(f"BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"IMAGES_PER_GPU: {config.IMAGES_PER_GPU}")
    print(f"GPU_COUNT: {config.GPU_COUNT}")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCH_SET,
                layers='all')

    dataset_train.clear_cache()
    dataset_val.clear_cache()

    print("\n" + "=" * 70)
    print("Training completed! Evaluating checkpoints for best F1 model...")
    print("=" * 70)

    checkpoints = []
    logs_dir = model.model_dir

    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.startswith('mask_rcnn_plaques_') and file.endswith('.h5'):
                try:
                    epoch = int(file.split('_')[-1].split('.')[0])
                    checkpoints.append((epoch, os.path.join(root, file)))
                except:
                    continue

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        print(f"Found {len(checkpoints)} checkpoints")

        best_f1 = 0
        best_epoch = 0
        best_model_path = None

        for epoch, ckpt_path in checkpoints:
            if epoch % checkpoint_manager.eval_interval == 0 or epoch == checkpoints[-1][0]:
                print(f"\nEvaluating epoch {epoch}...")
                model.load_weights(ckpt_path, by_name=True)
                metrics = checkpoint_manager.compute_validation_metrics()

                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")

                checkpoint_manager.metrics_history['precision'].append(metrics['precision'])
                checkpoint_manager.metrics_history['recall'].append(metrics['recall'])
                checkpoint_manager.metrics_history['f1_score'].append(metrics['f1'])
                checkpoint_manager.metrics_history['accuracy'].append(metrics['accuracy'])

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_epoch = epoch
                    best_model_path = ckpt_path

        if best_model_path:
            os.makedirs(checkpoint_manager.best_f1_dir, exist_ok=True)
            dest_path = os.path.join(
                checkpoint_manager.best_f1_dir,
                f'best_f1_model_epoch{best_epoch:04d}_f1{best_f1:.4f}.h5'
            )
            shutil.copy(best_model_path, dest_path)

            info = {
                'epoch': best_epoch,
                'f1_score': float(best_f1),
                'model_path': dest_path,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(os.path.join(checkpoint_manager.best_f1_dir, 'best_f1_info.json'), 'w') as f:
                json.dump(info, f, indent=2)

            latest_path = os.path.join(checkpoint_manager.best_f1_dir, 'latest_best_f1.h5')
            shutil.copy(best_model_path, latest_path)

            print(f"\n✓ Best F1 model saved!")
            print(f"  Epoch: {best_epoch}")
            print(f"  F1 Score: {best_f1:.4f}")
            print(f"  Path: {dest_path}")

    checkpoint_manager.save_history()
    checkpoint_manager.plot_history()

    print(checkpoint_manager.get_summary())


def train(model):
    """Train the model with dual saving strategy."""
    train_with_dual_saving(model)


def finalize_training(model):
    print("\n" + "=" * 70)
    print("Finalizing training - Selecting best models...")
    print("=" * 70)

    logs_dir = model.model_dir

    best_loss_dir = os.path.join(logs_dir, 'best_by_loss')
    best_f1_dir = os.path.join(logs_dir, 'best_by_f1')

    best_loss_info = None
    best_f1_info = None

    loss_info_path = os.path.join(best_loss_dir, 'best_loss_info.json')
    f1_info_path = os.path.join(best_f1_dir, 'best_f1_info.json')

    if os.path.exists(loss_info_path):
        with open(loss_info_path, 'r') as f:
            best_loss_info = json.load(f)

    if os.path.exists(f1_info_path):
        with open(f1_info_path, 'r') as f:
            best_f1_info = json.load(f)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("=" * 70)

    if best_loss_info:
        print("\nBest Model by Validation Loss:")
        print(f"   Epoch: {best_loss_info['epoch']}")
        print(f"   Val Loss: {best_loss_info['val_loss']:.4f}")
        print(f"   Path: {best_loss_info['model_path']}")

    if best_f1_info:
        print("\nBest Model by F1 Score:")
        print(f"   Epoch: {best_f1_info['epoch']}")
        print(f"   F1 Score: {best_f1_info['f1_score']:.4f}")
        print(f"   Precision: {best_f1_info['precision']:.4f}")
        print(f"   Recall: {best_f1_info['recall']:.4f}")
        print(f"   Accuracy: {best_f1_info['accuracy']:.4f}")
        print(f"   Path: {best_f1_info['model_path']}")

    print("\n" + "=" * 70)
    print("All model files saved to:", logs_dir)
    print("=" * 70)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask is not None and mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def display_detections(image, boxes, class_ids, scores, class_names, title="Detections"):
    fig, ax = plt.subplots(1, figsize=(12, 12))

    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

    colors = visualize.random_colors(len(class_names))

    for i in range(len(boxes)):
        if scores[i] < 0.5:
            continue

        y1, x1, y2, x2 = boxes[i]
        class_id = class_ids[i]
        class_name = class_names[class_id]
        score = scores[i]
        color = colors[class_id % len(colors)]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color,
                                 facecolor='none')
        ax.add_patch(rect)

        caption = f"{class_name} {score:.2f}"
        ax.text(x1, y1 - 5, caption, size=11, color='white',
                bbox={'facecolor': color, 'alpha': 0.7, 'pad': 2})

    plt.tight_layout()
    return plt


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        if 'masks' in r and r['masks'] is not None:
            splash = color_splash(image, r['masks'])
        else:
            # splash = visualize.draw_boxes(image, r['rois'], r['class_ids'],
            #                               r['scores'], model.config.class_names)
            plt_fig = display_detections(image, r['rois'], r['class_ids'],
                                         r['scores'], model.config.class_names,
                                         title="Detection Results")
            file_name = "detection_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            plt_fig.savefig(file_name)
            plt.close()
            splash = image
            print(f"Detection results saved to {file_name}")
            return

        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                if 'masks' in r and r['masks'] is not None:
                    splash = color_splash(image, r['masks'])
                else:
                    splash = visualize.draw_boxes(image, r['rois'], r['class_ids'],
                                                  r['scores'], model.config.class_names)
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def loss_visualize_old(epoch, tra_loss, val_loss):
    plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Epoch_Loss")
    plt.plot(epoch, tra_loss, label='train_loss', color='r', linestyle='-', marker='o')
    plt.plot(epoch, val_loss, label='val_loss', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(RESULT_DIR, 'loss.jpg'))
    plt.show()


def loss_visualize(epoch, tra_loss, val_loss,
                   tra_rpn_cls=None, tra_rpn_bbox=None,
                   tra_mrcnn_cls=None, tra_mrcnn_bbox=None,
                   val_rpn_cls=None, val_rpn_bbox=None,
                   val_mrcnn_cls=None, val_mrcnn_bbox=None):

    plt.style.use("ggplot")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Total Loss")
    plt.plot(epoch, tra_loss, label='train_loss', color='r', linestyle='-', marker='o')
    plt.plot(epoch, val_loss, label='val_loss', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # RPN cls loss
    if tra_rpn_cls is not None and val_rpn_cls is not None:
        plt.subplot(2, 3, 2)
        plt.title("RPN Class Loss")
        plt.plot(epoch, tra_rpn_cls, label='train', color='r')
        plt.plot(epoch, val_rpn_cls, label='val', color='b')
        plt.legend()
        plt.xlabel('epoch')

    # RPN bbox loss
    if tra_rpn_bbox is not None and val_rpn_bbox is not None:
        plt.subplot(2, 3, 3)
        plt.title("RPN BBox Loss")
        plt.plot(epoch, tra_rpn_bbox, label='train', color='r')
        plt.plot(epoch, val_rpn_bbox, label='val', color='b')
        plt.legend()
        plt.xlabel('epoch')

    # MRCNN cls loss
    if tra_mrcnn_cls is not None and val_mrcnn_cls is not None:
        plt.subplot(2, 3, 4)
        plt.title("MRCNN Class Loss")
        plt.plot(epoch, tra_mrcnn_cls, label='train', color='r')
        plt.plot(epoch, val_mrcnn_cls, label='val', color='b')
        plt.legend()
        plt.xlabel('epoch')

    # MRCNN bbox loss
    if tra_mrcnn_bbox is not None and val_mrcnn_bbox is not None:
        plt.subplot(2, 3, 5)
        plt.title("MRCNN BBox Loss")
        plt.plot(epoch, tra_mrcnn_bbox, label='train', color='r')
        plt.plot(epoch, val_mrcnn_bbox, label='val', color='b')
        plt.legend()
        plt.xlabel('epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'loss_all.jpg'))


if __name__ == '__main__':
    import argparse as args

    # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect plaques.')
    # parser.add_argument("command", required=False, default="train",
    #                     metavar="<command>",
    #                     help="'train' or 'splash'")
    # parser.add_argument('--dataset', required=False, default=dataset_dir,
    #                     metavar="/path/to/samples/Plaques/VOC2COCO/",
    #                     help='Directory of the plaques dataset')
    # parser.add_argument('--weights', required=False, default=model_test_dir,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    # args = parser.parse_args()

    args.command = MODE_SET
    args.dataset = dataset_dir
    args.weights = MODEL_SET
    args.logs = DEFAULT_LOGS_DIR
    args.image = " "
    args.video = " "

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    print("\n" + "=" * 70)
    print("TESTING DATASET LOADING BEFORE MODEL CREATION")
    print("=" * 70)

    test_dataset = PlaquesDataset()
    test_dataset.load_plaques(dataset_dir, "train")
    test_dataset.prepare()

    total_bboxes = 0
    for i in range(min(10, len(test_dataset.image_info))):
        img_info = test_dataset.image_info[i]
        num_bboxes = len(img_info.get('polygons', []))
        total_bboxes += num_bboxes
        print(f"Image {i}: {img_info.get('path', 'unknown')} - {num_bboxes} bboxes")
        if num_bboxes > 0:
            print(f"  First bbox: {img_info['polygons'][0]}")

    print(f"\nTotal bboxes in first 10 images: {total_bboxes}")

    if total_bboxes == 0:
        print("\n ERROR: No bounding boxes found in dataset!")
        print("   Please fix load_plaques function before training.")
        sys.exit(1)

    print("=" * 70 + "\n")


    if args.command == "train":
        config = PlaquesConfig()
    else:
        class InferenceConfig(PlaquesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # # Create model
    # import keras.backend
    # K = keras.backend.backend()
    # if K == 'tensorflow':
    #     keras.backend.set_image_dim_ordering('tf')

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.command == "train":
        if args.weights.lower() == "none" or args.weights.lower() == "scratch":
            print("=" * 50)
            print("Training from scratch - no pre-trained weights loaded")
            print("All layers will be randomly initialized")
            print("=" * 50)
        elif args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
            # Exclude the last layers because they require a matching number of classes
            if config.USE_MASK:
                exclude_layers = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                                  "mrcnn_bbox", "mrcnn_mask", "conv1"]
            else:
                exclude_layers = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                                  "mrcnn_bbox", "conv1"]
            model.load_weights(weights_path, by_name=True, exclude=exclude_layers)
        elif args.weights.lower() == "last":
            weights_path = model.find_last()
            if weights_path:
                model.load_weights(weights_path, by_name=True)
            else:
                print("No last weights found, training from scratch")
        elif args.weights.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
            if weights_path:
                model.load_weights(weights_path, by_name=True)
            else:
                print("No ImageNet weights found, training from scratch")
        else:
            weights_path = args.weights
            if os.path.exists(weights_path):
                model.load_weights(weights_path, by_name=True)
            else:
                print(f"Weights file {weights_path} not found, training from scratch")
    else:
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif args.weights.lower() == "last":
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        if weights_path and os.path.exists(weights_path):
            model.load_weights(weights_path, by_name=True)
        else:
            print(f"Error: No valid weights found for inference mode")
            sys.exit(1)

    if args.command == "train":
        train(model)
        finalize_training(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


    x_epoch, y_tra_loss, r_closs, r_bloss, m_closs, m_bloss, \
    y_val_loss, v_r_closs, v_r_bloss, v_m_closs, v_m_bloss = modellib.call_back()
    loss_data = {
        'epoch': x_epoch,
        'train_total_loss': y_tra_loss,
        'train_rpn_class_loss': r_closs,
        'train_rpn_bbox_loss': r_bloss,
        'train_mrcnn_class_loss': m_closs,
        'train_mrcnn_bbox_loss': m_bloss,
        'val_total_loss': y_val_loss,
        'val_rpn_class_loss': v_r_closs,
        'val_rpn_bbox_loss': v_r_bloss,
        'val_mrcnn_class_loss': v_m_closs,
        'val_mrcnn_bbox_loss': v_m_bloss,
    }

    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(RESULT_DIR, 'loss_all.csv'), index=False)

    if x_epoch[-1] == EPOCH_SET:
        loss_visualize(x_epoch, y_tra_loss, y_val_loss,
                       tra_rpn_cls=r_closs, tra_rpn_bbox=r_bloss,
                       tra_mrcnn_cls=m_closs, tra_mrcnn_bbox=m_bloss,
                       val_rpn_cls=v_r_closs, val_rpn_bbox=v_r_bloss,
                       val_mrcnn_cls=v_m_closs, val_mrcnn_bbox=v_m_bloss)
