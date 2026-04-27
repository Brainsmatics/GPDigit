'''
Model evaluation:
Evaluate all models in the target folder,
including metric calculation and visualisation of evaluation results.
'''


import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from collections import defaultdict
from tqdm import tqdm
from glob import glob
import re
import warnings

warnings.filterwarnings('ignore')

# Add project path
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
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
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_NMS_THRESHOLD = 0.7
    TOP_DOWN_PYRAMID_SIZE = 256
    DETECTION_MAX_INSTANCES = 300
    PRE_NMS_LIMIT = 1000
    POST_NMS_ROIS_INFERENCE = 400


class InferenceConfig(PlaquesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1



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

        image = np.expand_dims(image, axis=-1)
        return image
    except Exception as e:
        print(f"  Error loading image: {e}")
        return None


def predict_single_model(model_path, image_dir, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)

    existing_files = glob(os.path.join(output_dir, "*.txt"))
    if len(existing_files) > 0:
        print(f"  The prediction results already exist ({len(existing_files)} files), skip prediction")
        return True

    try:
        print(f"  Loading model weights...")
        model = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(model_path), config=config)
        model.load_weights(model_path, by_name=True)

        image_files = [f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        success_count = 0
        for file_name in tqdm(image_files, desc="  Predicting..."):
            try:
                image_path = os.path.join(image_dir, file_name)
                image = load_image_as_grayscale(image_path)

                if image is None:
                    continue

                results = model.detect([image], verbose=0)
                r = results[0]

                image_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(output_dir, f'{image_name}.txt')

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
                else:
                    pd.DataFrame().to_csv(output_path, sep=' ', index=False, header=False)

                success_count += 1

            except Exception as e:
                print(f"    Error handling {file_name}: {e}")
                continue

        print(f"  Completed: {success_count}/{len(image_files)} ")
        return True

    except Exception as e:
        print(f"  Prediction failed: {e}")
        return False


def parse_ground_truth(json_path):
    """Parsing ground truth JSON files in COCO format"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    gt_by_image = defaultdict(list)

    for img in data['images']:
        img_id = img.get('image_id') or img.get('id')
        if img_id is None:
            img_name = img.get('file_name', '').replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            img_id = img_name
        else:
            img_name = img.get('file_name', '').replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
            img_id = img_name

        for ann in img.get('annotations', []):
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                gt_by_image[img_id].append({
                    'bbox': [x1, y1, x2, y2],
                    'category_id': ann.get('category_id', 1)
                })

    return dict(gt_by_image)

def parse_predictions(pred_dir):
    """Parse the text file containing the model’s predictions"""
    pred_by_image = defaultdict(list)

    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith('.txt'):
            continue
        image_name = pred_file.replace('.txt', '')
        file_path = os.path.join(pred_dir, pred_file)
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None)
            for _, row in df.iterrows():
                if len(row) >= 8:
                    x_min, y_min, x_max, y_max = row[2], row[3], row[4], row[5]
                    score = float(row[7])
                    pred_by_image[image_name].append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'score': score
                    })
        except Exception as e:
            print(f"Error reading {pred_file}: {e}")

    return dict(pred_by_image)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def compute_ap(precision, recall):
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def compute_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return {'ap': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}

    pred_boxes = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
    gt_matched = [False] * len(gt_boxes)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
        if best_iou >= iou_threshold and best_idx != -1:
            tp[i] = 1
            gt_matched[best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    recall = tp_cum / (len(gt_boxes) + 1e-6)
    ap = compute_ap(precision, recall)

    final_precision = precision[-1] if len(precision) > 0 else 0
    final_recall = recall[-1] if len(recall) > 0 else 0
    f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)

    return {
        'ap': ap, 'precision': final_precision, 'recall': final_recall, 'f1': f1,
        'tp': int(np.sum(tp)), 'fp': int(np.sum(fp)), 'fn': len(gt_boxes) - int(np.sum(tp))
    }


def create_batches(image_list, batch_size=50):
    batches = []
    for i in range(0, len(image_list), batch_size):
        batches.append(image_list[i:i + batch_size])
    return batches


def evaluate_batch(gt_by_image, pred_by_image, image_ids, iou_threshold=0.5):
    batch_gt = []
    batch_pred = []
    for image_id in image_ids:
        if image_id in gt_by_image:
            for gt in gt_by_image[image_id]:
                batch_gt.append(gt)
        if image_id in pred_by_image:
            for pred in pred_by_image[image_id]:
                batch_pred.append(pred)
    metrics = compute_metrics(batch_gt, batch_pred, iou_threshold)
    return {
        'mAP': metrics['ap'], 'precision': metrics['precision'], 'recall': metrics['recall'],
        'f1': metrics['f1'], 'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn'],
        'num_images': len(image_ids)
    }


def evaluate_model_predictions(gt_by_image, pred_dir, batch_size=50, iou_threshold=0.5):
    pred_by_image = parse_predictions(pred_dir)
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    all_image_ids = sorted(all_image_ids)

    if len(all_image_ids) == 0:
        return None

    batches = create_batches(all_image_ids, batch_size)
    batch_results = []

    for batch_images in batches:
        result = evaluate_batch(gt_by_image, pred_by_image, batch_images, iou_threshold)
        batch_results.append(result)

    avg_mAP = np.mean([r['mAP'] for r in batch_results])
    avg_precision = np.mean([r['precision'] for r in batch_results])
    avg_recall = np.mean([r['recall'] for r in batch_results])
    avg_f1 = np.mean([r['f1'] for r in batch_results])

    total_images = sum([r['num_images'] for r in batch_results])
    weighted_avg_mAP = sum(
        [r['mAP'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_precision = sum(
        [r['precision'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_recall = sum(
        [r['recall'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_f1 = sum([r['f1'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0

    total_tp = sum([r['tp'] for r in batch_results])
    total_fp = sum([r['fp'] for r in batch_results])
    total_fn = sum([r['fn'] for r in batch_results])
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (
            overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    return {
        'avg_mAP': avg_mAP, 'avg_precision': avg_precision, 'avg_recall': avg_recall, 'avg_f1': avg_f1,
        'weighted_avg_mAP': weighted_avg_mAP, 'weighted_avg_precision': weighted_avg_precision,
        'weighted_avg_recall': weighted_avg_recall, 'weighted_avg_f1': weighted_avg_f1,
        'overall_precision': overall_precision, 'overall_recall': overall_recall, 'overall_f1': overall_f1,
        'total_images': total_images, 'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
        'num_batches': len(batch_results), 'batch_results': batch_results
    }



def plot_results(all_epoch_results, output_dir):
    if not all_epoch_results:
        return

    all_epoch_results.sort(key=lambda x: x['epoch'])
    epochs = [r['epoch'] for r in all_epoch_results]

    weighted_mAP = [r['results']['weighted_avg_mAP'] * 100 for r in all_epoch_results]
    weighted_precision = [r['results']['weighted_avg_precision'] * 100 for r in all_epoch_results]
    weighted_recall = [r['results']['weighted_avg_recall'] * 100 for r in all_epoch_results]
    weighted_f1 = [r['results']['weighted_avg_f1'] for r in all_epoch_results]

    overall_precision = [r['results']['overall_precision'] * 100 for r in all_epoch_results]
    overall_recall = [r['results']['overall_recall'] * 100 for r in all_epoch_results]
    overall_f1 = [r['results']['overall_f1'] for r in all_epoch_results]

    simple_mAP = [r['results']['avg_mAP'] * 100 for r in all_epoch_results]
    simple_f1 = [r['results']['avg_f1'] for r in all_epoch_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation Results (Epoch 1-30)', fontsize=16)

    ax1 = axes[0, 0]
    ax1.plot(epochs, weighted_mAP, 'b-o', linewidth=2, markersize=6, label='Weighted Average')
    ax1.plot(epochs, simple_mAP, 'r--s', linewidth=2, markersize=6, label='Simple Average')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontsize=12)
    ax1.set_title('Mean Average Precision vs Epoch', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    ax2 = axes[0, 1]
    ax2.plot(epochs, weighted_precision, 'g-s', linewidth=2, markersize=6, label='Precision (Weighted)')
    ax2.plot(epochs, weighted_recall, 'r-^', linewidth=2, markersize=6, label='Recall (Weighted)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Precision and Recall vs Epoch', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    ax3 = axes[1, 0]
    ax3.plot(epochs, weighted_f1, 'b-o', linewidth=2, markersize=6, label='Weighted F1')
    ax3.plot(epochs, overall_f1, 'g-^', linewidth=2, markersize=6, label='Overall F1')
    ax3.plot(epochs, simple_f1, 'r--s', linewidth=2, markersize=6, label='Simple F1')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('F1 Score vs Epoch', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    ax4 = axes[1, 1]
    detections = []
    for r in all_epoch_results:
        pred_dir = r['pred_dir']
        total_dets = 0
        pred_files = glob(os.path.join(pred_dir, "*.txt"))
        for f in pred_files:
            try:
                df = pd.read_csv(f, sep='\s+', header=None)
                total_dets += len(df)
            except:
                pass
        avg_dets = total_dets / len(pred_files) if pred_files else 0
        detections.append(avg_dets)

    ax4.plot(epochs, detections, 'c-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Average Detections per Image', fontsize=12)
    ax4.set_title('Average Detections vs Epoch', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n The graph has been saved:  {plot_path}")
    plt.close()


def find_best_models(results, output_dir):
    best_by_map = max(results, key=lambda x: x['results']['weighted_avg_mAP'])
    best_by_f1 = max(results, key=lambda x: x['results']['weighted_avg_f1'])
    best_by_overall_f1 = max(results, key=lambda x: x['results']['overall_f1'])

    with open(os.path.join(output_dir, 'best_models.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Summary of the Best Models\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Maximum weighted average mAP:\n")
        f.write(f"  Epoch: {best_by_map['epoch']}\n")
        f.write(f"  mAP: {best_by_map['results']['weighted_avg_mAP'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_map['results']['weighted_avg_f1']:.4f}\n\n")

        f.write(f"Highest weighted average F1 score:\n")
        f.write(f"  Epoch: {best_by_f1['epoch']}\n")
        f.write(f"  mAP: {best_by_f1['results']['weighted_avg_mAP'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_f1['results']['weighted_avg_f1']:.4f}\n\n")

        f.write(f"Highest overall F1 score:\n")
        f.write(f"  Epoch: {best_by_overall_f1['epoch']}\n")
        f.write(f"  Precision: {best_by_overall_f1['results']['overall_precision'] * 100:.2f}%\n")
        f.write(f"  Recall: {best_by_overall_f1['results']['overall_recall'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_overall_f1['results']['overall_f1']:.4f}\n")

    print(f"\n Best model:")
    print(f"  Highest mAP: Epoch {best_by_map['epoch']} ({best_by_map['results']['weighted_avg_mAP'] * 100:.2f}%)")
    print(f"  Highest F1 (weighted): Epoch {best_by_f1['epoch']} ({best_by_f1['results']['weighted_avg_f1']:.4f})")
    print(f"  Maximum F1 (global): Epoch {best_by_overall_f1['epoch']} ({best_by_overall_f1['results']['overall_f1']:.4f})")


def save_results_to_csv_v3(results, output_dir):
    if not results:
        return

    weighted_df = pd.DataFrame([
        {
            'model_name': r['model_name'],
            'mAP(%)': r['results']['weighted_avg_mAP'] * 100,
            'Precision(%)': r['results']['weighted_avg_precision'] * 100,
            'Recall(%)': r['results']['weighted_avg_recall'] * 100,
            'F1': r['results']['weighted_avg_f1']
        }
        for r in results
    ])
    weighted_df.to_csv(os.path.join(output_dir, 'results_weighted_average.csv'), index=False)

    overall_df = pd.DataFrame([
        {
            'model_name': r['model_name'],
            'Precision(%)': r['results']['overall_precision'] * 100,
            'Recall(%)': r['results']['overall_recall'] * 100,
            'F1': r['results']['overall_f1'],
            'Total_TP': r['results']['total_tp'],
            'Total_FP': r['results']['total_fp'],
            'Total_FN': r['results']['total_fn']
        }
        for r in results
    ])
    overall_df.to_csv(os.path.join(output_dir, 'results_overall.csv'), index=False)

    simple_df = pd.DataFrame([
        {
            'model_name': r['model_name'],
            'mAP(%)': r['results']['avg_mAP'] * 100,
            'Precision(%)': r['results']['avg_precision'] * 100,
            'Recall(%)': r['results']['avg_recall'] * 100,
            'F1': r['results']['avg_f1']
        }
        for r in results
    ])
    simple_df.to_csv(os.path.join(output_dir, 'results_simple_average.csv'), index=False)

    print(f"\n The results have been saved to a CSV file")


def plot_results_v3(all_results, output_dir):
    if not all_results:
        return

    x_indices = list(range(1, len(all_results) + 1))
    model_names = [r['model_name'] for r in all_results]

    weighted_mAP = [r['results']['weighted_avg_mAP'] * 100 for r in all_results]
    weighted_precision = [r['results']['weighted_avg_precision'] * 100 for r in all_results]
    weighted_recall = [r['results']['weighted_avg_recall'] * 100 for r in all_results]
    weighted_f1 = [r['results']['weighted_avg_f1'] for r in all_results]

    overall_precision = [r['results']['overall_precision'] * 100 for r in all_results]
    overall_recall = [r['results']['overall_recall'] * 100 for r in all_results]
    overall_f1 = [r['results']['overall_f1'] for r in all_results]

    simple_mAP = [r['results']['avg_mAP'] * 100 for r in all_results]
    simple_f1 = [r['results']['avg_f1'] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)

    ax1 = axes[0, 0]
    ax1.plot(x_indices, weighted_mAP, 'b-o', linewidth=2, markersize=6, label='Weighted Average')
    ax1.plot(x_indices, simple_mAP, 'r--s', linewidth=2, markersize=6, label='Simple Average')
    ax1.set_xlabel('Model Index', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontsize=12)
    ax1.set_title('Mean Average Precision', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)

    ax2 = axes[0, 1]
    ax2.plot(x_indices, weighted_precision, 'g-s', linewidth=2, markersize=6, label='Precision (Weighted)')
    ax2.plot(x_indices, weighted_recall, 'r-^', linewidth=2, markersize=6, label='Recall (Weighted)')
    ax2.set_xlabel('Model Index', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Precision and Recall', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)

    ax3 = axes[1, 0]
    ax3.plot(x_indices, weighted_f1, 'b-o', linewidth=2, markersize=6, label='Weighted F1')
    ax3.plot(x_indices, overall_f1, 'g-^', linewidth=2, markersize=6, label='Overall F1')
    ax3.plot(x_indices, simple_f1, 'r--s', linewidth=2, markersize=6, label='Simple F1')
    ax3.set_xlabel('Model Index', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('F1 Score', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)

    ax4 = axes[1, 1]
    detections = []
    for r in all_results:
        pred_dir = r['pred_dir']
        total_dets = 0
        pred_files = glob(os.path.join(pred_dir, "*.txt"))
        for f in pred_files:
            try:
                df = pd.read_csv(f, sep='\s+', header=None)
                total_dets += len(df)
            except:
                pass
        avg_dets = total_dets / len(pred_files) if pred_files else 0
        detections.append(avg_dets)

    ax4.plot(x_indices, detections, 'c-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Model Index', fontsize=12)
    ax4.set_ylabel('Average Detections per Image', fontsize=12)
    ax4.set_title('Average Detections', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nThe graph has been saved: {plot_path}")
    plt.close()


def find_best_models_v3(results, output_dir):
    if not results:
        return

    best_by_map = max(results, key=lambda x: x['results']['weighted_avg_mAP'])
    best_by_f1 = max(results, key=lambda x: x['results']['weighted_avg_f1'])
    best_by_overall_f1 = max(results, key=lambda x: x['results']['overall_f1'])

    with open(os.path.join(output_dir, 'best_models.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Summary of the Best Models\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Maximum weighted average mAP:\n")
        f.write(f"  Model: {best_by_map['model_name']}\n")
        f.write(f"  mAP: {best_by_map['results']['weighted_avg_mAP'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_map['results']['weighted_avg_f1']:.4f}\n\n")

        f.write(f"Highest weighted average F1 score:\n")
        f.write(f"  Model: {best_by_f1['model_name']}\n")
        f.write(f"  mAP: {best_by_f1['results']['weighted_avg_mAP'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_f1['results']['weighted_avg_f1']:.4f}\n\n")

        f.write(f"Highest overall F1 score:\n")
        f.write(f"  Model: {best_by_overall_f1['model_name']}\n")
        f.write(f"  Precision: {best_by_overall_f1['results']['overall_precision'] * 100:.2f}%\n")
        f.write(f"  Recall: {best_by_overall_f1['results']['overall_recall'] * 100:.2f}%\n")
        f.write(f"  F1: {best_by_overall_f1['results']['overall_f1']:.4f}\n")

    print(f"\nBest model:")
    print(f"  Highest mAP: {best_by_map['model_name']} ({best_by_map['results']['weighted_avg_mAP'] * 100:.2f}%)")
    print(f"  Highest F1 (weighted):{best_by_f1['model_name']} ({best_by_f1['results']['weighted_avg_f1']:.4f})")
    print(f"  Maximum F1 (global): {best_by_overall_f1['model_name']} ({best_by_overall_f1['results']['overall_f1']:.4f})")



def evaluate_per_image(gt_by_image, pred_by_image, iou_threshold=0.5):
    per_image_results = []

    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())

    for image_id in all_image_ids:
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])

        metrics = compute_metrics(gt_boxes, pred_boxes, iou_threshold)

        per_image_results.append({
            'image_id': image_id,
            'ap': metrics['ap'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        })

    return per_image_results


def evaluate_model_per_image(gt_by_image, pred_dir, iou_threshold=0.5):
    pred_by_image = parse_predictions(pred_dir)
    per_image_results = evaluate_per_image(gt_by_image, pred_by_image, iou_threshold)
    return per_image_results


def save_per_image_results(all_per_image_results, output_dir):
    for model_result in all_per_image_results:
        model_name = model_result['model_name']
        per_image_data = model_result['per_image_results']

        if per_image_data:
            df = pd.DataFrame(per_image_data)
            df = df.sort_values('image_id')
            csv_path = os.path.join(output_dir, f'per_image_{model_name}.csv')
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

    all_ap_data = []
    for model_result in all_per_image_results:
        model_name = model_result['model_name']
        for img_result in model_result['per_image_results']:
            all_ap_data.append({
                'model_name': model_name,
                'image_id': img_result['image_id'],
                'ap': img_result['ap'],
                'precision': img_result['precision'],
                'recall': img_result['recall'],
                'f1': img_result['f1'],
                'tp': img_result['tp'],
                'fp': img_result['fp'],
                'fn': img_result['fn'],
                'num_gt': img_result['num_gt'],
                'num_pred': img_result['num_pred']
            })

    if all_ap_data:
        df_all = pd.DataFrame(all_ap_data)
        summary_path = os.path.join(output_dir, 'all_models_per_image_summary.csv')
        df_all.to_csv(summary_path, index=False)
        print(f"\nThe summary document has been saved:  {summary_path}")
        return df_all

    return None


def plot_boxplot_per_model(all_per_image_results, output_dir):
    if not all_per_image_results:
        return

    model_names = []
    ap_values_list = []

    for model_result in all_per_image_results:
        model_name = model_result['model_name']
        ap_values = [img['ap'] for img in model_result['per_image_results'] if img['num_gt'] > 0 or img['num_pred'] > 0]

        if ap_values:
            model_names.append(model_name)
            ap_values_list.append(ap_values)

    if not model_names:
        print("There is insufficient data to plot a box plot.")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 0.8), 8))

    bp = ax.boxplot(ap_values_list, labels=model_names, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    for i, values in enumerate(ap_values_list, 1):
        x = np.random.normal(i, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.5, s=20, c='red', edgecolors='none')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Per-Image AP Distribution Across Models', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'per_image_ap_boxplot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nThe box plot has been saved: {plot_path}")
    plt.close()


def plot_combined_boxplot(all_per_image_results, output_dir):
    if not all_per_image_results:
        return

    data_for_plot = []
    model_names = []

    for model_result in all_per_image_results:
        model_name = model_result['model_name']
        short_name = model_name.replace('.h5', '').replace('plaques', '')
        if len(short_name) > 15:
            short_name = short_name[:15] + '...'

        ap_values = [img['ap'] for img in model_result['per_image_results']]

        if ap_values:
            model_names.append(short_name)
            data_for_plot.append(ap_values)

    if not data_for_plot:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    bp1 = ax1.boxplot(data_for_plot, labels=model_names, patch_artist=True,
                      showmeans=True, meanline=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Average Precision (AP)', fontsize=12)
    ax1.set_title('Per-Image AP Distribution by Model', fontsize=14)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    try:
        import seaborn as sns
        all_aps = []
        all_models = []
        for model_result in all_per_image_results:
            model_name = model_result['model_name']
            short_name = model_name.replace('.h5', '').replace('plaques', '')
            if len(short_name) > 15:
                short_name = short_name[:15] + '...'
            for img in model_result['per_image_results']:
                all_aps.append(img['ap'])
                all_models.append(short_name)

        ax2 = axes[1]
        sns.violinplot(x=all_models, y=all_aps, ax=ax2, palette='Set3', alpha=0.7)
        sns.swarmplot(x=all_models, y=all_aps, ax=ax2, color='red', alpha=0.5, size=3)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Average Precision (AP)', fontsize=12)
        ax2.set_title('Per-Image AP Distribution (Violin Plot)', fontsize=14)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
    except ImportError:
        ax2 = axes[1]
        ax2.axis('off')
        stats_data = []
        for i, (name, aps) in enumerate(zip(model_names, data_for_plot)):
            aps_array = np.array(aps)
            stats_data.append([
                name,
                f"{np.mean(aps_array):.3f}",
                f"{np.median(aps_array):.3f}",
                f"{np.std(aps_array):.3f}",
                f"{np.min(aps_array):.3f}",
                f"{np.max(aps_array):.3f}"
            ])

        table = ax2.table(cellText=stats_data,
                          colLabels=['Model', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title('AP Statistics Summary', fontsize=14, pad=20)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_image_ap_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"The comparison image has been saved: {plot_path}")
    plt.close()


def plot_ap_histogram(all_per_image_results, output_dir):
    if not all_per_image_results:
        return

    n_models = len(all_per_image_results)
    if n_models == 0:
        return

    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, model_result in enumerate(all_per_image_results):
        ax = axes[idx]
        model_name = model_result['model_name']
        ap_values = [img['ap'] for img in model_result['per_image_results']]

        if ap_values:
            ax.hist(ap_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(np.mean(ap_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(ap_values):.3f}')
            ax.axvline(np.median(ap_values), color='green', linestyle=':',
                       label=f'Median: {np.median(ap_values):.3f}')
            ax.set_xlabel('AP', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name[:30]}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])

    for idx in range(len(all_per_image_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('AP Distribution Histograms by Model', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'ap_histograms.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"The histogram has been saved: {plot_path}")
    plt.close()



def evaluate_per_image_batch(gt_by_image, pred_by_image, image_ids, iou_threshold=0.5):
    batch_per_image_results = []

    for image_id in image_ids:
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])

        metrics = compute_metrics(gt_boxes, pred_boxes, iou_threshold)

        batch_per_image_results.append({
            'image_id': image_id,
            'ap': metrics['ap'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        })

    return batch_per_image_results


def evaluate_model_with_batch_save(gt_by_image, pred_dir, batch_size=50, iou_threshold=0.5,
                                   batch_output_dir=None):
    """
    Evaluate the model in batches and save the results for each image.

    Parameters:
        gt_by_image: ground truth dictionary
        pred_dir: directory for prediction results
        batch_size: batch size
        iou_threshold: IoU threshold
        batch_output_dir: directory for saving batch results

    Returns:
        eval_result: Overall evaluation result
        all_per_image_results: List of evaluation results for all images
        batch_summaries: Summary information for each batch
    """

    pred_by_image = parse_predictions(pred_dir)

    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    all_image_ids = sorted(all_image_ids)

    if len(all_image_ids) == 0:
        return None, [], []

    batches = create_batches(all_image_ids, batch_size)

    batch_results = []
    all_per_image_results = []
    batch_summaries = []

    for batch_idx, batch_images in enumerate(batches):
        result = evaluate_batch(gt_by_image, pred_by_image, batch_images, iou_threshold)
        batch_results.append(result)

        batch_per_image = evaluate_per_image_batch(gt_by_image, pred_by_image, batch_images, iou_threshold)
        all_per_image_results.extend(batch_per_image)

        batch_summary = {
            'batch_idx': batch_idx,
            'num_images': len(batch_images),
            'batch_mAP': result['mAP'],
            'batch_precision': result['precision'],
            'batch_recall': result['recall'],
            'batch_f1': result['f1'],
            'batch_tp': result['tp'],
            'batch_fp': result['fp'],
            'batch_fn': result['fn'],
            'images_in_batch': batch_images
        }
        batch_summaries.append(batch_summary)

        if batch_output_dir:
            os.makedirs(batch_output_dir, exist_ok=True)

            batch_df = pd.DataFrame(batch_per_image)
            batch_csv_path = os.path.join(batch_output_dir, f'batch_{batch_idx:03d}_per_image.csv')
            batch_df.to_csv(batch_csv_path, index=False)

            summary_df = pd.DataFrame([{
                'batch_idx': batch_idx,
                'num_images': len(batch_images),
                'mAP': result['mAP'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'tp': result['tp'],
                'fp': result['fp'],
                'fn': result['fn']
            }])
            summary_csv_path = os.path.join(batch_output_dir, f'batch_{batch_idx:03d}_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)

            images_txt_path = os.path.join(batch_output_dir, f'batch_{batch_idx:03d}_images.txt')
            with open(images_txt_path, 'w', encoding='utf-8') as f:
                for img in batch_images:
                    f.write(f"{img}\n")

    avg_mAP = np.mean([r['mAP'] for r in batch_results])
    avg_precision = np.mean([r['precision'] for r in batch_results])
    avg_recall = np.mean([r['recall'] for r in batch_results])
    avg_f1 = np.mean([r['f1'] for r in batch_results])

    total_images = sum([r['num_images'] for r in batch_results])
    weighted_avg_mAP = sum(
        [r['mAP'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_precision = sum(
        [r['precision'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_recall = sum(
        [r['recall'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0
    weighted_avg_f1 = sum([r['f1'] * r['num_images'] for r in batch_results]) / total_images if total_images > 0 else 0

    total_tp = sum([r['tp'] for r in batch_results])
    total_fp = sum([r['fp'] for r in batch_results])
    total_fn = sum([r['fn'] for r in batch_results])
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (
                                                                                                              overall_precision + overall_recall) > 0 else 0

    eval_result = {
        'avg_mAP': avg_mAP, 'avg_precision': avg_precision, 'avg_recall': avg_recall, 'avg_f1': avg_f1,
        'weighted_avg_mAP': weighted_avg_mAP, 'weighted_avg_precision': weighted_avg_precision,
        'weighted_avg_recall': weighted_avg_recall, 'weighted_avg_f1': weighted_avg_f1,
        'overall_precision': overall_precision, 'overall_recall': overall_recall, 'overall_f1': overall_f1,
        'total_images': total_images, 'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
        'num_batches': len(batch_results), 'batch_results': batch_results
    }

    return eval_result, all_per_image_results, batch_summaries


def save_batch_summary_report(batch_summaries, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    all_batches_df = pd.DataFrame([{
        'batch_idx': s['batch_idx'],
        'num_images': s['num_images'],
        'mAP': s['batch_mAP'],
        'precision': s['batch_precision'],
        'recall': s['batch_recall'],
        'f1': s['batch_f1'],
        'tp': s['batch_tp'],
        'fp': s['batch_fp'],
        'fn': s['batch_fn']
    } for s in batch_summaries])

    summary_path = os.path.join(output_dir, f'{model_name}_batches_summary.csv')
    all_batches_df.to_csv(summary_path, index=False)
    print(f"  Batch summary saved:  {summary_path}")


def merge_all_batches_per_image(batch_output_dir, model_name):
    all_batch_files = glob(os.path.join(batch_output_dir, "batch_*_per_image.csv"))

    if not all_batch_files:
        return None

    all_dfs = []
    for f in sorted(all_batch_files):
        batch_idx = re.search(r'batch_(\d+)_per_image', os.path.basename(f))
        batch_num = int(batch_idx.group(1)) if batch_idx else 0

        df = pd.read_csv(f)
        df['batch_idx'] = batch_num
        all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_path = os.path.join(batch_output_dir, f'{model_name}_all_images_merged.csv')
        merged_df.to_csv(merged_path, index=False)
        return merged_df
    return None


def plot_batch_boxplots(all_models_batch_data, output_dir):
    if not all_models_batch_data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    data_for_boxplot = []
    labels = []

    for model_data in all_models_batch_data:
        model_name = model_data['model_name']
        batch_output_dir = model_data['batch_output_dir']

        for batch_idx in range(model_data['num_batches']):
            batch_file = os.path.join(batch_output_dir, f'batch_{batch_idx:03d}_per_image.csv')
            if os.path.exists(batch_file):
                df = pd.read_csv(batch_file)
                ap_values = df['ap'].values.tolist()
                if ap_values:
                    data_for_boxplot.append(ap_values)
                    short_name = model_name.replace('.h5', '').replace('plaques', '')[:10]
                    labels.append(f"{short_name}_b{batch_idx}")

    if data_for_boxplot:
        bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True, showmeans=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax1.set_xlabel('Model_Batch', fontsize=10)
        ax1.set_ylabel('AP', fontsize=12)
        ax1.set_title('AP Distribution by Batch (All Models)', fontsize=12)
        ax1.tick_params(axis='x', rotation=90, labelsize=8)
        ax1.set_ylim([0, 1.05])
        ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[0, 1]
    for model_data in all_models_batch_data:
        model_name = model_data['model_name']
        short_name = model_name.replace('.h5', '').replace('plaques', '')[:15]

        batch_mAPs = [s['batch_mAP'] for s in model_data['batch_summaries']]
        batch_indices = list(range(len(batch_mAPs)))

        ax2.plot(batch_indices, batch_mAPs, '-o', markersize=6, label=short_name)

    ax2.set_xlabel('Batch Index', fontsize=12)
    ax2.set_ylabel('Batch mAP', fontsize=12)
    ax2.set_title('Batch-level mAP Comparison', fontsize=12)
    ax2.legend(fontsize=8, loc='best')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if all_models_batch_data:
        model_ap_lists = []
        model_labels = []

        for model_data in all_models_batch_data:
            model_name = model_data['model_name']
            batch_output_dir = model_data['batch_output_dir']

            all_aps = []
            for batch_idx in range(model_data['num_batches']):
                batch_file = os.path.join(batch_output_dir, f'batch_{batch_idx:03d}_per_image.csv')
                if os.path.exists(batch_file):
                    df = pd.read_csv(batch_file)
                    all_aps.extend(df['ap'].values.tolist())

            if all_aps:
                model_ap_lists.append(all_aps)
                short_name = model_name.replace('.h5', '').replace('plaques', '')[:15]
                model_labels.append(short_name)

        if model_ap_lists:
            bp3 = ax3.boxplot(model_ap_lists, labels=model_labels, patch_artist=True, showmeans=True)
            for patch in bp3['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)
            ax3.set_xlabel('Model', fontsize=12)
            ax3.set_ylabel('AP', fontsize=12)
            ax3.set_title('Per-Image AP Distribution by Model', fontsize=12)
            ax3.tick_params(axis='x', rotation=45, labelsize=9)
            ax3.set_ylim([0, 1.05])
            ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    for model_data in all_models_batch_data:
        model_name = model_data['model_name']
        short_name = model_name.replace('.h5', '').replace('plaques', '')[:15]

        batch_sizes = [s['num_images'] for s in model_data['batch_summaries']]
        batch_mAPs = [s['batch_mAP'] for s in model_data['batch_summaries']]

        ax4.scatter(batch_sizes, batch_mAPs, s=50, alpha=0.7, label=short_name)

    ax4.set_xlabel('Number of Images in Batch', fontsize=12)
    ax4.set_ylabel('Batch mAP', fontsize=12)
    ax4.set_title('Batch mAP vs Batch Size', fontsize=12)
    ax4.legend(fontsize=8, loc='best')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Batch-wise Evaluation Analysis', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'batch_analysis_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n The batch analysis chart has been saved: {plot_path}")
    plt.close()


def create_batch_ap_heatmap(all_models_batch_data, output_dir):
    if not all_models_batch_data:
        return

    model_names = []
    heatmap_data = []
    max_batches = 0

    for model_data in all_models_batch_data:
        model_name = model_data['model_name']
        short_name = model_name.replace('.h5', '').replace('plaques', '')[:15]
        model_names.append(short_name)

        batch_mAPs = [s['batch_mAP'] for s in model_data['batch_summaries']]
        heatmap_data.append(batch_mAPs)
        max_batches = max(max_batches, len(batch_mAPs))

    for i in range(len(heatmap_data)):
        if len(heatmap_data[i]) < max_batches:
            heatmap_data[i].extend([np.nan] * (max_batches - len(heatmap_data[i])))

    heatmap_array = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(max(10, max_batches * 1.5), max(6, len(model_names) * 0.8)))

    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(max_batches))
    ax.set_xticklabels([f'B{i}' for i in range(max_batches)])
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(max_batches):
            if not np.isnan(heatmap_array[i, j]):
                text = ax.text(j, i, f'{heatmap_array[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Batch mAP Heatmap Across Models', fontsize=14)

    plt.colorbar(im, ax=ax, label='mAP')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'batch_ap_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved: {plot_path}")
    plt.close()




def main():
    print("=" * 70)
    print("Comprehensive model evaluation system.")
    print("=" * 70)


    model_dir = "D:/gxgong/net-code/2DMethod_test/logs/plaques20260413T1559/pick6"  # Model file directory
    image_dir = "G:/datasets_net/2D_data/mini_batch/image/val"  # Validation set image directory
    gt_path = "G:/datasets_net/2D_data/mini_batch/image/val/instances.json"  # Ground Truth
    output_base_dir = os.path.join(model_dir, "batch_evaluation")  # Evaluate the output directory

    batch_size = 50  # Batch size for batch evaluation
    iou_threshold = 0.5

    os.makedirs(output_base_dir, exist_ok=True)


    gt_by_image = parse_ground_truth(gt_path)

    config = InferenceConfig()

    model_files = glob(os.path.join(model_dir, "*.h5"))

    if len(model_files) == 0:
        print(f"  Error: No .h5 model files were found in {model_dir}.")
        return

    models_info = []
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        model_name_without_ext = model_name.replace('.h5', '')

        match = re.search(r'(\d+)', model_name)
        if match:
            sort_key = int(match.group(1))
        else:
            sort_key = len(models_info)

        folder_name = model_name_without_ext

        models_info.append({
            'sort_key': sort_key,
            'path': model_path,
            'name': model_name,
            'folder_name': folder_name
        })

    models_info.sort(key=lambda x: x['sort_key'])

    print(f"Find {len(models_info)} models:")
    for info in models_info:
        print(f"  - {info['name']} -> Prediction folder: {info['folder_name']}")


    all_results = []
    # all_per_image_results = []
    all_models_batch_data = []

    for idx, info in enumerate(models_info, 1):
        model_path = info['path']
        model_name = info['name']
        folder_name = info['folder_name']

        print(f"\n{'=' * 50}")
        print(f"[{idx}/{len(models_info)}] Evaluation model: {model_name}")
        print(f"{'=' * 50}")

        pred_dir = os.path.join(output_base_dir, folder_name)

        success = predict_single_model(model_path, image_dir, pred_dir, config)

        if not success:
            print(f"  {model_name} Prediction failed; skip evaluation")
            continue

        model_batch_dir = os.path.join(output_base_dir, "batch_results", folder_name)
        os.makedirs(model_batch_dir, exist_ok=True)

        eval_result = evaluate_model_predictions(gt_by_image, pred_dir, batch_size, iou_threshold)

        if eval_result is None:
            print(f"  Evaluation failed")
            continue

        # per_image_results = evaluate_model_per_image(gt_by_image, pred_dir, iou_threshold)

        print(f"\n  Calculating evaluation metrics (saving in batches)...")
        eval_result, per_image_results, batch_summaries = evaluate_model_with_batch_save(
            gt_by_image, pred_dir, batch_size, iou_threshold, model_batch_dir
        )

        if eval_result is None:
            print(f"  Evaluation failed")
            continue

        save_batch_summary_report(batch_summaries, model_batch_dir, folder_name)

        merged_df = merge_all_batches_per_image(model_batch_dir, folder_name)

        all_results.append({
            'model_name': model_name,
            'folder_name': folder_name,
            'model_path': model_path,
            'results': eval_result,
            'pred_dir': pred_dir
        })

        all_models_batch_data.append({
            'model_name': model_name,
            'folder_name': folder_name,
            'batch_summaries': batch_summaries,
            'per_image_results': per_image_results,
            'batch_output_dir': model_batch_dir,
            'num_batches': len(batch_summaries)
        })


        # all_per_image_results.append({
        #     'model_name': model_name,
        #     'folder_name': folder_name,
        #     'per_image_results': per_image_results
        # })

        print(f"    Weighted average - mAP: {eval_result['weighted_avg_mAP'] * 100:.2f}%, "
              f"Precision: {eval_result['weighted_avg_precision'] * 100:.2f}%, "
              f"Recall: {eval_result['weighted_avg_recall'] * 100:.2f}%, "
              f"F1: {eval_result['weighted_avg_f1']:.4f}")
        print(f"    Overall summary - Precision: {eval_result['overall_precision'] * 100:.2f}%, "
              f"Recall: {eval_result['overall_recall'] * 100:.2f}%, "
              f"F1: {eval_result['overall_f1']:.4f}")


    if all_results:
        save_results_to_csv_v3(all_results, output_base_dir)

        plot_results_v3(all_results, output_base_dir)

        find_best_models_v3(all_results, output_base_dir)

        # if all_per_image_results:
        #     per_image_dir = os.path.join(output_base_dir, "per_image_results")
        #     os.makedirs(per_image_dir, exist_ok=True)
        #     save_per_image_results(all_per_image_results, per_image_dir)
        #     plot_boxplot_per_model(all_per_image_results, per_image_dir)
        #     plot_combined_boxplot(all_per_image_results, per_image_dir)
        #     plot_ap_histogram(all_per_image_results, per_image_dir)


        if all_models_batch_data:

            batch_analysis_dir = os.path.join(output_base_dir, "batch_analysis")
            os.makedirs(batch_analysis_dir, exist_ok=True)
            plot_batch_boxplots(all_models_batch_data, batch_analysis_dir)
            create_batch_ap_heatmap(all_models_batch_data, batch_analysis_dir)

        print(f"\nAll results have been saved to: {output_base_dir}")



if __name__ == "__main__":
    main()