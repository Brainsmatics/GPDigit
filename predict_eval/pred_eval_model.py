'''
Model evaluation:
Including the calculation of indicators and the visualisation of evaluation results.
'''


import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import shutil

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config


MODEL_PATH = "logs/best_by_f1/latest_best_f1.h5"  # Path to the trained model
DATASET_DIR = "G:/datasets_net/2D_data/mini_batch/image/"  # Data set path
LOGS_DIR = "./evaluation_results"  # Directory for storing evaluation results

# Evaluation criteria
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
NUM_FP_IMAGES = 10  # Number of visualisations


# ==================== 配置类 ====================
class EvaluationConfig(Config):
    """Evaluation configuration (to be consistent with the training configuration)"""
    NAME = "plaques"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MASK = False
    USE_MINI_MASK = False
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 1
    MEAN_PIXEL = np.array([127.5])
    BACKBONE = "resnet101"

    # Anchor: adapt to plaque size
    RPN_ANCHOR_SCALES = (6, 12, 24, 48, 96)
    RPN_NMS_THRESHOLD = 0.7
    POST_NMS_ROIS_INFERENCE = 300



class PlaquesDataset(utils.Dataset):

    def __init__(self):
        super().__init__()
        self._preloaded_boxes = []
        self._image_cache = {}

    def load_plaques(self, dataset_dir, subset):
        self.add_class("plaques", 1, "object")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        json_path = os.path.join(dataset_dir, "instances.json")

        print(f"\nLoading {subset} dataset from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = data.get("images", [])
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

            for ann in annotations:
                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        polygons.append(bbox)

            width = int(img_data.get("width", 512))
            height = int(img_data.get("height", 512))

            valid_images.append({
                'file_name': file_name,
                'image_path': image_path,
                'width': width,
                'height': height,
                'polygons': polygons
            })

        for idx, img_info in enumerate(valid_images):
            self.add_image(
                "plaques",
                image_id=img_info['file_name'],
                path=img_info['image_path'],
                width=img_info['width'],
                height=img_info['height'],
                polygons=img_info['polygons']
            )

            polygons = img_info['polygons']
            num_instances = len(polygons)

            if num_instances == 0:
                self._preloaded_boxes.append(np.zeros((0, 4), dtype=np.int32))
            else:
                boxes = np.zeros((num_instances, 4), dtype=np.int32)
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

        print(f"  Loaded {len(valid_images)} images with {sum(len(b) for b in self._preloaded_boxes)} boxes")

    def load_image(self, image_id):
        if image_id in self._image_cache:
            return self._image_cache[image_id]

        image = super().load_image(image_id)
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
        self._image_cache[image_id] = image
        return image

    def get_preloaded_boxes(self, image_id):
        return self._preloaded_boxes[image_id].copy()

    def clear_cache(self):
        self._image_cache.clear()


def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.maximum(0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou_matrix = inter / (union + 1e-7)

    return iou_matrix


def compute_iou_single(box1, box2):
    y1, x1, y2, x2 = box1
    y1g, x1g, y2g, x2g = box2

    inter_y1 = max(y1, y1g)
    inter_x1 = max(x1, x1g)
    inter_y2 = min(y2, y2g)
    inter_x2 = min(x2, x2g)

    if inter_y2 > inter_y1 and inter_x2 > inter_x1:
        inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
        area1 = (y2 - y1) * (x2 - x1)
        area2 = (y2g - y1g) * (x2g - x1g)
        iou = inter_area / (area1 + area2 - inter_area)
    else:
        iou = 0
    return iou


def compute_detection_metrics(dataset, model, iou_threshold=0.5, conf_threshold=0.5):
    print(f"\nComputing detection metrics (IoU={iou_threshold}, Conf={conf_threshold})...")

    all_true_boxes = []
    all_pred_boxes = []
    all_pred_scores = []

    val_size = len(dataset.image_info)

    for i in range(val_size):
        image = dataset.load_image(i)
        true_boxes = dataset.get_preloaded_boxes(i)
        results = model.detect([image], verbose=0)[0]

        mask = results['scores'] >= conf_threshold
        pred_boxes = results['rois'][mask]
        pred_scores = results['scores'][mask]

        all_true_boxes.append(true_boxes)
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = sum(len(boxes) for boxes in all_true_boxes)

    for true_boxes, pred_boxes, pred_scores in zip(all_true_boxes, all_pred_boxes, all_pred_scores):
        if len(pred_boxes) == 0:
            total_fn += len(true_boxes)
            continue

        if len(true_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, true_boxes)
        matched_gt = set()
        sorted_indices = np.argsort(pred_scores)[::-1]

        for pred_idx in sorted_indices:
            best_gt_idx = np.argmax(iou_matrix[pred_idx])
            best_iou = iou_matrix[pred_idx][best_gt_idx]

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        total_fn += len(true_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_gt': total_gt,
        'total_pred': total_tp + total_fp
    }


def compute_pr_curve(dataset, model, iou_threshold=0.5, num_thresholds=50):
    print("\nComputing PR curve...")

    all_scores = []
    all_matched = []

    val_size = len(dataset.image_info)

    for i in range(val_size):
        image = dataset.load_image(i)
        true_boxes = dataset.get_preloaded_boxes(i)
        results = model.detect([image], verbose=0)[0]

        pred_boxes = results['rois']
        pred_scores = results['scores']

        if len(pred_boxes) == 0:
            continue

        if len(true_boxes) == 0:
            for score in pred_scores:
                all_scores.append(score)
                all_matched.append(0)
            continue

        iou_matrix = compute_iou_matrix(pred_boxes, true_boxes)
        matched_gt = set()
        sorted_indices = np.argsort(pred_scores)[::-1]

        for pred_idx in sorted_indices:
            best_gt_idx = np.argmax(iou_matrix[pred_idx])
            best_iou = iou_matrix[pred_idx][best_gt_idx]

            all_scores.append(pred_scores[pred_idx])
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                all_matched.append(1)
                matched_gt.add(best_gt_idx)
            else:
                all_matched.append(0)

    if len(all_scores) == 0:
        print("  No detections found!")
        return None, None, None

    sorted_indices = np.argsort(all_scores)[::-1]
    all_matched = np.array(all_matched)[sorted_indices]

    tp_cumsum = np.cumsum(all_matched)
    fp_cumsum = np.cumsum(1 - all_matched)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = tp_cumsum / (len(all_matched) + 1e-8)

    precisions = np.concatenate([[1], precisions])
    recalls = np.concatenate([[0], recalls])

    return precisions, recalls, all_scores


def find_optimal_threshold(precisions, recalls, scores):
    if precisions is None:
        return None

    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    if best_idx == 0 or best_idx - 1 >= len(scores):
        best_thresh = 0.5
    else:
        best_thresh = scores[best_idx - 1] if best_idx > 0 else 0.5

    return {
        'threshold': best_thresh,
        'precision': precisions[best_idx],
        'recall': recalls[best_idx],
        'f1': f1_scores[best_idx]
    }


def visualize_false_positives(model, dataset, save_dir, num_images=10,
                              iou_threshold=0.5, conf_threshold=0.5):
    print(f"\nVisualizing false positives (saving to {save_dir})...")

    os.makedirs(save_dir, exist_ok=True)

    analyzed = 0
    image_ids = list(range(len(dataset.image_info)))
    random.shuffle(image_ids)

    fp_stats = []

    for image_id in image_ids:
        if analyzed >= num_images:
            break

        image = dataset.load_image(image_id)
        image_display = image.astype(np.uint8)
        if image_display.ndim == 2 or image_display.shape[-1] == 1:
            image_display = np.squeeze(image_display)

        gt_boxes = dataset.get_preloaded_boxes(image_id)
        results = model.detect([image], verbose=0)[0]

        mask = results['scores'] >= conf_threshold
        pred_boxes = results['rois'][mask]
        pred_scores = results['scores'][mask]

        if len(pred_boxes) == 0:
            continue

        fps = []
        fp_scores = []
        tps = []

        if len(gt_boxes) == 0:
            fps = pred_boxes
            fp_scores = pred_scores
        else:
            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            matched_gt = set()
            sorted_indices = np.argsort(pred_scores)[::-1]

            for pred_idx in sorted_indices:
                best_gt_idx = np.argmax(iou_matrix[pred_idx])
                best_iou = iou_matrix[pred_idx][best_gt_idx]

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    tps.append(pred_boxes[pred_idx])
                    matched_gt.add(best_gt_idx)
                else:
                    fps.append(pred_boxes[pred_idx])
                    fp_scores.append(pred_scores[pred_idx])

        if len(fps) == 0:
            continue

        analyzed += 1
        fp_stats.append({
            'image_id': image_id,
            'num_fp': len(fps),
            'num_tp': len(tps),
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        })

        fig, ax = plt.subplots(1, figsize=(14, 14))

        if image_display.ndim == 2:
            ax.imshow(image_display, cmap='gray')
        else:
            ax.imshow(image_display)

        for gt_box in gt_boxes:
            y1, x1, y2, x2 = gt_box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green',
                                     facecolor='none', linestyle='-')
            ax.add_patch(rect)

        for i, fp_box in enumerate(fps):
            y1, x1, y2, x2 = fp_box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor='red',
                                     facecolor='red', alpha=0.3)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'FP (conf={fp_scores[i]:.2f})',
                    fontsize=10, color='red', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        for tp_box in tps:
            y1, x1, y2, x2 = tp_box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='blue',
                                     facecolor='none', linestyle='-')
            ax.add_patch(rect)

        ax.set_title(f'Image {image_id}\n'
                     f'GT: {len(gt_boxes)} | Pred: {len(pred_boxes)} | TP: {len(tps)} | FP: {len(fps)}',
                     fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'fp_image_{image_id}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Image {image_id}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}, "
              f"TP={len(tps)}, FP={len(fps)}")

    if fp_stats:
        with open(os.path.join(save_dir, 'fp_statistics.json'), 'w') as f:
            json.dump(fp_stats, f, indent=2)

        avg_fp = np.mean([s['num_fp'] for s in fp_stats])
        print(f"\n  Average FP per image: {avg_fp:.2f}")

    return fp_stats


def plot_pr_curve(precisions, recalls, optimal, save_path):
    if precisions is None:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')

    if optimal:
        plt.plot(optimal['recall'], optimal['precision'], 'ro', markersize=12,
                 label=f"Optimal (F1={optimal['f1']:.3f}, thresh={optimal['threshold']:.3f})")

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  PR curve saved to: {save_path}")


def evaluate_different_thresholds(dataset, model, iou_threshold=0.5):
    print("\nEvaluating different confidence thresholds...")

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    results = []

    for thresh in thresholds:
        metrics = compute_detection_metrics(dataset, model, iou_threshold, thresh)
        results.append({
            'threshold': thresh,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn']
        })
        print(f"  Thresh={thresh:.2f}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    return results


def plot_threshold_analysis(threshold_results, save_path):
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1_scores = [r['f1'] for r in threshold_results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2, markersize=8)
    plt.plot(thresholds, recalls, 'g-s', label='Recall', linewidth=2, markersize=8)
    plt.plot(thresholds, f1_scores, 'r-^', label='F1 Score', linewidth=2, markersize=8)

    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance vs Confidence Threshold', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 标注最佳F1点
    best_idx = np.argmax(f1_scores)
    plt.plot(thresholds[best_idx], f1_scores[best_idx], 'ro', markersize=15,
             label=f"Best F1={f1_scores[best_idx]:.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Threshold analysis saved to: {save_path}")


def generate_report(metrics, optimal, threshold_results, save_dir):
    report_path = os.path.join(save_dir, 'evaluation_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. CURRENT CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  IoU Threshold: {IOU_THRESHOLD}\n")
        f.write(f"  Confidence Threshold: {CONF_THRESHOLD}\n\n")

        f.write("2. DETECTION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"  True Positives: {metrics['tp']}\n")
        f.write(f"  False Positives: {metrics['fp']}\n")
        f.write(f"  False Negatives: {metrics['fn']}\n")
        f.write(f"  Total GT: {metrics['total_gt']}\n")
        f.write(f"  Total Predictions: {metrics['total_pred']}\n\n")

        if optimal:
            f.write("3. OPTIMAL THRESHOLD (from PR curve)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Optimal Confidence Threshold: {optimal['threshold']:.3f}\n")
            f.write(f"  Expected Precision: {optimal['precision']:.3f}\n")
            f.write(f"  Expected Recall: {optimal['recall']:.3f}\n")
            f.write(f"  Expected F1 Score: {optimal['f1']:.3f}\n\n")

        f.write("4. THRESHOLD COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}\n")
        f.write("-" * 50 + "\n")
        for r in threshold_results:
            f.write(f"{r['threshold']:>10.2f} | {r['precision']:>10.4f} | "
                    f"{r['recall']:>10.4f} | {r['f1']:>10.4f}\n")

        f.write("\n5. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")

        if metrics['precision'] < 0.5:
            f.write("️ Low Precision detected!\n")
            f.write("  - Consider increasing DETECTION_MIN_CONFIDENCE\n")
            f.write("  - Check for labeling errors in validation set\n")
            f.write("  - Review false positives visualization\n")

        if metrics['recall'] < 0.5:
            f.write("️ Low Recall detected!\n")
            f.write("  - Consider decreasing DETECTION_MIN_CONFIDENCE\n")
            f.write("  - May need more training data\n")
            f.write("  - Check if objects are too small for anchors\n")

        if optimal and optimal['threshold'] != CONF_THRESHOLD:
            f.write(f"\n Suggested confidence threshold: {optimal['threshold']:.3f}\n")
            f.write(f"  Current threshold: {CONF_THRESHOLD}\n")
            f.write(f"  Update config.DETECTION_MIN_CONFIDENCE = {optimal['threshold']:.3f}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\nReport saved to: {report_path}")




def main():
    print("=" * 70)
    print("MASK R-CNN MODEL EVALUATION")
    print("=" * 70)

    # Create a save directory
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("\n1. Loading configuration...")
    config = EvaluationConfig()
    config.display()

    print(f"\n2. Creating model and loading weights from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model file not found: {MODEL_PATH}")
        print("   Please update MODEL_PATH in the script")
        return

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=LOGS_DIR)
    model.load_weights(MODEL_PATH, by_name=True)
    print("   Model loaded successfully")

    print("\n3. Loading validation dataset...")
    dataset_val = PlaquesDataset()
    dataset_val.load_plaques(DATASET_DIR, "val")
    dataset_val.prepare()
    print(f"   Loaded {len(dataset_val.image_info)} validation images")

    print("\n4. Computing detection metrics...")
    metrics = compute_detection_metrics(
        dataset_val, model,
        iou_threshold=IOU_THRESHOLD,
        conf_threshold=CONF_THRESHOLD
    )

    print(f"\n    Results (IoU={IOU_THRESHOLD}, Conf={CONF_THRESHOLD}):")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall: {metrics['recall']:.4f}")
    print(f"      F1 Score: {metrics['f1']:.4f}")
    print(f"      TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

    print("\n5. Computing PR curve...")
    precisions, recalls, scores = compute_pr_curve(dataset_val, model, IOU_THRESHOLD)
    optimal = find_optimal_threshold(precisions, recalls, scores)

    if optimal:
        print(f"\n    Optimal threshold: {optimal['threshold']:.3f}")
        print(f"      Expected Precision: {optimal['precision']:.3f}")
        print(f"      Expected Recall: {optimal['recall']:.3f}")
        print(f"      Expected F1: {optimal['f1']:.3f}")

    pr_curve_path = os.path.join(LOGS_DIR, 'pr_curve.png')
    plot_pr_curve(precisions, recalls, optimal, pr_curve_path)

    threshold_results = evaluate_different_thresholds(dataset_val, model, IOU_THRESHOLD)

    threshold_plot_path = os.path.join(LOGS_DIR, 'threshold_analysis.png')
    plot_threshold_analysis(threshold_results, threshold_plot_path)

    fp_dir = os.path.join(LOGS_DIR, 'false_positives')
    visualize_false_positives(
        model, dataset_val, fp_dir,
        num_images=NUM_FP_IMAGES,
        iou_threshold=IOU_THRESHOLD,
        conf_threshold=CONF_THRESHOLD
    )

    generate_report(metrics, optimal, threshold_results, LOGS_DIR)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {LOGS_DIR}")
    print(f"  - PR Curve: {pr_curve_path}")
    print(f"  - Threshold Analysis: {threshold_plot_path}")
    print(f"  - False Positives: {fp_dir}")
    print(f"  - Report: {os.path.join(LOGS_DIR, 'evaluation_report.txt')}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Edit the path directly here.
    MODEL_PATH = "your_model_path.h5"
    DATASET_DIR = "your_dataset_path/"
    LOGS_DIR = "./evaluation_results"

    main()