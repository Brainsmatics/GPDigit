"""
Ablation test
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import random
import gc
import warnings
import traceback
warnings.filterwarnings('ignore')


USE_GPU = True  # False : CPU

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("✓ Running on GPU: 0")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("✓ Running on CPU")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
parent_dir = os.path.dirname(project_root)

sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "Plaques"))

print(f"Current dir: {current_dir}")
print(f"Project root: {project_root}")


print("\n" + "="*50)
print("Importing modules...")
print("="*50)

try:
    from mrcnn import utils
    from mrcnn import visualize
    from mrcnn.config import Config
    print("✓ Imported mrcnn base modules")
except ImportError as e:
    print(f"✗ Import mrcnn error: {e}")
    sys.exit(1)

try:
    from Plaques import PlaquesDataset, PlaquesConfig
    print("✓ Imported PlaquesDataset and PlaquesConfig from Plaques.py")
except ImportError as e:
    print(f"✗ Import from Plaques.py error: {e}")
    sys.exit(1)



def diagnose_dataset(dataset_dir):
    print("\n" + "="*70)
    print("DATASET DIAGNOSIS")
    print("="*70)

    val_dataset = PlaquesDataset()
    val_dataset.load_plaques(dataset_dir, "val")
    val_dataset.prepare()

    total_images = len(val_dataset.image_info)
    print(f"\nTotal validation images: {total_images}")

    has_bbox_count = 0
    no_bbox_count = 0
    bbox_counts = []
    bbox_distribution = {}

    for i in range(total_images):
        image_info = val_dataset.image_info[i]
        original_class_ids = image_info.get('original_class_ids', [])
        num_bboxes = len(original_class_ids)

        if num_bboxes > 0:
            has_bbox_count += 1
        else:
            no_bbox_count += 1

        bbox_counts.append(num_bboxes)
        bbox_distribution[num_bboxes] = bbox_distribution.get(num_bboxes, 0) + 1

    print(f"\nAnnotation Statistics:")
    print(f"  Images WITH bounding boxes: {has_bbox_count} ({has_bbox_count/total_images*100:.1f}%)")
    print(f"  Images WITHOUT bounding boxes: {no_bbox_count} ({no_bbox_count/total_images*100:.1f}%)")
    print(f"  Average bounding boxes per image: {sum(bbox_counts)/total_images:.2f}")
    print(f"  Min bounding boxes: {min(bbox_counts)}")
    print(f"  Max bounding boxes: {max(bbox_counts)}")

    print(f"\nBounding Box Distribution:")
    for num in sorted(bbox_distribution.keys())[:10]:
        print(f"  {num} bboxes: {bbox_distribution[num]} images")

    print(f"\nFirst 5 images:")
    for i in range(min(5, total_images)):
        image_info = val_dataset.image_info[i]
        original_class_ids = image_info.get('original_class_ids', [])
        polygons = image_info.get('polygons', [])
        print(f"  Image {i}: {os.path.basename(image_info['path'])} - {len(polygons)} bboxes")
        if len(polygons) > 0:
            print(f"    First bbox: {polygons[0]}")

    return val_dataset, has_bbox_count, no_bbox_count


def compute_iou(box1, box2):
    """
    box: [y1, x1, y2, x2]
    """
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_ap(precision, recall):
    """
    Average Precision (AP)
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def compute_pr_curve(all_predictions, total_gt, iou_threshold=0.5):
    """
    Precision-Recall

    Returns:
        precision: precision
        recall: recall
        ap: Average Precision
    """
    if total_gt == 0:
        return [], [], 0.0

    all_predictions.sort(key=lambda x: x['score'], reverse=True)

    tp = 0
    fp = 0
    matched_gt_set = set()
    precision = []
    recall = []

    for pred in all_predictions:
        if pred['matched_gt_idx'] >= 0 and pred['matched_gt_idx'] not in matched_gt_set:
            tp += 1
            matched_gt_set.add(pred['matched_gt_idx'])
        else:
            fp += 1

        precision.append(tp / (tp + fp))
        recall.append(tp / total_gt)

    ap = compute_ap(precision, recall)

    return precision, recall, ap


def compute_detection_metrics_iou(model, val_dataset, batch_indices,
                                   iou_threshold=0.5, conf_threshold=0.5,
                                   verbose=False):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_iou_scores = []
    all_conf_scores = []
    detailed_results = []

    all_predictions = []
    total_gt = 0

    for img_idx in batch_indices:
        try:
            image = val_dataset.load_image(img_idx)
            true_boxes = val_dataset.get_preloaded_boxes(img_idx)

            results = model.detect([image], verbose=0)[0]
            pred_boxes = results['rois']
            pred_scores = results['scores']

            mask = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]

            num_true = len(true_boxes)
            num_pred = len(pred_boxes)
            total_gt += num_true

            img_result = {
                'image_idx': img_idx,
                'num_true': num_true,
                'num_pred': num_pred,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'matched_iou': []
            }

            if num_true == 0 and num_pred == 0:
                pass
            elif num_true == 0:
                total_fp += num_pred
                img_result['fp'] = num_pred

                for pred_idx in range(num_pred):
                    all_predictions.append({
                        'score': pred_scores[pred_idx],
                        'matched_gt_idx': -1
                    })
            elif num_pred == 0:
                total_fn += num_true
                img_result['fn'] = num_true
            else:

                iou_matrix = np.zeros((num_pred, num_true))
                for i in range(num_pred):
                    for j in range(num_true):
                        iou_matrix[i, j] = compute_iou(pred_boxes[i], true_boxes[j])

                matched_gt = set()
                sorted_indices = np.argsort(pred_scores)[::-1]

                for pred_idx in sorted_indices:
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx in range(num_true):
                        if gt_idx in matched_gt:
                            continue
                        iou = iou_matrix[pred_idx, gt_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                        all_iou_scores.append(best_iou)
                        all_conf_scores.append(pred_scores[pred_idx])
                        img_result['tp'] += 1
                        img_result['matched_iou'].append(best_iou)
                        all_predictions.append({
                            'score': pred_scores[pred_idx],
                            'matched_gt_idx': best_gt_idx
                        })
                    else:
                        total_fp += 1
                        img_result['fp'] += 1
                        all_predictions.append({
                            'score': pred_scores[pred_idx],
                            'matched_gt_idx': -1
                        })

                unmatched_fn = num_true - len(matched_gt)
                total_fn += unmatched_fn
                img_result['fn'] = unmatched_fn

            detailed_results.append(img_result)

            if verbose and img_idx % 50 == 0:
                print(f"    Image {img_idx}: GT={num_true}, Pred={num_pred}, "
                      f"TP={img_result['tp']}, FP={img_result['fp']}, FN={img_result['fn']}")

        except Exception as e:
            print(f"    Error on image {img_idx}: {e}")
            continue

    precision_pr, recall_pr, ap = compute_pr_curve(all_predictions, total_gt, iou_threshold)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_gt': total_gt,
        'total_pred': total_tp + total_fp,
        'avg_iou': np.mean(all_iou_scores) if all_iou_scores else 0,
        'avg_conf': np.mean(all_conf_scores) if all_conf_scores else 0,
        'num_matched': len(all_iou_scores),
        'detailed_results': detailed_results,
        'pr_curve': {'precision': precision_pr, 'recall': recall_pr}
    }

    return metrics



def create_variant_config_class(use_per_level, use_top_level):
    class VariantConfig(PlaquesConfig):
        pass

    VariantConfig.USE_PER_LEVEL_ENHANCEMENT = use_per_level
    VariantConfig.USE_TOP_LEVEL_FUSION = use_top_level
    VariantConfig.NAME = f"ablation_p{int(use_per_level)}_t{int(use_top_level)}"

    if USE_GPU:
        VariantConfig.GPU_COUNT = 1
        VariantConfig.IMAGES_PER_GPU = 1
    else:
        VariantConfig.GPU_COUNT = 0
        VariantConfig.IMAGES_PER_GPU = 1

    VariantConfig.BATCH_SIZE = 1
    VariantConfig.USE_MASK = False
    VariantConfig.USE_MINI_MASK = False
    VariantConfig.IMAGE_RESIZE_MODE = "square"
    VariantConfig.IMAGE_MIN_DIM = 512
    VariantConfig.IMAGE_MAX_DIM = 512
    VariantConfig.IMAGE_CHANNEL_COUNT = 1
    VariantConfig.NUM_CLASSES = 2
    VariantConfig.DETECTION_MIN_CONFIDENCE = 0.5
    VariantConfig.DETECTION_NMS_THRESHOLD = 0.3
    VariantConfig.POST_NMS_ROIS_INFERENCE = 500
    VariantConfig.DETECTION_MAX_INSTANCES = 100

    return VariantConfig


def evaluate_variant(variant_name, model_path, use_per_level, use_top_level,
                     description, dataset_dir, output_dir,
                     iou_threshold=0.5, conf_threshold=0.5,
                     num_images=None, seed=42):

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {variant_name}")
    print(f"  Description: {description}")
    print(f"  USE_PER_LEVEL: {use_per_level}")
    print(f"  USE_TOP_LEVEL: {use_top_level}")
    print(f"  Model: {model_path}")
    print(f"{'=' * 70}")

    import tensorflow as tf

    if USE_GPU:
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.gpu_options.visible_device_list = "0"
    else:
        session_config = tf.ConfigProto(
            device_count={'GPU': 0},
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2
        )

    graph = tf.Graph()
    sess = tf.Session(graph=graph, config=session_config)

    result = {
        'variant_name': variant_name,
        'success': False,
        'metrics': None,
        'batch_metrics': [],
        'error': None
    }

    try:
        with graph.as_default():
            with sess.as_default():
                tf.keras.backend.set_session(sess)
                import mrcnn.model as modellib

                ConfigClass = create_variant_config_class(use_per_level, use_top_level)
                config = ConfigClass()

                print(f"\n  Config created:")
                print(f"    USE_PER_LEVEL_ENHANCEMENT: {config.USE_PER_LEVEL_ENHANCEMENT}")
                print(f"    USE_TOP_LEVEL_FUSION: {config.USE_TOP_LEVEL_FUSION}")

                val_dataset = PlaquesDataset()
                val_dataset.load_plaques(dataset_dir, "val")
                val_dataset.prepare()
                total_images = len(val_dataset.image_info)
                print(f"  Loaded {total_images} validation images")

                if num_images is None or num_images > total_images:
                    num_images = total_images

                random.seed(seed)
                all_indices = list(range(total_images))
                random.shuffle(all_indices)
                eval_indices = all_indices[:num_images]

                print(f"  Evaluating {len(eval_indices)} images (IoU={iou_threshold}, Conf={conf_threshold})")

                model = modellib.MaskRCNN(mode="inference", config=config, model_dir=output_dir)

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                model.load_weights(model_path, by_name=True)
                print(f"  ✓ Loaded weights")

                batch_size = 50
                num_batches = (len(eval_indices) + batch_size - 1) // batch_size
                all_batch_metrics = []

                print(f"\n  Starting evaluation...")
                print(f"  {'Batch':<8} {'Images':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AP':<10}")
                print(f"  {'-' * 60}")


                for batch_id in range(num_batches):
                    start_idx = batch_id * batch_size
                    end_idx = min(start_idx + batch_size, len(eval_indices))
                    batch_indices = eval_indices[start_idx:end_idx]

                    batch_metrics = compute_detection_metrics_iou(
                        model, val_dataset, batch_indices,
                        iou_threshold=iou_threshold,
                        conf_threshold=conf_threshold,
                        verbose=False
                    )

                    all_batch_metrics.append(batch_metrics)


                    print(f"  {batch_id + 1:<8} {len(batch_indices):<8} "
                          f"{batch_metrics['precision']:.4f}    "
                          f"{batch_metrics['recall']:.4f}    "
                          f"{batch_metrics['f1']:.4f}    "
                          f"{batch_metrics['ap']:.4f}")


                    if batch_metrics['detailed_results']:
                        batch_detail_file = os.path.join(output_dir, f"{variant_name}_batch{batch_id}_details.json")
                        with open(batch_detail_file, 'w') as f:
                            save_data = batch_metrics['detailed_results']
                            json.dump(save_data, f, indent=2)


                total_tp = sum(m['tp'] for m in all_batch_metrics)
                total_fp = sum(m['fp'] for m in all_batch_metrics)
                total_fn = sum(m['fn'] for m in all_batch_metrics)

                overall_ap = np.mean([m['ap'] for m in all_batch_metrics])
                overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                    if (overall_precision + overall_recall) > 0 else 0

                overall_metrics = {
                    'variant': variant_name,
                    'use_per_level': use_per_level,
                    'use_top_level': use_top_level,
                    'description': description,
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1,
                    'ap': overall_ap,
                    'tp': total_tp,
                    'fp': total_fp,
                    'fn': total_fn,
                    'total_gt': total_tp + total_fn,
                    'total_pred': total_tp + total_fp,
                    'num_images': len(eval_indices),
                    'iou_threshold': iou_threshold,
                    'conf_threshold': conf_threshold
                }

                result['success'] = True
                result['metrics'] = overall_metrics
                result['batch_metrics'] = all_batch_metrics

                print(f"\n Final Results for {variant_name}:")
                print(f"    AP: {overall_ap:.4f}")
                print(f"    F1: {overall_f1:.4f}")
                print(f"    Precision: {overall_precision:.4f}")
                print(f"    Recall: {overall_recall:.4f}")
                print(f"    TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

                del model
                sess.close()

    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f"\n  ✗ Error: {e}")
        print(traceback.format_exc())

    finally:
        tf.keras.backend.clear_session()
        gc.collect()

    return result


class AblationEvaluator:
    def __init__(self, dataset_dir, model_paths, output_dir,
                 iou_threshold=0.5, conf_threshold=0.5,
                 num_images=None):
        self.dataset_dir = dataset_dir
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.num_images = num_images

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"ablation_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.results_dir = os.path.join(self.session_dir, 'results')
        self.details_dir = os.path.join(self.session_dir, 'details')
        self.plots_dir = os.path.join(self.session_dir, 'plots')

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.details_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        print(f"\n✓ Results will be saved to: {self.session_dir}")

        self.variant_configs = {
            "baseline": {
                "use_per_level": False,
                "use_top_level": False,
                "description": "No enhancement modules"
            },
            "per_level_only": {
                "use_per_level": True,
                "use_top_level": False,
                "description": "Per-level Enhancement only"
            },
            "top_level_only": {
                "use_per_level": False,
                "use_top_level": True,
                "description": "Top-level Fusion only"
            },
            "full": {
                "use_per_level": True,
                "use_top_level": True,
                "description": "Both modules"
            }
        }

    def find_checkpoint(self, model_path):
        if os.path.isfile(model_path) and model_path.endswith('.h5'):
            return model_path
        elif os.path.isdir(model_path):
            h5_files = []
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.h5'):
                        h5_files.append(os.path.join(root, file))
            if h5_files:
                h5_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return h5_files[0]
        return None

    def run(self):
        print("\n" + "="*70)
        print("STARTING ABLATION STUDY")
        print("="*70)
        print(f"IoU Threshold: {self.iou_threshold}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Number of images: {self.num_images if self.num_images else 'all'}")
        print("="*70)

        all_results = []

        for variant_name, model_path in self.model_paths.items():
            if variant_name not in self.variant_configs:
                print(f"\n⚠ Unknown variant: {variant_name}, skipping...")
                continue

            ckpt_file = self.find_checkpoint(model_path)
            if ckpt_file is None:
                print(f"\n⚠ No checkpoint found for {variant_name} at {model_path}")
                continue

            config = self.variant_configs[variant_name]

            result = evaluate_variant(
                variant_name=variant_name,
                model_path=ckpt_file,
                use_per_level=config['use_per_level'],
                use_top_level=config['use_top_level'],
                description=config['description'],
                dataset_dir=self.dataset_dir,
                output_dir=self.details_dir,
                iou_threshold=self.iou_threshold,
                conf_threshold=self.conf_threshold,
                num_images=self.num_images
            )

            all_results.append(result)

            if result['success']:
                result_file = os.path.join(self.results_dir, f"{variant_name}_metrics.json")
                with open(result_file, 'w') as f:
                    json.dump(result['metrics'], f, indent=2)

        self.generate_report(all_results)
        return all_results

    def generate_report(self, results):
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)

        successful = [r for r in results if r['success']]

        if not successful:
            print("No successful results!")
            return

        report_data = []
        for r in successful:
            m = r['metrics']
            report_data.append({
                'Variant': r['variant_name'],
                'USE_PER_LEVEL': m['use_per_level'],
                'USE_TOP_LEVEL': m['use_top_level'],
                'AP': f"{m['ap']:.4f}",
                'F1': f"{m['f1']:.4f}",
                'Precision': f"{m['precision']:.4f}",
                'Recall': f"{m['recall']:.4f}",
                'TP': m['tp'],
                'FP': m['fp'],
                'FN': m['fn'],
                'Total_GT': m['total_gt'],
                'Total_Pred': m['total_pred']
            })

        df_report = pd.DataFrame(report_data)
        csv_path = os.path.join(self.session_dir, 'ablation_results.csv')
        df_report.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")

        print("\nResults Summary:")
        print(df_report[['Variant', 'AP', 'F1', 'Precision', 'Recall']].to_string(index=False))

        report_path = os.path.join(self.session_dir, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ABLATION STUDY REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write("="*80 + "\n\n")
            f.write(df_report.to_string(index=False) + "\n\n")

            best_f1_idx = np.argmax([float(r['metrics']['f1']) for r in successful])
            best = successful[best_f1_idx]
            f.write("\n" + "="*80 + "\n")
            f.write("BEST MODEL BY F1 SCORE\n")
            f.write("="*80 + "\n")
            f.write(f"Variant: {best['variant_name']}\n")
            f.write(f"F1 Score: {best['metrics']['f1']:.4f}\n")
            f.write(f"AP: {best['metrics']['ap']:.4f}\n")
            f.write(f"Precision: {best['metrics']['precision']:.4f}\n")
            f.write(f"Recall: {best['metrics']['recall']:.4f}\n")

            best_ap_idx = np.argmax([float(r['metrics']['ap']) for r in successful])
            best_ap = successful[best_ap_idx]
            f.write("\n" + "="*80 + "\n")
            f.write("BEST MODEL BY AP\n")
            f.write("="*80 + "\n")
            f.write(f"Variant: {best_ap['variant_name']}\n")
            f.write(f"AP: {best_ap['metrics']['ap']:.4f}\n")
            f.write(f"F1 Score: {best_ap['metrics']['f1']:.4f}\n")

        print(f"\n✓ Report saved to: {report_path}")
        self.plot_metrics_comparison(successful)

    def plot_metrics_comparison(self, results):
        try:
            import matplotlib.pyplot as plt

            variants = [r['variant_name'] for r in results]
            ap_scores = [r['metrics']['ap'] for r in results]
            f1_scores = [r['metrics']['f1'] for r in results]
            precision = [r['metrics']['precision'] for r in results]
            recall = [r['metrics']['recall'] for r in results]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(variants)]

            x = np.arange(len(variants))
            width = 0.35

            bars1 = axes[0].bar(x - width/2, ap_scores, width, label='AP', color='red', alpha=0.7)
            bars2 = axes[0].bar(x + width/2, f1_scores, width, label='F1', color='blue', alpha=0.7)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(variants, rotation=45, ha='right')
            axes[0].set_ylabel('Score')
            axes[0].set_title('AP vs F1 Score')
            axes[0].set_ylim([0, 1])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            for bar, score in zip(bars1, ap_scores):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, score in zip(bars2, f1_scores):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)

            axes[1].bar(x - width, precision, width, label='Precision', color='green', alpha=0.7)
            axes[1].bar(x, recall, width, label='Recall', color='orange', alpha=0.7)
            axes[1].bar(x + width, ap_scores, width, label='AP', color='red', alpha=0.7)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(variants, rotation=45, ha='right')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Precision vs Recall vs AP')
            axes[1].set_ylim([0, 1])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, 'metrics_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Metrics comparison plot saved to: {plot_path}")

        except Exception as e:
            print(f"⚠ Could not generate plot: {e}")



if __name__ == "__main__":
    DATASET_DIR = "../image/"

    VARIANT_MODEL_PATHS = {
        "baseline": "2D_method_plaques_0140.h5",
        "per_level_only": "a.h5",
        "top_level_only": "b.h5",
        "full": "c.h5"
    }

    OUTPUT_DIR = "../ablation_results/"

    IOU_THRESHOLD = 0.5
    CONF_THRESHOLD = 0.5
    NUM_IMAGES = None  # None: Evaluate all; Specific figure: Number of images evaluated

    print("\n" + "="*80)
    print("ABLATION STUDY CONFIGURATION")
    print("="*80)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"IoU Threshold: {IOU_THRESHOLD}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"Number of images: {NUM_IMAGES if NUM_IMAGES else 'all'}")
    print(f"Using GPU: {USE_GPU}")
    print("="*80)

    # 诊断数据集
    val_dataset, has_bbox, no_bbox = diagnose_dataset(DATASET_DIR)

    if no_bbox == 0:
        print("\n WARNING: All validation images have bounding boxes!")

    evaluator = AblationEvaluator(
        dataset_dir=DATASET_DIR,
        model_paths=VARIANT_MODEL_PATHS,
        output_dir=OUTPUT_DIR,
        iou_threshold=IOU_THRESHOLD,
        conf_threshold=CONF_THRESHOLD,
        num_images=NUM_IMAGES
    )

    try:
        results = evaluator.run()
        print("\n" + "="*80)
        print("✓ ABLATION STUDY COMPLETED SUCCESSFULLY!")
        print(f"✓ Results saved to: {evaluator.session_dir}")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()