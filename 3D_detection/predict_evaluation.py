'''
Model prediction and evaluation.
Supports both single-model and batch-model modes.
'''

import torch
import numpy as np
import os
import time
from darknet import Darknet
from util import *
import warnings
from torch.cuda.amp import autocast
import pandas as pd
import json
import glob
from pathlib import Path

warnings.filterwarnings("ignore")

CFGFILE = "./cfg/yolov3_drop.cfg"

# MODE = "single"  # Evaluate only a single model
# MODE = "batch"   # Batch evaluate all models in the folder
MODE = "batch"

SINGLE_WEIGHT = "3D_plaque_detection.pth"

# Batch mode configuration
WEIGHTS_FOLDER = "./models/"  # The path to the folder containing the models
WEIGHTS_PATTERN = "*.pth"
# WEIGHTS_LIST = ["model1.pth", "model2.pth", "model3.pth"]
WEIGHTS_LIST = []  # Automatic search


DATA_PATH = "./valid/image/"
INP_DIM = 128

CONFIDENCE = 0.5
NMS_THRESH = 0.7
IOU_THRESHOLD = 0.5


BATCH_SIZE = 8
NUM_WORKERS = 4


OUTPUT_DIR = "./models/evaluation"
SAVE_DETAILED = True  # Whether to save the detailed forecast results
SAVE_IMAGE_PREDICTIONS = True  # Save the prediction results for each image in a .txt file
# txt：class_id confidence x_center y_center z_center width height length



class Predictor:
    def __init__(self, weights_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights_path = weights_path
        self.model_name = Path(weights_path).stem

        self.model = Darknet(CFGFILE)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.anchors = (
            [
                (48, 48, 64), (56, 56, 72), (64, 64, 80),
                (72, 72, 88), (80, 80, 96), (88, 88, 96),
                (96, 96, 96), (80, 96, 80), (96, 80, 80)
            ],
            [
                (24, 24, 32), (28, 28, 40), (32, 32, 48),
                (36, 36, 52), (40, 40, 56), (44, 44, 60),
                (48, 48, 64), (40, 56, 40), (56, 40, 40)
            ],
            [
                (8, 8, 12), (10, 10, 16), (12, 12, 20),
                (14, 14, 24), (16, 16, 28), (18, 18, 30),
                (20, 20, 32), (16, 28, 16), (28, 16, 16)
            ]
        )

    def calculate_3d_iou(self, box1, box2):
        # [center_x, center_y, center_z, width, height, length]

        x1_min = box1[0] - box1[3] / 2
        x1_max = box1[0] + box1[3] / 2
        y1_min = box1[1] - box1[4] / 2
        y1_max = box1[1] + box1[4] / 2
        z1_min = box1[2] - box1[5] / 2
        z1_max = box1[2] + box1[5] / 2

        x2_min = box2[0] - box2[3] / 2
        x2_max = box2[0] + box2[3] / 2
        y2_min = box2[1] - box2[4] / 2
        y2_max = box2[1] + box2[4] / 2
        z2_min = box2[2] - box2[5] / 2
        z2_max = box2[2] + box2[5] / 2

        inter_xmin = max(x1_min, x2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymin = max(y1_min, y2_min)
        inter_ymax = min(y1_max, y2_max)
        inter_zmin = max(z1_min, z2_min)
        inter_zmax = min(z1_max, z2_max)

        if inter_xmax > inter_xmin and inter_ymax > inter_ymin and inter_zmax > inter_zmin:
            inter_volume = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin) * (inter_zmax - inter_zmin)
            box1_volume = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
            box2_volume = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
            iou = inter_volume / (box1_volume + box2_volume - inter_volume + 1e-6)
        else:
            iou = 0

        return iou

    def convert_to_binary_class(self, labels):
        binary_labels = []
        for label in labels:
            binary_label = [1.0] + list(label[1:7])
            binary_labels.append(binary_label)
        return binary_labels

    def calculate_metrics_per_image(self, predictions, targets):
        binary_targets = self.convert_to_binary_class(targets)

        if len(predictions) == 0:
            return {
                'tp': 0, 'fp': 0, 'fn': len(binary_targets),
                'precision': 0, 'recall': 0, 'accuracy': 0, 'f1_score': 0,
                'num_predictions': 0, 'num_targets': len(binary_targets)
            }

        predictions_sorted = sorted(predictions, key=lambda x: x[0], reverse=True)
        targets_matched = [False] * len(binary_targets)
        tp = 0
        fp = 0

        for pred in predictions_sorted:
            pred_conf = pred[0]
            pred_bbox = pred[1:7]

            best_iou = 0
            best_target_idx = -1

            for idx, target in enumerate(binary_targets):
                if targets_matched[idx]:
                    continue
                target_bbox = target[1:7]
                iou = self.calculate_3d_iou(pred_bbox, target_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = idx

            if best_iou >= IOU_THRESHOLD and best_target_idx >= 0:
                tp += 1
                targets_matched[best_target_idx] = True
            else:
                fp += 1

        fn = len(binary_targets) - sum(targets_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall,
            'accuracy': accuracy, 'f1_score': f1,
            'num_predictions': len(predictions), 'num_targets': len(binary_targets)
        }

    def save_image_predictions(self, image_name, predictions, output_dir):
        pred_dir = os.path.join(output_dir, 'predictions_txt')
        os.makedirs(pred_dir, exist_ok=True)

        txt_path = os.path.join(pred_dir, f"{image_name}.txt")
        with open(txt_path, 'w') as f:
            if predictions:
                for pred in predictions:
                    # class_id confidence x_center y_center z_center width height length
                    # class_id=1
                    f.write(f"1 {pred[0]:.6f} {pred[1]:.6f} {pred[2]:.6f} {pred[3]:.6f} "
                            f"{pred[4]:.6f} {pred[5]:.6f} {pred[6]:.6f}\n")

        return txt_path

    def predict_batch(self, dataloader, output_dir=None):
        all_images_metrics = []
        all_predictions_binary = []
        all_targets_binary = []
        image_predictions_details = {}

        all_image_paths = None
        if hasattr(dataloader.dataset, 'img_files'):
            all_image_paths = dataloader.dataset.img_files
        elif hasattr(dataloader.dataset, 'images'):
            all_image_paths = [str(p) for p in dataloader.dataset.images]

        global_img_idx = 0

        for batch_idx, sample_batched in enumerate(dataloader):
            batch_start_time = time.time()

            inputs = sample_batched['input_img']
            labels = sample_batched['label']

            batch_image_paths = None
            if all_image_paths is not None:
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(all_image_paths))
                batch_image_paths = all_image_paths[start_idx:end_idx]

            inputs = inputs.to(self.device)

            batch_tp = 0
            batch_fp = 0
            batch_fn = 0
            batch_num_pred = 0
            batch_num_target = 0
            batch_precision = 0
            batch_recall = 0
            batch_f1 = 0

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    final_pre = self.model(inputs, torch.cuda.is_available())

                    scaled_predictions = []

                    for i in range(len(final_pre)):
                        pred = final_pre[i]
                        batch_size = pred.size(0)
                        nG = [4, 8, 16][i]
                        nA = 18

                        if i == 0:
                            expected = 1152
                            if pred.size(1) > expected:
                                pred = pred[:, :expected, :]
                        elif i == 1:
                            expected = 9216
                            if pred.size(1) > expected:
                                pred = pred[:, :expected, :]
                        else:
                            expected = 73728
                            if pred.size(1) > expected:
                                pred = pred[:, :expected, :]

                        pred = pred.view(batch_size, nA, nG, nG, nG, 7 + 4)
                        pred = pred.permute(0, 1, 5, 2, 3, 4).contiguous()
                        scaled_predictions.append(pred)

                    write_res = write_results(
                        scaled_predictions,
                        confidence=CONFIDENCE,
                        num_classes=4,
                        nms_conf=NMS_THRESH
                    )


                    predictions_by_image = {i: [] for i in range(len(inputs))}

                    if not isinstance(write_res, int) and len(write_res) > 0:
                        for pred in write_res:
                            global_img_id = int(pred[0])
                            bbox = pred[1:7].cpu().numpy()
                            obj_conf = float(pred[7])
                            class_conf = float(pred[8])
                            pred_cls = int(pred[9])

                            img_in_batch_idx = global_img_id - global_img_idx

                            if 0 <= img_in_batch_idx < len(inputs):
                                if pred_cls >= 1 and pred_cls <= 4:
                                    x_center = (bbox[0] + bbox[3]) / 2
                                    y_center = (bbox[1] + bbox[4]) / 2
                                    z_center = (bbox[2] + bbox[5]) / 2
                                    width = bbox[3] - bbox[0]
                                    height = bbox[4] - bbox[1]
                                    length = bbox[5] - bbox[2]

                                    final_conf = obj_conf * class_conf

                                    predictions_by_image[img_in_batch_idx].append(
                                        [final_conf, x_center, y_center, z_center, width, height, length]
                                    )

                    targets_by_image = {i: [] for i in range(labels.shape[0])}
                    for b in range(labels.shape[0]):
                        for t in range(labels.shape[1]):
                            if labels[b, t].sum() != 0:
                                target = labels[b, t].cpu().numpy()
                                if target[0] >= 1 and target[0] <= 4:
                                    targets_by_image[b].append(target)

                    for img_idx in range(len(inputs)):
                        img_predictions = predictions_by_image[img_idx]
                        img_targets = targets_by_image[img_idx]

                        if batch_image_paths is not None and img_idx < len(batch_image_paths):
                            img_path = batch_image_paths[img_idx]
                            img_name = Path(img_path).stem
                        else:
                            img_name = f"image_{global_img_idx + img_idx}"

                        img_metrics = self.calculate_metrics_per_image(img_predictions, img_targets)
                        img_metrics['image_name'] = img_name
                        img_metrics['batch_idx'] = batch_idx
                        img_metrics['global_img_idx'] = global_img_idx + img_idx
                        img_metrics['image_idx_in_batch'] = img_idx
                        all_images_metrics.append(img_metrics)

                        if output_dir and SAVE_IMAGE_PREDICTIONS:
                            txt_path = self.save_image_predictions(img_name, img_predictions, output_dir)
                            img_metrics['predictions_file'] = txt_path

                        for pred in img_predictions:
                            all_predictions_binary.append(pred)
                        for target in img_targets:
                            all_targets_binary.append(target[1:7])

                        image_predictions_details[img_name] = {
                            'predictions': img_predictions,
                            'targets': img_targets,
                            'metrics': img_metrics
                        }

                    if len(inputs) > 0 and all_images_metrics:
                        start_idx = len(all_images_metrics) - len(inputs)
                        end_idx = len(all_images_metrics)
                        batch_metrics_list = all_images_metrics[start_idx:end_idx]
                        batch_tp = sum(m['tp'] for m in batch_metrics_list)
                        batch_fp = sum(m['fp'] for m in batch_metrics_list)
                        batch_fn = sum(m['fn'] for m in batch_metrics_list)
                        batch_num_pred = sum(m['num_predictions'] for m in batch_metrics_list)
                        batch_num_target = sum(m['num_targets'] for m in batch_metrics_list)

                        batch_precision = batch_tp / (batch_tp + batch_fp) if (batch_tp + batch_fp) > 0 else 0
                        batch_recall = batch_tp / (batch_tp + batch_fn) if (batch_tp + batch_fn) > 0 else 0
                        batch_f1 = 2 * batch_precision * batch_recall / (batch_precision
                                                                         + batch_recall) if (batch_precision + batch_recall) > 0 else 0

            batch_time = time.time() - batch_start_time

            print(f"batch {batch_idx + 1:3d}/{len(dataloader)} | "
                  f"time: {batch_time:.2f}s | "
                  f"pred: {batch_num_pred:3d} | "
                  f"target: {batch_num_target:3d} | "
                  f"TP: {batch_tp:3d} | "
                  f"FP: {batch_fp:3d} | "
                  f"FN: {batch_fn:3d} | "
                  f"P: {batch_precision:.4f} | "
                  f"R: {batch_recall:.4f} | "
                  f"F1: {batch_f1:.4f}")

            global_img_idx += len(inputs)

        return all_images_metrics, all_predictions_binary, all_targets_binary, image_predictions_details

    def calculate_ap(self, image_predictions_details):
        all_tp_fp = []
        total_targets = 0

        for img_name, img_info in image_predictions_details.items():
            predictions = img_info['predictions']
            targets = img_info['targets']

            if len(predictions) == 0:
                total_targets += len(targets)
                continue

            binary_targets = self.convert_to_binary_class(targets)

            predictions_sorted = sorted(predictions, key=lambda x: x[0], reverse=True)
            targets_matched = [False] * len(binary_targets)

            for pred in predictions_sorted:
                pred_bbox = pred[1:7]

                best_iou = 0
                best_target_idx = -1

                for idx, target in enumerate(binary_targets):
                    if targets_matched[idx]:
                        continue
                    target_bbox = target[1:7]
                    iou = self.calculate_3d_iou(pred_bbox, target_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = idx

                if best_iou >= IOU_THRESHOLD and best_target_idx >= 0:
                    all_tp_fp.append(1)  # TP
                    targets_matched[best_target_idx] = True
                else:
                    all_tp_fp.append(0)  # FP

            total_targets += len(binary_targets)

        if len(all_tp_fp) == 0 or total_targets == 0:
            return 0.0, [], []

        tp_cumsum = np.cumsum(all_tp_fp)
        fp_cumsum = np.cumsum([1 - x for x in all_tp_fp])

        recalls = tp_cumsum / total_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        recalls = np.concatenate(([0], recalls))
        precisions = np.concatenate(([1], precisions))

        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        return ap, recalls.tolist(), precisions.tolist()

    def evaluate(self, dataloader, output_dir):

        model_output_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        images_metrics, all_predictions, all_targets, image_details = self.predict_batch(dataloader, model_output_dir)

        total_tp = sum(m['tp'] for m in images_metrics)
        total_fp = sum(m['fp'] for m in images_metrics)
        total_fn = sum(m['fn'] for m in images_metrics)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision +
                                                               overall_recall) if (overall_precision + overall_recall) > 0 else 0

        ap, recalls_curve, precisions_curve = self.calculate_ap(image_details)

        print("\n" + "-" * 100)
        print(f"{self.model_name} :")
        print("-" * 100)
        print(f"  Precision: {overall_precision:.4f} ({overall_precision * 100:.2f}%)")
        print(f"  Recall:    {overall_recall:.4f} ({overall_recall * 100:.2f}%)")
        print(f"  Accuracy:  {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
        print(f"  F1-Score:  {overall_f1:.4f}")
        print(f"  AP:        {ap:.4f} ({ap * 100:.2f}%)")
        print("-" * 100)

        images_df = pd.DataFrame(images_metrics)
        images_df = images_df[[
            'image_name', 'num_predictions', 'num_targets',
            'tp', 'fp', 'fn', 'precision', 'recall', 'accuracy', 'f1_score'
        ]]
        images_df.to_csv(os.path.join(model_output_dir, 'per_image_metrics.csv'), index=False)

        batch_summary = []
        for batch_idx in sorted(set(m['batch_idx'] for m in images_metrics)):
            batch_images = [m for m in images_metrics if m['batch_idx'] == batch_idx]
            batch_tp = sum(m['tp'] for m in batch_images)
            batch_fp = sum(m['fp'] for m in batch_images)
            batch_fn = sum(m['fn'] for m in batch_images)
            batch_precision = batch_tp / (batch_tp + batch_fp) if (batch_tp + batch_fp) > 0 else 0
            batch_recall = batch_tp / (batch_tp + batch_fn) if (batch_tp + batch_fn) > 0 else 0
            batch_f1 = 2 * batch_precision * batch_recall / (
                    batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0

            batch_summary.append({
                'batch_idx': batch_idx,
                'num_images': len(batch_images),
                'total_predictions': sum(m['num_predictions'] for m in batch_images),
                'total_targets': sum(m['num_targets'] for m in batch_images),
                'tp': batch_tp,
                'fp': batch_fp,
                'fn': batch_fn,
                'precision': batch_precision,
                'recall': batch_recall,
                'f1_score': batch_f1
            })

        batch_df = pd.DataFrame(batch_summary)
        batch_df.to_csv(os.path.join(model_output_dir, 'batch_summary.csv'), index=False)

        results = {
            'model_name': self.model_name,
            'weights_path': self.weights_path,
            'overall': {
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
                'total_predictions': sum(m['num_predictions'] for m in images_metrics),
                'total_targets': sum(m['num_targets'] for m in images_metrics),
                'precision': overall_precision, 'recall': overall_recall,
                'accuracy': overall_accuracy, 'f1_score': overall_f1, 'ap': ap
            },
            'pr_curve': {'recalls': recalls_curve, 'precisions': precisions_curve}
        }

        with open(os.path.join(model_output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(model_output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"  TP: {total_tp}\n")
            f.write(f"  FP: {total_fp}\n")
            f.write(f"  FN: {total_fn}\n")
            f.write(f"  Precision: {overall_precision:.4f}\n")
            f.write(f"  Recall: {overall_recall:.4f}\n")
            f.write(f"  Accuracy: {overall_accuracy:.4f}\n")
            f.write(f"  F1-Score: {overall_f1:.4f}\n")
            f.write(f"  AP: {ap:.4f}\n\n")

        if SAVE_IMAGE_PREDICTIONS:
            print(f"  - predictions_txt/: A text file containing the prediction results for each image")

        return {
            'model_name': self.model_name,
            'precision': overall_precision,
            'recall': overall_recall,
            'accuracy': overall_accuracy,
            'f1_score': overall_f1,
            'ap': ap,
            'total_images': len(images_metrics)
        }


def get_weights_list():
    if MODE == "single":
        if not os.path.exists(SINGLE_WEIGHT):
            print(f"ERROR: {SINGLE_WEIGHT}")
            return []
        return [SINGLE_WEIGHT]

    elif MODE == "batch":
        if WEIGHTS_LIST:
            weights_list = [w for w in WEIGHTS_LIST if os.path.exists(w)]
            if len(weights_list) < len(WEIGHTS_LIST):
                print(f"{len(WEIGHTS_LIST) - len(weights_list)} no exist")
            return weights_list
        else:
            pattern = os.path.join(WEIGHTS_FOLDER, WEIGHTS_PATTERN)
            weights_list = glob.glob(pattern)
            if not weights_list:
                print(f" {WEIGHTS_FOLDER} no match {WEIGHTS_PATTERN}")
            return sorted(weights_list)

    else:
        return []


def main():

    if not os.path.exists(DATA_PATH):
        return

    weights_list = get_weights_list()
    if not weights_list:
        return

    print(f"\nMode: {MODE}")
    if MODE == "batch":
        print(f"Folder: {WEIGHTS_FOLDER}")

    dataset = ListDataset(DATA_PATH, img_size=INP_DIM)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    total_start_time = time.time()

    for i, weights_path in enumerate(weights_list, 1):
        try:
            predictor = Predictor(weights_path)
            result = predictor.evaluate(dataloader, OUTPUT_DIR)
            all_results.append(result)

        except Exception as e:
            print(f"\n ERROR: {weights_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    if len(all_results) > 1:
        comparison_df = pd.DataFrame(all_results)
        comparison_df = comparison_df.set_index('model_name')

        print("\n", comparison_df.round(4))

        best_ap = max(all_results, key=lambda x: x['ap'])
        best_f1 = max(all_results, key=lambda x: x['f1_score'])
        best_precision = max(all_results, key=lambda x: x['precision'])
        best_recall = max(all_results, key=lambda x: x['recall'])

        print("\n" + "-" * 100)
        print("Best model:")
        print(f"  Highest AP: {best_ap['model_name']} (AP={best_ap['ap']:.4f})")
        print(f"  Highest F1: {best_f1['model_name']} (F1={best_f1['f1_score']:.4f})")
        print(f"  Highest Precision: {best_precision['model_name']} (Precision={best_precision['precision']:.4f})")
        print(f"  Highest Recall: {best_recall['model_name']} (Recall={best_recall['recall']:.4f})")
        print("-" * 100)

        comparison_file = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file)

        comparison_json = os.path.join(OUTPUT_DIR, 'model_comparison.json')
        with open(comparison_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 100)



if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()