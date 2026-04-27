"""
Visualisation of the prediction results.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import colorsys
from datetime import datetime
import json


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def save_boxes_only(image, boxes, save_dir="./results", prefix="boxes_only"):
    os.makedirs(save_dir, exist_ok=True)

    if len(boxes) == 0:
        print(f"  No boxes to save for {prefix}")
        return

    colors = random_colors(len(boxes))

    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.imshow(image.astype(np.uint8))
    ax.axis('off')
    ax.set_ylim(image.shape[0] + 10, -10)
    ax.set_xlim(-10, image.shape[1] + 10)

    for i, box in enumerate(boxes):
        if not np.any(box):
            continue
        y1, x1, y2, x2 = box
        color = colors[i]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, linestyle='solid',
            edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

    jpg_path = os.path.join(save_dir, f"{prefix}.jpg")
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=150)

    png_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)

    plt.close(fig)
    print(f"  Saved: {jpg_path}")
    print(f"  Saved: {png_path}")


def save_boxes_with_confidence(image, boxes, scores, save_dir="./results", prefix="boxes_with_conf"):
    os.makedirs(save_dir, exist_ok=True)

    if len(boxes) == 0:
        print(f"  No boxes to save for {prefix}")
        return

    colors = random_colors(len(boxes))

    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.imshow(image.astype(np.uint8))
    ax.axis('off')
    ax.set_ylim(image.shape[0] + 10, -10)
    ax.set_xlim(-10, image.shape[1] + 10)

    for i, box in enumerate(boxes):
        if not np.any(box):
            continue
        y1, x1, y2, x2 = box
        color = colors[i]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, linestyle='solid',
            edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        score = scores[i] if scores is not None else 0
        ax.text(x1, y1 - 5, f"{score:.2f}",
                fontsize=8, color='white',
                bbox=dict(facecolor=color, alpha=0.7, pad=1))

    jpg_path = os.path.join(save_dir, f"{prefix}.jpg")
    png_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
    plt.close(fig)


def save_single_instance_boxes(image, boxes, save_dir="./results/single_instances"):
    os.makedirs(save_dir, exist_ok=True)

    if len(boxes) == 0:
        print("  No instances to save")
        return

    colors = random_colors(len(boxes))

    for i, box in enumerate(boxes):
        if not np.any(box):
            continue

        y1, x1, y2, x2 = box
        color = colors[i]

        fig, ax = plt.subplots(1, figsize=(16, 16))
        ax.imshow(image.astype(np.uint8))
        ax.axis('off')
        ax.set_ylim(image.shape[0] + 10, -10)
        ax.set_xlim(-10, image.shape[1] + 10)

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, linestyle='solid',
            edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        jpg_path = os.path.join(save_dir, f"instance_{i:03d}.jpg")
        png_path = os.path.join(save_dir, f"instance_{i:03d}.png")
        plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150, transparent=True)
        plt.close(fig)

    print(f"  Saved {len(boxes)} individual instance visualizations to {save_dir}")


def save_all_visualizations(image, boxes, scores=None, save_dir="./detection_results", prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving visualizations to: {save_dir}")

    original_dir = os.path.join(save_dir, "original")
    os.makedirs(original_dir, exist_ok=True)
    orig_jpg = os.path.join(original_dir, f"{prefix}original.jpg")
    orig_png = os.path.join(original_dir, f"{prefix}original.png")
    plt.imsave(orig_jpg, image.astype(np.uint8))
    plt.imsave(orig_png, image.astype(np.uint8))
    print(f"  Saved original image")

    boxes_dir = os.path.join(save_dir, "boxes_only")
    save_boxes_only(image, boxes, boxes_dir, f"{prefix}boxes_only")

    if scores is not None and len(scores) > 0:
        conf_dir = os.path.join(save_dir, "with_confidence")
        save_boxes_with_confidence(image, boxes, scores, conf_dir, f"{prefix}with_conf")

    instances_dir = os.path.join(save_dir, "single_instances")
    save_single_instance_boxes(image, boxes, instances_dir)


def inference_and_visualize(model, image_path, save_dir="./detection_results",
                            confidence_threshold=0.5):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    prefix = f"{timestamp}{img_name}_"

    result_dir = os.path.join(save_dir, img_name)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\nRunning inference on: {image_path}")
    results = model.detect([image], verbose=0)
    r = results[0]

    boxes = r['rois']
    scores = r['scores']

    if len(scores) > 0:
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]

    print(f"  Detected {len(boxes)} objects (threshold={confidence_threshold})")

    for i, (box, score) in enumerate(zip(boxes, scores)):
        y1, x1, y2, x2 = box
        print(f"    {i + 1}: box=[{x1},{y1},{x2},{y2}], score={score:.3f}")

    save_all_visualizations(image, boxes, scores, result_dir, prefix)

    return {
        'boxes': boxes,
        'scores': scores,
        'image_path': image_path,
        'save_dir': result_dir
    }


def batch_inference_and_visualize(model, image_dir, save_dir="./detection_results",
                                  confidence_threshold=0.5, extensions=['.jpg', '.png', '.jpeg']):
    """
    Batch process all images in the folder
    """
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])

    print(f"\nFound {len(image_files)} images in {image_dir}")

    results_summary = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        result = inference_and_visualize(model, img_path, save_dir, confidence_threshold)
        if result:
            results_summary.append(result)

    summary_path = os.path.join(save_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(results_summary),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence_threshold': confidence_threshold,
            'results': [
                {
                    'image': r['image_path'],
                    'num_detections': len(r['boxes']),
                    'save_dir': r['save_dir']
                } for r in results_summary
            ]
        }, f, indent=2)

    print(f"\nBatch processing completed! Summary saved to {summary_path}")
    return results_summary


def test_model_on_validation_set(model, dataset_val, save_dir, num_samples=10, confidence_threshold=0.5):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n" + "=" * 70)
    print(f"Testing model on validation set ({min(num_samples, len(dataset_val.image_ids))} samples)")
    print("=" * 70)

    for i in range(min(num_samples, len(dataset_val.image_ids))):
        image = dataset_val.load_image(i)
        image_id = dataset_val.image_ids[i]

        image_info = dataset_val.image_info[i]
        gt_boxes = dataset_val.get_preloaded_boxes(i) if hasattr(dataset_val, 'get_preloaded_boxes') else []

        results = model.detect([image], verbose=0)[0]
        pred_boxes = results['rois']
        pred_scores = results['scores']

        if len(pred_scores) > 0:
            keep = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

        img_name = f"val_image_{i:03d}_{image_id}"
        save_boxes_only(image, pred_boxes, save_dir, f"{img_name}_pred")

        if len(gt_boxes) > 0:
            save_boxes_only(image, gt_boxes, save_dir, f"{img_name}_gt")

        print(f"  Image {i}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}")

    print(f"\nTest visualizations saved to: {save_dir}")