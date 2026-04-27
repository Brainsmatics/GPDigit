'''
Visualisation of prediction results
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = "../"
PLAQUES_DIR = os.path.join(ROOT_DIR, "samples", "Plaques")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PLAQUES_DIR)

from mrcnn import utils
import mrcnn.model as modellib
try:
    import Plaques
except Exception as e:
    import traceback
    traceback.print_exc()


MODEL_DIR = ROOT_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "2D_plaque_detection.h5")
IMAGE_DIR = "../"
FMTEST_DIR = os.path.join(IMAGE_DIR, "visual_test")

if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)


class InferenceConfig(Plaques.PlaquesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"


def draw_detections_on_grayscale(image, detections, output_path, line_width=3):

    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image.copy()

    # Convert the greyscale image to a 3-channel RGB image so that a coloured border can be drawn
    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 165, 0),
        (255, 192, 203),
        (0, 128, 128)
    ]

    if detections.shape[0] > 0:
        for i, det in enumerate(detections):
            y1, x1, y2, x2 = det[:4].astype(np.int32)
            class_id = int(det[4])
            score = det[5]

            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(gray_rgb.shape[0], y2), min(gray_rgb.shape[1], x2)

            color = colors[i % len(colors)]

            cv2.rectangle(gray_rgb, (x1, y1), (x2, y2), color, line_width)

    cv2.imwrite(output_path, gray_rgb)

    return gray_rgb


def visualize_detections_simple(image, boxes, class_ids, scores, output_path, line_width=3):

    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image.copy()

    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]

    for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
        if score > 0.5:
            y1, x1, y2, x2 = box.astype(np.int32)
            y1, x1 = max(0, y1), max(0, x1)
            y2, x2 = min(gray_rgb.shape[0], y2), min(gray_rgb.shape[1], x2)

            color = colors[i % len(colors)]
            cv2.rectangle(gray_rgb, (x1, y1), (x2, y2), color, line_width)

    cv2.imwrite(output_path, gray_rgb)
    print(f"检测结果已保存至: {output_path}")

    return gray_rgb


if __name__ == "__main__":

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)

    dataset_val = Plaques.PlaquesDataset()
    dataset_val.load_plaques(IMAGE_DIR, "val")
    dataset_val.prepare()

    img_id = 57
    img, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val, config, img_id, use_mini_mask=False
    )
    info = dataset_val.image_info[img_id]


    results = model.detect([img], verbose=1)
    r = results[0]

    # Method 1: Use a custom greyscale visualisation function
    output_path1 = os.path.join(FMTEST_DIR, f'{img_id}_detection_grayscale.jpg')
    result_img = draw_detections_on_grayscale(
        img,
        r['rois'],
        output_path1,
        line_width=3
    )

    # Method 2: If you need to save a version that includes category information (optional)
    # output_path2 = os.path.join(FMTEST_DIR, f'{img_id}_detection_colored_boxes.jpg')
    # result_img2 = visualize_detections_simple(
    #     img,
    #     r['rois'],
    #     r['class_ids'],
    #     r['scores'],
    #     output_path2,
    #     line_width=3
    # )

    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title(f'Detection Results - Image {img_id}')
    plt.tight_layout()
    plt.savefig(os.path.join(FMTEST_DIR, f'{img_id}_display.jpg'), bbox_inches='tight')


    print(f"\n检测统计:")
    print(f"检测到 {len(r['rois'])} 个目标")
    for i, (box, class_id, score) in enumerate(zip(r['rois'], r['class_ids'], r['scores'])):
        if score > 0.5:
            print(f"  目标 {i + 1}: 类别={dataset_val.class_names[class_id]}, "
                  f"置信度={score:.3f}, 位置=({box[1]:.0f},{box[0]:.0f},{box[3]:.0f},{box[2]:.0f})")