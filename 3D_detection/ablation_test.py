"""
Ablation experiment: Dropblock, FH(HBNet)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
import json
import csv
import glob
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from util import ListDataset, write_results, Loss, eval_gpu_batch
from darknet import Darknet, DropBlock3D
import warnings
warnings.filterwarnings("ignore")


class Config:
    train_path = "./train/image/"
    val_path = "./valid/image/"

    output_dir = "./ablation_new/"
    cfg_full = "./cfg/yolov3_drop.cfg"
    cfg_no_drop = "./cfg/yolov3_no_drop.cfg"

    batch_size = 4
    num_epochs = 50
    lr = 0.0005
    inp_dim = 128
    num_class = 4
    confidence = 0.5
    nms_th = 0.45
    max_norm = 10
    warmup_epochs = 5
    warmup_factor = 0.15

    eval_batch_size = 10
    eval_num_batches = 10
    ap_iou_threshold = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed = 42

np.random.seed(Config.random_seed)
random.seed(Config.random_seed)
torch.manual_seed(Config.random_seed)

# ==================== Anchors ====================
ANCHORS = [
    [(48,48,64), (56,56,72), (64,64,80), (72,72,88), (80,80,96), (88,88,96),
     (96,96,96), (80,96,80), (96,80,80)],
    [(24,24,32), (28,28,40), (32,32,48), (36,36,52), (40,40,56), (44,44,60),
     (48,48,64), (40,56,40), (56,40,40)],
    [(8,8,12), (10,10,16), (12,12,20), (14,14,24), (16,16,28), (18,18,30),
     (20,20,32), (16,28,16), (28,16,16)]
]


def compute_iou_3d(box1, box2):
    """box: [cx, cy, cz, w, h, l]"""
    x1 = max(box1[0] - box1[3]/2, box2[0] - box2[3]/2)
    y1 = max(box1[1] - box1[4]/2, box2[1] - box2[4]/2)
    z1 = max(box1[2] - box1[5]/2, box2[2] - box2[5]/2)
    x2 = min(box1[0] + box1[3]/2, box2[0] + box2[3]/2)
    y2 = min(box1[1] + box1[4]/2, box2[1] + box2[4]/2)
    z2 = min(box1[2] + box1[5]/2, box2[2] + box2[5]/2)
    inter = max(0, x2-x1) * max(0, y2-y1) * max(0, z2-z1)
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    return inter / (vol1 + vol2 - inter + 1e-6)

def compute_ap(precisions, recalls):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
        ap += p / 11
    return ap

def compute_detection_ap(detections, groundtruths, iou_threshold=0.5):
    if len(groundtruths) == 0:
        return 1.0 if len(detections) == 0 else 0.0

    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    gt_matched = [False] * len(groundtruths)

    for i, det in enumerate(detections):
        conf, det_bbox = det
        best_iou, best_idx = 0, -1
        for j, gt_bbox in enumerate(groundtruths):
            iou = compute_iou_3d(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= iou_threshold and not gt_matched[best_idx]:
            tp[i] = 1
            gt_matched[best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / len(groundtruths)

    precisions = np.concatenate(([1], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    return compute_ap(precisions, recalls)

def convert_detections(detections_tensor):
    if isinstance(detections_tensor, int):
        return []
    dets = []
    for i in range(detections_tensor.shape[0]):
        x1, y1, z1, x2, y2, z2 = detections_tensor[i, :6].cpu().numpy()
        w, h, l = x2-x1, y2-y1, z2-z1
        cx, cy, cz = x1+w/2, y1+h/2, z1+l/2
        dets.append([float(detections_tensor[i, 6]), [cx, cy, cz, w, h, l]])
    return dets

def convert_groundtruths(labels_tensor):
    if labels_tensor.numel() == 0:
        return []
    valid = labels_tensor[:, 0] >= 0
    labels_tensor = labels_tensor[valid]
    gts = []
    for i in range(labels_tensor.shape[0]):
        cx, cy, cz, w, h, l = labels_tensor[i, 1:7].cpu().numpy().tolist()
        gts.append([cx, cy, cz, w, h, l])
    return gts


def create_model_with_hbnet_control(cfg_path, use_hbnet=True):
    """Create a model and control HBNet(FH) using monkey patches."""
    from HBNet import hb_net
    import types

    model = Darknet(cfg_path)

    if not use_hbnet:
        original_forward = model.forward

        def patched_forward(self, x, CUDA):
            original_hb_net = hb_net.forward

            def fake_hb_net(f1, f2, f3, scale_name):
                if scale_name == 'scale1':
                    return torch.zeros(f1.shape[0], 99, 4, 4, 4, device=f1.device)
                elif scale_name == 'scale2':
                    return torch.zeros(f1.shape[0], 99, 8, 8, 8, device=f1.device)
                else:  # scale3
                    return torch.zeros(f1.shape[0], 99, 16, 16, 16, device=f1.device)

            hb_net.forward = fake_hb_net
            detections = original_forward(x, CUDA)
            hb_net.forward = original_hb_net
            return detections

        model.forward = types.MethodType(patched_forward, model)

    model.to(Config.device)
    return model


def disable_all_dropblocks(model):
    for module in model.modules():
        if isinstance(module, DropBlock3D):
            module.forward = lambda x: x

def get_dropblock_positions(model):
    positions = []
    for name, module in model.named_modules():
        if isinstance(module, DropBlock3D):
            positions.append({'name': name, 'drop_prob': getattr(module, 'drop_prob', 0.0),
                              'block_size': getattr(module, 'block_size', 0)})
    return positions

def disable_specific_dropblocks(model, indices_to_disable):
    idx = 0
    for module in model.modules():
        if isinstance(module, DropBlock3D):
            if idx in indices_to_disable:
                module.forward = lambda x: x
            idx += 1

def train_model(model, train_loader, val_loader, config, model_name):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config.lr, eps=1e-5, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    eval_epochs = [5, 10, 13, 15]

    print(f"\n {model_name}")
    for epoch in range(config.num_epochs):
        if epoch < config.warmup_epochs:
            warmup_lr = config.lr * (config.warmup_factor + (1 - config.warmup_factor) * (epoch+1) / config.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        model.train()
        running_loss = 0.0
        for i_batch, sample in enumerate(train_loader):
            inputs = sample['input_img'].to(config.device)
            labels = sample['label'].to(config.device)
            optimizer.zero_grad()

            with autocast():
                preds = model(inputs, CUDA=torch.cuda.is_available())
                total_loss = 0.0
                for i, pred in enumerate(preds):
                    expected_sizes = [1152, 9216, 73728]
                    if pred.size(1) > expected_sizes[i]:
                        pred = pred[:, :expected_sizes[i], :]

                    nG = [4, 8, 16][i]
                    nA = 18
                    pred = pred.view(pred.size(0), nA, nG, nG, nG, 7 + config.num_class)
                    pred = pred.permute(0, 1, 5, 2, 3, 4).contiguous()

                    scale_weights = [0.3, 0.4, 0.3][i]
                    losses = Loss(pred, labels.float(), ANCHORS[i], inp_dim=config.inp_dim,
                                  num_anchors=18, num_classes=config.num_class, epoch=epoch)
                    total_loss += losses[0] * scale_weights

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()

            if i_batch % 20 == 0:
                print(f"  Epoch {epoch} Batch {i_batch}: loss={total_loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if epoch + 1 in eval_epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sample in val_loader:
                    inputs = sample['input_img'].to(config.device)
                    labels = sample['label'].to(config.device)
                    preds = model(inputs, CUDA=torch.cuda.is_available())
                    total_loss = 0.0
                    for i, pred in enumerate(preds):
                        expected_sizes = [1152, 9216, 73728]
                        if pred.size(1) > expected_sizes[i]:
                            pred = pred[:, :expected_sizes[i], :]

                        nG = [4, 8, 16][i]
                        nA = 18
                        pred = pred.view(pred.size(0), nA, nG, nG, nG, 7 + config.num_class)
                        pred = pred.permute(0, 1, 5, 2, 3, 4).contiguous()

                        scale_weights = [0.3, 0.4, 0.3][i]
                        losses = Loss(pred, labels.float(), ANCHORS[i], inp_dim=config.inp_dim,
                                      num_anchors=18, num_classes=config.num_class, epoch=epoch)
                        total_loss += losses[0] * scale_weights
                    val_loss += total_loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, LR={current_lr:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, os.path.join(config.output_dir, f"{model_name}_best.pth"))
        else:
            print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, LR={current_lr:.6f} (skip val)")

    return best_weights


def evaluate_model_multi_batch(model, dataset, batch_size=10, num_batches=10):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    batch_results = []

    for batch_idx in range(num_batches):
        start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
        batch_indices = indices[start:end]
        if len(batch_indices) < batch_size:
            continue

        batch_f1, batch_prec, batch_rec, batch_ap = [], [], [], []

        for idx in batch_indices:
            sample = dataset[idx]
            img = sample['input_img'].unsqueeze(0).to(Config.device)
            labels = sample['label'].unsqueeze(0).to(Config.device)

            with torch.no_grad():
                preds = model(img, CUDA=torch.cuda.is_available())
                detections = write_results(preds, confidence=Config.confidence,
                                           num_classes=Config.num_class, nms_conf=Config.nms_th)

            if isinstance(detections, int):
                f1, prec, rec = 0.0, 0.0, 0.0
                ap = 0.0
            else:
                f1, prec, rec = eval_gpu_batch(detections, labels,
                                               img_width=Config.inp_dim,
                                               img_height=Config.inp_dim,
                                               img_long=Config.inp_dim)
                ap = compute_detection_ap(convert_detections(detections),
                                         convert_groundtruths(labels.squeeze(0)),
                                         Config.ap_iou_threshold)

            batch_f1.append(f1)
            batch_prec.append(prec)
            batch_rec.append(rec)
            batch_ap.append(ap)

        batch_avg = {
            'batch_id': batch_idx,
            'f1_mean': np.mean(batch_f1),
            'precision_mean': np.mean(batch_prec),
            'recall_mean': np.mean(batch_rec),
            'ap_mean': np.mean(batch_ap),
            'num_samples': len(batch_indices)
        }
        batch_results.append(batch_avg)
        print(f"  batch{batch_idx+1}: F1={batch_avg['f1_mean']:.4f}, "
              f"Prec={batch_avg['precision_mean']:.4f}, Rec={batch_avg['recall_mean']:.4f}, AP={batch_avg['ap_mean']:.4f}")

    return {
        'f1_mean': np.mean([r['f1_mean'] for r in batch_results]),
        'f1_std': np.std([r['f1_mean'] for r in batch_results]),
        'precision_mean': np.mean([r['precision_mean'] for r in batch_results]),
        'precision_std': np.std([r['precision_mean'] for r in batch_results]),
        'recall_mean': np.mean([r['recall_mean'] for r in batch_results]),
        'recall_std': np.std([r['recall_mean'] for r in batch_results]),
        'ap_mean': np.mean([r['ap_mean'] for r in batch_results]),
        'ap_std': np.std([r['ap_mean'] for r in batch_results]),
        'batch_results': batch_results
    }


def dropblock_sensitivity_analysis(full_model, val_dataset):
    positions = get_dropblock_positions(full_model)
    n = len(positions)
    if n >= 5:
        groups = {'shallow': [0,1], 'middle': [2,3], 'deep': [4]}
    else:
        third = max(1, n // 3)
        groups = {'shallow': list(range(third)),
                  'middle': list(range(third, min(2*third, n))),
                  'deep': list(range(2*third, n))}

    baseline = evaluate_model_multi_batch(full_model, val_dataset,
                                          Config.eval_batch_size, Config.eval_num_batches)
    results = {'baseline': baseline}
    for name, indices in groups.items():
        model_copy = copy.deepcopy(full_model)
        disable_specific_dropblocks(model_copy, indices)
        results[name] = evaluate_model_multi_batch(model_copy, val_dataset,
                                                   Config.eval_batch_size, Config.eval_num_batches)
    return results


def plot_comparison(results):
    models = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('f1_mean', 'F1 Score'), ('precision_mean', 'Precision'),
               ('recall_mean', 'Recall'), ('ap_mean', 'AP')]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for ax, (key, title) in zip(axes.flat, metrics):
        means = [results[m][key] for m in models]
        stds = [results[m][key.replace('mean', 'std')] for m in models]
        ax.bar(models, means, yerr=stds, capsize=5, color=colors)
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'ablation_comparison.png'), dpi=150)
    plt.close()

def plot_sensitivity(results):
    groups = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('f1_mean', 'F1 Score'), ('precision_mean', 'Precision'),
               ('recall_mean', 'Recall'), ('ap_mean', 'AP')]

    for ax, (key, title) in zip(axes.flat, metrics):
        vals = [results[g][key] for g in groups]
        ax.bar(groups, vals, capsize=5)
        ax.set_ylabel(title)
        ax.set_title(f'DropBlock Sensitivity - {title}')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'dropblock_sensitivity.png'), dpi=150)
    plt.close()


def main():
    os.makedirs(Config.output_dir, exist_ok=True)

    for p in [Config.train_path, Config.val_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path does not exist: {p}")

    train_dataset = ListDataset(Config.train_path, img_size=Config.inp_dim, mode="train")
    val_dataset = ListDataset(Config.val_path, img_size=Config.inp_dim, mode="train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"\nTraining set: {len(train_dataset)},  Validation set: {len(val_dataset)}")

    configs = [
        {'name': 'Full',           'cfg': Config.cfg_full,     'use_hbnet': True,  'dropblock': True},
        {'name': 'OnlyHBNet',      'cfg': Config.cfg_full,  'use_hbnet': True,  'dropblock': False},
        {'name': 'OnlyDropBlock',  'cfg': Config.cfg_full,     'use_hbnet': False, 'dropblock': True},
        {'name': 'Baseline',       'cfg': Config.cfg_full,  'use_hbnet': False, 'dropblock': False}
    ]

    all_results = {}
    all_models = {}

    for cfg in configs:
        name = cfg['name']
        print(f"\n{'='*50}\nsetting: {name}\n{'='*50}")

        # Create a model and control HBNet
        model = create_model_with_hbnet_control(cfg['cfg'], use_hbnet=cfg['use_hbnet'])

        # DropBlock
        if not cfg['dropblock']:
            disable_all_dropblocks(model)

        best_weights = train_model(model, train_loader, val_loader, Config, name)
        model.load_state_dict(best_weights)

        print(f"\nEval {name}...")
        results = evaluate_model_multi_batch(model, val_dataset,
                                             Config.eval_batch_size, Config.eval_num_batches)
        all_results[name] = results
        all_models[name] = model

        csv_path = os.path.join(Config.output_dir, f"{name}_batch_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['batch_id', 'f1_mean', 'precision_mean',
                                                    'recall_mean', 'ap_mean', 'num_samples'])
            writer.writeheader()
            writer.writerows(results['batch_results'])

        json_path = os.path.join(Config.output_dir, f"{name}_summary.json")
        with open(json_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'batch_results'}, f, indent=2)

    plot_comparison(all_results)

    print("\nDropBlock Sensitivity analysis...")
    sensitivity = dropblock_sensitivity_analysis(all_models['Full'], val_dataset)
    with open(os.path.join(Config.output_dir, "dropblock_sensitivity.json"), 'w') as f:
        json.dump(sensitivity, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    plot_sensitivity(sensitivity)

    print(f"\n Results saved to: {Config.output_dir}")

if __name__ == '__main__':
    main()