'''
Mini-batch training.
The original intention was to compare the training performance of the original data with that of the augmented data.
'''

import time
import numpy as np
from util import *
import multiprocessing

multiprocessing.freeze_support()
import argparse
import os
from darknet import Darknet
import pandas as pd
import random
import glob
import warnings

warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


batch_size = 4
confidence = 0.5
nms_th = 0.45
lr = 0.0001
cfgfile = "./cfg/yolov3_drop.cfg"
num_class = 4
train_epoch = 70
train_path = './train_mini/image/'
val_path = './valid_mini/image/'
inp_dim = 128
max_norm = 10
grad_check = 1000
weights_path_t = "mini_batch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Darknet(cfgfile)
model = model.to(device)
scaler = GradScaler()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    eps=1e-5,
    weight_decay=0.001
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=train_epoch,
    eta_min=1e-6
)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


def compute_metrics(predictions, targets, num_classes=4):
    if isinstance(predictions, int) or len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'detection_rate': 0.0
        }

    pred_classes = predictions[:, 0].long()
    target_classes = targets[targets.sum(dim=1) > 0, 0].long()

    if len(target_classes) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'detection_rate': 0.0
        }

    if len(pred_classes) > 0:
        detection_rate = min(len(pred_classes) / len(target_classes), 1.0)

        if len(pred_classes) == len(target_classes):
            correct = (pred_classes == target_classes).sum().item()
            accuracy = correct / len(pred_classes)
            precision = accuracy
            recall = accuracy
            f1 = accuracy if accuracy > 0 else 0.0
        else:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        detection_rate = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detection_rate': detection_rate
    }


def evaluate_model(model, dataloader, device, confidence, nms_th, num_class, inp_dim):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sample_batched in dataloader:
            inputs = sample_batched['input_img'].to(device)
            labels = sample_batched['label']

            Final_pre = model(inputs, CUDA)

            eval_results = write_results(Final_pre, confidence=confidence,
                                         num_classes=num_class, nms_conf=nms_th)

            if not isinstance(eval_results, int):
                all_predictions.append(eval_results.cpu())

            valid_mask = labels.sum(dim=2) != 0
            valid_labels = labels[valid_mask]
            if len(valid_labels) > 0:
                all_targets.append(valid_labels.cpu())

    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, 0)
    else:
        all_predictions = torch.tensor([])

    if len(all_targets) > 0:
        all_targets = torch.cat(all_targets, 0)
    else:
        all_targets = torch.tensor([])

    metrics = compute_metrics(all_predictions, all_targets, num_class)

    return metrics


if __name__ == '__main__':
    multiprocessing.freeze_support()

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            ListDataset(train_path, img_size=inp_dim),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        ),
        'val': torch.utils.data.DataLoader(
            ListDataset(val_path, img_size=inp_dim),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
    }

    if not os.path.isdir('./checkpoint_small'):
        os.makedirs("./checkpoint_small")
    if not os.path.isdir('./model_weights_small'):
        os.makedirs("./model_weights_small")

    log_file = open("training_log_small.txt", mode="w")
    log_file.write("Epoch,Phase,Loss,Accuracy,Precision,Recall,F1_Score,Detection_Rate,Learning_Rate\n")
    log_file.close()

    summary_file = open("training_summary.txt", mode="w")
    summary_file.write(
        "Epoch,Train_Loss,Val_Accuracy,Val_Precision,Val_Recall,Val_F1,Val_Detection_Rate,Best_F1_So_Far\n")
    summary_file.close()

    anchors = (
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

    since = time.time()
    best_val_f1 = 0.0
    best_epoch = -1


    for epoch in range(train_epoch):
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch + 1}/{train_epoch}')
        print(f'Learning Rate: {current_lr:.8f}')
        print(f'{"=" * 60}')

        model.train()
        running_loss = 0.0
        num_batches = 0

        for i_batch, sample_batched in enumerate(dataloaders['train']):
            inputs = sample_batched['input_img'].to(device)
            labels = sample_batched['label'].to(device)

            optimizer.zero_grad()

            with autocast():
                Final_pre = model(inputs, CUDA)

                loss_item = {"total_loss": 0, "x": 0, "y": 0, "z": 0,
                             "w": 0, "h": 0, "l": 0, "conf": 0}

                for i in range(3):
                    pred = Final_pre[i]
                    batch_size_current = pred.size(0)

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

                    nG = [4, 8, 16][i]
                    nA = 18
                    pred = pred.view(batch_size_current, nA, nG, nG, nG, 7 + num_class)
                    pred = pred.permute(0, 1, 5, 2, 3, 4).contiguous()

                    losses = Loss(pred, labels.float(), anchors[i],
                                  inp_dim=inp_dim, num_anchors=18,
                                  num_classes=num_class, epoch=epoch)

                    scale_weights = [0.3, 0.4, 0.3]
                    for j, (name, value) in enumerate(zip(loss_item.keys(), losses)):
                        loss_item[name] += value * scale_weights[i]

                loss = loss_item['total_loss']

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Batch {i_batch}: Loss is NaN/Inf, skipping...")
                    optimizer.zero_grad()
                    continue

                if loss.item() > 1e6:
                    print(f"  Batch {i_batch}: Loss too large ({loss.item():.2f}), skipping...")
                    optimizer.zero_grad()
                    continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

            if (i_batch + 1) % 10 == 0:
                print(f"  Batch {i_batch + 1}/{len(dataloaders['train'])}: Loss = {loss.item():.4f}")

        train_loss = running_loss / num_batches if num_batches > 0 else 0

        val_metrics = evaluate_model(model, dataloaders['val'], device,
                                     confidence, nms_th, num_class, inp_dim)

        scheduler.step()


        print(f"\nVal loss: {train_loss:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
        print(f"  Detection Rate: {val_metrics['detection_rate']:.4f}")


        log_file = open("training_log_small.txt", mode="a")
        log_file.write(f"{epoch + 1},train,{train_loss:.6f},0,0,0,0,0,{current_lr:.8f}\n")
        log_file.write(f"{epoch + 1},val,0,{val_metrics['accuracy']:.6f},"
                       f"{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},"
                       f"{val_metrics['f1_score']:.6f},{val_metrics['detection_rate']:.6f},"
                       f"{current_lr:.8f}\n")
        log_file.close()


        summary_file = open("training_summary.txt", mode="a")
        summary_file.write(f"{epoch + 1},{train_loss:.6f},{val_metrics['accuracy']:.6f},"
                           f"{val_metrics['precision']:.6f},{val_metrics['recall']:.6f},"
                           f"{val_metrics['f1_score']:.6f},{val_metrics['detection_rate']:.6f},"
                           f"{best_val_f1:.6f}\n")
        summary_file.close()

        # 1. Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_metrics": val_metrics
            }

            torch.save(checkpoint, f'./checkpoint_small/epoch_{epoch + 1:03d}_checkpoint.pth')

            torch.save(model.state_dict(), f'./model_weights_small/epoch_{epoch + 1:03d}_weights.pth')

            print(f" Save: (Epoch {epoch + 1})")

        # 2. Preserving the finest F1 models
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_epoch = epoch + 1

            torch.save(model.state_dict(),
                       f'./model_weights_small/best_model_f1_{best_val_f1:.4f}_epoch{best_epoch}.pth')

            checkpoint = {
                "epoch": best_epoch,
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "best_f1": best_val_f1
            }
            torch.save(checkpoint, f'./checkpoint_small/best_checkpoint_f1_{best_val_f1:.4f}_epoch{best_epoch}.pth')

            print(f"  F1 Score: {best_val_f1:.4f} (Epoch {best_epoch})")

        torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    print(f"\n{'=' * 80}")
    print(f"\nBest:")
    print(f"  F1 Score: {best_val_f1:.4f}")
    print(f"  Epoch: {best_epoch}")
    print(f"\nFiles:")
    print(f"  - training_log_small.txt")
    print(f"  - training_summary.txt")
    print(f"  - ./model_weights_small/")
    print(f"  - Checkpoint: ./checkpoint_small/")
    print(f"{'=' * 80}")