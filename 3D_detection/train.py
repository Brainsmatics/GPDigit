'''
Model training.
Refer to the current script for training parameters.
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


batch_size = 8
confidence = 0.5
nms_th = 0.45
lr = 0.0001
cfgfile = "./cfg/yolov3_drop.cfg"
num_class = 4
# It is still a binary classification task, but whilst the data is labelled with class labels,
# the actual model training does not take the classification results into account.
# If you wish to perform classification training,
# you will need to modify the relevant sections of the script relating to the CLS loss.

train_epoch = 200
train_path = './train/image/'
val_path = './valid/image/'
inp_dim = 128
max_norm = 10       # Gradient clipping
grad_check = 1000     # Gradient check threshold

weights_path = "./output/"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = torch.cuda.is_available()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA = torch.cuda.is_available()

model = Darknet(cfgfile)

start_epoch = -1   # Restart training: -1

# Warm-up training
warmup_epochs = 10
base_lr = lr
warmup_factor = 0.1
accumulation_epoch = 6
# During the warm-up phase, updates are performed once every two batches over the first six epochs.

model = model.to(device)
scaler = GradScaler()

# Use pre-training (import a pre-trained model or resume training from a checkpoint)
Resume = False       # Whether to resume training. False: First training session; True: Continue training.
if Resume:
    path_checkpoint = "./checkpoint/ckpt_best.pth"  # checkpoint path
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch']  # Set the starting epoch
    model.load_state_dict(checkpoint['net'])  # Load the model to learn the parameters

    # Recreate the optimiser
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.00001,
        weight_decay=0.0005
    )

    # Recreate the scheduler (to accommodate the new learning rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        verbose=True,
        min_lr=1e-7
    )

    # scaler
    scaler = GradScaler()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.3, last_epoch=start_epoch-1)
    # step_size：Interval between learning rate decay cycles (gamma): After each step, the learning rate is updated to lr*gamma

else:
    # load the model and weights for initialization
    # model.load_state_dict(torch.load(weights_path))
    start_epoch = -1
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, eps=1e-5,
                                 weight_decay=0.0005
                                 )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )



if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)




if __name__ == '__main__':
    multiprocessing.freeze_support()


    dataloaders = {
        'train': torch.utils.data.DataLoader(
            ListDataset(train_path, img_size=inp_dim),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        ),
        'val': torch.utils.data.DataLoader(
            ListDataset(val_path, img_size=inp_dim),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
    }
    since = time.time()

    anchors = (
        # Scale 1 – Large objects
        [
            (48, 48, 64), (56, 56, 72), (64, 64, 80),
            (72, 72, 88), (80, 80, 96), (88, 88, 96),
            (96, 96, 96), (80, 96, 80), (96, 80, 80)
        ],
        # Scale 2 – Medium-sized objects
        [
            (24, 24, 32), (28, 28, 40), (32, 32, 48),
            (36, 36, 52), (40, 40, 56), (44, 44, 60),
            (48, 48, 64), (40, 56, 40), (56, 40, 40)
        ],
        # Scale 3 – Small objects
        [
            (8, 8, 12), (10, 10, 16), (12, 12, 20),
            (14, 14, 24), (16, 16, 28), (18, 18, 30),
            (20, 20, 32), (16, 28, 16), (28, 16, 16)
        ]
    )


    best_model_wts = copy.deepcopy(model.state_dict())
    best_F1_score = 0.0
    best_train_loss = 100000.0
    best_val_loss = 100000.0

    num_epochs = train_epoch

    if Resume:
        txt_file = open("loss_resume.txt", mode="w+")
        txt_file.close()
    else:
        txt_file = open("loss.txt", mode="w+")
        txt_file.close()


    for epoch in range(start_epoch+1, num_epochs):
        if epoch < warmup_epochs:
            current_lr = base_lr * (warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f'Warm-up Epoch {epoch}: lr = {current_lr:.6f}')
        if epoch < accumulation_epoch:
            accumulation_steps = 2  #
            print(f"Warm-up Epoch {epoch}: accumulation_steps={accumulation_steps}")
        else:
            accumulation_steps = 1

        print('=================== Epoch {}/{} ========================'.format(epoch, num_epochs-1))
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.8f}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss, running_xyz_loss, running_whl_loss, running_conf_loss, running_cls_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            running_recall, running_precision, running_F1_score = 0.0, 0.0, 0.0

            # iterate over data
            for i_batch, sample_batched in enumerate(dataloaders[phase]):

                inputs, labels, paths= sample_batched['input_img'], sample_batched['label'], sample_batched['path']


                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    a = time.time()

                    with autocast():
                        Final_pre = model(inputs, CUDA)
                        c = time.time()

                        # total loss no longer has a CLS; it must be enabled in "util" when required.
                        # HBN is merely a module and does not return a loss value.
                        loss_item = {"total_loss": 0, "x": 0, "y": 0, "z": 0, "w": 0, "h": 0, "l": 0, "conf": 0}

                        for i in range(3):
                            pred = Final_pre[i]

                            batch_size = pred.size(0)

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
                            pred = pred.view(batch_size, nA, nG, nG, nG, 7 + num_class)
                            pred = pred.permute(0, 1, 5, 2, 3, 4).contiguous()


                            scale_weights = [0.3, 0.4, 0.3]  # with a focus on mesoscale (scale 2)
                            losses = Loss(pred, labels.float(), anchors[i],
                                          inp_dim=inp_dim, num_anchors=18,
                                          num_classes=num_class, epoch=epoch)
                            # for j, name in enumerate(loss_item):
                            #     loss_item[name] += losses[j] * scale_weights[i]
                            for j, (name, value) in enumerate(zip(loss_item.keys(), losses)):
                                loss_item[name] += value * scale_weights[i]

                        if phase == 'val' and epoch >= 20 and epoch % 10 == 0:
                            with torch.no_grad():
                                eval_results = write_results(Final_pre, confidence=confidence,
                                                             num_classes=num_class, nms_conf=nms_th)

                                if not isinstance(eval_results, int):
                                    F1_score, precision, recall = eval_gpu_batch(
                                        eval_results, labels,
                                        img_width=inp_dim, img_height=inp_dim, img_long=inp_dim
                                    )
                                else:
                                    F1_score, precision, recall = 0, 0, 0
                        else:
                            F1_score, precision, recall = 0, 0, 0


                        e = time.time()
                        loss = loss_item['total_loss']

                        # Loss value check
                        if torch.isnan(loss) or torch.isinf(loss):
                            optimizer.zero_grad()
                            continue

                        if loss.item() > 1e6:
                            optimizer.zero_grad()
                            continue

                        xyz_loss = loss_item['x']+loss_item['y']+loss_item['z']
                        whl_loss = loss_item['w']+loss_item['h']+loss_item['l']
                        conf_loss = loss_item['conf']

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()

                            valid_gradients = True
                            total_norm = 0
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        valid_gradients = False
                                        break
                                    grad_norm = param.grad.data.norm(2).item()
                                    if grad_norm > grad_check:
                                        pass
                                    param_norm = param.grad.data.norm(2)
                                    total_norm += param_norm.item() ** 2

                            # scaler.unscale_(optimizer)

                            if (i_batch + 1) % accumulation_steps == 0:
                                if valid_gradients:
                                    total_norm = total_norm ** 0.5

                                    if i_batch % 200 == 0:
                                        print(f"  Batch {i_batch}: loss={loss.item():.4f}, "
                                              f"grad_norm={total_norm:.4f}")
                                        print(f"    xy={xyz_loss.item():.4f}, wh={whl_loss.item():.4f}, "
                                              f"conf={conf_loss.item():.4f}")

                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                                    grad_valid = True
                                    for param in model.parameters():
                                        if param.grad is not None:
                                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                                grad_valid = False
                                                break

                                    if grad_valid:
                                        scaler.step(optimizer)
                                        scaler.update()
                                    else:
                                        scaler.update()

                                optimizer.zero_grad()

                                if i_batch % 10 == 0:
                                    torch.cuda.empty_cache()


                         # statistics
                        running_loss += loss.item()
                        running_xyz_loss += xyz_loss.item()
                        running_whl_loss += whl_loss.item()
                        running_conf_loss += conf_loss.item()
                        running_recall += recall
                        running_precision += precision
                        running_F1_score += F1_score



            time.sleep(0.001)

            epoch_loss = running_loss / ((i_batch+1)*batch_size)
            epoch_xyz_loss = running_xyz_loss / ((i_batch+1)*batch_size)
            epoch_whl_loss = running_whl_loss / ((i_batch + 1) * batch_size)
            epoch_conf_loss = running_conf_loss / ((i_batch + 1) * batch_size)
            epoch_cls_loss = running_cls_loss / ((i_batch + 1) * batch_size)

            epoch_recall = running_recall / (i_batch+1)
            epoch_precision = running_precision / (i_batch+1)
            epoch_F1_score = running_F1_score / (i_batch+1)


            print(
                '{} Loss: {:.4f} Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss,
                                                                                           epoch_recall, epoch_precision, epoch_F1_score))
            print(
                '{} xy: {:.4f} wh: {:.4f} conf: {:.4f} class: {:.4f}'.format(phase, epoch_xyz_loss, epoch_whl_loss,
                                                                             epoch_conf_loss,epoch_cls_loss))

            if Resume:
                txt_file = open("loss_resume.txt", mode="a+")
            else:
                txt_file = open("loss.txt", mode="a+")
            if phase == "train":
                txt_file.write('Epoch {}/{} \n'.format(epoch, num_epochs-1))
            txt_file.write('{} Loss: {:.4f} Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f} \n'.format(phase, epoch_loss, epoch_recall,
                                                                                           epoch_precision, epoch_F1_score))
            txt_file.write('{} xy: {:.4f} wh: {:.4f} conf: {:.4f} class: {:.4f} \n'.format(phase, epoch_xyz_loss, epoch_whl_loss,
                                                                                                            epoch_conf_loss,
                                                                                                            epoch_cls_loss))
            txt_file.close()

            # deep copy the model
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            if not os.path.isdir('./checkpoint'):
                os.makedirs("./checkpoint")

            # if phase == 'train':
            #     if epoch < 20:
            #         current_phase = 'phase1'
            #         save_condition = epoch_loss < phase_best[current_phase]['loss']
            #     elif 20 <= epoch <= 25:
            #         current_phase = 'phase2'
            #         save_condition = True  # 过渡期全保存
            #     elif 26 <= epoch < 50:
            #         current_phase = 'phase3'
            #         save_condition = epoch_loss < phase_best[current_phase]['loss']
            #     elif epoch == 50:
            #         current_phase = 'phase4'
            #         save_condition = True  # epoch=50强制保存
            #     else:
            #         current_phase = 'phase5'
            #         save_condition = epoch_loss < phase_best[current_phase]['loss']
            #
            #     if save_condition:
            #         phase_best[current_phase]['loss'] = epoch_loss
            #         phase_best[current_phase]['epoch'] = epoch
            #
            #         # current_phase
            #         torch.save(model.state_dict(), weights_path_t + f"_{current_phase}_epoch{epoch + 1}.pth")
            #         torch.save(checkpoint, f'./checkpoint/ckpt_{current_phase}_epoch{epoch + 1}.pth')

            if phase == 'train':
                if epoch_loss < best_train_loss:
                    best_train_loss = epoch_loss
                    torch.save(model.state_dict(), weights_path + "best_train_loss.pth")
                    torch.save(checkpoint, './checkpoint/ckpt_best_train_loss.pth')
                    print(f'>>> New best train loss: {best_train_loss:.4f} at epoch {epoch}')

            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), weights_path + "_best_val_loss.pth")
                    torch.save(checkpoint, './checkpoint/ckpt_best_val_loss.pth')
                    print(f'>>> New best val loss: {best_val_loss:.4f} at epoch {epoch}')

                torch.save(checkpoint, f'./checkpoint/ckpt_epoch_{epoch}.pth')


            if phase == 'val':
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(epoch_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if old_lr != new_lr:
                    print(f'lr: {old_lr:.8f} -> {new_lr:.8f}')

        # if epoch >= warmup_epochs:
        #     scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best F1 score: {:4f}'.format(best_F1_score))
