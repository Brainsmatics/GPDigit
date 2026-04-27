from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import SimpleITK as sitk
import csv
import os
import glob
# from libtiff import TIFF
from torch.utils.data import Dataset
import copy
import math


try:
    from torchvision.ops import nms as tv_nms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("torchvision NMS not available")


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=128, mode="train"):
        self.mode = mode
        self.img_files = sorted(glob.glob(list_path+"*.tif"))
        if self.mode == "train":
            self.label_files = [path.replace('image', 'label').replace('.tif', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size, img_size)
        self.max_objects = 300

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        img = img.transpose(2, 1, 0)
        w, h, l = img.shape
        img = img[:, :, :, None].repeat(3, 3)

        pad_img = img
        input_img = pad_img[:, :, :, ::-1].transpose((3, 0, 1, 2)).copy()
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float().div(255.0)

        if self.mode == "train":
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            labels = None
            if os.path.exists(label_path):
                labels_noobj = np.loadtxt(label_path).reshape(-1, 7)

                if index == 0:
                    print(f"label: {labels_noobj[0]}")

                labels_noobj[labels_noobj >= 128] = 127
                labels_noobj[labels_noobj < 0] = 0

                labels = np.ones((labels_noobj.shape[0], 7))
                # labels[:, 1:] = labels_noobj[:, :6]/128

                labels[:, 1] = labels_noobj[:, 0] / 128.0  # x_center
                labels[:, 2] = labels_noobj[:, 1] / 128.0  # y_center
                labels[:, 3] = labels_noobj[:, 2] / 128.0  # z_center
                labels[:, 4] = labels_noobj[:, 3] / 128.0  # width
                labels[:, 5] = labels_noobj[:, 4] / 128.0  # height
                labels[:, 6] = labels_noobj[:, 5] / 128.0  # length

                labels[:, 0] = labels_noobj[:, 6] - 1
                labels_copy = labels.copy().copy()

                if index == 0:
                    print(f"Range: min={labels.min()}, max={labels.max()}")

            # Fill matrix
            filled_labels = np.zeros((self.max_objects, 7))
            if labels is not None:
                filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

            filled_labels = torch.from_numpy(filled_labels)

            sample = {'input_img': input_img, 'orig_img': pad_img, 'label': filled_labels, 'path': img_path}
        else:
            sample = {'input_img': input_img, 'orig_img': pad_img, 'path': img_path}

        return sample


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn=None, gamma=2.0, alpha=0.75, reduction='mean'):
        super(FocalLoss, self).__init__()
        if loss_fcn is None:
            self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_fcn = loss_fcn
            self.loss_fcn.reduction = 'none'

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_targets(target, anchors, grid_size, num_anchors=18, num_classes=4):
    """
    Build targets for loss calculation
    num_anchors: 18 (9个anchor * 2)
    """
    nB = target.size(0)
    nA = num_anchors  # 18
    nG = grid_size

    mask = torch.zeros(nB, nA, nG, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG, nG)
    tz = torch.zeros(nB, nA, nG, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG, nG)
    th = torch.zeros(nB, nA, nG, nG, nG)
    tl = torch.zeros(nB, nA, nG, nG, nG)
    tconf = torch.zeros(nB, nA, nG, nG, nG)
    tcls = torch.zeros(nB, nA, nG, nG, nG, num_classes)  # Although it isn’t necessary, we should maintain consistency in the interface

    pos_count = 0

    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue

            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gz = target[b, t, 3] * nG
            gw = target[b, t, 4] * nG
            gh = target[b, t, 5] * nG
            gl = target[b, t, 6] * nG

            gi = int(gx)
            gj = int(gy)
            gk = int(gz)

            if gi >= nG or gj >= nG or gk >= nG or gi < 0 or gj < 0 or gk < 0:
                continue

            txi = gx - gi
            tyi = gy - gj
            tzi = gz - gk

            txi = max(0, min(1, txi))
            tyi = max(0, min(1, tyi))
            tzi = max(0, min(1, tzi))

            # The first nine are the main branches, and the last nine are the FH(HBNet) branches
            base_anchors = anchors[:9]

            cx = gi + 0.5
            cy = gj + 0.5
            cz = gk + 0.5

            best_iou = 0
            best_idx = 0

            for a_idx, anchor in enumerate(base_anchors):
                aw, ah, al = anchor

                x1_a = cx - aw / 2
                y1_a = cy - ah / 2
                z1_a = cz - al / 2
                x2_a = cx + aw / 2
                y2_a = cy + ah / 2
                z2_a = cz + al / 2

                x1_g = cx - gw / 2
                y1_g = cy - gh / 2
                z1_g = cz - gl / 2
                x2_g = cx + gw / 2
                y2_g = cy + gh / 2
                z2_g = cz + gl / 2

                inter_x1 = max(x1_a, x1_g)
                inter_y1 = max(y1_a, y1_g)
                inter_z1 = max(z1_a, z1_g)
                inter_x2 = min(x2_a, x2_g)
                inter_y2 = min(y2_a, y2_g)
                inter_z2 = min(z2_a, z2_g)

                inter_vol = max(0, inter_x2 - inter_x1) * \
                            max(0, inter_y2 - inter_y1) * \
                            max(0, inter_z2 - inter_z1)

                vol_a = (x2_a - x1_a) * (y2_a - y1_a) * (z2_a - z1_a)
                vol_g = (x2_g - x1_g) * (y2_g - y1_g) * (z2_g - z1_g)
                union_vol = vol_a + vol_g - inter_vol

                iou = inter_vol / (union_vol + 1e-16)

                if iou > best_iou:
                    best_iou = iou
                    best_idx = a_idx

            # Use the best-matching anchor (assigned to both the main branch and the FH branch)
            if best_iou > 0.3:
                # Main head (first 9)
                mask[b, best_idx, gi, gj, gk] = 1
                # FH (last 9, index +9)
                mask[b, best_idx + 9, gi, gj, gk] = 1

                pos_count += 2

                tx[b, best_idx, gi, gj, gk] = txi
                ty[b, best_idx, gi, gj, gk] = tyi
                tz[b, best_idx, gi, gj, gk] = tzi

                tx[b, best_idx + 9, gi, gj, gk] = txi
                ty[b, best_idx + 9, gi, gj, gk] = tyi
                tz[b, best_idx + 9, gi, gj, gk] = tzi

                tw[b, best_idx, gi, gj, gk] = gw / base_anchors[best_idx][0]
                th[b, best_idx, gi, gj, gk] = gh / base_anchors[best_idx][1]
                tl[b, best_idx, gi, gj, gk] = gl / base_anchors[best_idx][2]

                tw[b, best_idx + 9, gi, gj, gk] = gw / base_anchors[best_idx][0]
                th[b, best_idx + 9, gi, gj, gk] = gh / base_anchors[best_idx][1]
                tl[b, best_idx + 9, gi, gj, gk] = gl / base_anchors[best_idx][2]

                tconf[b, best_idx, gi, gj, gk] = 1
                tconf[b, best_idx + 9, gi, gj, gk] = 1

    return mask, tx, ty, tz, tw, th, tl, tconf, tcls



def Loss(input, target, anchors, inp_dim, num_anchors=18, num_classes=4, epoch=1):
    """
    Stable version Loss: Uses the existing Focal Loss + fixed weights
    Localisation loss weight: 1.0, Confidence loss weight: 0.2
    """

    if input.dim() == 3:
        batch_size = input.size(0)
        num_boxes = input.size(1)

        if num_boxes == 1152:  # 4x4x4
            nG = 4
            nA = 18
        elif num_boxes == 9216:  # 8x8x8
            nG = 8
            nA = 18
        elif num_boxes == 73728:  # 16x16x16
            nG = 16
            nA = 18
        else:
            boxes_per_anchor = num_boxes // num_anchors
            nG = round(boxes_per_anchor ** (1 / 3))

        try:
            input = input.view(batch_size, nA, nG, nG, nG, 7 + num_classes)
            input = input.permute(0, 1, 5, 2, 3, 4).contiguous()
        except RuntimeError as e:
            print(f"Reshape failed: {e}")
            raise e

    nA = input.size(1) if input.dim() == 6 else num_anchors
    nG = input.size(3) if input.dim() == 6 else int(round((input.size(1) / num_anchors) ** (1 / 3)))

    nB = input.size(0)
    stride = inp_dim / nG

    FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
    ByteTensor = torch.cuda.ByteTensor if input.is_cuda else torch.ByteTensor

    if len(anchors) < nA:
        repeats = nA // len(anchors)
        anchors = anchors * repeats
    elif len(anchors) > nA:
        anchors = anchors[:nA]

    prediction = input.permute(0, 1, 3, 4, 5, 2).contiguous()

    prediction = torch.where(torch.isinf(prediction), torch.zeros_like(prediction), prediction)
    prediction = torch.where(torch.isnan(prediction), torch.zeros_like(prediction), prediction)

    x = torch.sigmoid(prediction[..., 0])  # Center x offset
    y = torch.sigmoid(prediction[..., 1])  # Center y offset
    z = torch.sigmoid(prediction[..., 2])  # Center z offset

    w_log = prediction[..., 3]  # log scale width
    h_log = prediction[..., 4]  # log scale height
    l_log = prediction[..., 5]  # log scale length

    pred_conf = prediction[..., 6]  # conf logits

    scaled_anchors = FloatTensor([(a_w / stride, a_h / stride, a_l / stride)
                                  for a_w, a_h, a_l in anchors])

    mask, tx, ty, tz, tw, th, tl, tconf, _ = build_targets(
        target=target.cpu().data,
        anchors=scaled_anchors.cpu().data,
        grid_size=nG,
        num_anchors=nA,
        num_classes=num_classes)

    tx, ty, tz = tx.type(FloatTensor), ty.type(FloatTensor), tz.type(FloatTensor)
    tw, th, tl = tw.type(FloatTensor), th.type(FloatTensor), tl.type(FloatTensor)
    tconf = tconf.type(FloatTensor)
    mask = mask.type(ByteTensor).bool()

    smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

    focal_loss = FocalLoss(gamma=2.0, alpha=0.75, reduction='mean')

    loc_weight = 1.0  # Location loss weights
    conf_weight = 0.2  # Confidence loss weight

    pos_count = mask.sum().item()

    if pos_count > 0:
        # Positioning error (centre point + dimensions)-
        loss_x = smooth_l1_loss(x[mask], tx[mask])
        loss_y = smooth_l1_loss(y[mask], ty[mask])
        loss_z = smooth_l1_loss(z[mask], tz[mask])

        # Size loss (log space)
        loss_w = smooth_l1_loss(w_log[mask], torch.log(tw[mask] + 1e-8))
        loss_h = smooth_l1_loss(h_log[mask], torch.log(th[mask] + 1e-8))
        loss_l = smooth_l1_loss(l_log[mask], torch.log(tl[mask] + 1e-8))

        center_loss = (loss_x + loss_y + loss_z) / pos_count
        size_loss = (loss_w + loss_h + loss_l) / pos_count

        loc_loss = center_loss + 0.5 * size_loss

        # Focal Loss
        conf_loss = focal_loss(pred_conf, tconf)

        # total loss
        loss = loc_weight * loc_loss + conf_weight * conf_loss

    else:
        conf_loss = focal_loss(pred_conf, tconf)
        loss = conf_weight * conf_loss

        loss_x = loss_y = loss_z = loss_w = loss_h = loss_l = torch.tensor(0.0, device=input.device)

    loss = torch.clamp(loss, max=100.0, min=0.0)

    return (loss, loss_x, loss_y, loss_z, loss_w, loss_h, loss_l, conf_loss)



def convert_label(image_anno, img_width, img_height, img_long):

    """
    Function: convert image annotation : center x, center y, center_z, w, h, l (normalized) to x1, y1, z1, x2, y2, z2 for corresponding img
    """
    x_center = image_anno[:, 1]
    y_center = image_anno[:, 2]
    z_center = image_anno[:, 3]
    width = image_anno[:, 4]
    height = image_anno[:, 5]
    long = image_anno[:, 6]

    output = torch.zeros_like(image_anno)
    output[:,0] = image_anno[:,0]
    output[:, 1], output[:, 4] = x_center - width / 2, x_center + width / 2
    output[:, 2], output[:, 5] = y_center - height / 2, y_center + height / 2
    output[:, 3], output[:, 6] = z_center - long / 2, z_center + long / 2

    output[:, [1, 4]] *= img_width
    output[:, [2, 5]] *= img_height
    output[:, [3, 6]] *= img_long

    # return output.type(torch.FloatTensor)
    return output.type(torch.FloatTensor).to(image_anno.device)


def eval_gpu_batch(output, labels, img_width, img_height, img_long):
    nProposals = (output[:, 7] > 0.5).sum().item()

    nGT = 0
    nCorrect = 0
    nCls = 0

    unique_batches = torch.unique(output[:, 0]).long()

    for b in unique_batches:
        pred_mask = output[:, 0] == b
        prediction = output[pred_mask]

        if prediction.size(0) == 0:
            continue

        conf_mask = prediction[:, 7] > 0.5
        prediction = prediction[conf_mask]

        if prediction.size(0) == 0:
            continue

        label_b = labels[b]
        valid_mask = label_b.sum(dim=1) != 0
        valid_labels = label_b[valid_mask]  # [K, 7]

        if valid_labels.size(0) == 0:
            continue

        nGT += valid_labels.size(0)

        gt_all = convert_label(valid_labels, img_width, img_height, img_long)
        gt_boxes = gt_all[:, 1:7]  # [K, 6]
        gt_classes = gt_all[:, 0]  # [K]

        pred_boxes = prediction[:, 1:7]  # [M, 6]
        pred_classes = prediction[:, -1]  # [M]

        gt_exp = gt_boxes.unsqueeze(1)  # [K, 1, 6]
        pred_exp = pred_boxes.unsqueeze(0)  # [1, M, 6]

        inter_x1 = torch.max(gt_exp[..., 0], pred_exp[..., 0])
        inter_y1 = torch.max(gt_exp[..., 1], pred_exp[..., 1])
        inter_z1 = torch.max(gt_exp[..., 2], pred_exp[..., 2])
        inter_x2 = torch.min(gt_exp[..., 3], pred_exp[..., 3])
        inter_y2 = torch.min(gt_exp[..., 4], pred_exp[..., 4])
        inter_z2 = torch.min(gt_exp[..., 5], pred_exp[..., 5])

        inter_vol = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0) * \
                    torch.clamp(inter_z2 - inter_z1, min=0)

        gt_vol = (gt_exp[..., 3] - gt_exp[..., 0]) * \
                 (gt_exp[..., 4] - gt_exp[..., 1]) * \
                 (gt_exp[..., 5] - gt_exp[..., 2])

        pred_vol = (pred_exp[..., 3] - pred_exp[..., 0]) * \
                   (pred_exp[..., 4] - pred_exp[..., 1]) * \
                   (pred_exp[..., 5] - pred_exp[..., 2])

        union_vol = gt_vol + pred_vol - inter_vol

        iou_matrix = inter_vol / (union_vol + 1e-16)

        max_iou_per_gt, max_idx_per_gt = torch.max(iou_matrix, dim=1)  # [K]

        correct_mask = max_iou_per_gt > 0.5
        nCorrect += correct_mask.sum().item()

        if correct_mask.any():
            correct_gt_indices = torch.where(correct_mask)[0]
            for idx in correct_gt_indices:
                pred_idx = max_idx_per_gt[idx]
                if pred_classes[pred_idx] == gt_classes[idx]:
                    nCls += 1

    recall = float(nCorrect / nGT) if nGT > 0 else 1
    precision = float(nCorrect / nProposals) if nProposals > 0 else 0
    F1_score = 2 * recall * precision / (recall + precision + 1e-16)

    return F1_score, precision, recall


def eval(output, labels, img_width, img_height, img_long):
    """
    Evaluate the prediction results by F1_score, precision, recall(confidence>0.5, iou>0.5)
    :param output: prediction results after "write_results" transform
    :param labels: labels of the training/validation/data
    :param img_width: img_width
    :param img_height: img_height
    :param img_long: img_long
    :return:
    """
    nProposals = int((output[:, 7] > 0.5).sum().item())
    nGT = 0
    nCorrect = 0
    nCls = 0
    for b in range(labels.shape[0]):  # for each image
        prediction = output[output[:,0] == b]  # filter out the predictions of corresponding image
        for t in range(labels.shape[1]):  # for each object
            if labels[b, t].sum() == 0:  # if the row is empty
                continue
            nGT += 1
            gt_label = convert_label(labels[b, t].unsqueeze(0), img_width, img_height, img_long)
            gt_box = gt_label[:, 1:7]
            for i in range(prediction.shape[0]):
                pred_box = prediction[i, 1:7].unsqueeze(0)
                iou = bbox_iou(pred_box, gt_box)
                pred_label = prediction[i, -1]
                target_label = gt_label[0, 0]
                if iou > 0.5:
                    nCorrect += 1
                if iou > 0.5 and pred_label == target_label:
                    nCls += 1
    recall = float(nCorrect / nGT) if nGT else 1
    precision = float(nCorrect / nProposals) if nProposals else 0
    F1_score = 2 * recall * precision / (recall + precision + 1e-16)

    return F1_score, precision, recall


def unique(tensor):
    tensor_np = tensor.cpu().detach().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two 3D bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3], box1[:, 4], box1[:, 5]
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3], box2[:, 4], box2[:, 5]

    # get the cordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0) * \
                 torch.clamp(inter_rect_z2 - inter_rect_z1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    transform the output of network to the format of yolo
    prediction: [batch, 198, grid, grid, grid] (Two branches)
                [batch, 99, grid, grid, grid] (Single branches)
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(3)
    grid_size = inp_dim // stride
    bbox_attrs = 7 + num_classes
    num_anchors = len(anchors)  # 9


    if prediction.size(1) == 198:  # Two branches
        prediction = prediction.view(batch_size, 2, bbox_attrs * num_anchors, grid_size, grid_size, grid_size)
        prediction = prediction.view(batch_size * 2, bbox_attrs * num_anchors, grid_size, grid_size, grid_size)
        multiplier = 2
    else:
        prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size, grid_size, grid_size)
        multiplier = 1

    num_boxes = (grid_size ** 3) * num_anchors  # single branch

    # [batch*multiplier, num_boxes, bbox_attrs]
    prediction = prediction.view(batch_size * multiplier, bbox_attrs * num_anchors, grid_size ** 3)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size * multiplier, num_boxes, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride, a[2] / stride) for a in anchors]

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 2] = torch.sigmoid(prediction[:, :, 2])
    prediction[:, :, 6] = torch.sigmoid(prediction[:, :, 6])

    grid = np.arange(grid_size)
    a, b, c = np.meshgrid(grid, grid, grid)

    y_offset = torch.FloatTensor(a).view(-1, 1)
    x_offset = torch.FloatTensor(b).view(-1, 1)
    z_offset = torch.FloatTensor(c).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        z_offset = z_offset.cuda()

    x_y_z_offset = torch.cat((x_offset, y_offset, z_offset), 1).repeat(1, num_anchors).view(-1, 3).unsqueeze(0)
    x_y_z_offset = x_y_z_offset.repeat(batch_size * multiplier, 1, 1)
    prediction[:, :, :3] += x_y_z_offset

    # log space transform
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size * grid_size, 1).unsqueeze(0)
    anchors = anchors.repeat(batch_size * multiplier, 1, 1)
    prediction[:, :, 3:6] = torch.exp(prediction[:, :, 3:6]) * anchors

    prediction[:, :, 7: 7 + num_classes] = torch.sigmoid((prediction[:, :, 7: 7 + num_classes]))
    prediction[:, :, :6] *= stride

    if multiplier == 2:
        prediction = prediction.view(batch_size, 2, num_boxes, bbox_attrs)
        prediction = prediction.view(batch_size, 2 * num_boxes, bbox_attrs)
    else:
        prediction = prediction.view(batch_size, num_boxes, bbox_attrs)

    return prediction


def write_results(predictions, confidence, num_classes, nms_conf=0.2):
    """
    perform NMS to remove the redundant bboxes
    :param predictions: list of 3 tensors from different scales, each shape [batch, 2*num_boxes, 11]
    :param confidence: confidence threshold
    :param num_classes: num_classes
    :param nms_conf: the union iou threshold
    :return: concatenated detections from all scales
    """
    if not isinstance(predictions, list):
        predictions = [predictions]

    all_detections = []

    for pred in predictions:
        # pred shape: [batch, num_boxes, 11]
        detections = process_single_scale(pred, confidence, num_classes, nms_conf)
        if not isinstance(detections, int):
            all_detections.append(detections)

    if all_detections:
        return torch.cat(all_detections, dim=0)
    return 0


def process_single_scale(prediction, confidence, num_classes, nms_conf=0.2):
    # [batch, total_anchors, features]
    if prediction is None or prediction.numel() == 0:
        return 0

    if len(prediction.shape) == 6:
        batch_size, nA, nF, nG, nG, nG = prediction.shape
        prediction = prediction.permute(0, 1, 3, 4, 5, 2).contiguous()
        prediction = prediction.view(batch_size, nA * nG * nG * nG, nF)

    batch_size, total_anchors, nF = prediction.shape

    if nF < 7 + num_classes:
        return 0

    # Confidence filtering
    conf_mask = (prediction[:, :, 6] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Convert to vertex coordinates
    box_corner = prediction.clone()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 3] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 4] / 2
    box_corner[:, :, 2] = prediction[:, :, 2] - prediction[:, :, 5] / 2
    box_corner[:, :, 3] = prediction[:, :, 0] + prediction[:, :, 3] / 2
    box_corner[:, :, 4] = prediction[:, :, 1] + prediction[:, :, 4] / 2
    box_corner[:, :, 5] = prediction[:, :, 2] + prediction[:, :, 5] / 2
    prediction[:, :, :6] = box_corner[:, :, :6]

    output_final = []

    for batch_idx in range(batch_size):
        image_pred = prediction[batch_idx]

        # Filter out low-confidence boxes
        mask = image_pred[:, 6] > 0
        image_pred = image_pred[mask]

        if image_pred.size(0) == 0:
            continue

        # Exclude boxes with a volume of 0
        boxes_temp = image_pred[:, :6]
        volumes_temp = (boxes_temp[:, 3] - boxes_temp[:, 0]) * \
                       (boxes_temp[:, 4] - boxes_temp[:, 1]) * \
                       (boxes_temp[:, 5] - boxes_temp[:, 2])
        valid_mask = volumes_temp > 0
        image_pred = image_pred[valid_mask]

        if image_pred.size(0) == 0:
            continue

        # Get category information
        max_conf, max_conf_idx = torch.max(image_pred[:, 7:7 + num_classes], dim=1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_idx = max_conf_idx.float().unsqueeze(1)

        detections = torch.cat([image_pred[:, :7], max_conf, max_conf_idx], dim=1)

        # Sort by confidence
        conf_scores = detections[:, 6]
        sorted_scores, sort_idx = conf_scores.sort(descending=True)
        detections = detections[sort_idx]
        boxes = detections[:, :6]

        # Calculate the volume of all boxes in advance
        volumes = (boxes[:, 3] - boxes[:, 0]) * \
                  (boxes[:, 4] - boxes[:, 1]) * \
                  (boxes[:, 5] - boxes[:, 2])

        # vector 3D NMS
        keep = torch.ones(detections.size(0), dtype=torch.bool, device=detections.device)

        for i in range(detections.size(0)):
            if not keep[i]:
                continue

            if i < detections.size(0) - 1:
                box_i = boxes[i].unsqueeze(0)
                rest_boxes = boxes[i + 1:]

                inter_x1 = torch.max(box_i[:, 0], rest_boxes[:, 0])
                inter_y1 = torch.max(box_i[:, 1], rest_boxes[:, 1])
                inter_z1 = torch.max(box_i[:, 2], rest_boxes[:, 2])
                inter_x2 = torch.min(box_i[:, 3], rest_boxes[:, 3])
                inter_y2 = torch.min(box_i[:, 4], rest_boxes[:, 4])
                inter_z2 = torch.min(box_i[:, 5], rest_boxes[:, 5])

                inter_vol = torch.clamp(inter_x2 - inter_x1, min=0) * \
                            torch.clamp(inter_y2 - inter_y1, min=0) * \
                            torch.clamp(inter_z2 - inter_z1, min=0)

                union_vol = volumes[i] + volumes[i + 1:] - inter_vol
                ious = inter_vol / (union_vol + 1e-16)

                keep[i + 1:] = keep[i + 1:] & (ious <= nms_conf)

        nms_results = detections[keep]

        # Add a batch index
        if nms_results.size(0) > 0:
            batch_indices = torch.full(
                (nms_results.size(0), 1),
                batch_idx,
                device=nms_results.device,
                dtype=nms_results.dtype
            )
            nms_results = torch.cat([batch_indices, nms_results], dim=1)
            output_final.append(nms_results)

    if output_final:
        return torch.cat(output_final, dim=0)
    return 0

