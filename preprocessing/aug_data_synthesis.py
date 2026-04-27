'''
Perform copy_past (Cutmix) and pad_crop (Mosaic) operations.
'''

import os
import glob
import numpy as np
import SimpleITK as sitk
import cv2
from libtiff import TIFF 
import tifffile
import pandas as pd
import random
import math

img_dir = './data_aug_new/'    # ori path
txt_dir = './label_aug_new/'
output_dir = './output/'

def txt_input(txt_dir):
    # txt_list = list()
    outter_list = list()
    name_list = list()
    file_names = next(os.walk(txt_dir))[2]
    i = 0
    for file_name in file_names:
        txt_path = os.path.join(txt_dir, file_name)
        txt_session = list()
        with open(txt_path, "r") as f:
            a = f.readlines()
            if a != [' ']:
                for line in a:
                    txt_session.append([float(l) for l in line.split(" ")[0:7]])
        boxnum = len(txt_session)
        txt_array = np.array(txt_session)
        txt_array = np.append(txt_array, [[i]]*boxnum, axis=1)
        
        if len(txt_array) == 0:
            outter_array = []
        else:
            h, w = txt_array.shape
            outter_array = np.zeros((h, w))
        
        outter_array[:, 0] = txt_array[:, 0]-txt_array[:, 3]/2
        outter_array[:, 1] = txt_array[:, 1]-txt_array[:, 4]/2
        outter_array[:, 2] = txt_array[:, 2]-txt_array[:, 5]/2
        outter_array[:, 3] = txt_array[:, 0]+txt_array[:, 3]/2
        outter_array[:, 4] = txt_array[:, 1]+txt_array[:, 4]/2
        outter_array[:, 5] = txt_array[:, 2]+txt_array[:, 5]/2
        outter_array[:, 6] = txt_array[:, 6]
        outter_array[:, 7] = txt_array[:, 7]
            
        # txt_list.extend(txt_array)
        outter_list.extend(outter_array)
        img_name = os.path.join(img_dir, file_name.replace("txt", "tif"))
        name_list.append(img_name)
        i = i + 1
    outter_arr = np.array(outter_list)   # Vertex format
    return outter_arr, name_list

def bbox_iou(box1, box2):

    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1[0], box1[1], box1[2], box1[3], box1[4], box1[5]
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3], box2[:, 4], box2[:, 5]

    inter_rect_x1 = np.array([np.max((b1_x1, b2x1)) for b2x1 in b2_x1])
    inter_rect_y1 = np.array([np.max((b1_y1, b2y1)) for b2y1 in b2_y1])
    inter_rect_z1 = np.array([np.max((b1_z1, b2z1)) for b2z1 in b2_z1])
    inter_rect_x2 = np.array([np.min((b1_x2, b2x2)) for b2x2 in b2_x2])
    inter_rect_y2 = np.array([np.min((b1_y2, b2y2)) for b2y2 in b2_y2])
    inter_rect_z2 = np.array([np.min((b1_z2, b2z2)) for b2z2 in b2_z2])
    
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1)*(inter_rect_y2 - inter_rect_y1 + 1)*(inter_rect_z2 - inter_rect_z1 + 1)
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)*(b1_z2 - b1_z1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)*(b2_z2 - b2_z1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
    return iou


def copy_past(outter_arr, name_list):
    box_arr = outter_arr.copy()
    lens = box_arr.shape[0]
    out_list = [50, 70, 80, 100, 130, 160, 180, 200, 220, 250, 280, 300]  # Generate data containing varying numbers of boxes
    for out_num in out_list:
        for times in range(20):
            rand_array = np.arange(lens)
            np.random.shuffle(rand_array)
            row_rand = box_arr[rand_array[:]].copy()
            for i in range(lens):
                new_box = list()
                if row_rand.shape[0] >= (out_num + i):
                    ious = bbox_iou(row_rand[i, 0:6], row_rand[i + 1:, 0:6])
                    overlaps = np.where(abs(ious) <= 0.07)[0]
                    if len(overlaps) >= (out_num - 1):
                        new_box.append(row_rand[i])
                        ii = i
                        for k in overlaps:
                            box_ = row_rand[ii + 1 + k]
                            new_box.append(box_)
                            new_box_ = np.array(new_box)
                            ious_box = bbox_iou(box_, new_box_)
                            overlaps_ = np.where(abs(ious_box) > 0.07)[0]
                            if len(overlaps_) > 1:
                                del(new_box[-1])
                            else:
                                row_rand = np.delete(row_rand, ii + 1 + k, axis=0)
                                ii = ii - 1
                            new_box_lens = len(new_box)
                            if new_box_lens == out_num:
                                break

                        if new_box_lens > int(out_num/2):
                            new_box = np.array(new_box)
                            img_new = np.zeros((128, 128, 128)).astype('uint8')
                            for bbox in new_box:
                                img_name = name_list[int(bbox[7])]
                                img_ = sitk.ReadImage(img_name)
                                img_arr = sitk.GetArrayFromImage(img_)
                                x_min, y_min, z_min, x_max, y_max, z_max = bbox[0:6].astype("uint32")
                                box_cut = img_arr[z_min:z_max, y_min:y_max, x_min:x_max]
                                img_new[z_min:z_max, y_min:y_max, x_min:x_max] = box_cut
                            label_new = new_box[:, 0:7].copy()
                            label_new[:, 0] = (new_box[:, 3] + new_box[:, 0]) / 2
                            label_new[:, 1] = (new_box[:, 4] + new_box[:, 1]) / 2
                            label_new[:, 2] = (new_box[:, 5] + new_box[:, 2]) / 2
                            label_new[:, 3] = new_box[:, 3] - new_box[:, 0]
                            label_new[:, 4] = new_box[:, 4] - new_box[:, 1]
                            label_new[:, 5] = new_box[:, 5] - new_box[:, 2]
                            label_new[:, 6] = new_box[:, 6]
                            label_new[label_new < 0] = 0
                            label_new[label_new >= 128] = 127
                            label_new = pd.DataFrame(label_new)

                            img_path = output_dir + "image_cp_d"  # cp: copy_past
                            if "image_cp_d" not in os.listdir(output_dir):
                                os.mkdir(img_path)
                            label_path = output_dir + "label_cp_d"
                            if "label_cp_d" not in os.listdir(output_dir):
                                os.mkdir(label_path)

                            Img_path = img_path + "/" + str(times) + "_" + str(new_box_lens) + "_" + str(i) + "_cp_d.tif"
                            tifffile.imsave(Img_path, img_new)
                            Label_path = label_path + "/" + str(times) + "_" + str(new_box_lens) + "_" + str(i) + "_cp_d.txt"
                            label_new.to_csv(Label_path, sep=' ', index=False, header=False)



def img_input(file_names_):
    Img_arr = np.zeros((128, 256, 256)).astype('uint8')
    j,k = 0,0
    for file_name in file_names_:
        img_ = sitk.ReadImage(os.path.join(img_dir, file_name))
        img_arr = sitk.GetArrayFromImage(img_)
        h_before = k * 128
        h_after = k * 128 + 128
        w_before = j * 128
        w_after = j * 128 + 128
        Img_arr[0:128, h_before:h_after, w_before:w_after] = img_arr
        if j == 0:
            j = j + 1
        elif j == 1:
            j = 0
            k = k + 1
    return Img_arr

def txt_input_(file_names_):
    txt_list = list() 
    outter_list = list()
    j,k = 0,0
    for file_name in file_names_:
        txt_path = os.path.join(txt_dir, file_name.replace("tif", "txt"))
        txt_session = list()
        with open(txt_path, "r") as f:
            a = f.readlines()
            if a != [' ']:
                for line in a:
                    txt_session.append([float(l) for l in line.split(" ")[0:7]])
        txt_array = np.array(txt_session)
        txt_array[:, 0] = txt_array[:, 0] + j*128    # x_c
        txt_array[:, 1] = txt_array[:, 1] + k*128    # y_c
                
        if j == 0:
            j = j + 1
        elif j == 1:
            j = 0
            k = k + 1
        
        if len(txt_array) == 0:
            outter_array = []
        else:
            h, w = txt_array.shape
            outter_array = np.zeros((h, w))
        outter_array[:, 0] = txt_array[:, 0]-txt_array[:, 3]/2
        outter_array[:, 1] = txt_array[:, 1]-txt_array[:, 4]/2
        outter_array[:, 2] = txt_array[:, 2]-txt_array[:, 5]/2
        outter_array[:, 3] = txt_array[:, 0]+txt_array[:, 3]/2
        outter_array[:, 4] = txt_array[:, 1]+txt_array[:, 4]/2
        outter_array[:, 5] = txt_array[:, 2]+txt_array[:, 5]/2
        outter_array[:, 6] = txt_array[:, 6]
        
        txt_list.extend(txt_array)
        outter_list.extend(outter_array)
    txt_arr = np.array(txt_list)         # Format of the centre point
    outter_arr = np.array(outter_list)   # Vertex format
    return txt_arr, outter_arr


def pad_crop(img_dir):     
    file_names = next(os.walk(img_dir))[2]
    lens = len(file_names)
    for times in range(10):
        if times != 0:
            random.shuffle(file_names)
        for i in range(int(lens/4)):
            before = i*4
            after = i*4+4
            file_names_ = file_names[before:after]
            img_arr = img_input(file_names_)
            txt_arr, outter_arr = txt_input_(file_names_)
            img_new = np.zeros((128, 128, 128)).astype('uint8')
            img_new = img_arr[0:128, 64:192, 64:192]

            set_arr = np.zeros((txt_arr.shape[0], 4))
            set_arr[:, 0] = txt_arr[:, 0] + txt_arr[:, 3]/3          # left
            set_arr[:, 1] = txt_arr[:, 0] - txt_arr[:, 3]/3          # right
            set_arr[:, 2] = txt_arr[:, 1] + txt_arr[:, 4]/3          # up
            set_arr[:, 3] = txt_arr[:, 1] - txt_arr[:, 4]/3          # down

            index = list()
            for j in range(set_arr.shape[0]):
                if (set_arr[j, 0] >= 64) & (set_arr[j, 1] < 192) & (set_arr[j, 2] >= 64) & (set_arr[j, 3] < 192):
                    index.append(j)
            new_arr = outter_arr[index].copy()
            new_arr[:, 0:2] = new_arr[:, 0:2] - 64
            new_arr[:, 3:5] = new_arr[:, 3:5] - 64
            new_arr[new_arr < 0] = 0
            new_arr[new_arr >= 128] = 127
        
            h, w = new_arr.shape
            label_new = np.zeros((h, w))                             # x_c, y_c, z_c, w, h, l，cls
            label_new[:, 0] = (new_arr[:, 3] + new_arr[:, 0])/2
            label_new[:, 1] = (new_arr[:, 4] + new_arr[:, 1])/2
            label_new[:, 2] = (new_arr[:, 5] + new_arr[:, 2])/2
            label_new[:, 3] = new_arr[:, 3] - new_arr[:, 0]
            label_new[:, 4] = new_arr[:, 4] - new_arr[:, 1]
            label_new[:, 5] = new_arr[:, 5] - new_arr[:, 2]
            label_new[:, 6] = new_arr[:, 6]
            label_new = pd.DataFrame(label_new)
        
            img_path = output_dir + "image_pc_s"                   #pc: pad_crop
            if "image_pc_s" not in os.listdir(output_dir):
                os.mkdir(img_path)
            label_path = output_dir + "label_pc_s"
            if "label_pc_s" not in os.listdir(output_dir):
                os.mkdir(label_path)
                    
            Img_path = img_path + "/" + str(times) + "_" + str(i) + "_pc_s.tif"
            tifffile.imsave(Img_path, img_new)
            Label_path = label_path + "/" + str(times)+ "_" + str(i) + "_pc_s.txt"
            label_new.to_csv(Label_path, sep=' ',index=False, header=False)


if __name__ == "__main__":
    outter_arr, name_list = txt_input(txt_dir)
    copy_past(outter_arr, name_list)
    # pad_crop(img_dir)

