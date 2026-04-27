'''
Single-image postprocessing demo
'''
import os
import glob
import numpy as np
import pandas as pd
import cv2
import time
import math
import SimpleITK as sitk
import pandas as pd
import libtiff as TIFF
import tifffile
import imutils
from imutils.paths import list_images

input_dir = './fm-show/fm/'
output_dir = './fm-show/fm/process/'

def print_factor(x):
    num_all = []
    for i in range(1, x + 1):
        if x % i == 0:
            num_all.append(i)
    num_all = sorted(num_all)
    l = len(num_all)
    half = int(l/2)
    if l%2==0:
        a = num_all[half-1]
    if l%2==1:
        a = num_all[half]
    return a

def concat(img_list):
    img_concat = np.zeros((128,10000,10000)).astype('uint32')
    l, h, w = img_list[0].shape
    n = len(img_list)
    x = print_factor(n)
    y = n/x
    i = 0
    for img_arr in img_list:
        x_offset = int(i % x)
        y_offset = int(i / x)
        xmin = int(x_offset) * w
        ymin = int(y_offset) * h
        img_concat[:l, ymin:(ymin+h), xmin:(xmin+w)] = img_arr
        i += 1
    xmax = int(w*x)
    ymax = int(h*y)
    img_concat = img_concat[0:l, 0:ymax, 0:xmax]
    img_concat = np.array(img_concat).astype('uint32')
    return img_concat

def addall(img_list):
    l, h, w = img_list[0].shape
    img_add = np.zeros((l, h, w)).astype('float64')
    num = len(img_list)
    for img_arr in img_list:
        for i in range(l):
            for j in range(h):
                for k in range(w):
                    img_add[i, j, k] += img_arr[i, j, k]
    img_add = img_add/num
    img_add = np.array(img_add).astype('uint32')
    return img_add


if __name__ == "__main__":
    file_names = next(os.walk(input_dir))[2]
    file_names.sort(key=lambda x:(x.split('.')[0].split('_')[0], x.split('.')[0].split('_')[-1]))
    for i in range(54):
        img_list = []
        for file_name in file_names:
            file_suffix = os.path.splitext(file_name)[-1]
            if file_suffix != ".tif":
                continue
            if int(file_name.split("_")[0])==i:
                img_dir = os.path.join(input_dir, file_name)
                img = sitk.ReadImage(img_dir)
                img_arr = sitk.GetArrayFromImage(img)
                img_list.append(img_arr)
        img_concat = concat(img_list)
        img_addall = addall(img_list)
        concat_dir = output_dir+"concat/"+str(i)+"-concat.tif"
        tifffile.imsave(concat_dir, img_concat)
        addall_dir = output_dir+"add/"+str(i)+"-add.tif"
        tifffile.imsave(addall_dir, img_addall)
