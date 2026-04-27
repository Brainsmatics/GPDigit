'''
Process the labels drawn by Slicer in .json format (labels drawn on 512×3 blocks)
Break these down into 2D labels (in .txt files, which must subsequently be processed by txt_conver.py to generate .xml/.json files)
and 3D block (128×3) labels (in .txt files; redundancy must not be set)
'''
import os
import glob
import json
import sys
import numpy as np
import pandas as pd
import json
import math

base_dir = "./3D-slicer-test/new/"
output_path = "./3D-slicer-test/new/"
name_dict = {1:"diffuse", 2:"core", 3:"CAA", 4:"cotton"}

def load_json(image_size):
    df_2d_all = pd.DataFrame()
    df_3d_all = pd.DataFrame()
    file_names = next(os.walk(base_dir))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    i = 0
    for file_name in file_names:
        file_suffix = os.path.splitext(file_name)[-1]
        if file_suffix != ".json":
            continue
        cls = name_dict[int(file_name[0])]
        json_dir = os.path.join(base_dir, file_name)
        with open(json_dir, "r", encoding='utf-8') as f:
            bbox = json.load(f)
            center = bbox['markups'][0]['center']
            size = bbox['markups'][0]['size']
        x_c,y_c,z_c = center
        w,h,l = size
        x_r = w/2
        y_r = h/2
        z_r = l/2
        if (x_c<x_r)or((image_size-x_c)<x_r):
            if x_c<x_r:
                x_r = (x_r + x_c)/2
                x_c = x_r
            else:
                x_r = (image_size - x_c + x_r)/2
                x_c = image_size - x_r
            w = x_r*2
        if (y_c<y_r)or((image_size-y_c)<y_r):
            if y_c<y_r:
                y_r = (y_r + y_c)/2
                y_c = y_r
            else:
                y_r = (image_size - y_c + y_r)/2
                y_c = image_size - y_r
            h = y_r*2
        if (z_c<z_r)or((image_size-z_c)<z_r):
            if z_c<z_r:
                z_r = (z_r + z_c)/2
                z_c = z_r
            else:
                z_r = (image_size - z_c + z_r)/2
                z_c = image_size - z_r
            l =z_r*2
        x_min = x_c - x_r
        y_min = y_c - y_r
        z_min = z_c - z_r
        x_max = x_c + x_r
        y_max = y_c + y_r
        z_max = z_c + z_r
        df_3d_all[i] = pd.Series({"x_min": int(x_min),
                                  "y_min": int(y_min),
                                  "z_min": int(z_min),
                                  "x_max": math.ceil(x_max),
                                  "y_max": math.ceil(y_max),
                                  "z_max": math.ceil(z_max),
                                  "x_center": int(x_c),
                                  "y_center": int(y_c),
                                  "z_center": int(z_c),
                                  "x_r": math.ceil(x_r),
                                  "y_r": math.ceil(y_r),
                                  "z_r": math.ceil(z_r),
                                  "class": cls })
        i = i+1
        df_2d_apart = pd.DataFrame()
        for j in range(math.ceil(l) + 1):
            df_2d_apart[j] = pd.Series({"x_min": int(x_min),
                                        "y_min": int(y_min),
                                        "x_max": math.ceil(x_max),
                                        "y_max": math.ceil(y_max),
                                        "width": math.ceil(w),
                                        "height": math.ceil(h),
                                        "z": int(z_min) + j,
                                        "class": cls })
        df_2d_all = df_2d_all.append(df_2d_apart.T) 
    df_3d_all = (df_3d_all.T).reset_index(drop=True)
    df_2d_all = df_2d_all.reset_index(drop=True)
    return df_3d_all, df_2d_all

def output_2d(df_2d_all, image_size, output_2d_path):
    for i in range(image_size):
        i = i+1
        df_2d = df_2d_all.loc[df_2d_all['z'] == i]
        df_2d_all = df_2d_all.drop(index=(df_2d.index.tolist()))
        df_2d = df_2d.reset_index(drop=True)
        #df_2d.to_excel(output_2d_path + str(i) + '.xls')
        df_2d.to_csv(output_2d_path + str(i) + '.txt', sep=' ',index=False, header=False)
        if (len(df_2d_all) == 0):
            break
    for j in range(image_size-i):
        open(output_2d_path + str(i+j+1) + '.txt', "w")

def output_3d(df_3d_all, image_size, cut_size, output_3d_path, output_yolo_path):
    l = len(df_3d_all)
    n = math.ceil(image_size/cut_size)
    block_3d = [[[]]for i in range(n*n*n)]
    block_yolo = [[[]]for i in range(n*n*n)]
    for m in range(l):
        x_min = df_3d_all.iloc[m]['x_min']
        x_max = df_3d_all.iloc[m]['x_max']
        y_min = df_3d_all.iloc[m]['y_min']
        y_max = df_3d_all.iloc[m]['y_max']
        z_min = df_3d_all.iloc[m]['z_min']
        z_max = df_3d_all.iloc[m]['z_max']
        x_center = df_3d_all.iloc[m]['x_center']
        y_center = df_3d_all.iloc[m]['y_center']
        z_center = df_3d_all.iloc[m]['z_center']
        x_r = df_3d_all.iloc[m]['x_r']
        y_r = df_3d_all.iloc[m]['y_r']
        z_r = df_3d_all.iloc[m]['z_r']
        cls = df_3d_all.iloc[m]['class']
                
        i = int(x_min / cut_size)
        j = int(y_min / cut_size)
        k = int(z_min / cut_size)
        block_num = k*n*n + j*n + i
                
        x_min_new = x_min - (i*cut_size)
        x_max_new = x_max - (i*cut_size)
        y_min_new = y_min - (j*cut_size)
        y_max_new = y_max - (j*cut_size)
        z_min_new = z_min - (k*cut_size)
        z_max_new = z_max - (k*cut_size)
        x_center_new = x_center - (i*cut_size)
        y_center_new = y_center - (j*cut_size)
        z_center_new = z_center - (k*cut_size)
    
        judge_x = judge_y = judge_z = 0
        if (x_max_new<=cut_size) & (y_max_new<=cut_size) & (z_max_new<=cut_size):
            bbox_3d = [x_min_new, y_min_new, z_min_new, x_max_new, y_max_new, z_max_new, cls]
            bbox_yolo = [x_center_new, y_center_new, z_center_new, x_r, y_r, z_r, cls]
            block_3d[block_num].append(bbox_3d)
            block_yolo[block_num].append(bbox_yolo)
        if (x_max_new>cut_size) or (y_max_new>cut_size) or (z_max_new>cut_size):
            if x_max_new>cut_size:
                judge_x = 1
                x_3d_min_1 = x_min_new
                x_3d_min_2 = 0
                x_3d_max_1 = cut_size
                x_3d_max_2 = x_max_new-cut_size
                x_r_1 = (cut_size - x_min_new + 1)/2
                x_r_2 = (x_max_new - cut_size + 1)/2
                x_yolo_1 = x_min_new + x_r_1
                x_yolo_2 = x_r_2
            else:
                x_3d_min_1 = x_3d_min_2 = x_min_new
                x_3d_max_1 = x_3d_max_2 = x_max_new
                x_r_1 = x_r_2 = x_r
                x_yolo_1 = x_yolo_2 = x_center_new         
            if y_max_new>cut_size:
                judge_y = 1
                y_3d_min_1 = y_min_new
                y_3d_min_2 = 0
                y_3d_max_1 = cut_size
                y_3d_max_2 = y_max_new-cut_size
                y_r_1 = (cut_size - y_min_new + 1)/2
                y_r_2 = (y_max_new - cut_size + 1)/2
                y_yolo_1 = y_min_new + y_r_1
                y_yolo_2 = y_r_2
            else:
                y_3d_min_1 = y_3d_min_2 = y_min_new
                y_3d_max_1 = y_3d_max_2 = y_max_new
                y_r_1 = y_r_2 = y_r
                y_yolo_1 = y_yolo_2 = y_center_new
            if z_max_new>cut_size:
                judge_z = 1
                z_3d_min_1 = z_min_new
                z_3d_min_2 = 0
                z_3d_max_1 = cut_size
                z_3d_max_2 = z_max_new-cut_size
                z_r_1 = (cut_size - z_min_new + 1)/2
                z_r_2 = (z_max_new - cut_size + 1)/2
                z_yolo_1 = z_min_new + z_r_1
                z_yolo_2 = z_r_2
            else:
                z_3d_min_1 = z_3d_min_2 = z_min_new
                z_3d_max_1 = z_3d_max_2 = z_max_new
                z_r_1 = z_r_2 = z_r
                z_yolo_1 = z_yolo_2 = z_center_new
            if ((judge_x==1)&(judge_y==judge_z==0))or((judge_y==1)&(judge_x==judge_z==0))or((judge_z==1)&(judge_y==judge_x==0)):
                bbox_3d_1 = [x_3d_min_1,  y_3d_min_1, z_3d_min_1, x_3d_max_1, y_3d_max_1, z_3d_max_1, cls]
                bbox_3d_2 = [x_3d_min_2,  y_3d_min_2, z_3d_min_2, x_3d_max_2, y_3d_max_2, z_3d_max_2, cls]
                bbox_yolo_1 = [x_yolo_1, y_yolo_1, z_yolo_1, x_r_1, y_r_1, z_r_1, cls]
                bbox_yolo_2 = [x_yolo_2, y_yolo_2, z_yolo_2, x_r_2, y_r_2, z_r_2, cls]
                if judge_x==1:
                    block_3d[block_num+1].append(bbox_3d_2)
                    block_yolo[block_num+1].append(bbox_yolo_2)
                if judge_y==1:
                    block_3d[block_num+n].append(bbox_3d_2)
                    block_yolo[block_num+n].append(bbox_yolo_2)
                if judge_z==1:
                    block_3d[block_num+n*n].append(bbox_3d_2)
                    block_yolo[block_num+n*n].append(bbox_yolo_2)
            elif (judge_x==judge_y==1)&(judge_z==0):
                bbox_3d_1 = [x_3d_min_1,  y_3d_min_1, z_3d_min_1, x_3d_max_1, y_3d_max_1, z_3d_max_1, cls]
                bbox_3d_2 = [x_3d_min_2,  y_3d_min_1, z_3d_min_2, x_3d_max_2, y_3d_max_1, z_3d_max_2, cls]
                bbox_3d_3 = [x_3d_min_1,  y_3d_min_2, z_3d_min_1, x_3d_max_1, y_3d_max_2, z_3d_max_1, cls]
                bbox_3d_4 = [x_3d_min_2,  y_3d_min_2, z_3d_min_2, x_3d_max_2, y_3d_max_2, z_3d_max_2, cls]
                bbox_yolo_1 = [x_yolo_1, y_yolo_1, z_yolo_1, x_r_1, y_r_1, z_r_1, cls]
                bbox_yolo_2 = [x_yolo_2, y_yolo_1, z_yolo_2, x_r_2, y_r_1, z_r_2, cls]
                bbox_yolo_3 = [x_yolo_1, y_yolo_2, z_yolo_1, x_r_1, y_r_2, z_r_1, cls]
                bbox_yolo_4 = [x_yolo_2, y_yolo_2, z_yolo_2, x_r_2, y_r_2, z_r_2, cls]
                block_3d[block_num+1].append(bbox_3d_2)
                block_yolo[block_num+1].append(bbox_yolo_2)
                block_3d[block_num+n].append(bbox_3d_3)
                block_yolo[block_num+n].append(bbox_yolo_3)
                block_3d[block_num+n+1].append(bbox_3d_4)
                block_yolo[block_num+n+1].append(bbox_yolo_4)
            elif ((judge_y==judge_z==1)&(judge_x==0))or((judge_x==judge_z==1)&(judge_y==0)):
                bbox_3d_1 = [x_3d_min_1,  y_3d_min_1, z_3d_min_1, x_3d_max_1, y_3d_max_1, z_3d_max_1, cls]
                bbox_3d_2 = [x_3d_min_2,  y_3d_min_2, z_3d_min_1, x_3d_max_2, y_3d_max_2, z_3d_max_1, cls]
                bbox_3d_3 = [x_3d_min_1,  y_3d_min_1, z_3d_min_2, x_3d_max_1, y_3d_max_1, z_3d_max_2, cls]
                bbox_3d_4 = [x_3d_min_2,  y_3d_min_2, z_3d_min_2, x_3d_max_2, y_3d_max_2, z_3d_max_2, cls]
                bbox_yolo_1 = [x_yolo_1, y_yolo_1, z_yolo_1, x_r_1, y_r_1, z_r_1, cls]
                bbox_yolo_2 = [x_yolo_2, y_yolo_2, z_yolo_1, x_r_2, y_r_2, z_r_1, cls]
                bbox_yolo_3 = [x_yolo_1, y_yolo_1, z_yolo_2, x_r_1, y_r_1, z_r_2, cls]
                bbox_yolo_4 = [x_yolo_2, y_yolo_2, z_yolo_2, x_r_2, y_r_2, z_r_2, cls]
                if judge_x==0:
                    block_3d[block_num+n].append(bbox_3d_2)
                    block_yolo[block_num+n].append(bbox_yolo_2)
                    block_3d[block_num+n*n].append(bbox_3d_3)
                    block_yolo[block_num+n*n].append(bbox_yolo_3)
                    block_3d[block_num+n*n+n].append(bbox_3d_4)
                    block_yolo[block_num+n*n+n].append(bbox_yolo_4)
                if judge_y==0:
                    block_3d[block_num+1].append(bbox_3d_2)
                    block_yolo[block_num+1].append(bbox_yolo_2)
                    block_3d[block_num+n*n].append(bbox_3d_3)
                    block_yolo[block_num+n*n].append(bbox_yolo_3)
                    block_3d[block_num+n*n+1].append(bbox_3d_4)
                    block_yolo[block_num+n*n+1].append(bbox_yolo_4)
            elif judge_x==judge_y==judge_z==1:
                bbox_3d_1 = [x_3d_min_1,  y_3d_min_1, z_3d_min_1, x_3d_max_1, y_3d_max_1, z_3d_max_1, cls]
                bbox_3d_2 = [x_3d_min_2,  y_3d_min_1, z_3d_min_1, x_3d_max_2, y_3d_max_1, z_3d_max_1, cls]
                bbox_3d_3 = [x_3d_min_1,  y_3d_min_2, z_3d_min_1, x_3d_max_1, y_3d_max_2, z_3d_max_1, cls]
                bbox_3d_4 = [x_3d_min_2,  y_3d_min_2, z_3d_min_1, x_3d_max_2, y_3d_max_2, z_3d_max_1, cls]
                bbox_3d_5 = [x_3d_min_1,  y_3d_min_1, z_3d_min_2, x_3d_max_1, y_3d_max_1, z_3d_max_2, cls]
                bbox_3d_6 = [x_3d_min_2,  y_3d_min_1, z_3d_min_2, x_3d_max_2, y_3d_max_1, z_3d_max_2, cls]
                bbox_3d_7 = [x_3d_min_1,  y_3d_min_2, z_3d_min_2, x_3d_max_1, y_3d_max_2, z_3d_max_2, cls]
                bbox_3d_8 = [x_3d_min_2,  y_3d_min_2, z_3d_min_2, x_3d_max_2, y_3d_max_2, z_3d_max_2, cls]
                bbox_yolo_1 = [x_yolo_1, y_yolo_1, z_yolo_1, x_r_1, y_r_1, z_r_1, cls]
                bbox_yolo_2 = [x_yolo_2, y_yolo_1, z_yolo_1, x_r_2, y_r_1, z_r_1, cls]
                bbox_yolo_3 = [x_yolo_1, y_yolo_2, z_yolo_1, x_r_1, y_r_2, z_r_1, cls]
                bbox_yolo_4 = [x_yolo_2, y_yolo_2, z_yolo_1, x_r_2, y_r_2, z_r_1, cls]
                bbox_yolo_5 = [x_yolo_1, y_yolo_1, z_yolo_2, x_r_1, y_r_1, z_r_2, cls]
                bbox_yolo_6 = [x_yolo_2, y_yolo_1, z_yolo_2, x_r_2, y_r_1, z_r_2, cls]
                bbox_yolo_7 = [x_yolo_1, y_yolo_2, z_yolo_2, x_r_1, y_r_2, z_r_2, cls]
                bbox_yolo_8 = [x_yolo_2, y_yolo_2, z_yolo_2, x_r_2, y_r_2, z_r_2, cls]
                block_3d[block_num+1].append(bbox_3d_2)
                block_yolo[block_num+1].append(bbox_yolo_2)
                block_3d[block_num+n].append(bbox_3d_3)
                block_yolo[block_num+n].append(bbox_yolo_3)
                block_3d[block_num+n+1].append(bbox_3d_4)
                block_yolo[block_num+n+1].append(bbox_yolo_4)
                block_3d[block_num+n*n].append(bbox_3d_5)
                block_yolo[block_num+n*n].append(bbox_yolo_5)
                block_3d[block_num+n*n+1].append(bbox_3d_6)
                block_yolo[block_num+n*n+1].append(bbox_yolo_6)
                block_3d[block_num+n*n+n].append(bbox_3d_7)
                block_yolo[block_num+n*n+n].append(bbox_yolo_7)
                block_3d[block_num+n*n+n+1].append(bbox_3d_8)
                block_yolo[block_num+n*n+n+1].append(bbox_yolo_8)
            block_3d[block_num].append(bbox_3d_1)
            block_yolo[block_num].append(bbox_yolo_1)
        
    for c in range(n*n*n):
        df_3d = pd.DataFrame(block_3d[c][1:])
        df_yolo = pd.DataFrame(block_yolo[c][1:])
        if len(df_3d)!=0:
            df_3d = df_3d.drop_duplicates()
            df_yolo = df_yolo.drop_duplicates()
            df_3d = df_3d.reset_index(drop=True)
            df_yolo = df_yolo.reset_index(drop=True)
            #df_3d.to_excel(output_3d_path+ str(c+1) + '.xls')
            df_3d.to_csv(output_3d_path + str(c+1) + '.txt', sep=' ',index=False, header=False)
            #df_yolo.to_excel(output_yolo_path+ str(c+1) + '.xls')
            df_yolo.to_csv(output_yolo_path + str(c+1) + '.txt', sep=' ',index=False, header=False)
        else:
            df_3d.to_csv(output_3d_path + str(c+1) + '.txt', sep=' ',index=False, header=False)
            df_yolo.to_csv(output_yolo_path + str(c+1) + '.txt', sep=' ',index=False, header=False)


if __name__ == "__main__": 
    output_2d_path = output_path + '2d/'
    output_3d_path = output_path + '3d/'
    output_yolo_path = output_path + 'yolo/'
    if '2d' not in os.listdir(output_path):
        os.mkdir(output_2d_path)
    if '3d' not in os.listdir(output_path): 
        os.mkdir(output_3d_path)
    if 'yolo' not in os.listdir(output_path): 
        os.mkdir(output_yolo_path)
    df_3d_all, df_2d_all = load_json()
    output_2d(df_2d_all, 512, output_2d_path)
    output_3d(df_3d_all, 512, 128, output_3d_path, output_yolo_path)

