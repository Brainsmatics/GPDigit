'''
Use json2txt.py to convert the 512×3 slicer labels into a 128×3 text file.
Then use this programme to convert the 128×3 labels back into slicer annotations, so that labels can continue to be plotted on the 128×3 image.
Subsequently, label_process.py can be used to process the new 128×3 annotated JSON output into labels suitable for 2D/3D networks.
'''
import os
import glob
import json
import sys
import math
import shutil

base_dir = "./new-3dslicer/block/yolo/"
json_dir = "./new-3dslicer/1-1.mrk.json"
output_dir = "./new-3dslicer/block/new-json/"
categories = {"diffuse":1, "core":2, "CAA":3, "cotton":4}

def load_txt(file_name):
    file_suffix = os.path.splitext(file_name)[-1]
    if file_suffix != ".txt":
        return 
    name = os.path.splitext(file_name)[0]
    txt_path = os.path.join(base_dir, file_name)
    object_num = 0
    for line in open(txt_path, "r"):
        if line != [' ']:
            object_num = object_num+1
            line = line.strip()
            info = line.split(" ")
            x_c = float(info[0])
            y_c = float(info[1])
            z_c = float(info[2])
            x_r = float(info[3])
            y_r = float(info[4])
            z_r = float(info[5])
            cls = info[6]
            if cls not in categories:
                new_id = len(categories)
                categories[cls] = new_id+1
            c_id = categories[cls]
            
            copy_path = output_dir + name + '/'
            if name not in os.listdir(output_dir):
                os.mkdir(copy_path) 
            copy_dir = os.path.join(copy_path, str(c_id) + '-' + str(object_num) + '.mrk.json')
            shutil.copy(json_dir,copy_dir)

            with open(copy_dir, "r", encoding='utf-8') as f:
                bbox = json.load(f)
            bbox['markups'][0]['center'] = [x_c,y_c,z_c]
            bbox['markups'][0]['controlPoints'][0]['position'] = [x_c, y_c, z_c]
            bbox['markups'][0]['size'] = [x_r*2, y_r*2, z_r*2]
            with open(copy_dir, "w", encoding='utf-8') as dump_f:
                json.dump(bbox, dump_f, indent=2)


if __name__ == "__main__": 
    file_names = next(os.walk(base_dir))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    for file_name in file_names: 
        load_txt(file_name)

