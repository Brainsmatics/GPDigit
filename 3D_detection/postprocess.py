"""
Output concatenation: use IoU to merge and eliminate redundancy.
"""
import os
import glob
import numpy as np
import pandas as pd


base_dir = "./predict/"
output_dir = "./predict/inall/"


def data_input(txt_dir, offset=True):
    outter_list = list()    # A list (in vertex format) containing all block plaque information
    for txt_path in glob.glob(os.path.join(txt_dir, "*.txt")):
        txt_session = list()
        with open(txt_path, "r") as f:
            a = f.readlines()
            if a != [' ']:
                # cls, x_c, y_c, z_c, w, h, l
                for line in a:
                    txt_session.append([float(l) for l in line.split(" ")[0:7]])
        txt_array = np.array(txt_session)
        if len(txt_array) == 0:
            outter_array = []
        else:
            h, w = txt_array.shape
            outter_array = np.zeros((h, w))
        
            if offset:
                n = int(os.path.basename(txt_path).split('.')[0])
                overlap = 28              # Redundancy is calculated based on 28-bit blocks
                x_offset = int((n-1)%5)
                # Set the corresponding values according to the actual assembly dimensions.
                y_offset = int(((n-1)%25)/5)
                z_offset = int((n-1)/25)
                # Here, we use the example of concatenating 128 data blocks to form 512 data blocks.
                txt_array[:, 1] += x_offset*(128-overlap)
                txt_array[:, 2] += y_offset*(128-overlap)
                txt_array[:, 3] += z_offset*(128-overlap)
                outter_array[:, 0] = txt_array[:, 0]
                outter_array[:, 1] = txt_array[:, 1]-txt_array[:, 4]/2
                outter_array[:, 2] = txt_array[:, 2]-txt_array[:, 5]/2
                outter_array[:, 3] = txt_array[:, 3]-txt_array[:, 6]/2
                outter_array[:, 4] = txt_array[:, 1]+txt_array[:, 4]/2
                outter_array[:, 5] = txt_array[:, 2]+txt_array[:, 5]/2
                outter_array[:, 6] = txt_array[:, 3]+txt_array[:, 6]/2

        outter_list.extend(outter_array)
    outter_arr = np.array(outter_list)
    return outter_arr


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


def match(outter_arr):
    box_arr = outter_arr[:, 1:7].copy().copy()
    box_index = 0
    match_arr = np.zeros((100000, 200))
    for src_index in range(box_arr.shape[0]):
        src_box = box_arr[src_index, :]
        n, _ = np.where(match_arr == src_index+1)
        if not len(n):
            match_arr[box_index, 0] = src_index+1
            n = [box_index]
            box_index += 1
        ious = bbox_iou(src_box, box_arr)
        near_index = np.where(ious > 0.5)[0]
        for iou_index in near_index:
            if not np.all(box_arr[iou_index] == src_box):
                target_index = iou_index + 1
                n1 = np.where(match_arr[n[0], :] == target_index)
                if not len(n1):
                    target_column = np.where(match_arr[n[0], :] == 0)[0]
                    match_arr[n[0], target_column] = target_index
                else:
                    continue
    num_plaque = np.where(match_arr[:,0] !=0 )[0][-1]+1
    match_arr = match_arr[:num_plaque,:]
    return match_arr


def post_process(txt_dir,block_name):
    outter_arr = data_input(txt_dir, True)
    match_arr = match(outter_arr)
    df_c = pd.DataFrame()
    df_out = pd.DataFrame()
    for num_plaque in range(match_arr.shape[0]):
        plaque3d_list = list()
        for i in range(match_arr.shape[1]):
            if match_arr[num_plaque, i] != 0:
                plaque3d = outter_arr[int(match_arr[num_plaque, i] - 1), :]
                plaque3d_list.append(plaque3d)
        plaque3d_arr = np.array(plaque3d_list)
        cls = np.argmax(np.bincount(plaque3d_arr[:, 0].astype(int)))
        xmin = np.min(plaque3d_arr[:, 1])
        ymin = np.min(plaque3d_arr[:, 2])
        zmin = np.min(plaque3d_arr[:, 3])
        xmax = np.max(plaque3d_arr[:, 4])
        ymax = np.max(plaque3d_arr[:, 5])
        zmax = np.max(plaque3d_arr[:, 6])
        x_c = (xmin+xmax)/2
        y_c = (ymin+ymax)/2
        z_c = (zmin+zmax)/2
        w = xmax-xmin
        h = ymax-ymin
        l = zmax-zmin
        df_out[num_plaque] = pd.Series({"x_min": xmin,
                                        "y_min": ymin,
                                        "z_min": ymin,
                                        "x_max": xmax,
                                        "y_max": ymax,
                                        "z_max": ymax,
                                        "cls": int(cls)})
        df_c[num_plaque] = pd.Series({"x_c": x_c,
                                      "y_c": y_c,
                                      "z_c": z_c,
                                      "w": w,
                                      "h": h,
                                      "l": l,
                                      "cls": int(cls)})
    save_dir = output_dir+block_name+".txt"
    df_out.T.to_csv(save_dir, sep=' ',index=False, header=False)
    save_c_dir = output_dir+block_name+"_c.txt"
    df_c.T.to_csv(save_c_dir, sep=' ',index=False, header=False)


if __name__ == "__main__": 
    # file_names = next(os.walk(base_dir))[2]
    # file_names.sort(key=lambda x:(x.split('.')[0]))
    # for file_name in file_names:
        # block_name = os.path.splitext(file_name)[0]
        # swc_dir = os.path.join(base_dir, file_name)
        # post_process(swc_dir,block_name)
    block_name = '10_3_13'
    post_process(base_dir, block_name)

