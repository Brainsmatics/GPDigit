"""
    Post-processing and stitching of 2D images.
"""
import os
import glob
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath("../")
base_dir = ROOT_DIR + "img/"             #路径结尾带/
output_dir = ROOT_DIR + "img_save/"           #路径结尾带/


def data_input(swc_dir, offset=True):
    swc_list = list()
    for swc_path in glob.glob(os.path.join(swc_dir, "*.txt")):
        swc_session = list()
        with open(swc_path, "r") as f:
            a = f.readlines()

            if a != [' ']:
                #
                # num, 1, xmin，ymin, xmax, ymax, z, cls, score, -1
                for line in a:
                    swc_session.append([float(l) for l in line.split(" ")[2:9]])
        swc_array = np.array(swc_session)
        if offset:
            n = int(os.path.basename(swc_path).split('.')[0])
            overlap = 28             # Set according to the actual redundancy value.
            # Configure this according to your requirements; here, we use 128 combined to form 512 as an example.
            x_offset = int((n-1)%5)
            y_offset = int(((n-1)%25)/5)
            z_offset = int((n-1)/25)
            
            swc_array[:, 0] += x_offset*(128-overlap)       #512 or 128
            swc_array[:, 2] += x_offset*(128-overlap)
            swc_array[:, 1] += y_offset*(128-overlap)
            swc_array[:, 3] += y_offset*(128-overlap)
            swc_array[:, 4] += z_offset*(128-overlap)
            swc_array[swc_array < 0] = 0
            swc_list.extend(swc_array)
    swc_arr = np.array(swc_list)
    swc_out = pd.DataFrame(swc_arr)
    save_dir = output_dir + "all.txt"
    swc_out.to_csv(save_dir, sep=' ',index=False, header=False)
    return swc_arr


def bbox_iou(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    inter_rect_x1 = np.array([np.max((b1_x1, b2x1)) for b2x1 in b2_x1])
    inter_rect_y1 = np.array([np.max((b1_y1, b2y1)) for b2y1 in b2_y1])
    inter_rect_x2 = np.array([np.min((b1_x2, b2x2)) for b2x2 in b2_x2])
    inter_rect_y2 = np.array([np.min((b1_y2, b2y2)) for b2y2 in b2_y2])

    inter_area1 = inter_rect_x2 - inter_rect_x1 + 1
    inter_area2 = inter_rect_y2 - inter_rect_y1 + 1
    inter_area1[inter_area1 < 0] = 0
    inter_area2[inter_area2 < 0] = 0

    inter_area = inter_area1 * inter_area2
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def match(swc_arr):
    """
    The rows of `match_arr` represent the patch sequence numbers,
    and the columns represent the row numbers in the original swc file (both are incremented by 1).
    """
    box_arr = swc_arr[:, 0:4].copy().copy()
    box_index = 0
    match_arr = np.zeros((10000, 300))
    for src_index in range(box_arr.shape[0]):
        src_box = box_arr[src_index, :]
        n, _ = np.where(match_arr == src_index+1)   # When src_index == 0, n must be empty
        if not len(n):
            match_arr[box_index, 0] = src_index+1   # src_index==0,match_arr=1
            n = [box_index]                         # Numbering starting from 0
            ious = bbox_iou(src_box, box_arr)
            near_index = np.where(ious > 0.5)[0]
            for iou_index in near_index:
                if not np.all(box_arr[iou_index] == src_box):
                    # Look for a box that is not your own; your own box has already been assigned to match_arr[box_index, 0].
                    target_index = iou_index + 1
                    # Remove the frame paired with the current frame, and increment the index of match_arr by 1
                    n1 = np.where(match_arr[n[0], :] == target_index)[0]
                    # For each matched bounding box, check whether it already exists among all the bounding boxes in the current plaque
                    if not len(n1):
                        # n1 is evaluated as null, i.e. the matching box is not present in the current patch;
                        # for the first matching box, it must be 0
                        target_column = np.where(match_arr[n[0], :] == 0)[0][0]
                        # Find the first number with a column value of 0; this will be used to store the new index
                        match_arr[n[0], target_column] = target_index
                    else:
                        continue
            box_index += 1     #box编号+1
        else:
            continue
    num_plaque = np.where(match_arr[:,0] !=0 )[0][-1]+1    # Find out how many plaques there are in total
    match_arr = match_arr[:num_plaque,:]
    return match_arr



def post_process(swc_dir,block_name):
    swc_arr = data_input(swc_dir, True)
    match_arr = match(swc_arr)
    region_dataframe = pd.DataFrame()
    for num_plaque in range(match_arr.shape[0]):      # Merge all bounding boxes of a patch
        plaque2d_list = list()
        for i in range(match_arr.shape[1]):
            if match_arr[num_plaque, i] != 0:
                plaque2d = swc_arr[int(match_arr[num_plaque, i] - 1), :]
                plaque2d_list.append(plaque2d)
        plaque2d_arr = np.array(plaque2d_list)
        xmin = np.min(plaque2d_arr[:, 0])
        ymin = np.min(plaque2d_arr[:, 1])
        xmax = np.max(plaque2d_arr[:, 2])
        ymax = np.max(plaque2d_arr[:, 3])
        z = np.max(plaque2d_arr[:, 4])
        cls = np.argmax(np.bincount(plaque2d_arr[:, 5].astype(int)))
        score = np.mean(plaque2d_arr[:, 6])
        region_dataframe[num_plaque] = pd.Series({"num": int(num_plaque),
                                                  "start": 1,
                                                  "x_min": int(xmin),
                                                  "y_min": int(ymin),
                                                  "x_max": int(xmax),
                                                  "y_max": int(ymax),
                                                  "z": int(z),
                                                  "cls": int(cls),
                                                  "score": score,
                                                  "end": -1})
    save_dir = output_dir+block_name+".txt"
    region_dataframe.T.to_csv(save_dir, sep=' ', index=False, header=False)
    #save_swc = output_dir+swc_name+".swc"
    #region_dataframe.T.to_csv(save_swc, sep=' ',index=False, header=False)



if __name__ == "__main__": 
    # file_names = next(os.walk(base_dir))[2]
    # file_names.sort(key=lambda x:(x.split('.')[0]))
    # for file_name in file_names:
    #     block_name = os.path.splitext(file_name)[0]
    #     swc_dir = os.path.join(base_dir, file_name)
    #     post_process(swc_dir,block_name)
    block_name = '10_3_13'
    post_process(base_dir, block_name)


