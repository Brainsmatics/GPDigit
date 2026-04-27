"""
    Post-processing is performed on the output files, resulting in two components:
    1) SWC files suitable for display in Amira: containing 3D information and radius
    2) SWC/TXT files suitable for subsequent processing and analysis:
           containing 3D information, patch centre coordinates, radius, classification results and scores
"""
import os
import glob
import numpy as np
import pandas as pd
import json
import PIL
from PIL import Image,ImageFilter,ImageDraw
from PIL import *
from skimage import data,filters
from tqdm import tqdm
import cv2
import time
import matplotlib.pyplot as plt
import multiprocessing
from skimage import measure,color
import imutils
from imutils.paths import list_images
import math

PIL.Image.MAX_IMAGE_PIXELS = 933120000


ROOT_DIR = os.path.abspath("../")
SWC_DIR = ROOT_DIR +'../'    # Output path
#num_process = 5
image_path = "../../"   # Path to the original image
prefix = "test_"                 # Prefix for original image file names
postfix = ".tif"                 # File extension of the original image
excel_path = SWC_DIR + 'process.xlsx'



def data_input(output_num):
    swc_path = swc_dir+output_num+'.swc'        # Path to the SWC file for the corresponding layer
    #swc_path = ROOT_DIR +'/output_process/test.swc' 
    region_dataframe = pd.DataFrame()
    with open(swc_path, "r") as f:
        a = f.readlines()
        if a != [' ']:
            for i, p_info in enumerate(a):
                if p_info.split(" ")[9] == "-1.0\n":
                    region_dataframe[i] = pd.Series({"x_min": float(p_info.split(" ")[2]),
                                                     "y_min": float(p_info.split(" ")[3]),
                                                     "x_max": float(p_info.split(" ")[4]),
                                                     "y_max": float(p_info.split(" ")[5]),
                                                     "z": float(p_info.split(" ")[6]),
                                                     "x_center": abs((float(p_info.split(" ")[2])+float(p_info.split(" ")[4]))/2),
                                                     "y_center": abs((float(p_info.split(" ")[3])+float(p_info.split(" ")[5]))/2),
                                                     "r_x": abs((float(p_info.split(" ")[4])-float(p_info.split(" ")[2]))/2),
                                                     "r_y": abs((float(p_info.split(" ")[5])-float(p_info.split(" ")[3]))/2),
                                                     "class": float(p_info.split(" ")[7]),
                                                     "score": float(p_info.split(" ")[8])})
                else:
                    continue
            return region_dataframe

        

def plaque_proposal(image_dataframe,output_num):
    """
    Perform inter-layer point tracking for each plaque, outputting information from
    all overlying layers for a single plaque, along with the individual details of that plaque.

    For each plaque, take the class of the central layer as its class;
    the score is updated using a weighted method.
    """
    plaque_proposal_all = pd.DataFrame()
    plaque_all = pd.DataFrame()
    # A summary of all 2D candidate bounding box information within each patch (excluding redundant candidate
    # bounding boxes within the same layer; retaining one bounding box per layer for each patch)

    image_dataframe_new = pd.DataFrame()
    dataframe_num = len(image_dataframe)
    for i in range(dataframe_num):
        # Select the eligible points, and set the plaque_num of all candidate points
        # within a plaque to the same value
        plaque_proposal = pd.DataFrame(columns = image_dataframe.columns)
        plaque_content = pd.DataFrame(columns = image_dataframe.columns)
        plaque_proposal = plaque_proposal.append(image_dataframe.iloc[0])    # A set of 2D candidate boxes for a patch
        plaque_content =  plaque_content.append(image_dataframe.iloc[0])
        # Without removing the redundant patch within the same layer from the set of 2D candidate boxes

        for j in range(output_num):
            x_center = plaque_proposal.iloc[j]['x_center']
            y_center = plaque_proposal.iloc[j]['y_center']
            r_x = plaque_proposal.iloc[j]['r_x']
            r_y = plaque_proposal.iloc[j]['r_y']
            z = plaque_proposal.iloc[j]['z']
            original_slice=image_dataframe.loc[(abs(image_dataframe['x_center']-x_center)<(r_x/2))
                                               &(abs(image_dataframe['y_center']-y_center)<(r_y/2))
                                               &((image_dataframe['z']-z)==0)]
            if (len(original_slice)>1):
                original_index = list(set(original_slice.index.values).intersection(set(plaque_proposal.index.values)))
                original_slice = original_slice.drop(index=original_index)
                plaque_content = plaque_content.append(original_slice)
            for n in range(1,4):
                plaque_slice=image_dataframe.loc[(abs(image_dataframe['x_center']-x_center)<(r_x/2))
                                                 &(abs(image_dataframe['y_center']-y_center)<(r_y/2))
                                                 &((image_dataframe['z']-z)==n)]
                if (len(plaque_slice)!=0):
                    break
            if (len(plaque_slice)==0):
                break
            plaque_content = plaque_content.append(plaque_slice)
            if (len(plaque_slice)>1):
                # Remove redundant candidate boxes on the same level, retaining the one closest to the centre point
                distance = [0]*len(plaque_slice)
                for l in range(len(plaque_slice)>1):
                    x0, y0 = plaque_slice.iloc[l]['x_center'], plaque_slice.iloc[l]['y_center']
                    x1, y1 = plaque_proposal.iloc[j]['x_center'], plaque_proposal.iloc[j]['y_center']
                    distance[l] = np.sqrt(np.square(x0-x1)+np.square(y0-y1))
                slice_index = np.argmin(distance)
                plaque_slice = plaque_slice.iloc[slice_index]
            plaque_proposal = plaque_proposal.append(plaque_slice)
        plaque_content = plaque_content.drop_duplicates()
        image_dataframe = image_dataframe.drop(index=(plaque_content.index.tolist()))
        plaque_num_list = [i]*len(plaque_content)
        plaque_num_list1 = [i]*len(plaque_proposal)
        plaque_content['plaque_num'] = plaque_num_list
        plaque_proposal['plaque_num'] = plaque_num_list1
        plaque_content.sort_values(by='z',inplace=True,ascending=True)
        plaque_proposal.sort_values(by='z',inplace=True,ascending=True)
        image_dataframe_new = image_dataframe_new.append(plaque_content)
        plaque_all = plaque_all.append(plaque_proposal)

        # Update the score for an individual patch: if more than half of the layers
        # agree with the central classification,
        # add 0.1 to the score; otherwise, no points are added. When the score exceeds 1, set it to 1.
        class_base = plaque_proposal.iloc[int(len(plaque_proposal)/2)]['class']
        score_count = 0
        for k in range(len(plaque_proposal)):
            if (plaque_proposal.iloc[k]['class']==class_base):
                score_count += 1
        if (score_count >= (int(len(plaque_proposal)/2))):
            score_weight = 0.1 + plaque_proposal.iloc[int(len(plaque_proposal)/2)]['score']
            if (score_weight>1):
                score_weight = 1
        else:
            score_weight = plaque_proposal.iloc[int(len(plaque_proposal)/2)]['score']

        # Summarise the information on all patches within a patch
        plaque_proposal_all[i] = pd.Series({"plaque_num": i,
                                            "x_min": plaque_proposal['x_min'].min(),
                                            "y_min": plaque_proposal['y_min'].min(),
                                            "x_max": plaque_proposal['x_max'].max(),
                                            "y_max": plaque_proposal['y_max'].max(),
                                            "z_min": int(plaque_proposal['z'].min()),
                                            "z_max": int(plaque_proposal['z'].max()),
                                            "x_center": (float(plaque_proposal['x_min'].min())+float(plaque_proposal['x_max'].max()))/2,
                                            "y_center": (float(plaque_proposal['y_min'].min())+float(plaque_proposal['y_max'].max()))/2,
                                            "z_center": int((plaque_proposal['z'].max()+plaque_proposal['z'].min())/2),
                                            "r_x": abs((float(plaque_proposal['x_max'].max())-float(plaque_proposal['x_min'].min()))/2),
                                            "r_y": abs((float(plaque_proposal['y_max'].max())-float(plaque_proposal['y_min'].min()))/2),
                                            "r_z": abs((int(plaque_proposal['z'].max())-int(plaque_proposal['z'].min()))/2),
                                            "class": plaque_proposal.iloc[int(len(plaque_proposal)/2)]['class'],
                                            "score": score_weight})  
        if (len(image_dataframe)==0):
            break
    return image_dataframe_new, plaque_all, plaque_proposal_all.T


def upgrade_r(plaque_proposal_all):
    # Update the value of r for each plaque (to be improved),
    # and record plaques that are not approximately spherical,
    # so that their centre positions and radii can be further updated in the next step
    r = pd.DataFrame()
    scale = pd.DataFrame()
    for i in range(len(plaque_proposal_all)):
        r_list = [int(plaque_proposal_all.iloc[i]['r_x']),int(plaque_proposal_all.iloc[i]['r_y']),int(plaque_proposal_all.iloc[i]['r_z'])]
        r_max_num = r_list.index(max(r_list))
        if r_max_num == 0:
            r_max = plaque_proposal_all.iloc[i]['r_x']
        elif r_max_num == 1:
            r_max = plaque_proposal_all.iloc[i]['r_y']
        else:
            r_max = plaque_proposal_all.iloc[i]['r_z']
        r[i] = pd.Series({"r": r_max})     # Find the maximum r in three directions

        if (plaque_proposal_all.iloc[i]['r_x']==0)or(plaque_proposal_all.iloc[i]['r_y']==0)or (plaque_proposal_all.iloc[i]['r_z']==0):
            scale[i] = pd.Series({"scale": 1})
            continue
        else:
            scale_xy = plaque_proposal_all.iloc[i]['r_x']/plaque_proposal_all.iloc[i]['r_y']
            scale_xz = plaque_proposal_all.iloc[i]['r_x']/plaque_proposal_all.iloc[i]['r_z']
            scale_yz = plaque_proposal_all.iloc[i]['r_y']/plaque_proposal_all.iloc[i]['r_z']
            if ((0.7<scale_xy<1.3)&(0.7<scale_xz<1.3)&(0.7<scale_yz<1.3)):
                scale[i] = pd.Series({"scale": 0})
            elif (0.7<scale_xy<1.3):                                         # Irregular in the z-direction
                scale[i] = pd.Series({"scale": 1})
            else:
                scale[i] = pd.Series({"scale": 2})         
    plaque_proposal_all_new=pd.concat([plaque_proposal_all, r.T, scale.T], axis=1)
    return plaque_proposal_all_new


def session_vol(image_path): 
    files = os.listdir(image_path)
    # files = glob.glob(pathname=image_path+'*.jpg')
    img_vol = 0
    for path in files:
        full_path = os.path.join(image_path, path)
        img = cv2.imread(full_path)
        height,width,_ = img.shape
        img_area = height*width
        # img = cv2.imread(full_path,0)
        # Unless the entire image consists of brain scans, when there is a background
        # _,c = np.where(img>=1)
        # img_area = len(c)
        img_vol += img_area 
    return img_vol



def calGrayHist(image):    
    rows, cols = image.shape
    grayHist = np.zeros([256])
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r,c])] += 1
    return grayHist



def threshEntroy(image):    
    rows,cols = image.shape

    grayHist = calGrayHist(image)

    normgrayHist = grayHist/float(rows*cols)
    zeroCumuMoment = np.zeros([256],np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k-1]+normgrayHist[k]
    entropy = np.zeros([256],np.float32)

    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k]*np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k-1]
            else:
                entropy[k] = entropy[k-1]-normgrayHist[k]*np.log10(normgrayHist[k])                

    ft = np.zeros([256],np.float32)
    ft1,ft2 = 0.,0.
    totalEntropy = entropy[255]
    for k in range(255):

        maxfornt = np.max(normgrayHist[:k+1])
        maxback = np.max(normgrayHist[k+1:256])
        if (maxfornt==0 or zeroCumuMoment[k]==0 or maxfornt==1 or zeroCumuMoment[k]==1 or totalEntropy==0):
            ft1 = 0
        else:
            ft1 = entropy[k]/totalEntropy*(np.log10(zeroCumuMoment[k])/np.log10(maxfornt))
        if(maxback==0 or 1-zeroCumuMoment[k]==0 or maxback==1 or 1-zeroCumuMoment[k]==1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1-zeroCumuMoment[k])/np.log10(maxback))
            else:
                ft2 = (1-entropy[k]/totalEntropy)*(np.log10(1-zeroCumuMoment[k])/np.log10(maxback))
        ft[k] = ft1+ft2

    thresloc = np.where(ft==np.max(ft))
    thresh = thresloc[0][0]

    threshold = np.copy(image)
    threshold[threshold>thresh] = 255
    threshold[threshold<=thresh] = 0

    return threshold, thresh, max(ft), entropy




def Area_outline(image, height, width):

    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #edged = cv2.Canny(image,1,2)

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    cnts = imutils.grab_contours(cnts)
    cnt_max = max(cnts, key=cv2.contourArea)
    
    mask_zero = np.zeros((height, width))
    mask = cv2.fillPoly( mask_zero,[cnt_max],255)
    _,c = np.where(mask==255)
    area_slice = len(c)

    # mask_outline = cv2.drawContours(image, [cnt_max], -1, 255, 1)
    # for i,c in enumerate(cnts):
    #     cv2.drawContours(image, [c], -1, 255, 1)
    # cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    # cv2.imshow("Image", image)
    # cv2.imshow("Edge", edged)
    # cv2.waitKey(0)
    return c, area_slice 


def process_all(output_num): 
    # Reading and consolidating data for a patch
    image_dataframe = pd.DataFrame()
    for i in range(output_num):
        region_dataframe = data_input(str(i))
        image_dataframe = image_dataframe.append(region_dataframe.T) 
    image_dataframe['plaque_num'] = '0'
    image_dataframe = image_dataframe.reset_index(drop=True)
    
    # Perform inter-slice registration and tracking for individual lesions,
    # and output information on all lesions in the patch
    image_dataframe_new, plaque_all, plaque_proposal_all = plaque_proposal(image_dataframe,output_num)
    plaque_proposal_all_new = upgrade_r(plaque_proposal_all)
    
    # Perform a statistical analysis of the plaques within
    # this patch and update the relevant information
    APnum_per_session = plaque_proposal_all_new.shape[0]

    whole_area = list()
    area_max_num = [0]*APnum_per_session
    r_z_new = [0]*APnum_per_session
    r_new = [0]*APnum_per_session
    plaque_vol = pd.DataFrame()
    for i in range(APnum_per_session):
        tic = time.time()

        x1 = int(plaque_proposal_all_new.iloc[i]['x_min'])
        x2 = int(plaque_proposal_all_new.iloc[i]['x_max'])
        y1 = int(plaque_proposal_all_new.iloc[i]['y_min'])
        y2 = int(plaque_proposal_all_new.iloc[i]['y_max'])
        z1 = int(plaque_proposal_all_new.iloc[i]['z_min'])
        z2 = int(plaque_proposal_all_new.iloc[i]['z_max'])
        plaque_class = int(plaque_proposal_all_new.iloc[i]['class'])
        plaque_scale = int(plaque_proposal_all_new.iloc[i]['scale'])

        area_slice_all = list() 
        area=0  
        for z_index in np.arange(z1, z2+1):
            if not os.path.exists(image_path + prefix + str(z_index).zfill(5) + postfix):
                print("not exist test_" + str(z_index).zfill(5) + ".tif")
                continue
            else:
                #im = Image.open(image_path + prefix + str(z_index).zfill(5) + postfix) 
                #region = im.crop((x1, y1, x2, y2))
                im = cv2.imread(image_path + prefix + str(z_index).zfill(5) + postfix, 0)
                region = im[y1:y2, x1:x2]
                #clrs = region.getcolors()
                #if len(clrs) == 1:
                #    area_slice = 0
                #    area += area_slice
                #    continue
                temp = np.unique(region)
                if len(temp) == 1:
                    area_slice = 0
                    area += area_slice
                    continue
                threshImg,thresh_entroy,max_ft,entropy = threshEntroy(np.array(region))
                dst = (threshImg)/255.0
                height, width = dst.shape

                if (plaque_class==4):
                    _,c = np.where(dst == 1)
                    area_slice = len(c)
                else:
                    c, area_slice = Area_outline(threshImg, height, width)
                area_slice_all.append(area_slice)
                area += area_slice
        plaque_vol[i] = pd.Series({"plaque_vol": area})
        whole_area.append(area)

        if (plaque_scale == 1):
            area_max_num[i] = area_slice_all.index(max(area_slice_all))+plaque_proposal_all_new.iloc[i]['z_min']
            r_z_list = [(area_max_num[i]-plaque_proposal_all_new.iloc[i]['z_min']+1),(plaque_proposal_all_new.iloc[i]['z_max']-area_max_num[i]+1)]
            r_z_new[i] = max(r_z_list)
            if ((max(r_z_list))>plaque_proposal_all_new.iloc[i]['r']):
                r_new[i] = max(r_z_list)
            else:
                r_new[i] = plaque_proposal_all_new.iloc[i]['r']  
        else:
            area_max_num[i] = plaque_proposal_all_new.iloc[i]['z_center']
            r_z_new[i] = plaque_proposal_all_new.iloc[i]['r_z']
            r_new[i] = plaque_proposal_all_new.iloc[i]['r']           
    whole_area = np.array(whole_area)
    average_vol = np.sum(whole_area) / APnum_per_session
    min_vol = np.min(whole_area)
    max_vol = np.max(whole_area)
    plaque_proposal_all_new['r_z'] = r_z_new               
    plaque_proposal_all_new['z_center'] = area_max_num  
    plaque_proposal_all_new['r'] = r_new

    r_all = plaque_proposal_all_new['r'].values.astype(np.float)
    min_index = np.argmin(r_all)
    max_index = np.argmax(r_all)
    mean_r = np.mean(r_all)
    min_x_center = plaque_proposal_all_new['x_center'].values[min_index]
    min_y_center = plaque_proposal_all_new['y_center'].values[min_index]
    min_z_center = plaque_proposal_all_new['z_center'].values[min_index]
    min_r = plaque_proposal_all_new['r'].values[min_index]
    max_x_center = plaque_proposal_all_new['x_center'].values[max_index]
    max_y_center = plaque_proposal_all_new['y_center'].values[max_index]
    max_z_center = plaque_proposal_all_new['z_center'].values[max_index]
    max_r = plaque_proposal_all_new['r'].values[max_index]
    plaque_proposal_all_final=pd.concat([plaque_proposal_all_new, plaque_vol.T], axis=1)

    img_vol = session_vol(image_path)

    ap_session_info = {
        "numbers_apart":APnum_per_session,
        "min_r":min_r,
        "min-volume": min_vol,
        "min_x":min_x_center,
        "min_y":min_y_center,
        "min_z":min_z_center,
        "max_r":max_r,
        "max_volume":max_vol,
        "max_x":max_x_center,
        "max_y":max_y_center,
        "max_z":max_z_center,
        "average_r":mean_r,
        "average_volume":average_vol,
        "volume_all":np.sum(whole_area),
        "plaque load":np.sum(whole_area)/img_vol,
        "plaque density":APnum_per_session/img_vol
        }
    return ap_session_info, plaque_proposal_all_final, image_dataframe_new, plaque_all



if __name__ == "__main__": 
    all_info = pd.DataFrame()
    output_num = len(glob.glob(pathname=swc_dir+'*.swc'))
    ap_session_info, plaque_proposal_all_final, image_dataframe_new, plaque_all = process_all(output_num)
    all_info = pd.Series(ap_session_info)
    with pd.ExcelWriter(excel_path) as writer:
        plaque_proposal_all_final.to_excel(writer, sheet_name='plaque')
        plaque_all.to_excel(writer, sheet_name='plaque_detail')
        image_dataframe_new.to_excel(writer, sheet_name='summary_all')
        all_info.to_excel(writer, sheet_name='analysis')
        

