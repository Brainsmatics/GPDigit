'''
Used for 3D data augmentation; performs synchronised processing on pre-processed 3D block images and labels (.txt)
Generates new augmented images and corresponding .txt labels
'''
import os
import glob
import cv2
from libtiff import TIFF 
import tifffile
import numpy as np
import pandas as pd
import random


base_dir = "./augmentation_test/"

image_3d = base_dir + "9_9_14-3d/"
label_3d = base_dir + "3dlabel_txt/test/"


def tiff_to_read(image_name): 
    tif = TIFF.open(image_name, mode = "r") 
    im_stack = list()
    for im in list(tif.iter_images()): 
        im_stack.append(im)
    return im_stack


def bbox_outcut(a1, a2, shape):  
    if a2<a1:
        a1,a2 = a2,a1
    if a1>shape:
        a1,a2 = shape,shape
    elif (a1<=shape)&(a2>shape):
        a2 = shape  
    return a1, a2


def function_img(image, image_num, choice):
    flipm, flipn, multiply, gaussian, affine = choice
    l,h,w = image.shape
    # mirror
    if flipm == 1:
        flipm_z = image[::-1, ...]
        flipm_x = image[...,::-1]
        flipm_y = image[:,::-1,:]
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipm_z.tif" ), flipm_z)
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipm_x.tif" ), flipm_x)
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipm_y.tif" ), flipm_y)
        
    # Flip
    if flipn == 1:
        flipn_z = image[::-1,::-1,...]
        flipn_x = image[...,::-1,::-1]
        flipn_y = image[::-1,...,::-1]
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipn_z.tif" ), flipn_z)
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipn_x.tif" ), flipn_x)
        tifffile.imsave(os.path.join(output_3d, str(image_num)+"_flipn_y.tif" ), flipn_y)
        
    # Adjust brightness / Add Gaussian noise
    # With an affine transformation, the label of the bounding box will change
    if (multiply ==1) or (gaussian ==1) or (affine == 1):
        if multiply ==1:
            multiply_img = np.zeros((l, h, w))
            multi_para = random.uniform(0.7, 1.3)
        if gaussian ==1:
            gaussian_img = np.zeros((l, h, w))
            noise_sigma = 0.5
            noise = np.random.randn(h, w) * noise_sigma
        if affine ==1:
            affine_img = np.zeros((l, h, w))
            matrix_1 = cv2.getRotationMatrix2D((w/2,h/2),45,0.9)
            matrix_2 = np.float32([[1,0,15],[0,1,15]])
            #pts1=np.float32([[0, 0],[0, h-1],[w-1, 0]])
            #pts2=np.float32([[0, 0],[100, height-100],[width-100, 100]])
            #matrix_3=cv2.getAffineTransform(pts1,pts2)
            
        for i in range(l):
            if gaussian ==1:
                gaussian_img[i] = (image[i] / 255.0) + noise
                gaussian_img[i] = np.clip(gaussian_img[i], 0, 1)
            if affine == 1:
                affine_img[i] = cv2.warpAffine(image[i], matrix_1, (w,h))
                affine_img[i] = cv2.warpAffine(affine_img[i], matrix_2, (w,h))
                #affine_img[i] = cv2.warpAffine(image[i], matrix_3, (w,h))
            for j in range(h):
                for k in range(w):
                    if multiply ==1:
                        multiply_img[i,j,k] = image[i,j,k] * multi_para
                        
        if multiply ==1:
            multiply_img = np.uint8(multiply_img)
            tifffile.imsave(os.path.join(output_3d, str(image_num)+"_multiply.tif" ), multiply_img)
        if gaussian ==1:
            gaussian_img = np.uint8(gaussian_img*255)
            tifffile.imsave(os.path.join(output_3d, str(image_num)+"_gaussian.tif" ), gaussian_img)
        if affine == 1:
            affine_img = np.uint8(affine_img)
            tifffile.imsave(os.path.join(output_3d, str(image_num)+"_affine.tif" ), affine_img)                


def function_bbox(image, bbsOnImg, image_num, choice):
    flipm, flipn, multiply, gaussian, affine = choice
    l,h,w = image.shape
    
    if flipm == 1:
        bbs_flipm_z = []
        bbs_flipm_x = []
        bbs_flipm_y = []
    if flipn == 1:
        bbs_flipn_z = []
        bbs_flipn_x = []
        bbs_flipn_y = [] 
    if affine == 1:
        bbs_affine = []
        
    for bb in bbsOnImg:
        x1,y1,z1,x2,y2,z2,label = bb
        # mirror
        if flipm == 1:
            zz1,zz2 = l-z2,l-z1
            zz1,zz2 = bbox_outcut(zz1,zz2,l)    
            bb_flipm_z = [x1,y1,zz1,x2,y2,zz2,label]
            bbs_flipm_z.append(bb_flipm_z)
            xx1,xx2 = w-x2,w-x1
            xx1,xx2 = bbox_outcut(xx1,xx2,w)
            bb_flipm_x = [xx1,y1,z1,xx2,y2,z2,label]
            bbs_flipm_x.append(bb_flipm_x)
            yy1,yy2 = h-y2,h-y1
            yy1,yy2 = bbox_outcut(yy1,yy2,h)
            bb_flipm_y = [x1,yy1,z1,x2,yy2,z2,label]
            bbs_flipm_y.append(bb_flipm_y)
        # flip
        if flipn == 1:
            zz1,zz2 = l-z2,l-z1
            yy1,yy2 = h-y2,h-y1
            zz1,zz2 = bbox_outcut(zz1,zz2,l)
            yy1,yy2 = bbox_outcut(yy1,yy2,h)
            bb_flipn_z = [x1,yy1,zz1,x2,yy2,zz2,label]
            bbs_flipn_z.append(bb_flipn_z)
            xx1,xx2 = w-x2,w-x1
            yy1,yy2 = h-y2,h-y1
            xx1,xx2 = bbox_outcut(xx1,xx2,w)
            yy1,yy2 = bbox_outcut(yy1,yy2,h)
            bb_flipn_x = [xx1,yy1,z1,xx2,yy2,z2,label]
            bbs_flipn_x.append(bb_flipn_x)
            xx1,xx2 = w-x2,w-x1
            zz1,zz2 = l-z2,l-z1
            xx1,xx2 = bbox_outcut(xx1,xx2,w)
            zz1,zz2 = bbox_outcut(zz1,zz2,l)
            bb_flipn_y = [xx1,y1,zz1,xx2,y2,zz2,label]
            bbs_flipn_y.append(bb_flipn_y)
        # affine
        if affine == 1:
            matrix_1 = cv2.getRotationMatrix2D((w/2,h/2),45,0.9)
            matrix_2 = np.float32([[1,0,15],[0,1,15]])
            #pts1=np.float32([[0, 0],[0, h-1],[w-1, 0]])
            #pts2=np.float32([[0, 0],[100, height-100],[width-100, 100]])
            #matrix_3=cv2.getAffineTransform(pts1,pts2)
            affine_bb = np.zeros((h, w))
            affine_bb[y1:y2,x1:x2] = 1
            affine_bb = cv2.warpAffine(affine_bb, matrix_1, (w,h))
            affine_bb = cv2.warpAffine(affine_bb, matrix_2, (w,h))
            #affine_bb = cv2.warpAffine(affine_bb, matrix_3, (w,h)) 
            index_y,index_x = np.where(affine_bb != 0)
            if len(index_x)!=0:
                xx1 = min(index_x)
                xx2 = max(index_x)
                yy1 = min(index_y)
                yy2 = max(index_y)
                xx1,xx2 = bbox_outcut(xx1,xx2,w)
                yy1,yy2 = bbox_outcut(yy1,yy2,h)
                bb_affine = [xx1,yy1,z1,xx2,yy2,z2,label]
                bbs_affine.append(bb_affine) 
               
    # save label
    if flipm == 1:
        bbs_flipm_z = pd.DataFrame(bbs_flipm_z)
        bbs_flipm_z.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipm_z.txt") , sep=' ',index=False, header=False)
        bbs_flipm_x = pd.DataFrame(bbs_flipm_x)
        bbs_flipm_x.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipm_x.txt") , sep=' ',index=False, header=False)
        bbs_flipm_y = pd.DataFrame(bbs_flipm_y)
        bbs_flipm_y.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipm_y.txt") , sep=' ',index=False, header=False)
    if flipn == 1:
        bbs_flipn_z = pd.DataFrame(bbs_flipn_z)
        bbs_flipn_z.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipn_z.txt") , sep=' ',index=False, header=False)
        bbs_flipn_x = pd.DataFrame(bbs_flipn_x)
        bbs_flipn_x.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipn_x.txt") , sep=' ',index=False, header=False)
        bbs_flipn_y = pd.DataFrame(bbs_flipn_y)
        bbs_flipn_y.to_csv(os.path.join(output_3d_label, str(image_num)+"_flipn_y.txt") , sep=' ',index=False, header=False)
    if (multiply ==1) or (gaussian ==1):
        bbsOnImg = pd.DataFrame(bbsOnImg)
        if multiply ==1:
            bbsOnImg.to_csv(os.path.join(output_3d_label, str(image_num)+"_multiply.txt") , sep=' ',index=False, header=False)
        if gaussian ==1:
            bbsOnImg.to_csv(os.path.join(output_3d_label, str(image_num)+"_gaussian.txt") , sep=' ',index=False, header=False)
    if affine == 1:
        bbs_affine = pd.DataFrame(bbs_affine)
        bbs_affine.to_csv(os.path.join(output_3d_label, str(image_num)+"_affine.txt") , sep=' ',index=False, header=False)
    
    

#show bbox
def drawontif(image_dir, bbs_dir, Thickness, output_3d_show):
    file_names = next(os.walk(bbs_dir))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    for file_name in file_names:
        file_suffix = os.path.splitext(file_name)[-1]
        if file_suffix != ".txt":
            continue
        txt_dir = os.path.join(bbs_dir, file_name)
        img_dir = os.path.join(image_dir, file_name.replace("txt", "tif").strip())
    
        img = tiff_to_read(img_dir)
        image = np.array(img)
        image_out = image
    
        for line in open(txt_dir, "r"):
            line = line.strip()
            info = line.split(" ")
            name = info[6]
            xmin = int(info[0])
            ymin = int(info[1])
            zmin = int(info[2])
            xmax = int(info[3])
            ymax = int(info[4])
            zmax = int(info[5])
            if xmin>xmax:
                xmin,xmax = xmax,xmin
            if ymin>ymax:
                ymin,ymax = ymax,ymin
            if zmin>zmax:
                zmin,zmax = zmax,zmin
            
            for i in range(zmin-1, zmax):
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(image_out[i], (xmin,ymin), (xmax,ymax), (255,0,255), Thickness)
                cv2.putText(image_out[i], name, (xmin,ymin), font, 1, (255,0,255), Thickness)
                    
        image_out = np.uint8(image_out)
        tifffile.imsave(os.path.join(output_3d_show, file_name.replace("txt", "tif").strip()), image_out)
    


def augmentation3d(output_3d, output_3d_label, choice):
    file_names = next(os.walk(label_3d))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    image_num = 0
    flipm, flipn, multiply, gaussian, affine = choice
    for file_name in file_names:
        file_suffix = os.path.splitext(file_name)[-1]
        if file_suffix != ".txt":
            continue
        txt_dir = os.path.join(label_3d, file_name)
        img_dir = os.path.join(image_3d, file_name.replace("txt", "tif").strip())
        image_num = image_num + 1
    
        img = tiff_to_read(img_dir)
        image = np.array(img)
    
        bbsOnImg = []
        for line in open(txt_dir, "r"):
            line = line.strip()
            info = line.split(" ")
            name = info[6]
            xmin = int(info[0])
            ymin = int(info[1])
            zmin = int(info[2])
            xmax = int(info[3])
            ymax = int(info[4])
            zmax = int(info[5])
            if xmin>xmax:
                xmin,xmax = xmax,xmin
            if ymin>ymax:
                ymin,ymax = ymax,ymin
            if zmin>zmax:
                zmin,zmax = zmax,zmin
            bb_3d=[xmin, ymin, zmin, xmax, ymax, zmax, name]
            bbsOnImg.append(bb_3d)
        
        function_img(image, image_num, choice)
        function_bbox(image, bbsOnImg, image_num, choice)




if __name__ == "__main__": 
    # 3D data augmentation
    output_3d = image_3d + 'augmentation/'
    output_3d_label = label_3d + 'augmentation/'
    if 'augmentation' not in os.listdir(image_3d):
        os.mkdir(output_3d)
    if 'augmentation' not in os.listdir(label_3d): 
        os.mkdir(output_3d_label)
    choice = [1, 1, 1, 1, 1]   # flipm, flipn, multiply, gaussian, affine; set 1
    augmentation3d(output_3d, output_3d_label, choice)

    # output_3d_show = output_3d  + 'show/'
    # if 'show' not in os.listdir(output_3d):
    #     os.mkdir(output_3d_show)
    # drawontif(output_3d, output_3d_label, 1, output_3d_show)
    
