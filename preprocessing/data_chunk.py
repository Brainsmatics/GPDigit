'''
Data blocks with redundancy.
Crop only the image; do not crop the label.
'''

import os
import glob
import numpy as np
import SimpleITK as sitk
import time


base_dir = './pick/'
output_dir = './pick/'


def crop_overlap(img, cut_size, overlap, output_path):
    # start = time.time()
    arr = sitk.GetArrayFromImage(img)
    size = arr.shape
    nl = int(size[0]/(cut_size-overlap))
    nh = int(size[1]/(cut_size-overlap))        
    nw = int(size[2]/(cut_size-overlap))
    
    rl = size[0]%(cut_size-overlap)
    if (rl>overlap):
        nl = nl+1
    rh = size[1]%(cut_size-overlap)             
    if (rh>overlap):
        nh = nh+1
    rw = size[2]%(cut_size-overlap)             
    if (rw>overlap):
        nw = nw+1
    
    num = 1
    for i in range(nl):
        print('i',i)
        for j in range(nh):
            print('j',j)
            for k in range(nw):
                print('k',k)
                
                lowerW = k*(cut_size-overlap)
                lowerH = j*(cut_size-overlap)
                lowerL = i*(cut_size-overlap)
                if k==(nw-1):
                    if (rw>overlap):
                        upperW = lowerW+rw
                    else:
                        upperW = lowerW+(cut_size-overlap)+rw
                else:
                    upperW = lowerW+cut_size
                if j==(nh-1):
                    if (rh>overlap):
                        upperH = lowerH+rh
                    else:
                        upperH = lowerH+(cut_size-overlap)+rh
                else:
                    upperH = lowerH+cut_size                       
                if i==(nl-1):
                    if (rl>overlap):
                        upperL = lowerL+rl
                    else:
                        upperL = lowerL+(cut_size-overlap)+rl
                else:
                    upperL = lowerL+cut_size
                    
                zeros = np.zeros(((cut_size,cut_size,cut_size)))
                croped = arr[lowerL:upperL, lowerH:upperH, lowerW:upperW]
                if croped.shape != zeros.shape:
                    size_c = croped.shape
                    zeros[0:size_c[0], 0:size_c[1], 0:size_c[2]] = croped
                    crop_arr = np.array(zeros,dtype='uint8')
                else:
                    crop_arr = croped
                print(crop_arr)
                croped_img = sitk.GetImageFromArray(crop_arr)
                rescalFilt = sitk.ShiftScaleImageFilter() 
                rescalFilt.SetScale(1.2)
                rescalFilt.SetShift(25)
                itkimage = rescalFilt.Execute(croped_img)
                
                save_path = output_path+"/"+str(num)+".tif"
                sitk.WriteImage(itkimage,save_path)

                num = num+1
    # end = time.time()
    # print(end-start)


if __name__ == "__main__": 
    file_names = next(os.walk(base_dir))[2]
    file_names.sort(key=lambda x:(x.split('.')[0]))
    for file_name in file_names:
        file_suffix = os.path.splitext(file_name)[-1]
        if file_suffix != ".tif":
            continue
        img_dir = os.path.join(base_dir, file_name)
        img = sitk.ReadImage(img_dir)
        f_name = os.path.splitext(file_name)[0]+"test"
        output_path = output_dir+f_name
        if f_name not in os.listdir(output_dir):
            os.mkdir(output_path)
        crop_overlap(img, 128, 28, output_path)

