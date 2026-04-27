'''
A 3D TIFF file is divided into single-layer 2D images along the axis
'''
import os
import glob
import numpy as np
from scipy import misc
from PIL import Image
from libtiff import TIFF 
import cv2
from scipy import misc

tiff_image_list = './BLA/'

def imadjust(img, low_in, high_in, low_out, high_out, gamma, c):
    '''
    low_in is the minimum pixel grey-scale threshold for the input image;
    low_out is the maximum pixel grey-scale threshold for the input image;
    (you will need to test these values in Fiji to select the most appropriate minimum and maximum values)

    low_out sets the output value for pixels in the input image that are below the minimum pixel grey-scale threshold
    (this is usually left unchanged at 0);
    low_out is the output value assigned to pixels in the input image that exceed the maximum pixel grey-scale threshold
     (typically left unchanged at 255; for 16-bit images, this is 65535);
    gamma is the transformation coefficient; a value of 1 indicates a linear transformation,
    a value greater than 1 indicates a convex transformation, and a value less than 1 indicates a concave transformation;

    c is the constant coefficient of the transformation, typically set to c = high_out / high_in
    '''
    #f = misc.imread(img).astype(np.uint8)
    h, w = img.shape
    img_out = np.zeros([h, w])
    # imadjust
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] <= low_in:
                img_out[y, x] = low_out
            elif img[y, x] >= high_in:
                img_out[y, x] = high_out
            else:
                img_out[y, x] = c * (img[y, x]**gamma)
    img_out = img_out.astype(np.uint8)
    #scipy.misc.imsave('figure.jpg', img_out)
    return img_out



if __name__ == "__main__": 
    file_names = next(os.walk(tiff_image_list))[2]
    for i in range(len(file_names)):
        if os.path.splitext(file_names[i])[1] == '.tif':
            file_name = os.path.splitext(file_names[i])[0]
            dir_name = tiff_image_list + file_name
            if file_name not in os.listdir(tiff_image_list):
                os.mkdir(dir_name)
            tif = TIFF.open(os.path.join(tiff_image_list, file_names[i]), mode = "r")
            j = 0
            for im in list(tif.iter_images()):
                j=j+1
                tiff_image_name = dir_name + '/' + str(j) + '.jpg'
                im_adjust = imadjust(im, 0, 150, 0, 255, 1, 1.7)  
                #gamma=1 LINEAR，c=high_out/high_in
                cv2.imwrite(tiff_image_name,im_adjust)

