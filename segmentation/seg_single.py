'''
Segment the foreground signal from the raw data blocks of the individual plaques that have been cropped out.
'''

import os
import cv2
import SimpleITK as sitk
from skimage import measure, color, data, filters, morphology, io
import tifffile
import imutils
import numpy as np

def volume_single(bbox_arr):
    volume = 0
    if len(bbox_arr.shape) == 2:
        l, h = bbox_arr.shape
        for i in range(l):
            for j in range(h):
                if bbox_arr[i, j] > 0:
                    volume += 1
    else:
        l, h, w = bbox_arr.shape
        for i in range(l):
            for j in range(h):
                for k in range(w):
                    if bbox_arr[i, j, k] > 0:
                        volume += 1
    return volume


def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5):
    """
    remove small object
    :param sitk_maskimg:input binary image
    :param rate:size rate
    :return:binary image
    """
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(sitk_maskimg.GetDirection())
    outmask_sitk.SetSpacing(sitk_maskimg.GetSpacing())
    outmask_sitk.SetOrigin(sitk_maskimg.GetOrigin())
    return outmask_sitk



def GetConnectedCompont(remove):
    cc = sitk.ConnectedComponent(remove)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, remove)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 255
    outmask[labelmaskimage != maxlabel] = 0
    outmasksitk = sitk.GetImageFromArray(outmask)
    outmasksitk.SetSpacing(remove.GetSpacing())
    outmasksitk.SetOrigin(remove.GetOrigin())
    outmasksitk.SetDirection(remove.GetDirection())

    # afteropen = sitk.BinaryMorphologicalOpening(outmasksitk!=0,[1,1,1])
    compont = sitk.BinaryFillhole(outmasksitk)
    return compont


def Area_outline(image, height, width):
    # blurred = cv2.GaussianBlur(image,(3, 3),1,2)
    # edged = cv2.Canny(blurred ,1,2)

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.RETR_LIST：Indicates that all contours are detected;
    # the detected contours do not form a hierarchical relationship, and are stored in a linked list
    # cv2.CHAIN_APPROX_NONE：Outputs the outline using the Freeman chain code;
    # all other methods output a polygon (a sequence of vertices).
    # Store all contour points where the difference in pixel positions between any two adjacent points does not exceed 1,
    # i.e. max(abs(x1-x2), abs(y2-y1)) == 1
    cnts = imutils.grab_contours(cnts)  # Return the contours in cnts
    if len(cnts) == 0:
        cnt_max = []
        mask = []
    else:
        cnt_max = max(cnts, key=cv2.contourArea)  # Return the outline with the largest area
        mask_zero = np.zeros((height, width))
        mask = cv2.fillPoly(mask_zero, [cnt_max], 255)
        # _,c = np.where(mask==255)
    # area_slice = len(c)

    # mask_outline = cv2.drawContours(image, [cnt_max], -1, 255, 1)
    # for i,c in enumerate(cnts):
    #     cv2.drawContours(image, [c], -1, 255, 1)
    # cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    # cv2.imshow("Image", image)
    # cv2.imshow("Edge", edged)
    # cv2.waitKey(0)
    return cnt_max, mask


def Outline_3D(compont_arr):
    im_outline = compont_arr
    i = 0
    for im in compont_arr:
        h, w = im.shape
        cnt_max, mask = Area_outline(im, h, w)
        if len(mask) == 0:
            mask = np.zeros((h, w))
        im_outline[i] = mask
        i = i + 1
    return im_outline


def Fuzzy(im_outline):
    mask_arr = sitk.GetImageFromArray(im_outline)
    sitk_src_gaus = sitk.DiscreteGaussianImageFilter()
    sitk_src_gaus.SetVariance(3)
    sitk_src_gaus.SetMaximumError(0.2)
    sitk_src_gaus = sitk_src_gaus.Execute(mask_arr)
    return sitk_src_gaus


def calGrayHist_3D_16(image, l, h, w):
    # For 16-bit images
    lows, rows, cols = l, h, w
    grayHist = np.zeros([65536])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
                # gray = int(image[l,r,c])
                # if gray > 255:
                #     gray = 255
                # if gray < 0:
                #     gray = 0
                # grayHist[gray] += 1
    return grayHist

def safe_log10(x, epsilon=1e-10):
    x_safe = np.where(x > 0, x, epsilon)
    return np.log10(x_safe)


def threshEntroy_3D_16(image, l, h, w):
    # 16-bit images, maximum entropy 3D segmentation.
    lows, rows, cols = l, h, w
    grayHist = calGrayHist_3D_16(image, l, h, w)
    normgrayHist = grayHist / float(lows * rows * cols)
    zeroCumuMoment = np.zeros([65536], np.float32)
    for k in range(65536):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([65536], np.float32)

    for k in range(65536):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])

    ft = np.zeros([65536], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[65535]
    for k in range(65535):

        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:65536])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (safe_log10(1 - zeroCumuMoment[k]) / safe_log10(maxback))
                # ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2

    thresloc = np.where(ft == np.max(ft))
    if thresloc[0].size == 0:
        threshold = np.copy(image)
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]

        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0

    # return threshold, thresh, max(ft), entropy
    return threshold.astype(np.uint8)



def calGrayHist_3D(image, l, h, w):
    lows, rows, cols = l, h, w
    grayHist = np.zeros([256])  # 8bit 256   16bit  65536
    for l in range(lows):
        for r in range(rows):
            for c in range(cols):
                grayHist[int(image[l, r, c])] += 1
                # gray = int(image[l,r,c])
                # if gray > 255:
                #     gray = 255
                # if gray < 0:
                #     gray = 0
                # grayHist[gray] += 1
    return grayHist



def threshEntroy_3D(image, l, h, w):
    # 8-bit
    lows, rows, cols = l, h, w
    grayHist = calGrayHist_3D(image, l, h, w)
    normgrayHist = grayHist / float(lows * rows * cols)
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])
    ft = np.zeros([256], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[255]
    for k in range(255):
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:65536])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    thresloc = np.where(ft == np.max(ft))
    if thresloc[0].size == 0:
        threshold = np.copy(image)
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
    # return threshold, thresh, max(ft), entropy
    return threshold


def calGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256])  # 8bit 256
    for r in range(rows):
        for c in range(cols):
            grayHist[int(image[r, c])] += 1
    return grayHist


def threshEntroy(image):
    # 2D version
    rows, cols = image.shape

    grayHist = calGrayHist(image)
    normgrayHist = grayHist / float(rows * cols)
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = normgrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k - 1] + normgrayHist[k]
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            if normgrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = -normgrayHist[k] * np.log10(normgrayHist[k])
        else:
            if normgrayHist[k] == 0:
                entropy[k] = entropy[k - 1]
            else:
                entropy[k] = entropy[k - 1] - normgrayHist[k] * np.log10(normgrayHist[k])
    ft = np.zeros([256], np.float32)
    ft1, ft2 = 0., 0.
    totalEntropy = entropy[255]
    for k in range(255):
        maxfornt = np.max(normgrayHist[:k + 1])
        maxback = np.max(normgrayHist[k + 1:256])
        if (maxfornt == 0 or zeroCumuMoment[k] == 0 or maxfornt == 1 or zeroCumuMoment[k] == 1 or totalEntropy == 0):
            ft1 = 0
        else:
            ft1 = entropy[k] / totalEntropy * (np.log10(zeroCumuMoment[k]) / np.log10(maxfornt))
        if (maxback == 0 or 1 - zeroCumuMoment[k] == 0 or maxback == 1 or 1 - zeroCumuMoment[k] == 1):
            ft2 = 0
        else:
            if totalEntropy == 0:
                ft2 = (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
            else:
                ft2 = (1 - entropy[k] / totalEntropy) * (np.log10(1 - zeroCumuMoment[k]) / np.log10(maxback))
        ft[k] = ft1 + ft2
    thresloc = np.where(ft == np.max(ft))
    if thresloc[0].size == 0:
        threshold = np.copy(image)
        threshold[threshold > 10] = 255
    else:
        thresh = thresloc[0][0]
        # 阈值处理
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0

    # return threshold, thresh, max(ft), entropy
    return threshold


def out_line(image, file_name, output_dir):
    h,w = image.shape
    image = np.array(image,dtype='uint8')  # uint8
    image_thresh = threshEntroy(image)              # Maximum entropy method
    # thresh = filters.threshold_otsu(image)   # otsu
    # image_thresh = (image>=thresh)*255.0       # otsu
    # image_thresh = (image < thresh) * 255.0   # otsu  inverted
    image_thresh = np.array(image_thresh, dtype='uint8')
    thresh_file = os.path.join(output_dir, file_name + "_thresh.png")
    cv2.imwrite(thresh_file, image_thresh)
    # image_thresh = np.array(image_thresh,dtype='uint8')
    cnt_max,mask = Area_outline(image_thresh,h,w)
    label_file = os.path.join(output_dir, file_name + "_mask.png")
    mask = np.array(mask)
    if len(mask) == 0:
        volume = 0
    else:
        volume = volume_single(mask)
        cv2.imwrite(label_file, mask)
    return image_thresh, cnt_max, mask, volume


# Segment a single plaque - 2D
def seg_2D(image_arr, out_dir, img_name):
    # image_arr: original image
    # sigma = 1.0  # Gaussian blur intensity
    # smoothed = filters.gaussian(image_arr, sigma=sigma)

    threshImg = threshEntroy(image_arr)
    # thresh = filters.threshold_otsu(image_arr)   # otsu
    # threshImg = (image_arr > thresh).astype(np.uint8) * 255
    threshImg_img = sitk.GetImageFromArray(threshImg)
    sitk.WriteImage(threshImg_img, out_dir + img_name + '_thresh.tif')

    # Morphological operations (applying opening to remove minor noise)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(threshImg, cv2.MORPH_OPEN, kernel, iterations=2)
    return processed


# Segment a single plaque - 3D
def segmentation(bbox_arr):
    l, h, w = bbox_arr.shape
    threshImg_3D = threshEntroy_3D(bbox_arr, l, h, w)  # 阈值提取前景信号threshEntroy_3D，16bit的tdat要改成threshEntroy_3D_16

    # otsu
    # pixels = bbox_arr.reshape(1, -1).astype(np.uint8)
    # global_thresh, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # threshImg_3D = (bbox_arr > global_thresh).astype(np.uint8) * 255

    threshImg_3D_img = sitk.GetImageFromArray(threshImg_3D)

    remove = RemoveSmallConnectedCompont(threshImg_3D_img, rate=0.5)  # Remove small connected components
    compont = GetConnectedCompont(remove)  # Find the largest connected component
    compont_arr = np.array(sitk.GetArrayFromImage(compont), dtype='uint8')
    volume = volume_single(compont_arr)
    im_outline = Outline_3D(compont_arr)  # mask
    gaus = Fuzzy(im_outline)  # Gaussian edge blurring
    gaus_arr = np.array(sitk.GetArrayFromImage(gaus), dtype='uint8')

    return threshImg_3D, compont_arr, im_outline, gaus_arr, volume



if __name__ == "__main__":

    image_dir = "./segmentation/"
    out_dir = "./seg_out/"

    f_list = next(os.walk(image_dir))[2]
    for f_name in f_list:
        f_suffix = f_name.split(".")[-1]
        if f_suffix != "tif":
            continue

        img_name = f_name.split(".")[0]

        img_path = image_dir + f_name

        img_arr = tifffile.imread(img_path)
        dim = img_arr.ndim

        if dim == 3:
            threshImg_3D, compont_arr, im_outline, gaus_arr, volume = segmentation(img_arr)

            # SAVE
            threshImg_3D_path = out_dir + "/" + img_name + "-thresh.tif"
            # Save the thresholded image with the extracted foreground signal
            tifffile.imsave(threshImg_3D_path, threshImg_3D, photometric="minisblack")
            compont_3D_path = out_dir + "/" + img_name + "-compont.tif"
            # Save the connected component analysis diagram
            tifffile.imsave(compont_3D_path, compont_arr, photometric="minisblack")
            mask_3D_path = out_dir + "/" + img_name + "-mask.tif"
            # Save the mask image after filling the holes
            tifffile.imsave(mask_3D_path, im_outline, photometric="minisblack")
            gaus_3D_path = out_dir + "/" + img_name + "-gaus.tif"
            # Save the mask image with blurred edges
            tifffile.imsave(gaus_3D_path, gaus_arr, photometric="minisblack")

        if dim == 2:
            thresh, cnt, mask, volume = out_line(img_arr, img_name, out_dir)

