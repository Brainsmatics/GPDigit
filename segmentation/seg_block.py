'''
Perform patch retrieval, cropping and segmentation based on the results
of block-based predictions (.txt/.swc).

Additionally, perform redundant stitching between patches.
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


# Remove small connected components
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


# Find the largest connected component
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


# Extraction of the maximum outer contour
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


# Blurred three-dimensional outline
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
    grayHist = np.zeros([256])  # 8bit 256
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
    # # 2D version
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
        if thresh <= 10:
            thresh = 10
        threshold = np.copy(image)
        threshold[threshold > thresh] = 255
        threshold[threshold <= thresh] = 0
    # return threshold, thresh, max(ft), entropy
    return threshold


def out_line(image):
    h,w = image.shape
    image = np.array(image,dtype='uint8')  #uint8
    image_thresh = threshEntroy(image)
    # thresh = filters.threshold_otsu(image)   #otsu
    # image_thresh = (image>=thresh)*255.0       #otsu
    # image_thresh = (image < thresh) * 255.0   #otsu  inverted
    image_thresh = np.array(image_thresh, dtype='uint8')
    cnt_max,mask = Area_outline(image_thresh,h,w)
    mask = np.array(mask)
    gaus = Fuzzy_2D(mask)
    if len(mask) == 0:
        volume = 0
    else:
        volume = volume_single(mask)
    return image_thresh, cnt_max, mask, gaus, volume


def Fuzzy_2D(im_outline):
    mask_arr = sitk.GetImageFromArray(im_outline)
    sitk_src_gaus = sitk.DiscreteGaussianImageFilter()
    sitk_src_gaus.SetVariance(3)
    sitk_src_gaus.SetMaximumError(0.2)
    sitk_src_gaus = sitk_src_gaus.Execute(mask_arr)
    return sitk.GetArrayFromImage(sitk_src_gaus)


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


def txt_input_2D(swc_dir):
    swc_list = list()       # A list containing all patch plaque information (in centre-point format)
    with open(swc_dir, "r") as f:
        a = f.readlines()
        if a != [' ']:
            # x_min, y_min, x_max, y_max, w, h, img_name, cls
            for line in a:
                swc_list.append([float(l) for l in line.split(" ")[0:7]])
    swc_arr = np.array(swc_list)
    if len(swc_arr)==0:
        return swc_arr
    h,w = swc_arr.shape
    outter_arr = np.zeros((h,w))
    outter_arr[:, 0] = swc_arr[:, 0] - 2 # xmin
    outter_arr[:, 1] = swc_arr[:, 1] - 2 # ymin
    outter_arr[:, 2] = swc_arr[:, 2] + 2 # xmax
    outter_arr[:, 3] = swc_arr[:, 3] + 2 # ymax
    outter_arr[outter_arr < 0] = 0
    outter_arr[outter_arr > 512] = 512
    return outter_arr


# Segment all plaques in the 2D image.
def bbox_cut_2D(outter_arr, img_arr, img_name, output_dir):
    box_num = 0
    threshImg_seg = np.zeros((512, 512)).astype('uint8')  # The predicted image size is 512. Set this according to the actual image size.
    compont_seg = np.zeros((512, 512)).astype('uint8')
    mask_seg = np.zeros((512, 512)).astype('uint8')
    gaus_seg = np.zeros((512, 512)).astype('uint8')

    box_path = output_dir + img_name
    # if img_name in os.listdir(output_dir):
    #     return
    if img_name not in os.listdir(output_dir):
        os.mkdir(box_path)

    for bbox in outter_arr:
        box_num += 1
        xmin, ymin, xmax, ymax = bbox[0:4].astype("uint32")
        box_cut = img_arr[ymin:ymax, xmin:xmax]
        box_cut = box_cut.astype("uint8")
        if ((ymax - ymin) == 0) or ((xmax - xmin) == 0):
            continue
        else:
            thresh, cnt, mask, gaus, volume = out_line(box_cut)
            # if len(list(zip(*np.where(mask < 255)))) == 0:
            #     out = np.copy(box_cut)
            #     out[:, :, :] = 0
            #     thresh = out
            #     cnt = out
            #     mask = out
            #     gaus = out

            # Whether to save each bbox result should be determined on a case-by-case basis
            bbox_path = box_path + "/" + str(box_num) + ".tif"
            tifffile.imsave(bbox_path, box_cut, dtype=np.int8, photometric="minisblack")

            threshImg_3D_path = box_path + "/" + str(box_num) + "-thresh.tif"
            # Save the thresholded image with the extracted foreground signal
            tifffile.imsave(threshImg_3D_path, thresh, dtype=np.int8, photometric="minisblack")
            compont_3D_path = box_path + "/" + str(box_num) + "-compont.tif"
            # Save the connected component analysis diagram
            tifffile.imsave(compont_3D_path, cnt, dtype=np.int8, photometric="minisblack")
            mask_3D_path = box_path + "/" + str(box_num) + "-mask.tif"
            # Save the mask image after filling the holes
            tifffile.imsave(mask_3D_path, mask, dtype=np.int8, photometric="minisblack")
            gaus_3D_path = box_path + "/" + str(box_num) + "-gaus.tif"
            # Save the mask image with blurred edges
            tifffile.imsave(gaus_3D_path, gaus, dtype=np.int8, photometric="minisblack")

            threshImg_seg[ymin:ymax, xmin:xmax] = thresh
            compont_seg[ymin:ymax, xmin:xmax] = cnt
            mask_seg[ymin:ymax, xmin:xmax] = mask
            gaus_seg[ymin:ymax, xmin:xmax] = gaus

    return threshImg_seg, compont_seg, mask_seg, gaus_seg


def segmentation(bbox_arr):
    l, h, w = bbox_arr.shape
    threshImg_3D = threshEntroy_3D(bbox_arr, l, h, w)  # 16bit:threshEntroy_3D_16
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


# Joining the results of multiple block splits
def connect_3D(suffix, output_dir):
    import re
    regex = re.compile(r'\d+')
    file_names = next(os.walk(output_dir))[2]
    file_names.sort(key=lambda x: (x.split('.')[0]))
    # When stitching together 512-voxel blocks that have been cropped into 128-voxel blocks with redundancy,
    # if the original image prior to cropping is not a 512-voxel block,
    # the array size must be set to a value slightly larger than the original image size.

    img_seg = np.zeros((700, 700, 700)).astype('uint8')
    for file_name in file_names:
        if suffix not in file_name:
            continue
        img_path = os.path.join(output_dir, file_name)
        img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img)

        n = int(max(regex.findall(file_name)))
        overlap = 28   # Set according to the actual redundancy value
        # The numbering starts from 1, so subtract 1 first before calculating the offset
        x_offset = int((n - 1) % 5)
        # This involves concatenating data with a 512-voxel block size and a 128-voxel redundancy block.
        y_offset = int(((n - 1) % 25) / 5)
        z_offset = int((n - 1) / 25)
        xmin = int(x_offset) * (128 - overlap)
        ymin = int(y_offset) * (128 - overlap)
        zmin = int(z_offset) * (128 - overlap)
        img_seg[zmin:(zmin + 128), ymin:(ymin + 128), xmin:(xmin + 128)] += img_arr
    img_seg = img_seg[:512, :512, :512]
    return img_seg



def txt_input_3D(swc_dir):
    swc_list = list()
    with open(swc_dir, "r") as f:
        a = f.readlines()
        if a != [' ']:
            # cls, x_c, y_c, z_c, w, h, l
            for line in a:
                swc_list.append([float(l) for l in line.split(" ")[0:7]])
    swc_arr = np.array(swc_list)
    if len(swc_arr)==0:
        return swc_arr
    h,w = swc_arr.shape
    outter_arr = np.zeros((h,w))
    outter_arr[:, 0] = swc_arr[:, 0]-swc_arr[:, 3]/2-2    # xmin
    outter_arr[:, 1] = swc_arr[:, 1]-swc_arr[:, 4]/2-2    # ymin
    outter_arr[:, 2] = swc_arr[:, 2]-swc_arr[:, 5]/2-2    # zmin
    outter_arr[:, 3] = swc_arr[:, 0]+swc_arr[:, 3]/2+2    # xmax
    outter_arr[:, 4] = swc_arr[:, 1]+swc_arr[:, 4]/2+2    # ymax
    outter_arr[:, 5] = swc_arr[:, 2]+swc_arr[:, 5]/2+2    # zmax
    outter_arr[outter_arr < 0] = 0
    outter_arr[outter_arr > 128] = 128
    return outter_arr



def bbox_cut_3D(outter_arr, img_arr, img_name, output_dir):
    box_num = 0
    threshImg_seg = np.zeros((128,128,128)).astype('uint8')   # Set according to the actual image size
    compont_seg = np.zeros((128,128,128)).astype('uint8')
    mask_seg = np.zeros((128,128,128)).astype('uint8')
    gaus_seg = np.zeros((128,128,128)).astype('uint8')

    box_path = output_dir + img_name
    # if img_name in os.listdir(output_dir):
    #     return
    if img_name not in os.listdir(output_dir):
        os.mkdir(box_path)

    for bbox in outter_arr:
        box_num += 1
        xmin, ymin, zmin, xmax, ymax, zmax = bbox[0:6].astype("uint32")
        box_cut = img_arr[zmin:zmax, ymin:ymax, xmin:xmax]
        box_cut = box_cut.astype("uint8")
        if ((zmax - zmin) == 0) or ((ymax - ymin) == 0) or ((xmax - xmin) == 0):
            continue
        else:
            threshImg_3D, compont_arr, im_outline, gaus_arr, volume = segmentation(box_cut)
            # if len(list(zip(*np.where(compont_arr < 255)))) == 0:
            #     out = np.copy(box_cut)
            #     out[:, :, :] = 0
            #     threshImg_3D = out
            #     compont_arr = out
            #     im_outline = out
            #     gaus_arr = out

            # SAVE
            bbox_path = box_path + "/" + str(box_num) + ".tif"
            tifffile.imsave(bbox_path, box_cut, photometric="minisblack")

            threshImg_3D_path = box_path+"/"+str(box_num)+"-thresh.tif"
            # Save the thresholded image with the extracted foreground signal
            tifffile.imsave(threshImg_3D_path, threshImg_3D, photometric="minisblack")
            compont_3D_path = box_path + "/" + str(box_num) + "-compont.tif"
            # Save the connected component analysis diagram
            tifffile.imsave(compont_3D_path, compont_arr, photometric="minisblack")
            mask_3D_path = box_path + "/" + str(box_num) + "-mask.tif"
            # Save the mask image after filling the holes
            tifffile.imsave(mask_3D_path, im_outline, photometric="minisblack")
            gaus_3D_path = box_path + "/" + str(box_num) + "-gaus.tif"
            # Save the mask image with blurred edges
            tifffile.imsave(gaus_3D_path, gaus_arr, photometric="minisblack")


            threshImg_seg[zmin:zmax, ymin:ymax, xmin:xmax] = threshImg_3D
            compont_seg[zmin:zmax, ymin:ymax, xmin:xmax] = compont_arr
            mask_seg[zmin:zmax, ymin:ymax, xmin:xmax] = im_outline
            gaus_seg[zmin:zmax, ymin:ymax, xmin:xmax] = gaus_arr

    return threshImg_seg, compont_seg, mask_seg, gaus_seg



if __name__ == "__main__":

    image_dir = "./figs/seg_block/"
    label_dir = "./figs/seg_block/"
    out_dir = "./figs/seg_block/seg/"

    f_list = next(os.walk(image_dir))[2]
    for f_name in f_list:
        f_suffix = f_name.split(".")[-1]
        if f_suffix != "tif":
            continue

        img_name = f_name.split(".")[0]

        img_path = image_dir + f_name
        txt_path = label_dir + img_name + ".txt"

        img_arr = tifffile.imread(img_path)
        dim = img_arr.ndim

        if dim == 3:
            txt_arr = txt_input_3D(txt_path)

            threshImg_seg, compont_seg, mask_seg, gaus_seg = bbox_cut_3D(txt_arr, img_arr, img_name, out_dir)

            # Complete fig data (128-voxel) saved.
            threshImg_3D_path = out_dir + "/" + img_name + "-thresh.tif"
            tifffile.imsave(threshImg_3D_path, threshImg_seg, photometric="minisblack")
            compont_3D_path = out_dir + "/" + img_name + "-compont.tif"
            tifffile.imsave(compont_3D_path, compont_seg, photometric="minisblack")
            mask_3D_path = out_dir + "/" + img_name + "-mask.tif"
            tifffile.imsave(mask_3D_path, mask_seg, photometric="minisblack")
            gaus_3D_path = out_dir + "/" + img_name + "-gaus.tif"
            tifffile.imsave(gaus_3D_path, gaus_seg, photometric="minisblack")

            # Concatenate the results of all blocks containing the offset, performing the operation as required
            # for suffix in suffixes:
            #     img_seg = connect_3D(suffix)
            #     connect_path = output_dir+"connect_"+suffix
            #     tifffile.imsave(connect_path, img_seg)

        if dim == 2:
            txt_arr = txt_input_2D(txt_path)
            threshImg_seg, compont_seg, mask_seg, gaus_seg = bbox_cut_2D(txt_arr, img_arr, img_name, out_dir)

            # Complete fig data (512-pixel) saved.
            threshImg_3D_path = out_dir + "/" + img_name + "-thresh.tif"
            tifffile.imsave(threshImg_3D_path, threshImg_seg, photometric="minisblack")
            compont_3D_path = out_dir + "/" + img_name + "-compont.tif"
            tifffile.imsave(compont_3D_path, compont_seg, photometric="minisblack")
            mask_3D_path = out_dir + "/" + img_name + "-mask.tif"
            tifffile.imsave(mask_3D_path, mask_seg, photometric="minisblack")
            gaus_3D_path = out_dir + "/" + img_name + "-gaus.tif"
            tifffile.imsave(gaus_3D_path, gaus_seg, photometric="minisblack")





