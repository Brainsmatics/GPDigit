'''
Feature map visualisation
'''

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import keras
import pandas as pd

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils,config,visualize_old2
import mrcnn.model as modellib

sys.path.append(os.path.join(ROOT_DIR, "samples/Plaques/"))  # To find local version
import Plaques


#MODEL_DIR = os.path.join(ROOT_DIR, "/logs/plaques20221114T1009/")
#MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_plaques_5000.h5")
MODEL_DIR = "E:/study-new/2DMethod-logs/plaques20221114T1009"
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_plaques_5000.h5")
#RESULT_DIR = os.path.join(ROOT_DIR, "logs")
#IMAGE_DIR = os.path.join(ROOT_DIR, "/data/test")
RESULT_DIR = MODEL_DIR
IMAGE_DIR = "E:/study-new/2DMethod-logs/visual-test"
FMTEST_DIR = os.path.join(IMAGE_DIR, "test")             #自建

if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)

class InferenceConfig(Plaques.PlaquesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, prefix, num):
    feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []
    plt1 = plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    print(num_pic)
    for i in range(0, num_pic):
        plt_split = plt.figure()
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt_split.imshow(feature_map_split)
        name1 = '{}_{}_fm.png'.format(prefix, num_pic)
        plt_split.savefig(os.path.join(FMTEST_DIR, name1))

        plt1.subplot(row, col, i + 1)
        plt1.imshow(feature_map_split)
        # plt.pause(1)
        plt1.axis('off')  # 关闭坐标轴

    name2 = '{}_all_{}_fm.png'.format(prefix, num)
    plt1.savefig(os.path.join(FMTEST_DIR, name2))
    plt1.show()
    plt1.pause(1)

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt2 = plt.figure()
    plt2.imshow(feature_map_sum)
    plt2.pause(1)
    name3 = '{}_add_{}_fm.png'.format(prefix, num)
    plt2.savefig(os.path.join(FMTEST_DIR, name3))



if __name__ == "__main__":

    # base_model = VGG19(weights='imagenet', include_top=False)
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(MODEL_PATH, by_name=True)
    # weights
    fw = open(RESULT_DIR + "/weights.txt", 'w')
    visualize.display_weight_stats_python(model,RESULT_DIR,fw)
    fw.close()
    model_test = Model(inputs=model.mold_inputs, outputs=model.build(mode="inference", config=config)[1:])  #把原始model更改为另一个模式的model

    img_name = 'test.jpg'
    img = image.load_img(os.path.join(IMAGE_DIR, img_name))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    rpn_fm, mrcnn_fm = model_test.predict(x)
    print(rpn_fm.shape)
    print(mrcnn_fm.shape)

    Prefix1 = "rpn"
    feature_rpn = rpn_fm.reshape(rpn_fm.shape[1:])
    visualize_feature_map(feature_rpn, Prefix1, 6)

    Prefix2 = "mrcnn"
    feature_mrcnn = mrcnn_fm.reshape(mrcnn_fm.shape[1:])
    visualize_feature_map(feature_mrcnn, Prefix2, 5)
