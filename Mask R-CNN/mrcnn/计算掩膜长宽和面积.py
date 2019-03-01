#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append('/home/yu/Mask_RCNN/mrcnn')  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import Maritime config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#from pycocotools.coco import COCO
import maritime

from matplotlib.pyplot import imshow


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/yu/Mask_RCNN/model/resnet101-110-every.h5")


# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "test_images/wheat/000002")
IMAGE_DIR = '/home/yu/Mask_RCNN/test_images/wheat'
# path = '/home/yu/Mask_RCNN/test_images/wheat/000002'
class InferenceConfig(maritime.MaritimeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # PRN_NMS_THRESHOLD = 0.5
config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['undetected', 'wheat']
colors = [[0, 0, 0], [0, 255, 0]]


def list2csv(list, save_path=None, columns=None):
    data = pd.DataFrame(columns=columns, data=list)
    data.to_csv(save_path)


def get_mask(path):
    # Override this function to load a mask from your dataset.
    # Otherwise, it returns an empty mask.

    labels_path = os.path.join(path, 'labels')
    print(labels_path)
    # #Get all .png files in the folder
    file_path = os.path.join(labels_path, '*.png')
    file_path = sorted(glob.glob(file_path))
    print(file_path)
    for pic in file_path:
        mask = skimage.io.imread(str(pic))
    return mask


def compute_mAP(local,ground_truth):
    overlap_area=0
    mask_area=0
    FP=0
    FN=0
    for i in range(1536):
        for j in range(2048):
            if ground_truth[i][j]:
                mask_area+=1


            if local[i][j] == ground_truth[i][j] and ground_truth[i][j] :
                overlap_area+=1
            if local[i][j] and ground_truth[i][j] != local[i][j]:
                FP+=1
            if local[i][j] != ground_truth[i][j] and ground_truth[i][j]:
                FN+=1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP=overlap_area
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    f1_measure=2*P*R/(P+R)
    return P, R, f1_measure


def main():
    # 对于每张图片的预测结果放入到指定列表中
    information = []
    all_instance_true_list = []
    for id in range(0, 1):
        # 定义一个列表，将ROI, W, H, area, percent, confidence存到此中
        mask_information = []
        # 从测试图片中依次进行预测
        if id < 10:
            image_folder_path = os.path.join(IMAGE_DIR, '00000{}'.format(id))
        else:
            image_folder_path = os.path.join(IMAGE_DIR, '0000{}'.format(id))

        original_mask = get_mask(image_folder_path)
        file_names = next(os.walk(image_folder_path))[2]
        print(file_names)
        image_path = os.path.join(image_folder_path, random.choice(file_names))
        print(image_path)
        image = skimage.io.imread(image_path)

        print("file_names---->", file_names, "image_path---->", image_path)
        h, w = image.shape[:2]
        print(w, h)
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        mask_ndarray = r['masks']
        mask_list = mask_ndarray.tolist()

        # ---------------------
        import pandas as pd

        pic = pd.DataFrame(mask_list)
        h, w = pic.shape[:2]
        print("w和h", w, h)

        roi_ndarray = r['rois']
        roi_list = roi_ndarray.tolist()
        print("y1,x1,y2,x2", roi_list)
        print(len(roi_list))
        # print(roi_list[0])
        # 加入ROI
        # for i in range(len(roi_list[0])):
        #     mask_information.append(roi_list[0][i])
        bbox_width = roi_list[0][3] - roi_list[0][1]
        bbox_heigh = roi_list[0][2] - roi_list[0][0]
        mask_information = roi_list[0].copy()
        mask_information.append(bbox_width)
        mask_information.append(bbox_heigh)
        # val_roi_list.append(roi_list)

        # ------计算mask面积和百分比-----
        count = 0
        No_count = 0
        for i in range(len(r['masks'])):
            for j in range(len(r['masks'][i])):
                if r['masks'][i][j].any() == True:
                    count += 1
                else:
                    No_count += 1
        print("r['masks']的长度为:", i, len(r['masks']))
        precent = count/(w*h)
        print('掩膜的占比为：{}%'.format(precent*100))
        # pixel = 200/1411
        pixel = 1.032031261388592636e-01
        mask_area = w*h*pixel*pixel*precent
        print("面积为：{}平方毫米".format(mask_area))
        # 加入面积占比和面积
        mask_information.append(precent*100)
        mask_information.append(mask_area)

        # -------得到每列的值然后判断------
        count = 0
        No_count = 0
        Max_line_list = []
        for i in range(w):
            for j in range(h):
                if r['masks'][j][i].any() == True:
                    count += 1
                else:
                    No_count += 1
            # 每训练一次
            Max_line_list.append(count)
            # 每记录一次 count清零一次
            count = 0
        mask_width = max(Max_line_list)*pixel

        # # 绘制每列的mask宽度
        # index = np.arange(0, len(Max_line_list), 1)
        # plt.figure()
        # plt.plot(index, Max_line_list)
        # plt.show()
        # --------得到每行的值然后判断----------
        count = 0
        No_count = 0
        Max_row_list = []
        for i in range(h):
            for j in range(w):
                if r['masks'][i][j].any() == True:
                    count += 1
                else:
                    No_count += 1
            Max_row_list.append(count)
            # 每记录一次 count清零一次
            count = 0
        mask_height = max(Max_row_list)*pixel

        # # 绘制每列的mask宽度
        # index = np.arange(0, len(Max_row_list), 1)
        # plt.figure()
        # plt.plot(index, Max_row_list)
        # plt.show()
        # -------------------
        # -----显示预测mask的长和宽
        print("预测得到mask的宽为：{}mm，高为：{}mm".format(mask_width, mask_height))
        # 加入宽、高
        mask_information.append(mask_width)
        mask_information.append(max(Max_line_list))
        mask_information.append(mask_height)
        mask_information.append(max(Max_row_list))
        print(r['class_ids'])
        # print(r)
        print("confidence------->", r['scores'])
        mask_information.append(r['scores'].tolist()[0])
        # # 显示全部信息
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])
        print(r['masks'].shape)
        instance_true = 0
        instance_true_list = []
        for instance_num in range(r['masks'].shape[2]):
            for i in range(r['masks'].shape[1]):
                for j in range(r['masks'].shape[0]):
                    if r['masks'][j][i][instance_num]== True:
                        instance_true += 1
                    else:
                        pass
            instance_true_list.append(instance_true)
            instance_true = 0
        print(instance_true_list)
        all_instance_true_list.append(instance_true_list)
        # visualize.display_instances_yzy(image, r['rois'], r['masks'], r['class_ids'],
                       #             class_names, r['scores'], save_num=id)
    df = pd.DataFrame(all_instance_true_list)
    df.to_csv('result.csv')
    file = open('file_name.txt', 'w')
    file.write(str(all_instance_true_list))
    file.close()


if __name__ == '__main__':
    main()

