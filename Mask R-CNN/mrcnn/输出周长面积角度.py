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
pixel = 1.032031261388592636e-01
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append('/home/yu/Mask_RCNN/mrcnn')  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import Maritime config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from pycocotools.coco import COCO
import maritime

from matplotlib.pyplot import imshow

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/yu/Mask_RCNN/model/resnet101-110-every.h5")

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "test_images/wheat/000002")
IMAGE_DIR = '/home/yu/Mask_RCNN/test_images/training'


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


def compute_mAP(local, ground_truth):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    for i in range(1536):
        for j in range(2048):
            if ground_truth[i][j]:
                mask_area += 1

            if local[i][j] == ground_truth[i][j] and ground_truth[i][j]:
                overlap_area += 1
            if local[i][j] and ground_truth[i][j] != local[i][j]:
                FP += 1
            if local[i][j] != ground_truth[i][j] and ground_truth[i][j]:
                FN += 1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP = overlap_area
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    f1_measure = 2 * P * R / (P + R)
    return P, R, f1_measure

import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd

# read image
# img_path = "/home/yu/Desktop/1.jpg"
# test_img = cv2.imread(img_path)

# read csv
filename = '/home/yu/Desktop/all_mask_list.csv'


def draw_mask(test_img, mask_location_list):
        mask = np.zeros([1536, 2048], dtype=np.uint8)
        for i in range(0, len(mask_location_list)):
            mask[int(mask_location_list[i][0])][int(mask_location_list[i][1])] = 255
        mask_BINARY = mask.copy() #拷贝掩模

        # 找轮廓
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("countours",contours[0])
        # 绘制轮廓
        cv2.drawContours(test_img, contours[0], -1, (255, 0, 255), 3) #画轮廓

        ##########2019-1-29##############
        # 将轮廓周长绘制出来
        file = open("/home/yu/result/zhouchang.txt", 'a')

        ##########2019-1-29##############
        ########## 2019-1-16 ##########
        # 计算轮廓的周长
        cnt = contours[0]
        zhouchang = cv2.arcLength(cnt, True)
        print("计算的轮廓周长---->", zhouchang)
        file.write(str(zhouchang*pixel) + "\n")
        file.close()
        # 计算轮廓面积
        # 计算轮廓所包含的面积
        area = cv2.contourArea(cnt)
        print("计算的轮廓面积---->", area)
        ##########2019-1-16##############


        rect = cv2.minAreaRect(contours[0])
        print("min_rectangle:", rect)  # 最小外接矩形返回值
        x, y = rect[0]  # 中心坐标
        w, h = rect[1]  # 长宽
        angle = rect[2]  # 旋转角度[-90,0)

        #########2019-1-19##############
        file_angle = open('/home/yu/result/angle.txt', 'a')
        file_angle.write(str(angle) + "\n")
        file_angle.close()
        #########2019-1-19#############
        box = cv2.boxPoints(rect)  # 矩形的定点
        print("box:", np.int0(box))
        box = np.int0(box)
        cv2.drawContours(test_img, [box], 0, (0, 255, 255), 2)  # red
        if angle != -0.0:
            # 判断bbox中最右边的值
            location = [box[0][0], box[1][0], box[2][0], box[3][0]]
            location_sorted = sorted(location)
            print(location_sorted)
            location1 = box[location.index(location_sorted[-1])]
            location2 = box[location.index(location_sorted[-2])]
            print(location1, location2)
            pt2 = (int((location1[0] + location2[0])/2), int((location1[1] + location2[1])/2))
            # pt2 = (int((box[2][0] + box[3][0])/2), int((box[2][1] + box[3][1])/2))
            # # 变长一些
            pt2_plus = (pt2[0] + int(pt2[0] - x), pt2[1] + int(pt2[1] - y))
            print(pt2)
            cv2.line(test_img, (int(x), int(y)), pt2_plus, (0, 0, 255), 5)
        else:
            # 判断bbox中最右边的值
            location = [box[0][0]+box[0][1], box[1][0]+box[1][1],
                        box[2][0]+box[2][1], box[3][0]+box[3][0]]
            location_sorted = sorted(location)
            print(location_sorted)
            location1 = box[location.index(location_sorted[-1])]
            location2 = box[location.index(location_sorted[-2])]
            print(location1, location2)
            pt2 = (int((location1[0] + location2[0]) / 2), int((location1[1] + location2[1]) / 2))
            # pt2 = (int((box[2][0] + box[3][0])/2), int((box[2][1] + box[3][1])/2))
            # # 变长一些
            pt2_plus = (pt2[0] + int(pt2[0] - x), pt2[1] + int(pt2[1] - y))
            print(pt2)
            cv2.line(test_img, (int(x), int(y)), pt2_plus, (0, 0, 255), 5)
        return test_img


# # 缩小图像

def run_draw_mask(test_img, mask_location_list, id):
    # height, width = test_img.shape[:2]
    # size = (int(width*0.5), int(height*0.5))

    test_img = draw_mask(test_img, mask_location_list)
    # show_img = cv2.resize(test_img, size, interpolation=cv2.INTER_AREA)
    # cv2.imshow("mask", show_img)
    cv2.imwrite("/home/yu//result/test{}.jpg".format(id), test_img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

def main():
    # 对于每张图片的预测结果放入到指定列表中
    information = []
    all_instance_true_list = []
    for id in range(60, 61, 1):
    # for id in [34]:
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
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'], id=id)
        all_mask_location_list = []
        img_path = "/home/yu/test{}.jpg".format(id)
        print("img_path为:", img_path)
        test_img = cv2.imread(img_path)
        # 每个籽粒的像素计数ssh
        count = 0
        No_count = 0
        # print(r['masks'].shape[2])
        # [2, 4, 6, 7]
        for k in range(r['masks'].shape[2]):
        # for k in [11]:
            mask_location_list = []
            for i in range(len(r['masks'])):
                for j in range(len(r['masks'][i])):
                    if r['masks'][i][j][k] == True:
                        count += 1
                        mask_location_list.append([i, j])
                    else:
                        No_count += 1
            # TODO 在这写绘制Mask的程序
            run_draw_mask(test_img, mask_location_list, id)
            # TODO 在这进行籽粒面积存储
            file_area = open('/home/yu/result/area.txt', 'a')
            file_area.write(str(count*pixel*pixel) + "\n")
            file_area.close()
            print("此籽粒的面积为:--->", count)
            count = 0


if __name__ == '__main__':
    main()
    import pandas as pd

    df1 = pd.read_csv('/home/yu/result/angle.txt')
    df2 = pd.read_csv('/home/yu/result/area.txt')
    df3 = pd.read_csv('/home/yu/result/zhouchang.txt')
    df4 = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    df = pd.merge(df4, df3, left_index=True, right_index=True, how='outer')

    df.to_csv('/home/yu/result/merge.txt')


