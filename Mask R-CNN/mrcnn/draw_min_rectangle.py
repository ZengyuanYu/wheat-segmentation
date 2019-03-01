#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
# draw min_rectangle
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd

# read image
img_path = "/home/yu/Desktop/1.jpg"
test_img = cv2.imread(img_path)

# read csv
filename = '/home/yu/Desktop/all_mask_list.csv'

# data = pd.read_csv(filename, header=None)
# print(data)
import csv

with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

# rows存放每一行的mask

#  draw mask
def draw_mask():
    for row in rows[1:2]:
        mask = np.zeros([1536, 2048], dtype=np.uint8)
        # print(x_y_list)
        mask_location = row[1:]
        for i in range(0, len(mask_location)):
            print("flAg", mask_location[i][0], mask_location[i][1])
            mask[int(mask_location[i][0])][int(mask_location[i][1])] = 255
        mask_BINARY = mask.copy() #拷贝掩模


        # 找轮廓
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("countours",contours[0]) #输出轮廓
        cv2.drawContours(test_img, contours[0], -1, (255, 0, 255), 3) #画轮廓

        rect = cv2.minAreaRect(contours[0])
        print("min_rectangle:", rect)  # 最小外接矩形返回值
        x, y = rect[0]  # 中心坐标
        w, h = rect[1]  # 长宽
        angle = rect[2]  # 旋转角度[-90,0)

        box = cv2.boxPoints(rect)  # 矩形的定点
        print("box:",np.int0(box))
        box = np.int0(box)
        cv2.drawContours(test_img, [box], 0, (0, 255, 255), 2)  # red
        # x_min, y_min, w_min, h_min = cv2.boundingRect(box)
        # print("x_min, y_min, w_min, h_min:", x_min, y_min, w_min, h_min)
        return test_img


# # 缩小图像

def run_draw_mask(test_img):
    height, width = test_img.shape[:2]
    size = (int(width*0.5), int(height*0.5))

    test_img = draw_mask()
    show_img = cv2.resize(test_img, size, interpolation=cv2.INTER_AREA)
    cv2.imshow("mask",show_img)
    cv2.imwrite("/home/yu/test5.jpg",test_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    run_draw_mask(test_img)
    # print("a")



