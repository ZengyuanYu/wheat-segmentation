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
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "/home/yu/Mask_RCNN/model/mask_rcnn_maritime_0065.h5")


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test_images/wheat_every/000000")

class InferenceConfig(maritime.MaritimeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['undetected', 'wheat']
colors = [[0, 0, 0], [0, 255, 0]]

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
h, w = image.shape[:2]
print(w, h)
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
print("r------->", r)
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
print("此张图片检测到了{}麦粒组".format(len(roi_list)))

count = 0
No_count = 0

for i in range(len(r['masks'])):
    for j in range(len(r['masks'][i])):
        for k in range(len(r['masks'][i][j])):
            if r['masks'][i][j][k]== True:
                count += 1
            else:
                No_count += 1
print("掩膜的像素数和非掩膜的像素数", count, No_count)
precent = count/(w*h)
print('掩膜的占比为：{}%'.format(precent*100))
pixel = 200/1411
area = w*h*pixel*pixel*precent
print("实例面积为：{}平方毫米".format(area))

# -------得到每列的值然后判断------
count = 0
No_count = 0
Max_line_list = []
for i in range(w):
    for j in range(h):
        for k in range(len(r['masks'][i][j])):
            if r['masks'][j][i].any() == True:
                count += 1
            else:
                No_count += 1
        Max_line_list.append(count)
    # 每记录一次 count清零一次
    count = 0
# print("Max_list的值为：", len(Max_list))
print("宽度最大为：{}mm".format(max(Max_line_list)*pixel))
new_line_list = [i for i in Max_line_list if i > 0]
print("new_list", len(new_line_list))

# 绘制每列的mask宽度
index = np.arange(0, len(Max_line_list), 1)
plt.figure()
plt.plot(index, Max_line_list)
plt.show()
# --------得到每行的值然后判断----------
count = 0
No_count = 0
Max_row_list = []
for i in range(h):
    for j in range(w):

        if r['masks'][i][j].any()== True:
            count += 1
        else:
            No_count += 1
    Max_row_list.append(count)
    # 每记录一次 count清零一次
    count = 0
# print("Max_list的值为：", len(Max_list))
print("长度最大为：{}mm".format(max(Max_row_list)*pixel))
# 绘制每列的mask宽度
index = np.arange(0, len(Max_row_list), 1)
plt.figure()
plt.plot(index, Max_row_list)
plt.show()
# -------------------
print(r['class_ids'])
# print(r)
print("confidence------->", r['scores'])

# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                         class_names, r['scores'])

path = '/home/yu/Mask_RCNN/test_images/wheat/'
# print("r['masks'].shape[2]---->", r['masks'].shape[2])

def get_mask(path):
    # Override this function to load a mask from your dataset.
    # Otherwise, it returns an empty mask.
    instance_mask = []
    labels_path = os.path.join(path, 'labels')

    # #Get all .png files in the folder
    file_path = os.path.join(labels_path, '*.png')
    file_path = sorted(glob.glob(file_path))
    print("mask路径", file_path)

    for pic in file_path:
        mask = skimage.io.imread(str(pic))
        instance_mask.append(mask)
    mask = np.stack(instance_mask, axis=2)
    return mask


# 得到图片的GT掩膜
original_mask = get_mask(path)
print(original_mask.shape)
print(r['masks'].shape)


def get_dict(local, ground_truth):

    all_pre_mask_area_list = []

    for m in range(local.shape[2]):
        pre_mask_area_list = []
        for k in range(local.shape[2]):
            pre_mask_area = 0
            for i in range(1536):
                for j in range(2048):
                    if local[i][j][m] and local[i][j][m] == ground_truth[i][j][k]:
                        pre_mask_area += 1
            pre_mask_area_list.append(pre_mask_area)

        all_pre_mask_area_list.append(pre_mask_area_list)

    return all_pre_mask_area_list


result_list = get_dict(r['masks'], original_mask)
result_dict = {}
for i in range(len(result_list)):
    for j in range(len(result_list[i])):
        if result_list[i][j] > 0:
            result_dict[j] = result_list[i][j]
print(result_dict)


def compute_ap(local,  ground_truth, result_dict):

    TP = 0
    FP = 0
    FN = 0
    TP_list = []
    FP_list = []
    FN_list = []
    P_list = []
    R_list = []
    F1_list = []
    key_list = []
    for key in result_dict:
        key_list.append(key)
    for k in range(local.shape[2]):
        for i in range(1536):
            for j in range(2048):
                # for k in range(len(local.shape[2])):

                # 真实结果为1且预测的结果为1 True Positive
                if local[i][j][k] == ground_truth[i][j][key_list[k]] and ground_truth[i][j][key_list[k]]:
                    TP += 1
                # 真实结果为0但预测结果为1 False Positive
                if local[i][j][k] and ground_truth[i][j][key_list[k]] != local[i][j][k]:
                    FP += 1
                # 真实结果为1但预测结果为0 False Nagative
                if local[i][j][k] != ground_truth[i][j][key_list[k]] and ground_truth[i][j][key_list[k]]:
                    FN += 1
        TP_list.append(TP)
        TP = 0
        FP_list.append(FP)
        FP = 0
        FN_list.append(FN)
        FN = 0

    # 计算ＡＰ(需要每一个都这样做)
    # 将每一个ＴＰ，ＦＮ，ＦＰ的值存入到列表中
    # 分别计算每个值对应的ap值　求平均
    # 画ＰＲ曲线
    # print("TP-{} FP-{}, FN-{}".format(TP, FP, FN))
    print("TP_list-->", TP_list, "\n", "FP_list-->", FP_list, "\n", "FN_list-->", FN_list)
    for i in range(len(TP_list)):
        P = TP_list[i]/(TP_list[i]+FP_list[i])
        R = TP_list[i]/(TP_list[i]+FN_list[i])
        f1_measure = 2*P*R/(P+R)
        P_list.append(P)
        R_list.append(R)
        F1_list.append(f1_measure)

    return P_list, R_list, F1_list


print(compute_ap(r['masks'], original_mask, result_dict))
percision_list, recall_list, F1_list = compute_ap(r['masks'], original_mask, result_dict)

sum = 0
for i in range(len(percision_list)):
    sum += percision_list[i]
ap = sum / len(percision_list)
print("得到ap值为： ", ap)
# def compute_f1_measure(local, ground_truth):
#     overlap_area=0
#     mask_area = 0
#     pre_mask_area = 0
#     TP = 0
#     FP = 0
#     FN = 0
#     for i in range(1536):
#         for j in range(2048):
#             # local.shape[2] == 图片中预测到的掩膜个数
#             for k in range(local.shape[2]):
#                 if ground_truth[i][j][k]:
#                     mask_area += 1
#                 if local[i][j][k]:
#                     pre_mask_area += 1
#                 # 真实结果为1且预测的结果为1 True Positive
#                 if local[i][j][k] == ground_truth[i][j][k] and ground_truth[i][j][k]:
#                     TP += 1
#                 # 真实结果为0但预测结果为1 False Positive
#                 if local[i][j][k] and ground_truth[i][j][k] != local[i][j][k]:
#                     FP += 1
#                 # 真实结果为1但预测结果为0 False Nagative
#                 if local[i][j][k] != ground_truth[i][j][k] and ground_truth[i][j][k]:
#                     FN += 1
#     print("True Positive（真正阳性）", overlap_area)
#     print("总共区域: ", mask_area)
#     print("TP-{}, FP-{}, FN-{}".format(TP, FP, FN))
#     P = TP/(TP+FP)
#     R = TP/(TP+FN)
#     f1_measure = 2*P*R/(P+R)
#     return P, R, f1_measure


# def compute_f1_measure_test(local, ground_truth):
#     overlap_area=0
#     mask_area = 0
#     pre_mask_area = 0
#     TP = 0
#     FP = 0
#     FN = 0
#     for i in range(h):
#         for j in range(w):
#
#             # local.shape[2] == 图片中预测到的掩膜个数
#             for k in range(local.shape[2]):
#                 # 得到mask的真实区域
#                 if ground_truth[i][j][k] == 1:
#                     mask_area += 1
#                 # 得到mask的预测区域
#                 if local[i][j][k] == 1:
#                     pre_mask_area += 1
#                 # 真实结果为1且预测的结果为1 True Positive
#                 if ground_truth[i][j][k] == 1 and local[i][j][k] == 1:
#                     TP += 1
#                 # 真实结果为0但预测结果为1 False Positive
#                 if ground_truth[i][j][k] == 0 and local[i][j][k] == 1:
#                     FP += 1
#                 # 真实结果为1但预测结果为0 False Nagative
#                 if ground_truth[i][j][k] == 1 and local[i][j][k] == 0:
#                     FN += 1
#     # print("True Positive（真正阳性）", overlap_area)
#     print("实际mask总共区域: ", mask_area, "预测mask的总共区域:", pre_mask_area)
#     print("TP-{}, FP-{}, FN-{}".format(TP, FP, FN))
#     P = TP/(TP+FP)
#     R = TP/(TP+FN)
#     f1_measure = 2*P*R/(P+R)
#     return P, R, f1_measure


# persicion, recall, F1 = compute_f1_measure(r['masks'], original_mask)
# print("查全率-->{}\n查准率-->{}\nF1-->{}".format(persicion, recall, F1))


# image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#         modellib.load_image_gt(dataset_val, inference_config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
#     # Run object detection
#     results = model.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps =\
#         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)