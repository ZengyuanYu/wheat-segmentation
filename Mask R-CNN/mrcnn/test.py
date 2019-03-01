#! /usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiaoyu
# import skimage.io
# import os
# import numpy as np
# import glob
#
#
# path = '/home/yu/Mask_RCNN/test_images/fish/'
#
# def get_mask(path):
#     # Override this function to load a mask from your dataset.
#     # Otherwise, it returns an empty mask.
#
#     labels_path = os.path.join(path, 'labels')
#     print(labels_path)
#     # #Get all .png files in the folder
#     file_path = os.path.join(labels_path,'*.png')
#     file_path = sorted(glob.glob(file_path))
#     print(file_path)
#     for pic in file_path:
#
#         mask = skimage.io.imread(str(pic))
#     return mask


#

# a = [1, 2, 3, 5]
# b = []
# c = 2
# b = a.copy()
# b.append(c)
# print(b)

# import numpy as np

# x = np.arange(9)

# print(np.split(x, 3, 2))
#
# import numpy as np
#
#
# def calc_ent(x):
#     x_value_list = []
#     for i in range(x[0]):
#         x_value_list.append(x[i+1][0])
#     # print(x_value_list)
#     ent = 0.0
#     count = 0
#     for i, x_value in enumerate(x_value_list):
#         if x[i+1][1] == x_value:
#             count += 1
#         else:
#             pass
#     p = float(count) / len(x)
#     logp = np.log2(p)
#     ent -= p * logp
#
#     return ent
#
#
# a = [5, [1, 1], [1, 1], [2, 0], [0, 0], [3, 0]]
# print(a)
# print(calc_ent(a))

# def calc_ent(x):
#     """
#         calculate shanno ent of x
#     """
#
#     x_value_list = set([x[i] for i in range(x.shape[0])])
#     print(x_value_list)
#     ent = 0.0
#     for x_value in x_value_list:
#         p = float(x[x == x_value].shape[0]) / x.shape[0]
#         logp = np.log2(p)
#         ent -= p * logp
#
#     return ent
#
# a = np.array([1, 1])
# calc_ent(a)
#
# import sys
#
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = map(int, line.split())
#         for v in values:
#             ans += v
#     print (ans)


# from scipy import *
#
#
# def asymmetricKL(P, Q):
#     return sum(P * log(P / Q))  # calculate the kl divergence between P and Q
#
# #
# # def symmetricalKL(P, Q):
# #     return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00
import sys





