import sys
sys.path.append('/home/yu/Mask_RCNN/')

import maritime
import matplotlib.pyplot as plt
import numpy as np
import utils

dataset_dir = "/home/yu/Mask_RCNN/Train_own_dataset/my_data"
dataset_type = "training"

#Prepare the dataset loader
dataset_train = maritime.MaritimeDataset()
dataset_train.load_maritime(dataset_dir, dataset_type)
dataset_train.prepare()

print(dataset_train.image_ids)

cumulative_red = 0
cumulative_green = 0
cumulative_blue = 0

print("Starting calculation of mean pixel values")

for image_idx in dataset_train._image_ids:
	if image_idx%200 == 0:
		print(image_idx)

	print (str(image_idx) + "/" + str(len(dataset_train._image_ids)))
	image = dataset_train.load_image(image_idx)
	#Split into separate masks
	split_colors = np.split(image, image.shape[2], 2)

	red = split_colors[0]
	green = split_colors[1]
	blue = split_colors[2]

	red_mean = np.mean(red)
	green_mean = np.mean(green)
	blue_mean = np.mean(blue)

	cumulative_red = cumulative_red + red_mean
	cumulative_green = cumulative_green + green_mean
	cumulative_blue = cumulative_blue + blue_mean

final_red = cumulative_red/len(dataset_train._image_ids)
final_green = cumulative_green/len(dataset_train._image_ids)
final_blue = cumulative_blue/len(dataset_train._image_ids)

print("Mean pixels red channel", final_red)
print("Mean pixels green channel", final_green)
print("Mean pixels blue channel", final_blue)

