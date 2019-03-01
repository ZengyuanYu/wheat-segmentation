import maritime
import matplotlib.pyplot as plt
import numpy as np
import utils
import argparse

maskColor = 1

parser = argparse.ArgumentParser(description='Show image with masks')
parser.add_argument("image",metavar='N',type = int,help="Image idx")
args = parser.parse_args()

dataset_dir = "/home/yu/Mask_RCNN/Train_own_dataset/my_data/"
dataset_type = "showdata"
save_dir = '/home/yu/Mask_RCNN/Result/GT'
# Which image to load
image_idx = args.image
print("Loading image: ", image_idx)
print(image_idx)
# Load config
config = maritime.MaritimeConfig()
# config.display()

# Prepare the dataset loader
dataset_train = maritime.MaritimeDataset()
dataset_train.load_maritime(dataset_dir, dataset_type)
dataset_train.prepare()

# Load data
image = dataset_train.load_image(image_idx)
mask, ids = dataset_train.load_mask(image_idx)

# mask shape (h, w, mask个数)
# Split into separate masks
# mask.shape = [h, w, mask个数]
split_masks = np.split(mask, mask.shape[2], 2)
print("split_masks", len(split_masks))
# Combine masks into a single
all_masks = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
all_masks = all_masks * 255

for mask_img in split_masks:
	mask_img = np.reshape(mask_img, [mask_img.shape[0], mask_img.shape[1]])
	all_masks[mask_img == maskColor] = np.random.randint(50, dtype=np.uint8)

# Show original image
plt.imshow(image)
plt.savefig(save_dir + '/' + "{}_image.jpg".format(image_idx))
# Show masks
plt.figure(2)
# plt.imshow(all_masks, cmap='gray')
plt.imshow(all_masks)
plt.savefig(save_dir + '/' + "{}_mask.jpg".format(image_idx))
# Show image with masks
plt.figure(3)
plt.imshow(image)
plt.imshow(all_masks, alpha=0.3)
plt.savefig(save_dir + '/' + "{}image+mask.jpg".format(image_idx))
plt.show()
