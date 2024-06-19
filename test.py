import cv2
import matplotlib.pyplot as plt

img_idx = cv2.imread('datasets/gt_trainval/gt/fog/train/GOPR0475/GOPR0475_frame_000041_gt_labelIds.png')
img_color = cv2.imread('datasets/gt_trainval/gt/fog/train/GOPR0475/GOPR0475_frame_000041_gt_labelColor.png')

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(img_idx)
ax[1].imshow(img_color)
plt.show()