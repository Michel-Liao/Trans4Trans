import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from segmentron.data.dataloader.transobj import TransObjSegmentation

dataset = TransObjSegmentation(mode="train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# dict = next(iter(dataloader))
img, mask, name = next(iter(dataloader))

img_squeezed = img.squeeze(0).numpy()
mask_squeezed = mask.squeeze(0).numpy()

# Real image using cv2
real_image = cv2.cvtColor(
    cv2.imread("datasets/transobj/8_285/frame0/image_0.png"), cv2.COLOR_BGR2RGB
)


# Real image using PIL
# real_image = np.asarray(Image.open("datasets/transobj/8_285/frame0/image_0.png"))
# plt.imshow(real_image)
# plt.show()

# print(f"Min is {np.min(real_image)}") # 0
# print(f"Max is {np.max(real_image)}") # 255


fig, ax = plt.subplots(1, 3, figsize=(25, 10))
fig.suptitle(name[0])
ax[0].imshow(img_squeezed)
ax[1].imshow(mask_squeezed)
ax[2].imshow(real_image)

filename = "test.png"
plt.savefig(filename)
print(f"Image saved as {filename}")
# plt.show()
