# Code modified from DVPO

import numpy as np
import torch

# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F

# import csv
import cv2

# import math
# import random
# import json
# import pickle


# import os.path as osp
from pathlib import Path

from .seg_data_base import SegmentationDataset

# ?: Where do they put their augmentations?
# from .augmentation import TransAugmentor

# from .rgbd_utils import *

# Imports from other T4T datasets
import os
import logging
import glob
import random
from PIL import Image


class TransObjSegmentation(SegmentationDataset):
    """Transparent Object Segmentation Dataset."""

    BASE_DIR = "TransObj"
    NUM_CLASS = 1

    def __init__(
        self,
        # name,
        root="datasets/transobj",
        split="test",
        mode=None,
        transform=None,
        **kwargs
    ):
        """
        Arguments:
            root (string): Directory with images and transparency data.
            transform (callable, optional): Optional augmentations to be applied to images.
        """
        # self.root_dir = Path(root_dir)
        # self.name = name

        # self.augmentation = None
        # self.sample = sample

        # self.n_frames = n_frames
        # self.fmin = fmin  # exclude very easy examples
        # self.fmax = fmax  # exclude very hard examples

        # if self.augmentation:
        #     self.augmentation = TransAugmentor()

        # ?: Not sure why make cache directory if it's never used.
        # # building dataset is expensive, cache so only needs to be performed once
        # cur_path = osp.dirname(osp.abspath(__file__))
        # if not os.path.isdir(osp.join(cur_path, "cache")):
        #     os.mkdir(osp.join(cur_path, "cache"))

        # self.scene_info = pickle.load(open("datasets/TartanAir.pickle", "rb"))[0]

        # self._build_dataset_index()

        # NOTE: All new code below
        super(TransObjSegmentation, self).__init__(
            root, split, mode, transform, **kwargs
        )
        # root = os.path.join(self.root, self.BASE_DIR)
        # root = os.path.join(self.root)

        root = Path(root)

        self.npz_paths, self.png_paths = self._load_data_paths(root)

        # assert os.path.exists(
        #     root
        # ), "Please put the data in {SEG_ROOT}/datasets/transobj"
        # self.images, self.masks = self._load_data_paths(self)
        # assert len(self.images) == len(self.masks)
        # if len(self.images) == 0:
        #     raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        logging.info(
            "Found {} images in the folder {}".format(len(self.png_paths), root)
        )

    def _load_data_paths(self, root):
        # ?: Better to use numpy arrays?
        npz_paths = []
        png_paths = []

        for subdir in root.iterdir():
            if subdir.is_dir():
                for frame_dir in subdir.iterdir():
                    sorted_npz_paths = sorted(frame_dir.glob("*.npz"))
                    sorted_png_paths = sorted(frame_dir.glob("*.png"))
                    if sorted_npz_paths and sorted_png_paths:
                        npz_paths.extend(sorted_npz_paths)
                        png_paths.extend(sorted_png_paths)

        return npz_paths, png_paths

    # ?: What does this do?
    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype("int32"))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BILINEAR)
        mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)

    # def _build_dataset_index(self):
    #     self.dataset_index = []
    #     for scene in self.scene_info:
    #         if not self.__class__.is_test_scene(scene):
    #             graph = self.scene_info[scene]["graph"]
    #             for i in graph:
    #                 if i < len(graph) - 65:
    #                     self.dataset_index.append((scene, i))
    #         else:
    #             print("Reserving {} for validation".format(scene))

    # @staticmethod
    # def image_read(image_path):
    #     return cv2.cvtColor(cv2.imread(image_path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def npz_read(npz_path, threshold=0.5):
        """Returns transparency array from npz file.

        Args:
            npz_path (str): Path to npz file.
            threshold (float, optional): Defaults to 0.5.

        Returns:
            ndarray: Binary mask of transparent pixels. [H, W]
        """
        # transparency_arr = np.load(npz_path)["denoising_mask_map"]

        # 1 to access the second layer
        transparency_arr = np.load(npz_path)["denoising_mask_map"][1]  # [H, W]
        threshold_arr = (transparency_arr > threshold).astype(np.uint8)

        # return threshold_arr.transpose(1, 2, 0)
        return threshold_arr

    def __getitem__(self, index):
        """Returns an image and its corresponding transparency mask.

        Arguments:
            index (int): Index of the image to be returned.
        """
        npz_path = self.npz_paths[index]
        png_path = self.png_paths[index]

        # img = torch.from_numpy(self.image_read(png_path))  # [H, W, C]
        # img = self.image_read(png_path)  # [H, W, C]
        img = Image.open(png_path).convert("RGB")

        # if this is a test image, return the image and its name

        if self.mode == "test":
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.png_paths[index])

        # ?: Not sure if this will work
        # mask = Image.open(self.masks[index]).convert("P")
        # mask = self.npz_read(npz_path)  # [H, W]
        mask = Image.fromarray(self.npz_read(npz_path), mode="P")

        # synchrosized transform
        if self.mode == "train":
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == "val":
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == "testval"
            img, mask = self._val_sync_transform_resize(img, mask)

        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        # return img, mask, os.path.basename(self.images[index])
        return img, mask, os.path.basename(self.png_paths[index])

        # if self.augmentation:
        #     im, transparency_arr = self.augmentation(im, transparency_arr)

        # return {"image": im, "transparency_array": transparency_arr}

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return "Transparent"

    def __len__(self):
        return len(self.npz_paths)

    # def __imul__(self, x):
    #     self.dataset_index *= x
    #     return self
