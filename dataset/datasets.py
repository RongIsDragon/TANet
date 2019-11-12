import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pylab as plt

class ModaDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), mirror=True, rotate=True, ignore_label=255):
        self.root = root
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.rotate = rotate
        self.img_ids = [i.strip() for i in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil((float(max_iters) / len(self.img_ids))))
        self.files = []
        for item in self.img_ids:
            name = osp.splitext(osp.basename(item))[0]
            img_file = osp.join(self.root, 'images', item)
            label_file = osp.join(self.root, 'labels', item)
            # print(img_file, label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        if np.max(label)>13:
            print(label)
        image -= self.mean

        if self.rotate:
            flip = np.random.choice(2) * 2 - 1
            if flip:
                h, w = image.shape[:2]
                center = (h/2, w/2)
                M = cv2.getRotationMatrix2D(center, angle=10, scale=1.0)
                image = cv2.warpAffine(image, M, (w, h))
                label = cv2.warpAffine(label, M, (w, h))


        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class newDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), mirror=True, rotate=True, ignore_label=255):
        self.root = root
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.rotate = rotate
        self.img_ids = [i.strip() for i in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil((float(max_iters) / len(self.img_ids))))
        self.files = []
        for item in self.img_ids:
            name = osp.splitext(osp.basename(item))[0]
            img_file = osp.join(self.root, 'bounding_box_train', item)
            # print(img_file)
            # print(img_file, label_file)
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        image = cv2.resize(image, (2 * size[1], 2*size[0]))
        print(image.shape)
        size = image.shape
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image -= self.mean

        if self.rotate:
            flip = np.random.choice(2) * 2 - 1
            if flip:
                h, w = image.shape[:2]
                center = (h/2, w/2)
                M = cv2.getRotationMatrix2D(center, angle=10, scale=1.0)
                image = cv2.warpAffine(image, M, (w, h))


        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]

        return image.copy(), np.array(size), name
