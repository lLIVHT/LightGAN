###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
from torchvision import  transforms
import os
import os.path
import cv2
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def store_dataset(dir):
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')

                images.append(img)
                all_path.append(path)

    return images, all_path


def default_loader(path):
    # 图像预处理：初步增强
    # im = cv2.imread(path)
    # im1 = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    # y = (im1[:, :, 0])/255
    # d = 0.5 / np.mean(y)
    # # y = y * y * y - 0.8 * d * y * y + 0.8 * d * y #法一
    # y = 1 - 1 / (np.exp(0.7*d * y)) #法2
    # im1[:, :, 0] = y*255
    # im1 = cv2.cvtColor(im1, cv2.COLOR_YCR_CB2RGB)
    # pil_image = transforms.ToPILImage()(im1)

    return Image.open(path).convert('RGB')
    # return pil_image


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
