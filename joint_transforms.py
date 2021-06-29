import numbers
import random

from PIL import Image, ImageOps
import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize(
                (tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop(
            (x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, img_real, img_rand=None): 
        if random.random() < 0.5:
            if img_rand is None:
                if isinstance(img, np.ndarray): # for cv2 Image
                    return cv2.flip(img, 1), cv2.flip(img_real, 1)
                else: # for PIL Image
                    return img.transpose(
                        Image.FLIP_LEFT_RIGHT), img_real.transpose(
                            Image.FLIP_LEFT_RIGHT)
            else:
                if isinstance(img, np.ndarray): # for cv2 Image
                    return cv2.flip(img, 1), cv2.flip(img_real, 1), cv2.flip(img_rand, 1)
                else: # for PIL Image
                    return img.transpose(
                        Image.FLIP_LEFT_RIGHT), img_real.transpose(
                            Image.FLIP_LEFT_RIGHT), img_rand.transpose(
                                Image.FLIP_LEFT_RIGHT)
        else:
            if img_rand is None:
                return img, img_real
            else:
                return img, img_real, img_rand


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(
            rotate_degree, Image.NEAREST)
