import cv2
import numpy as np

from torchvision.transforms import Compose
from torchvision.transforms import ToTensor


def similar_transform(image, rotation=0, scale=1, tx=0, ty=0, flip=False):
    h, w = image.shape[0:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, scale)
    mat[:, 2] += (tx * w, ty * h)
    result = cv2.warpAffine(
        image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if flip:
        return result[:, ::-1]
    else:
        return result


class RandomRotation(object):
    def __init__(self, rotation_range=180):
        self.rotation_range = rotation_range

    def __call__(self, image):
        degrees = np.random.uniform(-self.rotation_range, self.rotation_range)
        return similar_transform(image, rotation=degrees)


class Downscale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        size = (self.size, self.size)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
