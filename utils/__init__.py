import cv2
import itertools
import numpy as np

from .transforms import crop_and_resize_face
from .video import VideoFile


def iter_by_step(iterable, step):
    return itertools.islice(iterable, None, None, step)


def load_jpeg_buffer(buf):
    buf = np.asarray(bytearray(buf))
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
