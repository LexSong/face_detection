import numpy as np
from zipfile import ZipFile
from torch.utils.data import Dataset

from ..utils import load_jpeg_buffer
from ..utils import crop_and_resize_face


class CelebA(Dataset):
    def __init__(self, zipfile, detections):
        self.zip = ZipFile(zipfile)
        self.detections = np.load(detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index):
        filename, box = self.detections[index]
        buf = self.zip.read(filename)
        image = load_jpeg_buffer(buf)
        return crop_and_resize_face(image, box)
