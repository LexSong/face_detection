import cv2
import numpy as np
from .api import S3FD
from .bbox import nms

detection_dtype = np.dtype([('box', float, 4), ('score', float)])


class FaceDetector(object):
    def __init__(self, s3fd_model):
        self.s3fd = S3FD(s3fd_model)

    def detect(
            self,
            image,
            resize_width=None,
            score_threshold=0.5,
            nms_threshold=0.3,
    ):
        if resize_width is not None:
            h, w = image.shape[0:2]
            resize_shape = (
                round(resize_width),
                round(resize_width * h / w),
            )
            image = cv2.resize(image, resize_shape)

        boxes = self.s3fd.detect(image, score_threshold)
        boxes = boxes[nms(boxes, nms_threshold)]
        boxes = boxes.ravel().view(detection_dtype)

        if resize_width is not None:
            boxes['box'] *= w / resize_width

        return boxes
