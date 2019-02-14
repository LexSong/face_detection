import cv2
import numpy as np


def get_center_crop_transform(center, size):
    cx, cy = center
    scale = size / 2
    mat = [[scale, 0, cx], [0, scale, cy]]
    mat = np.array(mat, dtype=float)
    return cv2.invertAffineTransform(mat)


def face_box_to_transform(box):
    x1, y1, x2, y2 = box
    cx = x1 * 0.5 + x2 * 0.5
    cy = y1 * 0.5 + y2 * 0.5
    size = ((x2 - x1) + (y2 - y1)) * 0.65
    return get_center_crop_transform((cx, cy), size)


def apply_transform(image, transform, size, flags=cv2.INTER_LINEAR):
    mat = np.array(transform)
    mat[:, 2] += 1
    mat *= size / 2
    return cv2.warpAffine(image, mat, (size, size), flags=flags)


def crop_and_resize_face(image, face_box, image_size=256):
    transform = face_box_to_transform(face_box)
    return apply_transform(image, transform, image_size)
