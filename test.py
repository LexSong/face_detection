import cv2
import numpy as np

from api import S3FD

model = "models/s3fd_convert.pth"
s3fd = S3FD(model)

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()

    imgshow = np.copy(img)
    bboxlist = s3fd.detect(img, 0.5)

    for b in bboxlist:
        x1, y1, x2, y2, s = b
        cv2.rectangle(imgshow, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 1)

    cv2.imshow('test', imgshow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
