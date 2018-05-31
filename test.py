import cv2
from api import S3FD

model = "models/s3fd_convert.pth"
s3fd = S3FD(model)

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    boxes = s3fd.detect(frame, 0.5)
    boxes = boxes[:, 0:4].astype(int)

    figure = frame.copy()
    for box in boxes:
        p1 = tuple(box[0:2])
        p2 = tuple(box[2:4])
        color = (0, 255, 0)
        cv2.rectangle(figure, p1, p2, color, 1)
    cv2.imshow('', figure)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
