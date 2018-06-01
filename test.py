import cv2
import sys
from tqdm import tqdm

from api import S3FD


class Video(object):
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def iter_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame


model = "models/s3fd_convert.pth"
s3fd = S3FD(model)

video = Video(sys.argv[1])

resize_shape = (640, int(640*video.frame_height/video.frame_width))

iter_frames = tqdm(video.iter_frames(), total=video.frame_count, ascii=True)

for frame in iter_frames:
    frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
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
