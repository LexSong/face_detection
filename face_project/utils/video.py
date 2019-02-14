import cv2


class VideoFile(object):
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(str(filename))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def __len__(self):
        return self.frame_count
