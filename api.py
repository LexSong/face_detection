import torch
import torch.nn.functional as F

import numpy as np

import net_s3fd
from bbox import nms

torch.backends.cudnn.bencmark = True


def decode(x, offset_var, size_var):
    offset = offset_var * x[0:2]
    size = np.exp(x[2:4] * size_var)

    return np.array((
        offset - size / 2,
        offset + size / 2
        ))


class S3FD(object):
    def __init__(self, model):
        self.net = net_s3fd.s3fd()
        self.net.load_state_dict(torch.load(model))
        self.net.cuda()
        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad = False

    def _detect(self, img):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float().cuda()

        net_output = self.net(img)
        net_output = [x.cpu() for x in net_output]

        def iter_layers(layer_list):
            x = iter(layer_list)
            return zip(x, x)

        layers = iter_layers(net_output)

        bboxlist = []
        for i, (scores, oreg) in enumerate(layers):
            scores = F.softmax(scores[0], dim=0)[1]
            scores = scores.numpy()

            oreg = oreg.numpy()
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor_size = stride * 4

            valid_indices = np.argwhere(scores >= 0.05)

            all_offsets = decode(oreg[0], offset_var=0.1, size_var=0.2) * anchor_size

            for hindex, windex in valid_indices:
                axc = stride * (windex + 0.5)
                ayc = stride * (hindex + 0.5)
                anchor_center = np.array([axc, ayc], dtype='float32')

                offsets = all_offsets[:, :, hindex, windex]

                x1, y1 = anchor_center + offsets[0]
                x2, y2 = anchor_center + offsets[1]

                score = scores[hindex, windex]
                bboxlist.append([x1, y1, x2, y2, score])

        if bboxlist:
            return np.array(bboxlist)
        else:
            return np.zeros((0, 5))

    def detect(self, img, threshold):
        bboxlist = self._detect(img)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        scores = bboxlist[:, 4]
        return bboxlist[scores >= threshold]
