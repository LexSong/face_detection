import torch
import torch.nn.functional as F

import numpy as np

import net_s3fd
from bbox import decode, nms

torch.backends.cudnn.bencmark = True


class S3FD(object):
    def __init__(self, model):
        self.net = net_s3fd.s3fd()
        self.net.load_state_dict(torch.load(model))
        self.net.cuda()
        self.net.eval()

    def _detect(self, img):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float().cuda()

        def iter_layers(layer_list):
            x = iter(layer_list)
            return zip(x, x)

        layers = iter_layers(self.net(img))

        bboxlist = []
        for i, (scores, oreg) in enumerate(layers):
            scores = F.softmax(scores, dim=1)
            scores = scores[0, 1].data.cpu().numpy()

            oreg = oreg.data.cpu()
            stride = 2**(i+2)    # 4,8,16,32,64,128

            valid_indices = np.argwhere(scores >= 0.05)

            for hindex, windex in valid_indices:
                axc = stride * (windex + 0.5)
                ayc = stride * (hindex + 0.5)
                loc = oreg[0:1, :, hindex, windex]
                priors = torch.Tensor(
                    [[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0]*1.0
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
