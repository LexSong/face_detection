import numpy as np
import torch
import torch.nn.functional as F

from . import net_s3fd


def decode(x, offset_var, size_var):
    offset = x[0:2] * offset_var
    size = np.exp(x[2:4] * size_var)

    return np.array((
        offset - size / 2,
        offset + size / 2,
    ))


class S3FD(object):
    def __init__(self, model):
        self.net = net_s3fd.s3fd()
        self.net.load_state_dict(torch.load(model))
        self.net.cuda()
        self.net.eval()

        for param in self.net.parameters():
            param.requires_grad = False

        self.img_offset = torch.Tensor((104, 117, 123)).cuda().float()

        self.decode_kwargs = {
            'offset_var': 0.1,
            'size_var': 0.2,
        }

    def detect(self, img, threshold):
        img = torch.from_numpy(img).cuda().float()
        img -= self.img_offset
        img = img.permute(2, 0, 1).unsqueeze(0)

        net_output = self.net(img)
        net_output = [x[0].cpu() for x in net_output]

        def iter_layers(layer_list):
            x = iter(layer_list)
            return zip(x, x)

        layers = iter_layers(net_output)

        bboxlist = []
        for i, (scores, offsets) in enumerate(layers):
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor_size = stride * 4

            scores = F.softmax(scores, dim=0)[1]
            scores = scores.numpy()

            offsets = offsets.numpy()
            offsets = anchor_size * decode(offsets, **self.decode_kwargs)

            valid_indices = np.argwhere(scores >= threshold)

            for y, x in valid_indices:
                center = (np.array((x, y)) + 0.5) * stride
                points = center + offsets[:, :, y, x]
                score = scores[y, x]

                bboxlist.append([*(points.ravel()), score])

        if bboxlist:
            return np.array(bboxlist)
        else:
            return np.zeros((0, 5))
