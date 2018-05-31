import torch
import torch.nn.functional as F

import cv2
import argparse
import numpy as np

import net_s3fd
from bbox import decode, nms

torch.backends.cudnn.bencmark = True


def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = torch.from_numpy(img).float().cuda()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2):
        olist[i*2] = F.softmax(olist[i*2], dim=1)
    for i in range(len(olist)//2):
        ocls, oreg = olist[i*2].data.cpu(), olist[i*2+1].data.cpu()
        stride = 2**(i+2)    # 4,8,16,32,64,128

        scores = ocls[0, 1].numpy()
        valid_indices = np.argwhere(scores >= 0.05)

        for hindex, windex in valid_indices:
            axc, ayc = stride/2+windex*stride, stride/2+hindex*stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor(
                [[axc/1.0, ayc/1.0, stride*4/1.0, stride*4/1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0]*1.0
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))
    return bboxlist


parser = argparse.ArgumentParser(description='PyTorch face detect')
parser.add_argument('--net', '-n', default='s3fd', type=str)
parser.add_argument('--model', default='', type=str)
parser.add_argument('--path', default='CAMERA', type=str)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


net = getattr(net_s3fd, args.net)()
if args.model != '':
    net.load_state_dict(torch.load(args.model))
else:
    print('Please set --model parameter!')
net.cuda()
net.eval()


if args.path == 'CAMERA':
    cap = cv2.VideoCapture(0)
while(True):
    if args.path == 'CAMERA':
        ret, img = cap.read()
    else:
        img = cv2.imread(args.path)

    imgshow = np.copy(img)
    bboxlist = detect(net, img)

    keep = nms(bboxlist, 0.3)
    bboxlist = bboxlist[keep, :]
    for b in bboxlist:
        x1, y1, x2, y2, s = b
        if s < 0.5:
            continue
        cv2.rectangle(imgshow, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.imshow('test', imgshow)

    if args.path == 'CAMERA':
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imwrite(args.path[:-4]+'_output.png', imgshow)
        if cv2.waitKey(0) or True:
            break
