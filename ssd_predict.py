import torch
from torch.autograd import Variable
import cv2
import time
import argparse
import sys
from os import path
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd

# device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
# parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_PLAIN

net = build_ssd('test', 300, 21)# initialize SSD
# net = net.to(device)
net.load_state_dict(torch.load(args.weights))
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

def predict(frame):    
    # print "predict"
    pt = []
    idxcls = []
    height, width = frame.shape[:2]
    x = torch.Tensor(transform(frame)[0]).permute(2, 0, 1)
    x = x.unsqueeze(0)
    # x = x.to(device)
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        # print (detections[0, i, j, 0])
        while detections[0, i, j, 0] >= 0.45:
            # class 0 - 10Hz : person, potted plant
            # class 1 - 12Hz : dog, car
            # class 2 - 15Hz : chair, aeroplane
            print (labelmap[i - 1])
            if (labelmap[i-1] == 'person'):
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(0)
            if (labelmap[i-1] == 'dog'):
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(1)
            if (labelmap[i-1] == 'chair'):
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(2)
            if (labelmap[i-1] == 'pottedplant'):
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(3)
            if (labelmap[i-1] == 'car') :
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(4)
            if (labelmap[i-1] == 'aeroplane'):
                pt.append((detections[0, i, j, 1:] * scale).cpu().numpy())
                idxcls.append(5)
            j += 1
            time.sleep(1.0)

    return pt, idxcls

