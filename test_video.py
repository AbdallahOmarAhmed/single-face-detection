import torch

from torchvision import transforms
import cv2
import torch.nn as nn
from torchvision import models
from data_aug.data_aug import *
from data_aug.bbox_util import *
import argparse

parser = argparse.ArgumentParser(description='inputs')
parser.add_argument('path', type=str, help='your video path')
args = parser.parse_args()
path = args.path

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')

transforms = Sequence([Resize(256)])
model = models.resnet34()
n = model.fc.in_features
model.fc = nn.Linear(n, 5)
model.load_state_dict(torch.load("model.pth"))
model.eval()
model.to(device)
sig = nn.Sigmoid()
box = torch.reshape(torch.tensor([0.,0.,0.,0.,1.]),(-1,5))


vidcap = cv2.VideoCapture(path)
success,frame = vidcap.read()
video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),\
 30,(256,256))
count = 0
while success:
    success,frame = vidcap.read()
    if type(frame) == type(None):
        break
    count += 1
    print(count)
    image, _ = transforms(frame, box)
    x = torch.from_numpy(image) / 255
    x = torch.unsqueeze(x, 0)
    x = x.permute(0, 3, 1, 2).to(device)
    out = torch.squeeze(sig(model(x)))
    k = list(map(int, out[0:4] * 256))
    if out[-1] >= 0.5:
        k = list(map(int, out[0:4] * 256))
        color = (150, 150, 150)
        cv2.rectangle(image, (k[0], k[1]), (k[2], k[3]), color, 2)
    video.write(image)

