import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_aug.data_aug import *
from data_aug.bbox_util import *
import random
import torch.nn as nn

dataPath = 'FDDB-folds/FDDB-fold-0'
pathEnd = '-ellipseList.txt'

class TrainDataset(Dataset):
    def __init__(self):
        self.X=[]
        self.Y=[]
        for i in range(9):
            with open(dataPath+str(i+1)+pathEnd) as f:
                while True :
                    imgName = f.readline().strip('\n')
                    if not imgName :
                        break
                    c = int(f.readline())
                    if c != 1 :
                        for i in range(c):
                            _ = f.readline()
                        continue
                    anno = f.readline().split()
                    anno = list(map(float, anno))
                    self.X.append(imgName)
                    self.Y.append(anno)
        no_faces = os.listdir('no_faces/')
        self.X = self.X + no_faces
        for i in range(len(no_faces)):
            z = [random.uniform(0.,256.),random.uniform(0.,256.),random.uniform(0.,256.),random.uniform(0.,256.),0.]
            self.Y.append(z)
        print('finished loading train dataset : ', len(self.X))
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        transforms = Sequence([Resize(256),RandomHorizontalFlip(0.5), RandomHSV(30, 20, 20)])#, RandomRotate(15)])
        k = self.Y[index]
        k = (k[3]-k[1],k[4]-k[0],k[3]+k[1],k[4]+k[0],k[-1])
        l = torch.unsqueeze(torch.tensor(k),0)
        labels = l.numpy()
        path = 'originalPics/'+self.X[index]+'.jpg' if k[4]==1 else 'no_faces/'+self.X[index]
        image = cv2.imread(path)
        x, y = transforms(image, labels)

        y[:,-1] *= 256
        y = torch.from_numpy(np.squeeze(y))/256
        x = torch.from_numpy(x)/255
        #self.x = self.myTransforms(self.x)/255
        return x,y

class TestDataset(Dataset):
    def __init__(self):
        self.X=[]
        self.Y=[]
        with open('FDDB-folds/FDDB-fold-10-ellipseList.txt') as f:
            while True :
                imgName = f.readline().strip('\n')
                if not imgName:
                    break
                c = int(f.readline())
                if c != 1 :
                    for i in range(c):
                        _ = f.readline()
                    continue
                anno = f.readline().split()
                anno = list(map(float, anno))
                self.X.append(imgName)
                self.Y.append(anno)
        no_faces = os.listdir('no_faces_test/')
        self.X = self.X + no_faces
        for i in range(len(no_faces)):
            z = [random.uniform(0., 256.), random.uniform(0., 256.), random.uniform(0., 256.), random.uniform(0., 256.),
                 0.]
            self.Y.append(z)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        transforms = Sequence([Resize(256)])
        k = self.Y[index]
        k = (k[3]-k[1],k[4]-k[0],k[3]+k[1],k[4]+k[0],k[-1])
        labels = np.expand_dims(np.array(k), axis=0)
        path = 'originalPics/'+self.X[index]+'.jpg' if k[-1]==1 else 'no_faces_test/'+self.X[index]
        image = cv2.imread(path)
        self.x, self.y = transforms(image, labels)

        self.y[:,-1] *= 256
        self.y = torch.from_numpy(np.squeeze(self.y))/256
        self.x = torch.from_numpy(self.x)/255
        #self.x = self.myTransforms(self.x)/255
        return self.x,self.y
#2326 torch.Size([0, 5])



def draw(frame, face_locations):
    k = face_locations
    frame = frame.numpy()
    k = list(map(int, k))
    color = (0,255,0)
    cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), color, 2)
    cv2.imshow('image', frame)
    cv2.waitKey(0)



#print(x[3][0][100])
#draw(x,y)
# print(x.shape)
# print(y)
# print(x[100])

