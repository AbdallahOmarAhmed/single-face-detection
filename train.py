import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import timm


device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

from DataSet import TrainDataset, TestDataset

# hayper parametars
num_epochs = 40
learning_rate = 0.0003
batch_size = 80

# load data
train_data = TrainDataset()
test_data = TestDataset()

train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1,shuffle=False, num_workers=0)

def area(inter):
    inter_area = abs(inter[0] - inter[2]) * abs(inter[1] - inter[3])
    return inter_area

def iou (box1, box2):
    if box2[4] == 0:
        return 1
    inter = torch.tensor([torch.max(box1[0], box2[0]), torch.max(box1[1], box2[1]), torch.min(box1[2], box2[2]), torch.min(box1[3], box2[3])])
    inter_area = area(inter)
    area1 =  area(box1)
    area2 = area(box2)
    all_area = area1 + area2 - inter_area
    iou = inter_area/all_area
    return iou

def Loss(pred,ans):
    loss = 0
    for y,y_hat in zip(pred,ans) :
        loss += (y[-1]-y_hat[-1])**2
        if y_hat[-1] == 1:
            loss += sum((y[0:4]-y_hat[0:4])**2)
    return loss

#model = timm.create_model("resnet34",pretrained=True,num_classes=5)
model = models.resnet34(pretrained=True)
print('model created')
n = model.fc.in_features
model.fc = nn.Linear(n, 5)
optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#Loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=15, gamma=0.8)
#scheduler1 = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=1, gamma=2)
#scheduler2 = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=10, gamma=0.5)
model.to(device)
min = 0.2
sig = nn.Sigmoid()
for epoch in range(num_epochs):
    model.train()
    for x, y in train_dataloader:
        # forward
        x = x.permute(0,3,1,2)
        images = x.to(torch.float32).to(device)
        labels = y.to(torch.float32).to(device)
        output = model(images)
        output = sig(output)
        loss = Loss(output, labels)
        # backward
        optimaizer.zero_grad()
        loss.backward()
        optimaizer.step()
    #scheduler1.step() if epoch < 10 else scheduler2.step()
    scheduler.step()
    print(epoch + 1, '/', num_epochs, 'loss = ', loss.item())
    model.eval()
    # accuracy
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        loss2 = 0
        for x, y in test_dataloader:
            x = x.permute(0, 3, 1, 2)
            y = y.to(device)
            image = x.to(device)
            label = y[0][4]
            output = model(image)
            output = torch.sigmoid(output)
            loss2 += Loss(output, y)
            IoU = iou(output[0], y[0])
            pred = 1 if output[0][4] >= 0.5 else 0
            n_total += 1
            if pred == label and IoU >= 0.75:
                n_correct +=1
        acc = 100.0 * n_correct / n_total
        print('loss test : ', loss2.item())
        print('accuracy = ', acc)
        if loss2 < min:
            min = loss2
            torch.save(model.state_dict(), 'model.pth')
            print('saved!')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
