import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset import mydata
from torch.utils.data import Dataset
from PIL import Image
import os
from torch import nn


class my_net(nn.Module):  #加上模型的批标准化是为了loss不是nan，而且训练效率大幅增加，防止梯度是None
    def __init__(self):
        super(my_net,self).__init__()
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=1,padding=0,stride=1)#要计算padding和stride的值,假设stride=1
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, padding=0, stride=1)
        self.conv3  = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1,padding=0,stride=1)
        self.flatten = nn.Flatten(start_dim=1) #nn.Flatten默认从第二位开始展开，但是torch.flatten就全部展开
        self.liner1 = nn.Linear(3456,60)
        self.liner2 = nn.Linear(60,2)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(3)
        self.sof = nn.Softmax(dim=0)
        # self.my_net1 = nn.Sequential(self.conv1,self.maxpool,self.conv2)
        # self.my_net2 = nn.Sequential(self.liner1,self.liner2,self.liner3,self.sof)
    def forward(self,x):
        x2 = x.clone()
        operater1 = nn.Sequential(self.conv1,self.batch_norm1,self.conv2,self.batch_norm2,self.act)
        x1 = operater1(x)
        # print(x1)
        # print("x1形状：",x1.shape)

        # operater2 = nn.Sequential(self.conv3,self.act)
        # x2 = operater2(x)
        # x2 = x2[:, :1200]
        # print("x2形状：",x2.shape)
        # print(x2)
        x3 = torch.concat((x1,x2),dim=1)
        x3 = self.flatten(x3)
        #print(x3.shape)
        operater3 = nn.Sequential(self.liner1,self.liner2,self.sof)
        y = operater3(x3)
        #print(y.shape)
        return y


root_dir = "./dataset2/train"
root_dir1 = "./dataset2/val"
label_dir1 = "lie"
label_dir2 = "stand"
# label_dir1 = "safebelt"
# label_dir2 = "nosafebelt"
my_dataset = mydata(root_dir, label_dir1)
my_dataset1 = mydata(root_dir, label_dir2)
train_dataset = my_dataset + my_dataset1
my_dataset2 = mydata(root_dir1, label_dir1)
my_dataset3 = mydata(root_dir1, label_dir2)
test_dataset = my_dataset2 + my_dataset3
print(train_dataset.__len__())
print(test_dataset.__len__())
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
for data in train_loader:
    imgs, labels = data
    print(imgs.shape)
    print(labels)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
print(len(test_dataset))


def train():
    net = my_net()
    # 损失函数
    los_function = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=0.01)

    total_train = 0
    total_test = 0
    epoch = 101

    #进行训练
    for i in range(epoch):
        print("第{}轮训练开始".format(i))
        for data in train_loader:
            imgs, labels = data
            output = net(imgs)
            loss = los_function(output, labels)
            #print([x.grad for x in optim.param_groups[0]['params']])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train = total_train + 1
            if(total_train%10==0):
                print("训练次数{},loss={}".format(total_train,loss.item()))

    #进行验证
            total_test_loss = 0
            total_num=0
            with torch.no_grad():
                for data in test_loader:
                    imgs, labels = data
                    # print(labels)
                    #print(labels)
                    output = net(imgs)
                    output_pred = output.argmax(1)
                    #print(output)
                    #print(output_pred)
                    # print(output_pred)
                    test_loss = los_function(output, labels)
                    total_test_loss += test_loss
                    comparison = (output_pred == labels).bool()
                    total_num += comparison.sum()
                    # print(comparison.sum())
                # print(total_num)
                # print(temp3)
            print("test的损失是{}".format(total_test_loss))
            print("test的总体正确率是{}".format(float(total_num / len(test_dataset))))
        if (i % 10 == 0):
            torch.save(net, "epoch{}".format(i))


if __name__=="__main__":
    train()


