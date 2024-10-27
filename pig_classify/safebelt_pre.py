import torchvision
from torch import nn
import torch
import numpy as np
import os
from PIL import Image

class my_net(nn.Module):  #加上模型的批标准化是为了loss不是nan，而且训练效率大幅增加，防止梯度是None
    def __init__(self):
        super(my_net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,padding=0,stride=1)#要计算padding和stride的值,假设stride=1
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, padding=0, stride=1)
        self.flatten = nn.Flatten(start_dim=1) #nn.Flatten默认从第二位开始展开，但是torch.flatten就全部展开
        self.liner1 = nn.Linear(2888,60)
        self.liner2 = nn.Linear(60,10)
        self.liner3 = nn.Linear(10,2)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(3)
        self.sof = nn.Softmax(dim=0)
        # self.my_net1 = nn.Sequential(self.conv1,self.maxpool,self.conv2)
        # self.my_net2 = nn.Sequential(self.liner1,self.liner2,self.liner3,self.sof)
    def forward(self,x):
        operater1 = nn.Sequential(self.conv1,self.batch_norm1,self.maxpool,self.conv2,self.batch_norm2,self.sof,self.flatten)
        x = operater1(x)
        #print(x.shape)
        operater2 = nn.Sequential(self.liner1,self.liner2,self.liner3)
        x = operater2(x)
        #print(x.shape)
        return x

def predict(img_path_folder):
    classes = np.array(['lie', 'stand'])
    #img_path_folder = "./18"
    img_path = [os.path.join(img_path_folder, filename) for filename in os.listdir(img_path_folder)]
    num1=0
    num2=0
    for item in img_path:
        image = Image.open(item)
        net = my_net()
        net = torch.load("epoch70")
        image = image.convert('RGB')
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((24, 24)), torchvision.transforms.ToTensor()])
        image = transform(image)
        #print(image.shape)
        image =torch.reshape(image,(1,3,24,24))
        #print(image)
        #print(item)
        with torch.no_grad():
            output = net(image)
            #print(classes[output.argmax(1)])
            if(classes[output.argmax(1)]=="stand"):
                num1=num1+1
            if(classes[output.argmax(1)]=="lie"):
                num2=num2+1
    print(num1,num2)
    output = ""
    if(num1>num2):
        #print("stand")
        output = "stand"
    if(num1<num2):
        #print("lie")
        output = "lie"
    if(num1==num2):
        #print("unidentified")
        output = "unidentified"
    return output

def folder_predict():
    parent_directory = "./test"
    subfolders = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]
    for item in subfolders:
        item1 = parent_directory+'/'+item
        print("id ",item,": ",predict(item1))


if __name__=="__main__":
    #predict()
    folder_predict()