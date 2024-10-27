import torch
from torch.utils.data import Dataset
import torchvision
import cv2
from PIL import Image
import os

class mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_paths = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_paths[idx]  # 获取图片文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        #print(img.size)
        img = img.convert('RGB')
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((24, 24)), torchvision.transforms.ToTensor()])
        img = transform(img)
        #print("out img size:",img.shape)
        label = self.label2tensor(label)
        return img ,label

    def label2tensor(self,x):
        if(x=="lie"):
            return torch.tensor(0)
        elif(x=="stand"):
            return torch.tensor(1)

    def __len__(self):
        return len(self.img_paths)