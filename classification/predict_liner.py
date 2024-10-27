import torch
from torchvision import models, transforms
import torch.nn as nn
import os
from PIL import Image
from classification import Classification

classfication = Classification()


def predict(img_path_folder,txt_path):
    #print(img_path_folder)
    last_part = img_path_folder.split('/')[-1]  # 适用于Unix风格的路径
    #print(int(last_part))
    ba= []
    img_path = [os.path.join(img_path_folder, filename) for filename in os.listdir(img_path_folder)]
    #print(img_path)
    num1=0
    num2=0
    list=[]
    list_temp=[]
    i = 0
    for item in img_path:
        #print(item)
        list1=[]
        image = Image.open(item)
        class_name,pred = classfication.detect_image(image)
        pred = pred.tolist()
        #list1.append(pred)
        #print(pred)
        #print(class_name)
        if(class_name=="stand"):
            num1=num1+1
        if(class_name=="lie"):
            num2=num2+1
        list1.append(pred)
        with open(txt_path,
                  'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行到一个列表
            if i < len(lines):  # 确保i在行数范围内
                list2=[]
                line_content = lines[i]  # 获取第i行的内容
                #print(line_content)
                #print(f"第{i + 1}行的内容是: {line_content.strip()}")
                parts = line_content.strip().split(',')  # 使用逗号作为分隔符分割字符串
                first_number = float(parts[0])  # 将分割后的第一个元素转换为浮点数
                second_number = float(parts[1])
                #print(first_number)
                #print(second_number)
                list2.append(first_number)
                list2.append(second_number)
                list1.append(list2)
                list.append(list1)
            else:
                print(f"错误：文件中没有第{i + 1}行。")
            #print('list2:',list2)

            i+=1
    # print('list',list)
    # print(len(list))
    # print(num1,num2)
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
    return output,list


def getdataset(img_path):
    import stat
    #parent_directory = "F:/pycharmproject/segment-anything/classification-pytorch-main/test1"
    # os.chmod(parent_directory, stat.S_IRWXU)
    # subfolders = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]
    list_all = []
    #for item in subfolders:
    # print("item：",item)
    #item1 = parent_directory + '/' + item
    txt_path = img_path + '.txt'
    output, list0 = predict(img_path,txt_path)
    list_all.append(list0)
    #print("id ", item, ": ", output)
    # print("list_all:", list_all)
    inputs = torch.tensor(list_all)
    padding = 100 - inputs.size(1)
    # 创建一个填充的值，这里使用0，也可以使用其他值或通过某种方式生成
    padding_values = torch.zeros(1, padding, 2, 2)
    inputs = torch.cat((inputs, padding_values), dim=1)
    return inputs

# 定义模型结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # 将输入张量展平
        self.fc1 = nn.Linear(100 * 2 * 2 , 64)  # 第一个全连接层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(64, 3)  # 第二个全连接层，输出维度为3

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
def run(folder_path):
    model = SimpleNN()
    model = torch.load('best_behavior.pth')

    # 设置为评估模式
    model.eval()

    # 图像预处理
    import os
    for item in os.listdir(folder_path):
        if item[-1] != 't':
            #print(item)
            item1 = folder_path + '/' +item
            input = getdataset(item1)
            # print(input)
            # print(input.shape)

            # 进行预测
            with torch.no_grad():
                outputs = model(input)

            id = item.split('-')[0]
            #print(id)
            frame = item.split('-')[1]
            class_index = ["walk","sleep","stand_without_moving"]
            #print(outputs)
            output1 = outputs[0].tolist()
            # 使用max函数找出最大值
            max_value = max(output1)
            # 找出最大值的索引
            max_index = output1.index(max_value)
            #print(max_value,max_index)

            print('id:',id,',from',(int(frame)-1)*20,'frame','to',(int(frame))*20,'frame',',dynamic behavior:',class_index[max_index])

def main():
    import shutil
    source_folder = '../img10/img10'
    # 目标文件夹路径
    destination_folder = './test2'

    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有条目
    for item in os.listdir(source_folder):
        # 获取源文件夹中每个条目的完整路径
        source_item_path = os.path.join(source_folder, item)

        # 检查条目是否是文件夹或txt文件
        if os.path.isdir(source_item_path) or item.endswith('.txt'):
            # 构造目标路径
            destination_item_path = os.path.join(destination_folder, item)

            # 如果目标路径不存在，则移动条目
            if not os.path.exists(destination_item_path):
                shutil.move(source_item_path, destination_item_path)
                print(f"Moved: {source_item_path} to {destination_item_path}")
            else:
                print(f"Item already exists at destination: {destination_item_path}, skipped.")

    # 打印完成消息
    print("Move operation completed.")
    folder_path = "./test2"
    run(folder_path)
#"F:/pycharmproject/segment-anything/classification/test1/8"
if __name__ == "__main__":
    main()