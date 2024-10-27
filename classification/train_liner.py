import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from classification import Classification

classfication = Classification()


def predict(img_path_folder):
    print(img_path_folder)
    last_part = img_path_folder.split('/')[-1]  # 适用于Unix风格的路径
    print(int(last_part))
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
        print(pred)
        print(class_name)
        if(class_name=="stand"):
            num1=num1+1
        if(class_name=="lie"):
            num2=num2+1
        list1.append(pred)
        with open(r"F:/pycharmproject/segment-anything/classification/test1/{}.txt".format(int(last_part)),
                  'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行到一个列表
            if i < len(lines):  # 确保i在行数范围内
                list2=[]
                line_content = lines[i]  # 获取第i行的内容
                print(line_content)
                print(f"第{i + 1}行的内容是: {line_content.strip()}")
                parts = line_content.strip().split(',')  # 使用逗号作为分隔符分割字符串
                first_number = float(parts[0])  # 将分割后的第一个元素转换为浮点数
                second_number = float(parts[1])
                print(first_number)
                print(second_number)
                list2.append(first_number)
                list2.append(second_number)
                list1.append(list2)
                list.append(list1)
            else:
                print(f"错误：文件中没有第{i + 1}行。")
            print('list2:',list2)

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


def getdataset():
    import stat
    parent_directory = "F:/pycharmproject/segment-anything/classification/test1"
    os.chmod(parent_directory, stat.S_IRWXU)
    subfolders = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]
    list_all = []
    for item in subfolders:
        # print("item：",item)
        item1 = parent_directory + '/' + item
        output, list0 = predict(item1)
        list_all.append(list0)
        print("id ", item, ": ", output)
    # print("list_all:", list_all)
    inputs = torch.tensor(list_all)
    padding = 100 - inputs.size(1)
    # 创建一个填充的值，这里使用0，也可以使用其他值或通过某种方式生成
    padding_values = torch.zeros(7, padding, 2, 2)
    inputs = torch.cat((inputs, padding_values), dim=1)
    return inputs

# 定义一个简单的神经网络模型
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
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 假设我们有一些随机生成的输入数据和目标数据
# inputs = torch.tensor(getdataset())
# padding = 100 - inputs.size(1)
# # 创建一个填充的值，这里使用0，也可以使用其他值或通过某种方式生成
# padding_values = torch.zeros(7,padding, 2, 2)
# # 将原始张量和填充值合并
# print(inputs.shape)
# print(padding_values.shape)
inputs = getdataset()
print(inputs.shape)

targets = torch.tensor([[1.0, 0, 0],
        [0, 1.0, 0],
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0],
        [0, 0, 1.0]])  # 100个目标张量，每个目标的维度是3*1

print(targets.shape)
#print(targets)
# 训练模型
num_epochs = 30  # 训练的轮数
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清除之前的梯度

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets.view(-1, 3))  # 计算损失

    # 反向传播和优化
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    # 打印损失值
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model, 'best_behavior.pth')
print("Complete model saved successfully.")


# import stat
# if __name__ =="__main__":
#     parent_directory = "F:/pycharmproject/segment-anything/classification-pytorch-main/test1"
#     os.chmod(parent_directory, stat.S_IRWXU)
#     subfolders = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]
#     list_all=[]
#     for item in subfolders:
#         #print("item：",item)
#         item1 = parent_directory + '/' + item
#         output,list0 = predict(item1)
#         list_all.append(list0)
#         print("id ", item, ": ", output)
#     print("list_all:",list_all)


