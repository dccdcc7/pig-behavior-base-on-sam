# pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple/
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import tensor
import torch.utils.data as Data
import math
from matplotlib import pyplot
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv("Test_Data.csv")  # 1 3 7 是 预测列
data.dropna(axis=0, how='any')
# data = data.fillna(0)
# print(data.head())
# print(data.columns)
data_x = data[
    ['2#泵排出压力(bar)', '吸入真空(bar)', '管路平均浓度(%)', '横移速度(m/min)','绞刀电机电流(A)', ]].values
data_y = data[[ '管路流速(m/s)']].values

# print(len(data_y))
# 四个数据划分为一组 用前三个预测后一个
data_4_x = []
data_4_y = []
for i in range(0, len(data_y) - 4, 4):
    data_4_x.append(data_x[i:i + 3])
    data_4_y.append(data_y[i + 4])
print(len(data_4_x), len(data_4_y))
x_train, x_test, y_train, y_test = train_test_split(np.array(data_4_x), np.array(data_4_y), test_size=0.2)


class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


Batch_Size = 8  #
DataSet = DataSet(np.array(x_train), list(y_train))
train_size = int(len(x_train) * 0.8)
test_size = len(y_train) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(DataSet, [train_size, test_size])
TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True, drop_last=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out


def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Transformer(nn.Module):
    """标准的Transformer编码器-解码器结构"""

    def __init__(self, n_encoder_inputs, n_decoder_inputs, Sequence_length, d_model=512, dropout=0.1, num_layer=8):
        """
        初始化
        :param n_encoder_inputs:    输入数据的特征维度
        :param n_decoder_inputs:    编码器输入的特征维度，其实等于编码器输出的特征维度
        :param d_model:             词嵌入特征维度
        :param dropout:             dropout
        :param num_layer:           Transformer块的个数
         Sequence_length:           transformer 输入数据 序列的长度
        """
        super(Transformer, self).__init__()

        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        self.linear = torch.nn.Linear(d_model, 1)
        self.ziji_add_linear = torch.nn.Linear(Sequence_length, 1)

    def encode_in(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (torch.arange(0, in_sequence_len, device=src.device).unsqueeze(0).repeat(batch_size, 1))
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        pos_decoder = (torch.arange(0, out_sequence_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1))
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        # print("out.shape:",out.shape)# torch.Size([batch, 3, 1]) # 原本代码中的输出
        # 上边的这个输入可以用于很多任务的输出 可以根据任务进行自由的变换
        # 下面是自己修改的
        # 使用全连接变成 [batch,1] 构成了基于transformer的回归单值预测
        out = out.squeeze(2)
        out = self.ziji_add_linear(out)
        return out


model = Transformer(n_encoder_inputs=5, n_decoder_inputs=5, Sequence_length=3).to(device)  # 3 表示Sequence_length  transformer 输入数据 序列的长度


def _test():
    with torch.no_grad():
        val_epoch_loss = []
        # for i in range(0, len(x_test),batch):# batch是 1 测试用1测试就行
        for index, (inputs, targets) in enumerate(TrainDataLoader):
            # inputs = x_test[i:i+batch]
            # targets = y_test[i:i+batch]
            # if len(inputs) == batch:  # 最后一个batch可能不足长度 舍弃
            inputs = inputs.float()
            targets = targets.float()
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            tgt_in = torch.rand((Batch_Size, 3, 5)).to(device)
            outputs = model(inputs, tgt_in)
            loss = criterion(outputs.float(), targets.float())
            val_epoch_loss.append(loss.item())
    return np.mean(val_epoch_loss)


epochs = 2
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss().to(device)

val_loss = []
train_loss = []
best_test_loss = 10000000
for epoch in tqdm(range(epochs)):
    train_epoch_loss = []
    # for i in range(0, len(x_train),batch):# batch是 1
    for index, (inputs, targets) in enumerate(TrainDataLoader):
        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.tensor(inputs).to(device)
        targets = torch.tensor(targets).to(device)
        print("inputs",inputs.shape) # [batch,3，16]
        print("targets",targets.shape) # targets torch.Size([batch])
        tgt_in = torch.rand((Batch_Size, 3,5)).to(device)  # 输入数据的维度是[batch,序列长度，每个单元的维度]
        outputs = model(inputs, tgt_in).to(device)
        # print("outputs.shape:",outputs.shape) # outputs.shape [batch, 3, 1]
        loss = criterion(outputs.float(), targets.float())
        print("loss:", loss)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
    train_loss.append(np.mean(train_epoch_loss))
    val_epoch_loss = _test()
    val_loss.append(val_epoch_loss)
    print("epoch:", epoch, "train_epoch_loss:", train_epoch_loss, "val_epoch_loss:", val_epoch_loss)
    # 保存下来最好的模型：
    if val_epoch_loss < best_test_loss:
        best_test_loss = val_epoch_loss
        best_model = model
        print("best_test_loss -------------------------------------------------", best_test_loss)
        torch.save(best_model.state_dict(), 'best_Transformer_trainModel.pth')

# 画一下loss图
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=len(val_loss), xmin=0)
plt.ylim(ymax=max(max(train_loss), max(val_loss)), ymin=0)
# 画两条（0-9）的坐标轴并设置轴标签x，y
x1 = [i for i in range(0, len(train_loss), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y1 = val_loss  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
x2 = [i for i in range(0, len(train_loss), 1)]
y2 = train_loss
colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积
# 画散点图
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='val_loss')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='train_loss')
plt.legend()
plt.show()

# 加载模型预测------
model = Transformer(n_encoder_inputs=5, n_decoder_inputs=5, Sequence_length=3).to(device)
model.load_state_dict(torch.load('best_Transformer_trainModel.pth'))
model.to(device)
model.eval()
# 在对模型进行评估时，应该配合使用with torch.no_grad() 与 model.eval()：
y_pred = []
y_true = []
with torch.no_grad():
    with torch.no_grad():
        val_epoch_loss = []
        for index, (inputs, targets) in enumerate(TrainDataLoader):
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()
            tgt_in = torch.rand((Batch_Size, 3, 5)).to(device)
            outputs = model(inputs, tgt_in)
            outputs = list(outputs.cpu().numpy().reshape([1, -1])[0])  # 转化为1行列数不指定
            targets = list(targets.cpu().numpy().reshape([1, -1])[0])
            y_pred.extend(outputs)
            y_true.extend(targets)


y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(y_true.shape)
print(y_pred.shape)
# 画折线图显示----
dataframe = pd.DataFrame({'pred': y_pred,
                                            'true': y_true
                          })
dataframe.to_csv("bijiao2.csv", index=False, sep=',')

print("y_pred", y_pred)
print("y_true", y_true)
len_ = [i for i in range(len(y_pred[0:1000]))]
plt.xlabel('标签', fontsize=8)
plt.ylabel('值', fontsize=8)
plt.plot(len_, y_true[0:1000], label='y_true', color="blue")
plt.plot(len_, y_pred[0:1000], label='y_pred', color="yellow")
plt.title("真实值预测值画图")
plt.show()
