import pandas as pd
# data = pd.read_csv("Test_Data.csv")  # 1 3 7 是 预测列
# data.dropna(axis=0, how='any')
# # data = data.fillna(0)
# # print(data.head())
# # print(data.columns)
# data_x = data[
#     ['2#泵排出压力(bar)', '吸入真空(bar)', '管路平均浓度(%)', '横移速度(m/min)','绞刀电机电流(A)', '管路流速(m/s)']].values
#
# # print(len(data_y))
# # 四个数据划分为一组 用前三个预测后一个
# data_4_x = []
# data_4_y = []
# for i in range(0, len(data_x) - 10, 10):
#     data_4_x.append(data_x[i:i +9])
#     data_4_y.append(data_x[i+9])
#     break
import cv2
image = cv2.imread("../output/1/1_1.png",cv2.IMREAD_GRAYSCALE)
print(image)
print(image.shape)
image = image.flatten()
print(image)
print(image.shape)