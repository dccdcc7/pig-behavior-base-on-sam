'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，可以参考一下txt_annotation.py文件。
'''

import os
from PIL import Image
from .classification import Classification

classfication = Classification()

def Predict1(img_path_folder):
    #print(img_path_folder)
    img_path = [os.path.join(img_path_folder, filename) for filename in os.listdir(img_path_folder)]
    num1=0
    num2=0
    ba=[]
    for item in img_path:
        #print(item)
        image = Image.open(item)
        class_name,pred = classfication.detect_image(image)
        pred = pred.tolist()
        ba.append(pred)
        # print(class_name)
        if(class_name=="stand"):
            num1=num1+1
        if(class_name=="lie"):
            num2=num2+1
    print(ba)
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



def Predict(img):
    image = Image.open(img)
    class_name, pred = classfication.detect_image(image)
    # print(pred)
    # print(class_name)
    return class_name

def Predict1(img):
    class_name, pred = classfication.detect_image(img)
    print(pred)
    print(class_name)
    return class_name


if __name__ =="__main__":
    import stat
    parent_directory = "F:/pycharmproject/segment-anything/classification-pytorch-main/test1"
    os.chmod(parent_directory, stat.S_IRWXU)
    subfolders = [entry.name for entry in os.scandir(parent_directory) if entry.is_dir()]
    for item in subfolders:
        #print("item：",item)
        item1 = parent_directory + '/' + item
        print("id ", item, ": ", Predict1(item1))


