# This is the implementation of A New Method for Pig Behavior Detection - Tracking and Segmentation Them With Sam
The pig farming industry is vital to animal husbandry, with health assessments key to efficient production. Traditional methods relying on manual observation are inadequate for modern, large-scale operations. To address this, a new approach combining multi-target tracking and instance segmentation using improved YOLOv5_Pig+Bytetrack and Segment Anything models has been developed. This method provides precise measurements of pig movements and behaviors, crucial for disease prevention like African Swine Fever. It offers detailed health assessments and enhances farming efficiency, with higher precision than object detection alone, benefiting decision-making in pig farming and disease control.
# Our Framework
![image text](https://github.com/dccdcc7/pig-behavior-base-on-sam/edit/main/framework.png "DBSCAN Performance Comparison")

# usage
1. required packages
matplotlib>=3.3  
numpy>=1.23.5  
opencv-python>=4.6.0  
pillow>=10.3.0  
PyYAML>=5.3.1  
requests>=2.32.2  
scipy>=1.4.1  
thop>=0.1.1  # FLOPs computation  
torch>=1.8.0  
torchvision>=0.9.0  
tqdm>=4.66.3  
ultralytics==8.0.232  
pandas>=1.1.4  
seaborn>=0.11.0  
setuptools>=70.0.0  
onnxruntime  
onnx
2. install bytetrack
git clone https://github.com/ifzhang/ByteTrack.git  
cd ByteTrack  
pip3 install -r requirements.txt  
python3 setup.py develop  
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'  
pip3 install cython_bbox
3. install segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git  
4. install FastSAM
https://github.com/CASIA-IVA-Lab/FastSAM.git  
note: if your prompt is not linked with text, you don't need to install CLIP
