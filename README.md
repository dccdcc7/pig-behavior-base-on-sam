# This is the implementation of A New Method for Pig Behavior Detection - Tracking and Segmentation Them With Sam
The pig farming industry is vital to animal husbandry, with health assessments key to efficient production. Traditional methods relying on manual observation are inadequate for modern, large-scale operations. To address this, a new approach combining multi-target tracking and instance segmentation using improved YOLOv5_Pig+Bytetrack and Segment Anything models has been developed. This method provides precise measurements of pig movements and behaviors, crucial for disease prevention like African Swine Fever. It offers detailed health assessments and enhances farming efficiency, with higher precision than object detection alone, benefiting decision-making in pig farming and disease control.
# Our Framework
![image text](https://github.com/dccdcc7/pig-behavior-base-on-sam/blob/main/framework.png "Our Framework")
# inference output
![image text](https://github.com/dccdcc7/pig-behavior-base-on-sam/blob/main/pigimage1.png "Our Framework")
![image text](https://github.com/dccdcc7/pig-behavior-base-on-sam/blob/main/pigimage.png "Our Framework")
# Setup
1. required packages
```shell
conda create -n FastSAM python=3.8
conda activate pigseg
pip install matplotlib>=3.3  
pip install numpy>=1.23.5  
pip install opencv-python>=4.6.0  
pip install pillow>=10.3.0  
pip install PyYAML>=5.3.1  
pip install requests>=2.32.2  
pip install scipy>=1.4.1  
pip install thop>=0.1.1  # FLOPs computation  
pip install torch>=1.8.0  
pip install torchvision>=0.9.0  
pip install tqdm>=4.66.3  
pip install ultralytics==8.0.232  
pip install pandas>=1.1.4  
pip install seaborn>=0.11.0  
pip install setuptools>=70.0.0  
pip install onnxruntime  
pip install onnx
```
3. install bytetrack
```shell
git clone https://github.com/ifzhang/ByteTrack.git  
cd ByteTrack  
pip3 install -r requirements.txt  
python3 setup.py develop  
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'  
pip3 install cython_bbox
```
4. install segment-anything
```shell
pip install git+https://github.com/facebookresearch/segment-anything.git  
```
Click the links below to download the checkpoint for the corresponding model type we used
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

5. install FastSAM  
reference: https://github.com/CASIA-IVA-Lab/FastSAM.git  
note: if your prompt is not linked with text, you don't need to install CLIP
First download a [model checkpoint](#model-checkpoints).
# Usage
1. Pig Behavior Detection based on SAM
```shell
cd yolo-pyqt
python yolo1.py 
```
2. Pig Behavior Detection based on SAM
```shell
cd yolo-pyqt
python yolo1_fastsam.py 
```
