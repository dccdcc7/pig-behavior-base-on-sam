#without classification 2:48
#with classfication 2:57

import os, random, yaml, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import cv2
import torch
import numpy as np
import sys
sys.path.insert(0,'FastSAM-main')
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy

sys.path.append('F:/pycharmproject/segment-anything/yolo-pyqt')
from onnx11 import model_init,onnx_forward1
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box
import cv2
import numpy as np
from classification.predict import Predict1

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
colors = Colors()  # create instance for 'from utils.plots import colors'
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# predictor = SamPredictor(build_sam(checkpoint="checkpoint/sam_vit_b_01ec64.pth"))
# _ = predictor.model.to(device='cuda')
sam_checkpoint = "F:/pycharmproject/segment-anything/checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to(device=device)

# from fastsam import FastSAM, FastSAMPrompt



def ByteTrack_opt():
    parser = argparse.ArgumentParser("ByteTrack Param.")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=10, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fps", default=25, type=int, help="frame rate (fps)")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser.parse_args()

class base_model:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs):
        device = self.select_device(device)
        print(device)
        if model_path.endswith('pt'):
            model = torch.jit.load(model_path).to(device)
        elif model_path.endswith('onnx'):
            try:
                import onnxruntime as ort
            except:
                raise 'please install onnxruntime.'
            providers = ['CUDAExecutionProvider'] if device.type != 'cpu' else ['CPUExecutionProvider']
            model = ort.InferenceSession(model_path, providers=providers)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        im = self.processing(image)
        
        if self.model_path.endswith('pt'):
            result = self.model(im)[0]
        elif self.model_path.endswith('onnx'):
            result = self.model.run([i.name for i in self.model.get_outputs()], {'images':im})[0]
        return self.post_processing(result, im, image)
        
    def processing(self, img):
        image = letterbox(img, new_shape=tuple(self.imgsz), auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = np.array(image, dtype=np.float32)
        image /= 255
        
        if self.model_path.endswith('pt'):
            im = torch.from_numpy(image).float().to(self.device)
        elif self.model_path.endswith('onnx'):
            im = image
        return im
    
    def post_processing(self, result, im=None, img=None):
        pass
    
    def select_device(self, device):
        if device == -1:
            return torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
            device = torch.device('cuda:0')
        return device
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self, i,frame, det_result):
        #print(det_result)
        c = frame.shape[0]
        h = frame.shape[1]
        #print(c,h)
        if det_result.shape==(0,):
            return frame
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        prompt_list= np.empty((0,))
        prompt_list = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame1 = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                #frame = plot_one_box([(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c,tid], frame, (0, 0, 255), str(tid))
                #plot_one_box([0,0,100,100], frame,(0, 0, 255))
                #plot_one_box([tlwh[0], tlwh[1], tlwh[2], tlwh[3]], frame, (0, 0, 255), str(tid))
                #plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
                #prompt_list = np.append(prompt_list,[[(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c]])
                prompt_list.append([(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c,tid])
                content = str(i) + ',' + str(tid) + ',' + str((abs(tlwh[0]/640))*h) + ',' + str((tlwh[1]/640)*c) + ',' + str(
                    ((tlwh[2]) / 640) * h) + ',' + str(((tlwh[3])/640)*c) + ',' + '0' + ',' + '0' + ',' + '0' + '\n'
                #print(content)
                filename = 'car_output.txt'
                # 使用'w'模式打开文件，如果文件不存在则创建
                with open(filename, 'a') as file:
                    # 写入内容
                    file.write(content)
        #print("prompt_list is: == ",prompt_list)
        prompt_list1 = np.array(prompt_list)
        masks_all=[]
        predictor.set_image(frame)
        #prompt_list = torch.tensor(prompt_list)
        num = 0
        for mbox in prompt_list1:
            #print(mbox)
            #提示可以改成点+框
            bpoint = np.array([[(mbox[0] + mbox[2]) / 2, (mbox[1] + mbox[3]) / 2]])
            input_label = np.array([1])
            #masks, iou_predictions, low_res_masks = predictor.predict(point_labels=input_label,point_coords=bpoint,box=mbox[:4])
            masks, iou_predictions, low_res_masks = predictor.predict(box=mbox[:4])
            index_max = iou_predictions.argsort()[0]
            masks = np.concatenate(
                [masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)]],
                axis=0)
            masks1 = masks
            if masks1.dtype != np.uint8:
                masks1 = (masks1 * 255).astype(np.uint8)  # 乘以 255 并转换为 uint8
                masks1 = 255 - masks1
            # 保存掩膜图像
            # mask_image = masks1[index_max]  # 假设我们只对单一的掩膜图像感兴趣
            mask_image = masks1[index_max]  # 假设我们只对单一的掩膜图像感兴趣
            mask_image1 = mask_image[int(mbox[1]):int(mbox[3]), int(mbox[0]):int(mbox[2])]
            #output_cls = str(Predict1(mask_image1))
            #cv2.putText(frame, output_cls, (int(mbox[2]) - 20, int(mbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = plot_one_box([mbox[0],mbox[1],mbox[2],mbox[3]], frame, (0, 0, 255), str(mbox[-1]))
            #mask_image1 = mask_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # print(mask_image1.shape)
            # print('frame',frame)
            #folder_path = f'{output_path}{mbox[-1]}-{int(int(frame_counter) / 100) + 1}/'
            #print('mask目录：', folder_path)
            #os.makedirs(folder_path, exist_ok=True)
            #cv2.imwrite(f"{folder_path}{frame_counter}.jpg", mask_image1)
            masks = masks.astype(np.int32) * np.array(colors(mbox[-1]))[:, None, None]
            #把mask转成pil格式之后进行图像分类，不会提高太多运行时间
            #print(type(masks))
            #cv2.imwrite('1212_{}.jpg'.format(mbox[-1]),masks)
            masks_all.append(masks)
            num+=1
        #print("num is:",num)
        predictor.reset_image()
        if len(masks_all):
            masks_sum = masks_all[0].copy()
            for m in masks_all[1:]:
                masks_sum += m
        else:
            print("error")
            img = frame.copy()[..., ::-1]
            masks_sum = np.zeros_like(img).transpose(2, 0, 1)

        img = frame.copy()[..., ::-1]
        img = (img * 0.5 + (masks_sum.transpose(1, 2, 0) * 30) % 128).astype(np.uint8)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pic', 1280, 720)
        cv2.imshow('pic', img)
        # print(image.shape)
        # image.transpose((2, 0, 1))
        cv2.waitKey(1)
        return img

class yolov7(base_model):
    
    def post_processing(self, result, im=None, img=None):
        if self.model_path.endswith('pt'):
            result = non_max_suppression(result, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        elif self.model_path.endswith('onnx'):
            result = result[:, 1:]
            ratio, dwdh = letterbox(img, new_shape=tuple(self.imgsz), auto=False)[1:]
            result[:, :4] -= np.array(dwdh * 2)
            result[:, :4] /= ratio
            result[:, [4, 5]] = result[:, [5, 4]] # xyxy, cls, conf => xyxy, conf, cls
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])

        return img, result

class yolov5(base_model):
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs):
        super().__init__(model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs)
    
    def post_processing(self, result, im=None, img=None):
        if self.model_path.endswith('pt'):
            result = non_max_suppression(result, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        elif self.model_path.endswith('onnx'):
            result = non_max_suppression(torch.from_numpy(result), conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]
            result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
            
            for *xyxy, conf, cls in result:
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
        
        return img, result

class yolov8:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs) -> None:
        print(model_path)
        model = YOLO(model_path)
        model.info()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        
        result = next(self.model.predict(source=image, stream=True, iou=self.iou_thres, conf=self.conf_thres, imgsz=self.imgsz, save=False, device=self.device))
        result = result.boxes.data.cpu().detach().numpy()
        for *xyxy, conf, cls in result:
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
        
        return image, result
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self, frame, det_result):
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
        return frame

class rtdetr:
    def __init__(self, model_path, iou_thres, conf_thres, device, names, imgsz, **kwargs) -> None:
        model = RTDETR(model_path)
        model.info()
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        self.__dict__.update(locals())
    
    def __call__(self, data):
        if type(data) is str:
            image = cv2.imdecode(np.fromfile(data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = data
        
        result = next(self.model.predict(source=image, stream=True, iou=self.iou_thres, conf=self.conf_thres, imgsz=self.imgsz, save=False, device=self.device))
        result = result.boxes.data.cpu().detach().numpy()
        for *xyxy, conf, cls in result:
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
        
        return image, result
    
    def track_init(self, track_type):
        from track_utils.byte_tracker import BYTETracker, BaseTrack
        if track_type == 'ByteTrack':
            self.track_opt = ByteTrack_opt()
            self.tracker = BYTETracker(self.track_opt, frame_rate=self.track_opt.fps)
            BaseTrack._count = 0
    
    def track_processing(self, frame, det_result):
        if type(det_result) is torch.Tensor:
            det_result = det_result.cpu().detach().numpy()
        online_targets = self.tracker.update(det_result[:, :5], frame.shape[:2], [640, 640])
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                plot_one_box([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], frame, (0, 0, 255), str(tid))
        return frame

def test_yolov7():
    # read cfg
    with open('yolov7-tiny.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov7(**cfg)
    image_path = '1.jpg'
    # inference
    image, _ = yolo(image_path)
    cv2.imshow('pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_yolov5():
    # read cfg
    with open('yolov5s.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov5(**cfg)
    image_path = '2.jpg'
    # inference
    image, _ = yolo(image_path)
    cv2.imshow('pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_yolov5_track():
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('yolov5s.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov5(**cfg)
    yolo.track_init('ByteTrack')

    cap = cv2.VideoCapture('img.mp4')
    i=1
    while True:
        if(i % 6 == 0):
            ret, frame = cap.read()
            if frame is None:
                break

            image, result = yolo(frame.copy())
            image = yolo.track_processing(i,frame.copy(), result)
            # print(type(result))
            # print(result)
            print(image.size)
            cv2.imshow('pic', image)
            cv2.waitKey(1)
        i+=1


def test_yolov5_track1():
    #from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('F:/pycharmproject/segment-anything/yolo-pyqt/yolov5s.yaml') as f:
         cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # # print cfg
    print(cfg)
    # # init
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = "yolov5_output.mp4"
    fps = 25

    frame_size = (3840,2160)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    yolo = yolov5(**cfg)
    yolo.track_init('ByteTrack')
    #cap = cv2.VideoCapture('img19.mp4')
    cap = cv2.VideoCapture(0)


    i = 1
    Model = model_init()
    while True:
        if (i % 6 == 0):
            ret, frame = cap.read()
            if frame is None:
                break
            frame_size = frame.size

            #frame = frame.transpose(1,0,2)
            #print(frame.shape)
            image, result = onnx_forward1(Model,frame.copy())
            #print(result)
            image = yolo.track_processing(i, frame.copy(), result)
            #print(type(result))
            #print(result)
            #print(image.shape)

        i += 1
        print(i)
    cap.release()
    video_writer.release()


def test_yolov7_track():
    from track_utils.byte_tracker import BYTETracker
    # read cfg
    with open('yolov7-tiny.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # print cfg
    print(cfg)
    # init
    yolo = yolov7(**cfg)
    yolo.track_init('ByteTrack')
    
    cap = cv2.VideoCapture('1.mp4')
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        image, result = yolo(frame.copy())
        image = yolo.track_processing(frame.copy(), result)
        
        cv2.imshow('pic', image)
        cv2.waitKey(20)

if __name__ == "__main__":
    test_yolov5_track1()
