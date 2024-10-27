#total yime escaplied 4:56
import os, random, yaml, argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
random.seed(0)
import cv2
import torch
import numpy as np
import argparse
import sys
sys.path.insert(0,'FastSAM-main')
from fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import cv2

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="cat.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:[0]", help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
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
        promote_list_fastsam = []
        prompt_list = []
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame1 = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                promote_list_fastsam.append([(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c])
                prompt_list.append([(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c,tid])
                content = str(i) + ',' + str(tid) + ',' + str((abs(tlwh[0]/640))*h) + ',' + str((tlwh[1]/640)*c) + ',' + str(
                    ((tlwh[2]) / 640) * h) + ',' + str(((tlwh[3])/640)*c) + ',' + '0' + ',' + '0' + ',' + '0' + '\n'
                #print(content)
                filename = 'car_output.txt'
                # 使用'w'模式打开文件，如果文件不存在则创建
                with open(filename, 'a') as file:
                    # 写入内容
                    file.write(content)
        print("prompt_list_fastsam is: == ",promote_list_fastsam)
        #注意，使用fastsam的时候4k的视频是不能用的
        #current fastsam
        args = parse_args()
        model = FastSAM(args.model_path)
        args.point_prompt = ast.literal_eval(args.point_prompt)
        args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
        args.point_label = ast.literal_eval(args.point_label)
        input = Image.fromarray(frame)
        print(input)
        everything_results = model(
            input,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou
        )
        print(everything_results)
        bboxes = None
        points = None
        point_label = None
        # box_prompt = promote_list_fastsam
        box_prompt = promote_list_fastsam
        prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
        if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=box_prompt)
            bboxes = box_prompt
        elif args.text_prompt != None:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
            points = args.point_prompt
            point_label = args.point_label
        else:
            ann = prompt_process.everything_prompt()
        img = prompt_process.fastplot(
            annotations=ann,
            output_path=args.output + args.img_path.split("/")[-1],
            bboxes=bboxes,
            points=points,
            point_label=point_label,
            withContours=args.withContours,
            better_quality=args.better_quality,
        )
        print("return result is:", img.shape)
        img = cv2.UMat(img)
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > self.track_opt.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.track_opt.min_box_area and not vertical:
                img = plot_one_box([(tlwh[0]/640)*h, (tlwh[1]/640)*c, ((tlwh[0] + tlwh[2])/640)*h, ((tlwh[1] + tlwh[3])/640)*c,tid], img, (0, 0, 255), str(tid))
        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pic', 1280, 720)
        # # 在 'pic' 窗口中显示图像
        cv2.imshow('pic', img)
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
        # cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('pic', 1280, 720)
        # # # 在 'pic' 窗口中显示图像
        # video_writer.write(image)
        # cv2.imshow('pic', image)
        #print(image.shape)
        #image.transpose((2, 0, 1))
        # cv2.waitKey(1)
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

test_yolov5_track1()
