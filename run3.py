from segment_anything import build_sam, SamPredictor
import os
import cv2
import numpy as np
from collections import defaultdict
import time
from classification.predict import Predict
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
sam_checkpoint = "checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
sam.to(device=device)
# image = cv2.imread('/home/hadoop-vacv/yanfeng/data/dancetrack/train/dancetrack0001/img1/00000109.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# predictor.set_image(image)

# bbox = np.array([0,0,100,100], dtype=np.int32)

# masks, _, _ = predictor.predict(box=bbox)

# masks

# input_path = './img9'
# targets = [f for f in os.listdir(os.path.join(input_path, 'img1')) if
#            not os.path.isdir(os.path.join(input_path, 'img1', f))]
# targets = [os.path.join(input_path, 'img1', f) for f in targets]
# targets.sort()

bboxes_all = defaultdict(list)
#gt_path = os.path.join(input_path, 'gt', 'gt.txt')
gt_path = "./yolo-pyqt/car_output.txt"
#print(gt_path)
#print(gt_path)
# gt_path = os.path.join('/home/hadoop-vacv/yanfeng/project/MOTRv2/MOTRv3/exps/motrv2ch_uni5cost6g/run2/tracker0', 'dancetrack0004.txt')
for l in open(gt_path):
    #print(l)
    t, i, *xywh, mark, label = l.strip().split(',')[:8]
    print(t,i,mark,label)
    t, i, mark, label = map(int, (t, i, mark, label))
    if mark == 1:
        continue
    if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
        continue
    else:
        crowd = False
    x, y, w, h = map(int, map(float, (xywh)))
    print(x,y,w,h)
    bboxes_all[t].append([int(x), int(y),int(x+w),int(y+h), i])
    #bboxes_all[t].append([int((x/640)*1920), int((y/640)*1080),  int((w/640)*1920), int((h/640)*1080), i])


fps = 25
size = (3840, 2160)
videowriter = cv2.VideoWriter('tmp.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

video_path = "./yolo-pyqt/img11.mp4"
video = cv2.VideoCapture(video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)
output_video = 'pig_video.mp4'
video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_counter = 0
while True:
    # 读取下一帧
    ret, frame = video.read()

    # 如果读取成功，则处理帧
    if ret:
        # 帧计数器增加
        #print(frame_counter)
        frame_counter += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imshow('ss',frame)
        masks_all = []
        #print(os.path.basename(t)[:-4])
        bboxes = np.array(bboxes_all[frame_counter])
        predictor.set_image(frame)
        output_path = './img10/img10/'
        for bbox in bboxes:
            print(bbox)
            masks, iou_predictions, low_res_masks = predictor.predict(box=bbox[:4])
            index_max = iou_predictions.argsort()[0]
            masks = np.concatenate(
                [masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)]],
                axis=0)
            masks1 = masks
            if masks1.dtype != np.uint8:
                masks1 = (masks1 * 255).astype(np.uint8)  # 乘以 255 并转换为 uint8
                masks1 = 255 - masks1
            # 保存掩膜图像
            mask_image = masks1[index_max]  # 假设我们只对单一的掩膜图像感兴趣
            mask_image1 = mask_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # print(mask_image1.shape)
            # print('frame',frame)
            folder_path = f'{output_path}{bbox[-1]}-{int(int(frame_counter) / 100) + 1}/'
            print('mask目录：',folder_path)
            os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(f"{folder_path}{frame_counter}.jpg", mask_image1)
            masks = masks.astype(np.int32) * np.array(colors(bbox[4]))[:, None, None]
            # cv2.imwrite('{}.jpg'.format(bbox[-1]),masks)
            masks_all.append(masks)
        predictor.reset_image()

        if len(masks_all):
            masks_sum = masks_all[0].copy()
            for m in masks_all[1:]:
                masks_sum += m
        else:
            print("error")
            img = image.copy()[..., ::-1]
            masks_sum = np.zeros_like(img).transpose(2, 0, 1)

        img = image.copy()[..., ::-1]
        img = (img * 0.5 + (masks_sum.transpose(1, 2, 0) * 30) % 128).astype(np.uint8)
        for bbox in bboxes:
            folder_path = f'{output_path}{bbox[-1]}-{int(int(frame_counter) / 100) + 1}/'
            os.makedirs(folder_path, exist_ok=True)
            output_cls = Predict(f"{folder_path}/{frame_counter}.jpg")
            output_cls = str(output_cls)
            bbox = list(bbox)
            bbox.append(output_cls)
            bbox = np.array(bbox)
            # print(bbox)
            txt_output_path = f"{output_path}{bbox[-2]}-{int(int(frame_counter) / 100) + 1}.txt"
            with open(txt_output_path, 'a') as file:
                # 写入数字1和2，然后添加一个空格
                # file.write(str(((xmin + width) / 2) / 1920))
                file.write(str(((int(bbox[0]) + int(bbox[2])) / 2) / 3840))
                file.write(',')
                # file.write(str(((ymin + height) / 2) / 1080))
                file.write(str(((int(bbox[1]) + int(bbox[3])) / 2) / 2160))
                file.write('\n')
                # 写入换行符，以便开始新的一行
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=3)
            i = int(bbox[-2])
            text = 'id:' + str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (int(bbox[0]), int(bbox[1])), font, 1, (0, 255, 0), 2)
            # print(bbox[-1])
            output_class = str(bbox[-1])
            cv2.putText(img, output_class, (int(bbox[2]) - 20, int(bbox[1])), font, 1, (0, 255, 0), 2)
        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('pic', 1280, 720)
        cv2.imshow('pic',img)
        cv2.waitKey(20)
        video_writer.write(img)

        cv2.imwrite(f"{output_path}/{frame_counter}.jpg", img)
        print('picture目录：', f"{output_path}/{frame_counter}.jpg")

    video_writer.release()






# for t in targets:
#     print(f"Processing '{t}'...")
#     image = cv2.imread(t)
#     frame1 = t.split('\\')[-1]  # 这会得到 '113.jpg'
#     frame = frame1.split('.')[0]  # 这会得到 '113'
#     if image is None:
#         print(f"Could not load '{t}' as an image, skipping...")
#         continue
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     masks_all = []
#     print(os.path.basename(t)[:-4])
#     bboxes = np.array(bboxes_all[int(os.path.basename(t)[:-4])])
#     # predictor.set_image(image)
#     # masks, _, _ = predictor.predict(box=bboxes[:, :4])
#     predictor.set_image(image)
#     output_path = './img11/img11/'
#     for bbox in bboxes:
#         #print(bbox)
#         masks, iou_predictions, low_res_masks = predictor.predict(box=bbox[:4])
#         index_max = iou_predictions.argsort()[0]
#         masks = np.concatenate(
#             [masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)], masks[index_max:(index_max + 1)]],
#             axis=0)
#         masks1 = masks
#         if masks1.dtype != np.uint8:
#             masks1 = (masks1*255).astype(np.uint8)  # 乘以 255 并转换为 uint8
#             masks1 = 255 - masks1
#         # 保存掩膜图像
#         mask_image = masks1[index_max]  # 假设我们只对单一的掩膜图像感兴趣
#         mask_image1 = mask_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
#         #print(mask_image1.shape)
#         #print('frame',frame)
#         folder_path = f'{output_path}{bbox[-1]}-{int(int(frame)/100)+1}/'
#         os.makedirs(folder_path, exist_ok=True)
#         cv2.imwrite(f"{folder_path}{frame}.jpg", mask_image1)
#         masks = masks.astype(np.int32) * np.array(colors(bbox[4]))[:, None, None]
#         #cv2.imwrite('{}.jpg'.format(bbox[-1]),masks)
#         masks_all.append(masks)
#     predictor.reset_image()
#
#     if len(masks_all):
#         masks_sum = masks_all[0].copy()
#         for m in masks_all[1:]:
#             masks_sum += m
#     else:
#         masks_sum = np.zeros_like(img).transpose(2, 0, 1)
#
#     img = image.copy()[..., ::-1]
#     img = (img * 0.5 + (masks_sum.transpose(1, 2, 0) * 30) % 128).astype(np.uint8)
#     for bbox in bboxes:
#         folder_path = f'{output_path}{bbox[-1]}-{int(int(frame) / 100) + 1}/'
#         os.makedirs(folder_path, exist_ok=True)
#         output_cls = Predict(f"{folder_path}{frame}.jpg")
#         output_cls = str(output_cls)
#         bbox = list(bbox)
#         bbox.append(output_cls)
#         bbox = np.array(bbox)
#         # print(bbox)
#         txt_output_path = f"{output_path}{bbox[-2]}-{int(int(frame) / 100) + 1}.txt"
#         with open(txt_output_path, 'a') as file:
#             # 写入数字1和2，然后添加一个空格
#             # file.write(str(((xmin + width) / 2) / 1920))
#             file.write(str(((int(bbox[0]) + int(bbox[2])) / 2) / 2560))
#             file.write(',')
#             # file.write(str(((ymin + height) / 2) / 1080))
#             file.write(str(((int(bbox[1]) + int(bbox[3])) / 2) / 1440))
#             file.write('\n')
#             # 写入换行符，以便开始新的一行
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=3)
#         i = int(bbox[-2])
#         text = 'id:' + str(i)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, text, (int(bbox[0]), int(bbox[1])), font, 1, (0, 255, 0), 2)
#         # print(bbox[-1])
#         output_class = str(bbox[-1])
#         cv2.putText(img, output_class, (int(bbox[2]) - 20, int(bbox[1])), font, 1, (0, 255, 0), 2)
#
#     cv2.imwrite(f"{output_path}{frame}.jpg", img)
#
#     videowriter.write(img)
#
# videowriter.release()





