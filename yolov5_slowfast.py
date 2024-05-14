# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse, os, sys, platform, cv2, numpy
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
# import torch.backends.cudnn as cudnn

ROOT = Path(__file__).resolve().parents[0]  # YOLOv5 root directory
if str('yolov5') not in sys.path:
    sys.path.append(str('yolov5'))  # add ROOT to PATH

# from models.yolo import Model
# from utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors

from slowfast_predict import AVAVisualizerWithPrecomputedBox

class yolov5_inference():
    def __init__(self, 
            device='',
            weights='yolov5s.pt',  # model.pt path(s)
            # source='yolov5_6_2_0/data/images',  # file/dir/URL/glob, 0 for webcam
            # data='data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            classes=[0],
        ):
        # select device
        device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
        # os.environ['CUDA_VISIBLE_DEVICES'] = device 
        self.device = torch.device(device)

        # Load model
        # model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
        self.model = torch.load(weights, map_location='cpu')  # load

        self.model = (self.model.get('ema') or self.model['model']).to(self.device).float()  # FP32 model
        if not hasattr(self.model, 'stride'):
            self.model.stride = torch.tensor([32.])  # compatibility update for ResNet etc.
        self.model.eval()  # model in eval mode

        self.imgsz = [640, 640]  # check image size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

    def __call__(self, im0):
        # Padded resize
        im = letterbox(im0, self.imgsz, stride=32, auto=True)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = numpy.ascontiguousarray(im)

        # Run inference
        # im = torch.zeros(*imgsz, dtype=torch.float, device=device)  # input
        # for _ in range(1):
        #     model(im)  # warmup

        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False, max_det=1000)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        outputs = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, c in reversed(det):
                    # outputs.append([int(c), *list(map(int, xyxy)), float(conf)])
                    outputs.append([*list(map(float, xyxy))])
        return outputs



model_coco    = yolov5_inference(weights='yolov5/yolov5s.pt', classes=[0])

slowfast = AVAVisualizerWithPrecomputedBox()

# Dataloader
cap = cv2.VideoCapture("1046569042_da2-1-192_1.mp4")
was_read=True
while was_read:
    frames=[]
    seq_length=64
    while was_read and len(frames) < seq_length:
        was_read, frame = cap.read()
        if was_read:
            frames.append(frame)

    if len(frames) == seq_length:
        bboxes = model_coco(frames[seq_length//2])
        if bboxes is not None:
            slowfast_pred = slowfast.predict(frames, bboxes) 
            print(slowfast_pred)



 