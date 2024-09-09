# -*- coding: utf-8 -*-
#!/usr/bin/python3

# update 2022. 08. 08
# DEEPI.INC, Jongwon Kim
# YOLO V5 기반 객체 탐지

import random
color = [(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)) for j in range(80)]
# 필수 라이브러리
import cv2
import copy
import os
import glob
import numpy as np
from numpy import random
# import pytictoc
# YOLO V5 라이브러리
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device
#%% 딥러닝 모델 로드
def loadModel():
    wdir = 'best.pt'
    # 가중치 파일 경로 확인을 위한 절대 경로
    # GPU 확인
    device = select_device()
    # 구현 라이브러리
    backend = 'pytorch'
    # 모델 로드
    model = attempt_load(wdir, device=device)  # load FP32 model
    stride = int(model.stride.max()) 
    # 입력 이미지 크기
    imgsz = check_img_size(4032, s=stride)
    # imgsz = 2000
    # 학습 데이터 클래스 정보
    names = model.module.names if hasattr(model, 'module') else model.names  # class names
    # 객체 경계상자 색상 
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    # 모델 초기화 테스트
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    print('YOLO V5 모델 초기화 완료')
    return model

#%% 이미지 패딩 메서드
def letterbox(img, new_shape=(4032, 4032), color=(114, 114, 114), scaleFill=False, scaleup=True, stride=32):
    
    # Resize and pad image
    # 원본 이미지 해상도
    shape = img.shape[:2] 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 가로 세로 여백 보안
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)

#%% 리스케일링 메서드
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) 
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]  

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

#%% 리스케일링 메서드
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0]) 
        
        
#%% MAIN
def img_test():   
    # 모델 로드
    dis = 'DATASET/'
    # 저장 경로
    dirs = 'RESULT/'
    
    model = loadModel()
    # 카메라 영상 정보
    img_pth = sorted(glob.glob(dis+'/*.jpg'))
   
    #%% RUN
    for im in img_pth:
        # 이미지 캡처
        img = cv2.imread(im)
        # 원본 복사
        oimg = copy.deepcopy(img)
        # 이미지 전처리
        img = letterbox(img)[0]
        # img = cv2.resize(img,dsize=(2016,2016))
        h,w,ch = oimg.shape
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to('cuda')
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        # NMS
        pred = non_max_suppression(pred,0.15, 0.65, multi_label=False, agnostic=False)
        lb = []
        for i,det in enumerate(pred):
            if len(det):
                # 경계상자 리스케일링
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4], oimg.shape).round()
                for j in det:
                    if  True:
                        # 화면에 출력 할 경계상자
                        cv2.rectangle(oimg,(int(j[0]),int(j[1])),
                                      (int(j[2]),int(j[3])),color[int(j[5])],2)



                        x1, y1 = int(j[0]), int(j[1])
                        x2, y2 = int(j[2]), int(j[3])
                        cx, cy = round(float(x1 + abs(x1-x2)/2)/w,5), round(float(y1 + abs(y1-y2)/2)/h,5)
                        cw, ch = round(float(abs(x1-x2)/w),5), round(float(abs(y1-y2)/h),5)
                
                        lb.append(['%d %04f %04f %04f %04f %04f \n'%(int(j[5]),j[4],cx,cy,cw,ch)])
                   
        newpth = os.path.join(dirs,os.path.basename(im))            
        f = open(newpth.replace('jpg','txt'),'w')
        if len(lb) == 1:
            f.write(lb[0][0])
        else:
            for l in lb:
                f.write(l[0])
        f.close()        
                        
        cv2.imwrite(newpth,oimg)
        cv2.imshow('test',oimg)
        if cv2.waitKey(1) > 0 :
            break
img_test()