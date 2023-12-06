import cv2
import numpy as np
from math import sqrt
import cv2 as cv
import os
import time
import math
import gc
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import cv2
import numpy as np
from SIM import CustomDataset, SiameseNetwork


def getYOLOOutput(img, model ,CONFIDENCE=0.5):
    resultlist = model.predict(source=img, imgsz=512)
    results = resultlist[0]
    print(
        f"[INFO] YOLO took {results.speed['preprocess'] + results.speed['inference'] + results.speed['postprocess']} ms ")
    boxes = []
    dfg = []
    pc = []
    for box in results.boxes:
        if box.conf > CONFIDENCE:
            boxes.append(box)
    for box in boxes:
        if box.cls == 0:
            dfg.append([box.xywh[0][0], box.xywh[0][1], box.conf])
        else:
            pc.append([box.xywh[0][0], box.xywh[0][1], box.conf])

    return dfg, pc


def onePoint(x, y, angle):
    X = x * math.cos(angle) + y * math.sin(angle)
    Y = y * math.cos(angle) - x * math.sin(angle)
    return [int(X), int(Y)]


def extractROI(img, dfg, pc):
    (H, W) = img.shape[:2]
    if W > H:
        im = np.zeros((W, W, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = W
    else:
        im = np.zeros((H, H, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = H

    center = (edge / 2, edge / 2)

    x1 = float(dfg[0][0])
    y1 = float(dfg[0][1])
    x2 = float(dfg[1][0])
    y2 = float(dfg[1][1])
    x3 = float(pc[0][0])
    y3 = float(pc[0][1])

    x0 = (x1 + x2) / 2
    y0 = (y1 + y2) / 2

    unitLen = math.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    k1 = (y1 - y2) / (x1 - x2)  # line AB
    b1 = y1 - k1 * x1

    k2 = (-1) / k1
    b2 = y3 - k2 * x3

    tmpX = (b2 - b1) / (k1 - k2)
    tmpY = k1 * tmpX + b1

    vec = [x3 - tmpX, y3 - tmpY]
    sidLen = math.sqrt(np.square(vec[0]) + np.square(vec[1]))
    vec = [vec[0] / sidLen, vec[1] / sidLen]
    # print(vec)

    if vec[1] < 0 and vec[0] > 0:
        angle = math.pi / 2 - math.acos(vec[0])
    elif vec[1] < 0 and vec[0] < 0:
        angle = math.acos(-vec[0]) - math.pi / 2
    elif vec[1] >= 0 and vec[0] > 0:
        angle = math.acos(vec[0]) - math.pi / 2
    else:
        angle = math.pi / 2 - math.acos(-vec[0])
    # print(angle/math.pi*18)

    x0, y0 = onePoint(x0 - edge / 2, y0 - edge / 2, angle)

    x0 += edge / 2
    y0 += edge / 2

    M = cv.getRotationMatrix2D(center, angle / math.pi * 180, 1.0)
    tmp = cv.warpAffine(im, M, (edge, edge))
    ROI = tmp[int(y0 + unitLen / 2):int(y0 + unitLen * 3), int(x0 - unitLen * 5 / 4):int(x0 + unitLen * 5 / 4), :]
    ROI = cv.resize(ROI, (224, 224), interpolation=cv.INTER_CUBIC)
    return ROI




def ROI_pretreat(ROI):
    gray_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    blurred_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    rslt = blurred_img

    return rslt

'''
read a  image and return the image's ROI
imgpath:the image input
CONFIDENCE: the minimum confidence you accept to check whether the ROI region it recognize is true.

return : return a ROI nparray( a image can output by cv.imwrite())
'''
def getinput(imgpath,model,CONFIDENCE=0.5):

    imgPath1 = imgpath



    img1 = cv.imread(imgPath1)

    dfg1, pc1 = getYOLOOutput(img1, model,CONFIDENCE)
    gc.collect()

    '''
    Step 2: Construct the local coordinates based on detected points.
    '''

    if len(dfg1) < 2:
        print('Detect fail. Please re-take photo and input it.')
    else:
        if len(dfg1) > 2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg1) - 1):
                for j in range(i + 1, len(dfg1)):
                    d = sqrt(pow(dfg1[i][0] - dfg1[j][0], 2) + pow(dfg1[i][1] - dfg1[j][1], 2))
                    if d > maxD:
                        tmpdfg = [dfg1[i], dfg1[j]]
                        maxD = d
            dfg1 = tmpdfg

        pc1 = sorted(pc1, key=lambda x: x[-1], reverse=True)

    ROI1 = extractROI(img1, dfg1, pc1)
    return ROI1

'''
to compare 2 palm similarity,return True or False represented the palm comes from the same palm
ROI1: the ROI region from a image
ROI2: the ROI region from a image

return : true or false
'''
def compareROI(ROI1,ROI2):
    RGB_MEAN = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
    RGB_STD = [0.5, 0.5, 0.5]
    model = SiameseNetwork()

    model.load_state_dict(torch.load('./curr_3.pth', map_location=torch.device('cpu')))
    test_transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=Image.NEAREST),  # smaller side resized
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD), ])

    ROI1 = Image.fromarray(ROI1)
    ROI2 = Image.fromarray(ROI2)

    ROI1 = test_transform(ROI1)
    ROI2 = test_transform(ROI2)

    ROI1 = ROI1.unsqueeze(0)
    ROI2 = ROI2.unsqueeze(0)

    output = model(ROI1, ROI2)
    similarity = output[0][0]
    if similarity > 0.7:
        # similarity=1
        print('Palmprint Verification Success.')
        print(f"similarity is {similarity}")
        return True
    else:
        # similarity=0
        print('Palmprint Verification Fail.')
        return False


if __name__ == '__main__':

    '''
    step1: load the yolo model
    '''
    model = YOLO('./best.onnx', task='detect')
    print("[INFO] loading YOLO from disk...")

    '''
    step2: change the image into ROI image
    '''
    imgPath1 = 'src.jpg'
    imgPath2 = 'dst.jpg'
    ROI1 = getinput(imgPath1,model)
    ROI2 = getinput(imgPath2,model)

    '''
    step 3: compare 2 ROI
    '''
    result = compareROI(ROI1,ROI2)

