import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np

import matplotlib.pyplot as plt

#importing dependencies related to image transformations
import torchvision
from torchvision import transforms
from PIL import Image

#importing dependencies related to data loading
from torchvision import datasets
from torch.utils.data import DataLoader

import YOLO

def detect_face(img):
    face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)

    faces=face_clsfr.detectMultiScale(gray,1.3,3)
    print(f'Number of faces found = {len(faces)}')
    if len(faces) == 0:
        return None

    

    x,y,w,h = 0, 0, 0, 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    

    face_img=img[y:y+w,x:x+w]
    #plt.imshow(face_img)
    #cv2.imshow("face found          ",face_img)
    #cv2.waitKey()
    return face_img




def detect_mask(face_img):
    if face_img is None or face_img.all() == None:
        return "No mask detected because face found is 0."
    model = torch.load("model.pth")
    model.eval()
    transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])])
    transform = transforms.ToTensor()
    face_img = transform(face_img)
    face_img = torch.stack([face_img])
    model.eval()
    result=model(face_img)
    label = 1
    _, predicted = torch.max(result.data, 1)
    if predicted==label:
        print("Prediction ", predicted)
        return "No Mask"
    else:
        print("Prediction ", predicted)
        return"Mask"


    

def Test():
    img = cv2.imread("demo1.jpg")
    #half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    #cv2.imshow("Original image",half)

    detect = YOLO.humanDetect(img)
    cv2.imshow("human image",detect)
    cv2.waitKey()

    detect = detect_face(detect)
    cv2.imshow("face", detect)
    cv2.waitKey()

#Test()
    

