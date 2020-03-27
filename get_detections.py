# Code inspired by https://github.com/cfotache/pytorch_objectdetecttrack

from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import cv2
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor

def detect_image(img):
	# scale and pad image
	ratio = min(img_size/img.size[0], img_size/img.size[1])
	imw = round(img.size[0] * ratio)
	imh = round(img.size[1] * ratio)
	img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
		 transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
						(128,128,128)),
		 transforms.ToTensor(),
		 ])
	# convert image to Tensor
	image_tensor = img_transforms(img).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input_img = Variable(image_tensor.type(Tensor))
	# run inference on the model and get detections
	with torch.no_grad():
		detections = model(input_img)
		detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
	return detections[0]


colours = {'person': (255, 0, 255), 'sports ball': (255, 255, 0), 'baseball glove': (0, 255, 255)}

videopath = '../videos/slomo_1568156854_1_Cam3.mp4'
vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1], frame.shape[0]))

frame_count = 0
rows = []

while(ret):
# for ii in range(40):
	frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pilimg = Image.fromarray(frame1)
	detections = detect_image(pilimg)

	img = np.array(pilimg)
	pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
	pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
	unpad_h = img_size - pad_y
	unpad_w = img_size - pad_x
	if detections is not None:
		unique_labels = detections[:, -1].cpu().unique()

		for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
			box_h = ((y2 - y1) / unpad_h) * img.shape[0]
			box_w = ((x2 - x1) / unpad_w) * img.shape[1]
			y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
			x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

			c = classes[int(cls_pred)]
			colour = colours[c]

			cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), colour, 2)
			cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour)

			rows.append([frame_count, c, x1.numpy(), y1.numpy(), box_w.numpy(), box_h.numpy()])
			
	out.write(frame)
	frame_count += 1
	
	ret, frame = vid.read()
	
vid.release()
out.release()

df = pd.DataFrame(rows, columns = ['frame', 'class', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
df.to_csv('detections.csv')


