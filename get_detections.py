################################
# Pascale Walters
# pascale.walters@uwaterloo.ca
################################

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
import argparse


def detect_image(img):
	# Code from https://github.com/cfotache/pytorch_objectdetecttrack

	# scale and pad image
	ratio = min(img_size / img.size[0], img_size / img.size[1])
	imw = round(img.size[0] * ratio)
	imh = round(img.size[1] * ratio)
	img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
		 transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
						(128,128,128)),
		 transforms.ToTensor(),
		 ])
	# convert image to Tensor
	image_tensor = img_transforms(img).float()
	image_tensor = image_tensor.unsqueeze_(0).to(device = device)
	# run inference on the model and get detections
	with torch.no_grad():
		detections = model(image_tensor)
		detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
	return detections[0]


# Get input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input-video', '-i', help = 'Path to the video to analyze', required = True)
parser.add_argument('--output-dir', '-o', help = 'Path to the output directory', default = 'output')
parser.add_argument('--save-video', action = 'store_true', help = 'Store output detections in a video')
parser.add_argument('--config-dir', help = 'Path to directory that contains model weights', default = 'config')

args = parser.parse_args()

# Make output directory if it does not exist
if not os.path.isdir(args.output_dir):
	os.makedirs(args.output_dir)

# Ensure that the input video is in .mp4 format
if 'mp4' not in os.path.basename(args.input_video):
	print('Input video is not in mp4 format.')
	exit()

config_path = os.path.join(args.config_dir, 'yolov3.cfg')
weights_path = os.path.join(args.config_dir, 'yolov3.weights')
class_path = os.path.join(args.config_dir, 'coco.names')
img_size = 416
conf_thres = 0.8
nms_thres = 0.4

num_gpus = torch.cuda.device_count()
device = 'cuda' if num_gpus > 0 else 'cpu'

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device = device)
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor

colours = {'person': (255, 0, 255), 'sports ball': (255, 255, 0), 'baseball glove': (0, 255, 255),
			'baseball bat': (125, 255, 0)}

videopath = args.input_video
vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()

output_csv_path = os.path.join(args.output_dir, 
		os.path.basename(videopath).replace('.mp4', '_detections.csv'))

if args.save_video:
	output_video_path = os.path.join(args.output_dir, 
		os.path.basename(videopath).replace('.mp4', '_detections.avi'))
	out = cv2.VideoWriter(output_video_path, 
		cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1], frame.shape[0]))

frame_count = 0
rows = []

while(ret):
	# Get detections from the frame
	frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pilimg = Image.fromarray(frame1)
	detections = detect_image(pilimg).cpu()

	img = np.array(pilimg)
	pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
	pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
	unpad_h = img_size - pad_y
	unpad_w = img_size - pad_x

	# If the network finds detections, save them
	if detections is not None:
		unique_labels = detections[:, -1].cpu().unique()

		for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
			box_h = ((y2 - y1) / unpad_h) * img.shape[0]
			box_w = ((x2 - x1) / unpad_w) * img.shape[1]
			y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
			x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

			c = classes[int(cls_pred)]
			if c not in colours:
				continue
			colour = colours[c]

			# Draw a rectangle on the frame
			cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), colour, 2)
			cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour)

			# Add the detection to the csv
			rows.append([frame_count, c, x1.numpy(), y1.numpy(), box_w.numpy(), box_h.numpy()])
	
	if args.save_video:		
		out.write(frame)
	frame_count += 1
	
	ret, frame = vid.read()
	
vid.release()
if args.save_video:
	out.release()

# Save the detections to a CSV
df = pd.DataFrame(rows, 
	columns = ['frame', 'class', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
df.to_csv(output_csv_path)


