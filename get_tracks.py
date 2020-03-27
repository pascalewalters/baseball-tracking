import pandas as pd
import cv2
import numpy as np


def draw_box(row, frame, colour):
	x1 = int(row['bb_left'])
	y1 = int(row['bb_top'])
	box_w = int(row['bb_width'])
	box_h = int(row['bb_height'])

	cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), colour, 2)


# Read in detections
detections_file = 'detections.csv'
df = pd.read_csv(detections_file, index_col = 0)

baseball_detections = df.loc[df['class'] == 'sports ball']
player_detections = df.loc[df['class'] == 'person']

baseball_frames = list(baseball_detections['frame'])

# Find rows in detections file after first detection that don't have a baseball detection
first_baseball_detection = baseball_frames[0]
last_baseball_detection = baseball_frames[-1]

all_baseball_frames = df.loc[(df['frame'] >= first_baseball_detection) & (df['frame'] <= last_baseball_detection)]
all_baseball_frames = all_baseball_frames['frame'].unique()

# Add rows to baseball_detections dataframe
for f in all_baseball_frames:
	if f not in baseball_frames:
		det = {'frame': f, 'class': 'sports ball', 'bb_left': None,
			'bb_top': None, 'bb_width': None, 'bb_height': None}
		baseball_detections = baseball_detections.append(det, ignore_index = True)

baseball_detections = baseball_detections.sort_values(by = ['frame'], ignore_index = True)
baseball_frames = list(baseball_detections['frame'])

# Find release point (point where baseball bbox is above player bbox)
for f in baseball_frames:
	ball = baseball_detections.loc[baseball_detections['frame'] == f]
	player = player_detections.loc[player_detections['frame'] == f]

	ball_bottom = ball['bb_top'].values[0] + ball['bb_height'].values[0]
	player_top = player['bb_top'].values[0]

	if ball_bottom < player_top:
		release_frame = ball['frame'].values[0]
		break

# Interpolate frames with quadratic until release, then linear
before_release = baseball_detections.loc[baseball_detections['frame'] <= release_frame]
after_release = baseball_detections.loc[baseball_detections['frame'] > release_frame]

before_release = before_release.interpolate(method = 'quadratic')
after_release = after_release.interpolate(method = 'linear')

baseball_detections = pd.concat([before_release, after_release], ignore_index = True)

# Read in video
videopath = '../videos/slomo_1568156854_1_Cam3.mp4'
vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()

out = cv2.VideoWriter('ball_track1.avi',
	cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1], frame.shape[0]))

frame_count = 0
rows = []

while(ret):

	if frame_count in baseball_frames:
		row = baseball_detections.loc[baseball_detections['frame'] == frame_count]
		assert len(row) == 1

		if frame_count <= release_frame:
			draw_box(row, frame, (255, 255, 0))
		else:
			draw_box(row, frame, (0, 255, 255))

		rows.append([frame_count, not (frame_count <= release_frame), 
			row['bb_left'].values[0], row['bb_top'].values[0],
			row['bb_width'].values[0], row['bb_height'].values[0]])
			
	out.write(frame)
	frame_count += 1
	
	ret, frame = vid.read()
	
vid.release()
out.release()

df = pd.DataFrame(rows, columns = ['frame', 'ball_released', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
df.to_csv('ball_track.csv')

