import pandas as pd
import cv2


def crop_box(row, frame):
	x1 = int(row['bb_left'] - 5)
	y1 = int(row['bb_top'] - 5)
	box_w = int(row['bb_width'] + 10)
	box_h = int(row['bb_height'] + 10)

	return frame[y1:y1 + box_h, x1:x1 + box_w]


track_file = 'ball_track.csv'
df = pd.read_csv(track_file, index_col = 0)

baseball_frames = list(df['frame'])

# Read in video
videopath = '../videos/slomo_1568156854_1_Cam3.mp4'
vid = cv2.VideoCapture(videopath)
ret, frame = vid.read()

frame_count = 0

while(ret):

	if frame_count in baseball_frames:
		row = df.loc[df['frame'] == frame_count]
		assert len(row) == 1

		if row['ball_released'].values[0]:
			cropped = crop_box(row, frame)
			cv2.imwrite('../ball_track/{}.jpg'.format(str(frame_count).zfill(4)), cropped)

	frame_count += 1
	
	ret, frame = vid.read()
	
vid.release()

