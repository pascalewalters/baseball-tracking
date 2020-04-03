import cv2
import os
import numpy as np
import itertools


def normalize(im):
	rng = np.max(im) - np.min(im)
	amin = np.min(im)
	return (im - amin) * 255 / rng


ball_clip_dir = '../ball_track'
frames_names = os.listdir(ball_clip_dir)
frames_names.sort()
frames = []

largest_frame = (0, 0)

for f in frames_names:
	im = cv2.imread(os.path.join(ball_clip_dir, f))
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	frames.append(im)

	if im.shape > largest_frame:
		largest_frame = im.shape

largest_frame = (max(largest_frame), max(largest_frame))

resized_frames = []
circle_idx = []

for idx, f in enumerate(frames):
	resized_frame = cv2.resize(f, largest_frame)

	img = cv2.medianBlur(resized_frame, 5)
	cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
		param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)

	if circles is not None:
		assert len(circles) == 1
		circles = np.uint16(np.around(circles))
		mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
		for i in circles[0, :]:
			cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
		img = cv2.bitwise_and(img, img, mask = mask)

		circle_idx.append(idx)
		resized_frames.append(img)

# comparisons = list(itertools.product(resized_frames, repeat = 2))
# comparisons = list(itertools.combinations(range(len(resized_frames)), 2))
comparisons = list(itertools.combinations(range(len(circle_idx)), 2))

diffs = []

for im1, im2 in comparisons:

	if im1 == im2 + 1 or im1 == im2 - 1:
		continue

	# Normalize images?
	im1 = normalize(resized_frames[im1])
	im2 = normalize(resized_frames[im2])

	diff = np.subtract(im1, im2)
	m_norm = np.sum(np.square(diff))

	diffs.append(m_norm)

	# exit()

diffs = np.array(diffs)
sorted_idx = np.argsort(diffs)

print(diffs[sorted_idx[1]])

print(circle_idx[comparisons[sorted_idx[1]][0]])
print(circle_idx[comparisons[sorted_idx[1]][1]])

cv2.imshow('blah', frames[circle_idx[comparisons[sorted_idx[1]][0]]])
cv2.imshow('blah2', frames[circle_idx[comparisons[sorted_idx[1]][1]]])
cv2.waitKey()

# diffs = np.array(diffs, dtype = np.float)
# diffs *= 255.0 / np.float(diffs.max())
# diffs = diffs.reshape((len(resized_frames), len(resized_frames)))
# diffs = diffs.astype(np.uint8)

# cv2.imshow('blah', diffs)
# cv2.waitKey()
