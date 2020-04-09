# Baseball tracking during pitching ⚾️🐦

### Step 1: Baseball detection

The bounding box for the baseball is determined using a pre-trained YOLOv3 network (MS COCO dataset has a class for sports ball).
YOLOv3 is used because it is a small network that achieves good performance for detection.
The architecture is well-known and has many implementations and pre-trained networks available ([example](https://github.com/cfotache/pytorch_objectdetecttrack)).
Faster-RCNN could also be used, with potentially better results, as it works better for small objects.
However, it would likely have slower inference time.

A deep learning approach was selected because it achieves better performance than classical image processing techinques. It is also more likely to work from a variety of view points and illuminations, due to the wide variety of training samples in the dataset. Further improvements to the detection method could be done with a larger, more accurate detection network, which would also increase training and inference time. 

As seen in the demo video, the network does not perform well when the ball passes in front of the pitcher's leg. This is likely because there is not much contrast between the white ball and the white pants.

To run the detections script:

```
cd config/
./download_weights.sh
cd ..
python get_detections.py -i <path to input video> --save-video
```

This generates a csv file of detections and an output video of the overlayed bounding boxes.

If the detections are generated on the CPU the inference time can be quite long.

### Step 2: Baseball tracking and release point

Assume for tracking that there is only one baseball in the video and that this baseball is being used for the pitch.
Define the release point as the instant the bounding box of the baseball is above the player's bounding box.

![release point](img/release_point.gif)

The ball is missed in some detections. These are interpolated quadratically before the release point and linearly after the release point. These methods were selected based on qualitative results. For some pitches, this interpolation is not particularly effective, which may be due to the camera angle or the pitch type. Further research in this area would be interesting.

To run the tracking script:

```
python get_tracks.py -i <path to input video> --save-video
```

This generates a csv file of ball tracks with a column for the ball release point.
It also produces a video with the bounding boxes of the ball, coloured blue if it is before the release point and yellow if it is after the release point.

### Step 3: Spin rate and spin axis

[Ijiri *et al.*](http://www.sic.shibaura-it.ac.jp/~ijiri/files/ijiri_spinEstimation_SIVP2017.pdf) propose a method for calculating the spin rate and spin direction of pitched baseballs independently. From the previous step, bounding boxes of the ball as it travels through the air after being pitched have been determined.

The spin rate is defined in terms of the spin period. This is the time it takes for the ball, as it is rotating, to return to a similar appearance (i.e., similar position) as a previous time step. The spin rate would be the reciprocal of this spin period.
In order to compare time steps of the ball as it is pitched appearance descrptors are required. Ijiri *et al.* compare absolute pixel values with a weighting function to remove the effects of illumination on the ball. This requires that all ball shapes have the same dimensions. Since the ball increases in size as it approaches the camera, these effects must be controlled. 

Unfortunately, I was unable to get an implementation of this methodology working. The descriptors were not effective with this dataset. I also tried to use mean pixel values to measure the differences between frames, but this was also ineffective. Given more time, I think it would be interesting to try additional features such as histogram of oriented gradients or deep features. If it were possible to obtain a colour video, this may be more successful as the red seams of the baseball or other patterns on the ball could be extracted without the interference of shadows.

The spin axis is calculated by performing texture registration. A frame of the ball is warped to a frame of a ball in a later frame, using orthagonal projection and 3D rotation. The best angle that represents this transformation is selected as the spin axis for the pitch.

### Further research questions

An interesting research question for the pitcher video would be to calculate pose. Using deep learning, the positions of joints can be estimated, which can be used to predict the position of the player's body segments in space. Combining pose data with ball trajectory information (e.g., speed, spin rate) would allow for biomechanical analysis that could determine optimal technique and provide player feedback or assessment. 

