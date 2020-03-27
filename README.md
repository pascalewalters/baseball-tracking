# Baseball tracking during pitching ⚾️🐦

**Step 1:** Baseball detection

The bounding box for the baseball is determined using a pre-trained YOLOv3 network (MS COCO dataset has a class for sports ball).
YOLOv3 is used because it is a small network that achieves good performance for detection.
The architecture is well-known and has many implementations and pre-trained networks available (e.g., https://github.com/eriklindernoren/PyTorch-YOLOv3).
Faster-RCNN could also be used, with potentially better results, as it works better for small objects.

```
python get_detections.py
```

**Step 2:** Baseball tracking

SORT algorithm