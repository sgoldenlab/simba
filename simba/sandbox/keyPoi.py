import json

import cv2

DATA_PATH = r"C:\troubleshooting\coco_data\annotations.json"
IMAGE_PATH = r"C:\troubleshooting\coco_data\img\FRR_gq_Saline_0624_0.png"

with open(DATA_PATH) as json_data:
    data = json.load(json_data)


annot = data['annotations'][0]
img = cv2.imread(IMAGE_PATH)


top_left = (annot['bbox'][0], annot['bbox'][1])
bottom_right = (int(annot['bbox'][0] + annot['bbox'][2]), int(annot['bbox'][1] + annot['bbox'][3]))

img = cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)
cv2.imshow('sdasd', img)
cv2.waitKey(5000)