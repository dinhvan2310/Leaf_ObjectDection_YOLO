import os
from ultralytics import YOLO
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2

HOME = os.getcwd()

model = YOLO('./leafs-20240205T053521Z-001/leafs/weights/best.pt')

folder_path = './archive/train'
folder_path = './Leaf-OD-OVERALL-1/train/images'

train_path = os.listdir(folder_path)
train_path = [os.path.join(folder_path, i) for i in train_path]

# get 10 images randomly
train_path = random.sample(train_path, 10)

fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for i in range(2):
    for j in range(5):
        img = cv2.imread(train_path[i*5+j])
        results = model(train_path[i*5+j], conf=0.25)
        for r in results:
            for index, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy):
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f'{r.names[int(r.boxes.cls[index])]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        ax[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i, j].axis('off')
plt.show()

