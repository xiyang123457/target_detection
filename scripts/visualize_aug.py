import sys
import os
sys.path.insert(0, r"D:\target_detection")     # 让 scripts 可被导入

import cv2
import numpy as np
from scripts.VOC_dataset import VOCDataset, VOC_CLASSES
from scripts.augment import build_train_transform

root = r"D:\target_detection\data\VOCdevkit\VOC2012"
ds = VOCDataset(root,split="train",train = True)
t =build_train_transform()

out_dir = r"D:\target_detection\outputs"
os.makedirs(out_dir, exist_ok=True)

stems = ["2007_000027", "2007_000032", "2008_007069"]
for stem in stems:
    img = cv2.imread(os.path.join(ds.jpeg_dir, f"{stem}.jpg"))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    boxes, labels = ds._parse_xml(stem)
    for tun in range(3):        #每张图跑3次
        out = t(image=img,bboxes=boxes.tolist(),labels=labels.tolist())
        aug_img   = out["image"]                                        # RGB
        aug_boxes = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
        aug_labels= np.array(out["labels"], dtype=np.int64)
        #画图前RGB->BGR
        vis = cv2.cvtColor(aug_img,cv2.COLOR_RGB2BGR)
        for(x1,y1,x2,y2),label in zip(aug_boxes,aug_labels):
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(vis,VOC_CLASSES[label-1],(x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imwrite(os.path.join(out_dir,f"aug_{stem}_{tun}.jpg"),vis)
        print(f"{stem} run{tun}: {len(aug_boxes)} 个框")




