import cv2
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.VOC_dataset import VOCDataset,VOC_CLASSES

#建数据集
ds =VOCDataset(r"D:\target_detection\data\VOCdevkit\VOC2012", split="train")

#挑3张有代表性的图片
stems = ["2007_000027", "2007_000032", "2008_007069"]
out_dir = r"D:\target_detection\outputs"
os.makedirs(out_dir, exist_ok=True)

for stem in stems:
    #读图
    img = cv2.imread(os.path.join(ds.jpeg_dir, f"{stem}.jpg"))

    #复用_parse_xml函数,拿到框和类别
    boxes, labels = ds._parse_xml(stem)

    #遍历每个框，画矩形+写类别
    for(x1,y1,x2,y2),label in zip(boxes,labels):
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        #画框
        cv2.rectangle(img,(x1,y1),(x2,y2),color=(0,0,255),2)
        #类别编号
        name =VOC_CLASSES[label-1]
        cv2.putText(img,name,(x1,y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    #保存
    cv2.imwrite(os.path.join(out_dir,f"{stem}_boxes.jpg"),img)
    print(f"已保存{stem}_boxes.jpg ,共 {len(boxes)} 个框")
