import os                        # 拼路径、列目录、判断文件存在
import glob                      # 批量扫 *.xml —— 比 os.listdir 少写一步过滤
import xml.etree.ElementTree as ET   # 解析 XML —— 标准库，不用 pip
import numpy as np               # 组装 boxes/labels 数组
import cv2                       # 读图 —— 比 PIL 快，且 BGR 顺序正好能练"记得转 RGB"
import torch                     # 转 tensor —— Dataset 的标准返回类型
from torch.utils.data import Dataset   # 继承基类 —— 只用它的接口约定，没有实际功能
import random
from scripts.augment import hflip,resize,build_train_transform

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

class VOCDataset(Dataset):
    def __init__(self, voc_root,split="train",skip_difficult=False,train = False):
        self.voc_root = voc_root
        self.split = split
        self.skip_difficult = skip_difficult
        self.train = train #数据增强开关
        self.transform = build_train_transform() if train else None

        # 参数 voc_root: str，VOC2012 根目录
        #   例：r"D:\target_detection\data\VOCdevkit\VOC2012"
        # 参数 split: str，用哪个划分，"train"/"val"/"trainval"
        # 参数 skip_difficult: bool，是否跳过 difficult=1 的框
        #   为什么默认 False：训练时保留能见更多样本；评估时必须开 True 才和 VOC 官方口径一致

        self.ann_dir = os.path.join(voc_root, "Annotations")
        self.jpeg_dir = os.path.join(voc_root, "JPEGImages")
        self.set_dir = os.path.join(voc_root, "ImageSets", "Main")

        #第一步，扫描所有xml
        xml_files = glob.glob(os.path.join(self.ann_dir, "*.xml"))
        #排序后每次运行样本一致，方便调试
        xml_files = sorted(xml_files)
        #存图片的id
        all_stems = [os.path.splitext(os.path.basename(p))[0] for p in xml_files]

        #按split过滤
        if split == "all":         #不做过滤
            self.stems = all_stems
        else:
            split_file = os.path.join(self.set_dir, f"{split}.txt")
            with open(split_file, "r") as f:
                # 变量 keep：set[str]，该划分包含的图片 id
                #   为什么用 set 不用 list：下面要做 17125 次成员判断，
                #   set 是 O(1) 哈希查找，list 是 O(n) 遍历 —— 差三个数量级
                keep = {line.strip() for line in f if line.strip()}
                # 【坑】一定要 if line.strip() 过滤空行
                #   train.txt 末尾可能有空行，不过滤会在 keep 里塞进一个空字符串
                self.stems = [s for s in all_stems if s in keep]
        #检查
        if len(self.stems) == 0:
            raise ValueError(
                f"样本数为 0!检查 voc_root 和 split:\n"
                f"  voc_root = {voc_root}\n"
                f"  split    = {split}\n"
                f"  路径存在 = {os.path.exists(voc_root)}"
            )


    def _parse_xml(self,stem):
        #定位并解析xml
        xml_path = os.path.join(self.ann_dir, f"{stem}.xml")
        #   返回 root: xml.etree.ElementTree.Element，代表 <annotation> 根节点
        root = ET.parse(xml_path).getroot()
        #取出所有物体
        objs = root.findall("object")
        #逐条读取，边读变换算
        boxes_list = []
        label_list = []

        for obj in objs:
            if self.skip_difficult and int(obj.find("difficult").text)==1:
                continue
            #类别，编号0被“背景”占用，要加1
            label =VOC_CLASSES.index(obj.find("name").text) +1

            #坐标
            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text) -1
            y1 = float(bb.find("ymin").text) -1 
            x2 = float(bb.find("xmax").text) - 1
            y2 = float(bb.find("ymax").text) -1
            boxes_list.append([x1,y1,x2,y2])
            label_list.append(label)
        #组装成numpy并自检
        #为什么必须显式指定 dtype：不写 numpy 默认 float64（整数默认 int64 但平台相关），
        #  而 torch 默认 float32，不一致会报错
        #         reshape(-1,4) 把它变成 (0,4)，保证空图也能返回正确形状而不崩
        boxes = np.array(boxes_list,dtype=np.float32).reshape(-1,4)
        labels = np.array(label_list,dtype=np.int64)
        #最后的检查
        assert boxes.shape[0] == labels.shape[0], "框和类别数量不一致，配对出错" #第0维都是物体的个数

        return boxes,labels

    def __len__(self):
        return len(self.stems)
    #返回数据集有多少张图
    
    #给定下标idx，返回图像和标签
    def __getitem__(self,idx):
        stem = self.stems[idx]
        
        #读图，imread返回[H,W,3]
        #注意：cv2读出来的是BGR顺序，不是RGB
        img_path = os.path.join(self.jpeg_dir, f"{stem}.jpg")
        img = cv2.imread(img_path)
        
        #BGR->RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #读标签
        boxes,labels = self._parse_xml(stem)
        #只在训练时做增强
        # albumentations 要 list 进、list 出；format=pascal_voc 对应我们的 [x1,y1,x2,y2]
        if self.train:
            out = self.transform(image = img,bboxes = boxes.tolist(),labels = labels.tolist())
            img = out["image"]
            boxes = np.array(out["bboxes"],dtype=np.float32).reshape(-1,4)
            labels = np.array(out["labels"],dtype=np.int64)


        #转换成float32 
        img = img.astype(np.float32)
        #归一化到0~1
        img = img / 255.0
        #HWC->CHW(深度学习要)
        img = img.transpose(2,0,1)
        #转换成torch tensor
        img = torch.from_numpy(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
      
        #返回
        return img,{"boxes":boxes,"labels":labels}
        
