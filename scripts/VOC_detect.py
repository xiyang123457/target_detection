import glob  #批量查找文件
import os #拼路径
from collections import Counter
import numpy as np #数组运算
import matplotlib.pyplot as plt #画图
from xml.etree import ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
VOC = r"D:\target_detection\data\VOCdevkit\VOC2012"
ANN = os.path.join(VOC, "Annotations")

#累加器
cls_cnt = Counter()
per_img = [] #每张图的框数
areas = [] #每个框的面积
w_list = [] #每个框的宽
h_list = [] #每个框的高
n_diff = 0 #difficult = 1 的框数
n_trunc = 0 #truncated = 1 的框数

#遍历一次，收集统计量
for xml_path in glob.glob(os.path.join(ANN, "*.xml")):
    root = ET.parse(xml_path).getroot()
    objs =root.findall("object")
    per_img.append(len(objs))
    for obj in objs:
        cls_cnt[obj.find("name").text] += 1
        n_diff += int(obj.find("difficult").text)
        n_trunc += int(obj.findtext("truncated", 0))
        bb = obj.find("bndbox")
        x1 =float(bb.find("xmin").text)     # 变量 x1：float，框左上角 x 坐标
        y1 = float(bb.find("ymin").text)    # 变量 y1：框左上角 y 坐标
        x2 = float(bb.find("xmax").text)    # 变量 x2: 框右下角 x 坐标
        y2 = float(bb.find("ymax").text)    # 变量 y2：框右下角 y 坐标
        w =  x2-x1
        h = y2-y1
        w_list.append(w)
        h_list.append(h)
        areas.append(w * h)

#转numpy
areas = np.array(areas)
per_img = np.array(per_img)
print(f"{len(per_img)} 张图 / {len(areas)} 个框 / {len(cls_cnt)} 类")
print(f"每图框数  mean={per_img.mean():.2f}  median={np.median(per_img):.0f}  max={per_img.max()}")
print(f"等效边长中位数 {np.median(np.sqrt(areas)):.0f} px")
print(f"difficult {n_diff} | truncated {n_trunc}")
print("前5类:", cls_cnt.most_common(5))
print("后5类:", cls_cnt.most_common()[-5:])
small = (areas < 32 ** 2).mean() * 100
large = (areas > 96 ** 2).mean() * 100
print(f"small {small:.1f}% / medium {100 - small - large:.1f}% / large {large:.1f}%")
#画图
os.makedirs(OUT, exist_ok=True)
items = cls_cnt.most_common()                 # [(类别, 框数), ...] 已降序

# 图1：类别分布 —— 看长尾
plt.figure(figsize=(11, 4))                   # 20 个类别，画宽一点
plt.bar([k for k, _ in items], [v for _, v in items], color="#4C78A8")
plt.xticks(rotation=45, ha="right")           # 类别名横排会糊在一起，转 45 度
plt.ylabel("instances")
plt.tight_layout()                            # 自动留边距，否则旋转后的标签被裁掉
plt.savefig(os.path.join(OUT, "fig1_class.png"), dpi=140)
plt.close()                                   # 必须关掉，否则下一张图画在同一画布上

# 图2：每图框数 —— 看密度
plt.figure(figsize=(8, 4))
plt.hist(per_img, bins=range(0, per_img.max() + 2), color="#4C78A8")
plt.xlabel("objects per image")
plt.ylabel("images")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_per_image.png"), dpi=140)
plt.close()

# 图3：框宽高 —— 看尺度
plt.figure(figsize=(8, 4))
plt.hist(w_list, bins=50, alpha=0.6, label="width",range=(0,500))    # alpha 半透明，重叠处才看得见
plt.hist(h_list, bins=50, alpha=0.6, label="height",range=(0,500))
plt.legend()
plt.xlabel("pixels")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_size.png"), dpi=140)
plt.close()