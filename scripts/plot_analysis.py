"""画两张结果分析图：
1. fig_ap_by_class.png      —— 逐类 AP@0.5:0.95 排序条形图（回应"总 mAP 掩盖长尾类"）
2. fig_ap_vs_instances.png  —— 实例数(log) vs AP@0.5:0.95 散点（回应"样本多的类是否 AP 更高"）
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无窗口后端，只存图
import matplotlib.pyplot as plt

from scripts.VOC_dataset import VOC_CLASSES

ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"
ANN = os.path.join(ROOT, "Annotations")
OUT = r"D:\target_detection\outputs"

# 每类 AP@0.5:0.95（val 5823 张，5 epoch 全量微调，单次实验；来自 evaluate.py 输出）
# 【别搞错口径】这 20 个数的均值 = 0.434，正好等于 result.md 4.1 的 mAP@0.5:0.95 ≈ 0.43，
#   所以它们是 AP@0.5:0.95，不是 AP@0.5（早期版本这里把它标成了 "AP@0.5"）。
AP = {
    "aeroplane": 0.554, "bicycle": 0.439, "bird": 0.424, "boat": 0.330, "bottle": 0.371,
    "bus": 0.626, "car": 0.520, "cat": 0.450, "chair": 0.314, "cow": 0.387,
    "diningtable": 0.302, "dog": 0.418, "horse": 0.491, "motorbike": 0.489, "person": 0.508,
    "pottedplant": 0.234, "sheep": 0.540, "sofa": 0.330, "train": 0.525, "tvmonitor": 0.430,
}

# 每类实例数：只统计 train 划分（模型真正见过的样本数）
train_txt = os.path.join(ROOT, "ImageSets", "Main", "train.txt")
keep = {line.strip() for line in open(train_txt, "r", encoding="utf-8") if line.strip()}

cnt = {c: 0 for c in VOC_CLASSES}
for p in glob.glob(os.path.join(ANN, "*.xml")):
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem not in keep:                     # 只算 train 集
        continue
    root = ET.parse(p).getroot()
    for obj in root.findall("object"):
        cnt[obj.find("name").text] += 1

names = VOC_CLASSES
aps = [AP[c] for c in names]
counts = [cnt[c] for c in names]

os.makedirs(OUT, exist_ok=True)

# ---- 图 1：逐类 AP 排序条形图 ----
order = np.argsort(aps)                       # 升序
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([names[i] for i in order], [aps[i] for i in order], color="#4C78A8")
ax.set_xlabel("AP@0.5:0.95")
ax.set_title("Per-class AP@0.5:0.95 (sorted, val 5823)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_ap_by_class.png"), dpi=140)
plt.close()

# ---- 图 2：实例数(log) vs AP 散点 ----
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(counts, aps, color="#4C78A8")
for c, x, y in zip(names, counts, aps):
    ax.annotate(c, (x, y), fontsize=7, ha="left", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("train instance count (log)")
ax.set_ylabel("AP@0.5:0.95")
ax.set_title("Instance count vs AP@0.5:0.95")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_ap_vs_instances.png"), dpi=140)
plt.close()

print("已保存 fig_ap_by_class.png / fig_ap_vs_instances.png")
print("train 实例数总和:", sum(counts))
# 自检：逐类均值必然等于总体指标（两者都是"20 类求平均"）。
# 如果这个数不等于 result.md 4.1 的 mAP@0.5:0.95，说明 AP 表被接到了另一个口径上。
print(f"逐类 AP@0.5:0.95 均值 = {np.mean(aps):.4f}（应 ≈ 0.43）")
