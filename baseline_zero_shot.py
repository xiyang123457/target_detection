"""COCO 零样本 baseline：不换头、不训练，直接在 VOC val 上算 mAP。

为什么要跑这一组：
没有它，报告里的 mAP=0.7x 就只是一个孤零零的数字 —— 读者无法判断"高还是低"。
零样本给出下限锚点，微调的收益 = 微调 mAP − 零样本 mAP。

用法：
    python baseline_zero_shot.py --n 20      # 冒烟（先确认类别映射没接错）
    python baseline_zero_shot.py --n 1000    # 子集
    python baseline_zero_shot.py --n 0       # 全量 5823 张
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import os
import argparse
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection import MeanAveragePrecision

from scripts.VOC_dataset import VOCDataset, collate_fn
# 复用 evaluate.py 的指标口径与落盘格式：基线数字必须和微调数字【同口径、同字段】，
# 否则后面做汇总表时又得手工对齐一遍，那正是之前把 AP@0.5 和 AP@0.5:0.95 弄混的根源。
from scripts.evaluate import SCALAR_KEYS, collect_per_class, save_metrics, _num, _fmt, _versions

ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"

# VOC_CLASSES 下标 -> COCO category id（注意：是 id，不是 0~79 的连续下标）
VOC_TO_COCO = {
    0: 5, 1: 2, 2: 16, 3: 9, 4: 44, 5: 6, 6: 3, 7: 17, 8: 62, 9: 21,
    10: 67, 11: 18, 12: 19, 13: 4, 14: 1, 15: 64, 16: 20, 17: 63, 18: 7, 19: 72,
}
COCO_TO_VOC_LABEL = {cid: i + 1 for i, cid in VOC_TO_COCO.items()}   # COCO id -> VOC 1~20


def main():
    ap = argparse.ArgumentParser(description="COCO 零样本 baseline（不训练，直接测）")
    ap.add_argument("--n", type=int, default=1000, help="评估前 n 张；0 = 全量 5823 张（默认 1000）")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 零样本：直接用官方权重，不换头（91 类 COCO 输出）
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.transform.min_size = (480,)      # ← 和微调对齐分辨率，否则比的不是同一件事
    model.eval()

    val_ds = VOCDataset(ROOT, split="val", skip_difficult=True, train=False)
    n_req = None if args.n == 0 else args.n
    ds = Subset(val_ds, range(n_req)) if n_req else val_ds
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"待评估 {len(ds)} 张 | device = {device}")

    metric_main = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric_50   = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True,
                                       iou_thresholds=[0.5])

    n = 0
    t0 = time.time()
    with torch.no_grad():
        for images, targets in loader:
            preds = model([img.to(device) for img in images])
            mapped = []
            for p in preds:
                lab = p["labels"].tolist()
                keep = [c in COCO_TO_VOC_LABEL for c in lab]          # 只要 VOC 的 20 类
                mapped.append({
                    "boxes":  p["boxes"][keep].cpu(),
                    "scores": p["scores"][keep].cpu(),
                    "labels": torch.tensor([COCO_TO_VOC_LABEL[c] for c in lab if c in COCO_TO_VOC_LABEL]),
                })
            tgt = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric_main.update(mapped, tgt)
            metric_50.update(mapped, tgt)
            n += len(images)
            print(f"\r已评估 {n} 张", end="")
    print()
    elapsed = time.time() - t0

    res_main = metric_main.compute()
    res_50   = metric_50.compute()
    metric_main.reset()
    metric_50.reset()

    overall   = {new: _num(res_main[old]) for old, new in SCALAR_KEYS.items()}
    per_class = collect_per_class(res_main, res_50)   # 用 res["classes"] 映射，不按顺序 zip

    print("零样本 mAP@0.5      =", _fmt(overall["map_50"]))
    print("零样本 mAP@0.5:0.95 =", _fmt(overall["map_50_95"]))

    meta = {
        "checkpoint":      "COCO pretrained fasterrcnn_resnet50_fpn（91 类输出，未做 VOC 微调）",
        "checkpoint_file": "torchvision weights DEFAULT",
        "mode":            "zeroshot",       # 与 full / head_lr5e3 并列，汇总表按它分組
        "epoch":           None,             # 零样本没有轮次概念，落盘为 eNA
        "split":           "val",
        "skip_difficult":  True,
        "n_images":        n,
        "n_images_total":  len(val_ds),
        "subset":          bool(n_req and n < len(val_ds)),
        "batch_size":      loader.batch_size,
        "device":          str(device),
        "elapsed_sec":     round(elapsed, 1),
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versions":        _versions(),
        "note":            "COCO 91 类输出按 VOC_TO_COCO 映射为 VOC 20 类；不在映射内的 COCO 类别直接丢弃。"
                           "只测不训，min_size=480 与微调对齐。",
    }
    json_path, csv_path = save_metrics(overall, per_class, meta)
    print("已保存:", json_path)
    print("已保存:", csv_path)


if __name__ == "__main__":
    main()
