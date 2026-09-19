"""COCO 零样本 baseline：不换头、不训练，直接在 VOC val 上算 mAP。"""
import sys
sys.path.insert(0, r"D:\target_detection")

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchmetrics.detection import MeanAveragePrecision

from scripts.VOC_dataset import VOCDataset, VOC_CLASSES, collate_fn

# VOC_CLASSES 下标 -> COCO category id（注意：是 id，不是 0~79 的连续下标）
VOC_TO_COCO = {
    0: 5, 1: 2, 2: 16, 3: 9, 4: 44, 5: 6, 6: 3, 7: 17, 8: 62, 9: 21,
    10: 67, 11: 18, 12: 19, 13: 4, 14: 1, 15: 64, 16: 20, 17: 63, 18: 7, 19: 72,
}
COCO_TO_VOC_LABEL = {cid: i + 1 for i, cid in VOC_TO_COCO.items()}   # COCO id -> VOC 1~20

QUICK_N = 1000        # 先 1000 张验证映射；确认后改 None 跑满 5823


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 零样本：直接用官方权重，不换头（91 类 COCO 输出）
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.transform.min_size = (480,)      # ← 加这一行，和微调对齐分辨率
    model.eval()

    val_ds = VOCDataset(r"D:\target_detection\data\VOCdevkit\VOC2012",
                        split="val", skip_difficult=True, train=False)
    from torch.utils.data import Subset
    ds = Subset(val_ds, range(QUICK_N)) if QUICK_N else val_ds
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    n = 0
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
            metric.update(mapped, [{k: v.cpu() for k, v in t.items()} for t in targets])
            n += len(images)
            print(f"\r已评估 {n} 张", end="")
    print()

    res = metric.compute()
    print("零样本 mAP@0.5      =", round(res["map_50"].item(), 3))
    print("零样本 mAP@0.5:0.95 =", round(res["map"].item(), 3))


if __name__ == "__main__":
    main()
