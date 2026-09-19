"""生成"原图(绿框) vs 增强后(绿框)"的并排对照图，证明框同步正确。
选 3 张有代表性的图：单目标 / 多目标 / 密集。
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import os
import cv2
import numpy as np

from scripts.VOC_dataset import VOCDataset
from scripts.augment import build_train_transform

ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"
OUT = r"D:\target_detection\outputs"
STEMS = ["2007_000027", "2007_000032", "2008_007069"]   # 单目标 / 多目标 / 密集


def draw(img_bgr, boxes, color=(0, 255, 0), thick=2):
    """把框画到 BGR 图上（原地）。"""
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    return img_bgr


def pad_to_height(img, h):
    """底部补黑，让两张图高度一致以便并排。"""
    ph = h - img.shape[0]
    if ph <= 0:
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    ds = VOCDataset(ROOT, split="train", train=True)
    t = build_train_transform()

    for stem in STEMS:
        bgr = cv2.imread(os.path.join(ds.jpeg_dir, f"{stem}.jpg"))
        boxes, labels = ds._parse_xml(stem)

        # 原图
        left = draw(bgr.copy(), boxes)

        # 增强后（随机，每次不同）
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = t(image=rgb, bboxes=boxes.tolist(), labels=labels.tolist())
        aug_bgr = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
        aug_boxes = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)
        right = draw(aug_bgr.copy(), aug_boxes)

        # 并排
        h = max(left.shape[0], right.shape[0])
        side = np.hstack([pad_to_height(left, h), pad_to_height(right, h)])
        cv2.putText(side, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(side, "augmented", (left.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        save = os.path.join(OUT, f"fig_aug_compare_{stem}.png")
        cv2.imwrite(save, side)
        print("已保存", save)


if __name__ == "__main__":
    main()
