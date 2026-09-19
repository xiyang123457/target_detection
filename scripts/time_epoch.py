"""计时脚本：先跑 100 张，外推"每个 epoch 要多久"，用于决定 NUM_EPOCHS。

为什么要单独跑这个：train.py 要跑完一个 epoch 才知道耗时（可能几十分钟）；
测 100 张只要 1-2 分钟，就能外推出整 epoch 时间。
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import time
import torch
from torch.utils.data import DataLoader, Subset

from scripts.VOC_dataset import VOCDataset, collate_fn
from scripts.train import build_model        # 复用搭模型函数（train.py 有 __main__ 保护，导入不会触发训练）

ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"
WARMUP = 10          # 前 10 个 step 含 CUDA 初始化/缓存预热，不计入
BATCH = 4


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train = VOCDataset(ROOT, split="train", train=True)
    timing_ds = Subset(ds_train, range(100))          # 只取 100 张
    loader = DataLoader(timing_ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    model = build_model(device=device)
    model.train()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005,
    )

    t0 = None
    counted = 0
    for i, (images, targets) in enumerate(loader):
        images  = [x.to(device) for x in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i + 1 == WARMUP:                 # 第 WARMUP 个 step 之后才开始计时
            t0 = time.time()
        elif i + 1 > WARMUP:
            counted += 1

    dt = time.time() - t0
    sps = dt / counted
    steps_per_epoch = len(ds_train) / BATCH             # 5717 / 4 ≈ 1429
    epoch_min = sps * steps_per_epoch / 60

    print(f"实测 s/step = {sps:.3f}   （{counted} 个 step 计时）")
    print(f"每 epoch ≈ {epoch_min:.1f} 分钟   （{steps_per_epoch:.0f} step × {sps:.3f}s）")
    if device == "cuda":
        print(f"显存峰值 = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
