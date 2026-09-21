"""DataLoader + collate_fn 的冒烟验证（D4 学习步骤留档）。

用途
----
手动确认 VOCDataset 与 collate_fn 串起来之后，一个 batch 的形状 / dtype / 取值范围是否符合预期。
这不是单元测试，只是打印式自检；正式验证建议改用 pytest（见 report.md 附录 D）。

跑完应该看到
------------
    len(images) = 2
    images[i]: shape=(3, H, W) dtype=torch.float32 范围=[0.00,1.00]
    len(targets) = 2
    targets[i]: boxes 与 labels 的第一维相等 = True
    val batch 也 OK: 2 2

为什么 NUM_WORKERS 必须是 0
--------------------------
Windows 下多进程走 spawn，配合共享内存曾有卡死风险（见 report.md 附录 D）。
本脚本早期写成 2，是当时"先 0 跑通、再改 2 试多进程"的试验残留；
现已改回 0，与 train.py / evaluate.py 保持一致 —— 否则这里跑通、那边卡死，最难查。

为什么 collate_fn 从 VOC_dataset 导入，而不是在本文件里再定义一份
--------------------------------------------------------------
本项目早期 Dataset 与测试脚本各写了一份 collate_fn，改一处忘另一处就会造成
"训练正常、评估错位"这类静默 bug。现统一由 scripts/VOC_dataset.py 提供唯一实现，
本文件只 import，不再自带副本。
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import torch
from torch.utils.data import DataLoader
from scripts.VOC_dataset import VOCDataset, collate_fn   # collate_fn 的唯一实现在 VOC_dataset.py

NUM_WORKERS = 0        # 见文件头说明：Windows 下必须 0，与 train.py / evaluate.py 一致

if __name__ == "__main__":
    root = r"D:\target_detection\data\VOCdevkit\VOC2012"
    ds_train = VOCDataset(root,split="train",train=True)
    ds_val = VOCDataset(root,split="val",train=False)
    dl_train = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=2, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    it = iter(dl_train)     #拿迭代器
    batch = next(it)
    images,targets = batch
    print("=== Step 5 验证 ===")
    print("len(images) =", len(images), "(应=2)")
    for i, im in enumerate(images):
        print(f"  images[{i}]: shape={tuple(im.shape)} dtype={im.dtype} "
              f"范围=[{im.min():.2f},{im.max():.2f}]")
    print("len(targets) =", len(targets))
    for i, t in enumerate(targets):
        print(f"  targets[{i}]: boxes={tuple(t['boxes'].shape)} {t['boxes'].dtype}, "
              f"labels={tuple(t['labels'].shape)} {t['labels'].dtype}, "
              f"数量相等={t['boxes'].shape[0]==t['labels'].shape[0]}")

    v_images, v_targets = next(iter(dl_val))
    print("val batch 也 OK:", len(v_images), len(v_targets))
