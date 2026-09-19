import sys
sys.path.insert(0, r"D:\target_detection")

import torch
from torch.utils.data import DataLoader
from scripts.VOC_dataset import VOCDataset

NUM_WORKERS = 2        # ← Step 6：先 0 跑通，再改 2 试多进程

def collate_fn(batch):
    images, targets = zip(*batch)         # 按样本分组 → 按字段分组
    return list(images), list(targets)

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
  
