import sys

# 把项目根目录塞进"导入搜索路径"。
sys.path.insert(0, r"D:\target_detection")
import os
import cv2
import torch
from torch.utils.data import DataLoader,Subset
from scripts.VOC_dataset import VOCDataset, collate_fn,VOC_CLASSES
from scripts.train import build_model

# ---- 路径 ----
ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"       # VOC2012 根目录
RUNS = r"D:\target_detection\runs"                         # checkpoint 目录（已 gitignore）
OUT  = r"D:\target_detection\outputs"                      # 预测图输出目录（已 gitignore）
#权重
CKPT = os.path.join(RUNS, "fasterrcnn_resnet50_fpn_voc_epoch4.pth")

#先在小规模上跑通
QUICK_N =None


def load_model(device):
    model = build_model(device=device)
    model.load_state_dict(torch.load(CKPT,map_location=device))
    # map_location=device：权重里记的是保存时的设备，这里强制映射到当前设备，
    model.eval()
    print(f"已加载权重: {CKPT}")
    return model

def build_val_loader(n=None):
    #参数 n: int 或 None。给定 n 就只用前 n 张（快速验证），None 用全量。

    val_ds = VOCDataset(ROOT,split="val",skip_difficult=True,train=False)
    ds=Subset(val_ds,range(n)) if n else val_ds
    
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return val_ds,loader

#遍历验证集
def evaluate_map(model,loader,device):
    from torchmetrics.detection import MeanAveragePrecision   # 局部导入，不装也不影响脚本其余部分

    # box_format="xyxy"  —— 我们的框是 [x1,y1,x2,y2] 绝对像素（和模型输出一致）
    # iou_type="bbox"    —— 只算框的 mAP（不涉及分割掩码）
    # class_metrics=True —— 额外返回【每类 AP】，必须开：长尾数据集的总体 mAP 会掩盖弱势类
    metric = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",class_metrics=True)

    n = 0
    with torch.no_grad():
        for images, targets in loader:
            preds = model([img.to(device) for img in images])

            # 指标在 CPU 上算：.cpu() 把张量搬回内存，避免 GPU 显存越堆越多。
            metric.update(
                [{k: v.cpu() for k, v in p.items()} for p in preds],
                [{k: v.cpu() for k, v in t.items()} for t in targets],
            )
            n+=len(images)
            print(f"\r已评估 {n} 张", end="")     # \r 原地刷新，不刷屏
    print()
    return metric.compute()

def visualize(model,val_ds:VOCDataset,device,stems,score_thr = 0.5):
    """每张图画出 GT（绿框）和预测（红框 + 分数），存到 outputs/。

    参数 stems: list[str]，要可视化的图片 id
    参数 score_thr: float，只画置信度 >= 该值的预测框（否则图上一堆低分框看不清）
    """
    os.makedirs(OUT, exist_ok=True)            # 不存在就建，否则 imwrite 静默失败

    for stem in stems:
        #读原图
        bgr = cv2.imread(os.path.join(val_ds.jpeg_dir,f"{stem}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        #   ① BGR→RGB（模型是按 RGB 训练的）
        #   ② float 并 /255（0~255 整数 → 0~1 浮点）
        #   ③ HWC→CHW（模型要 [通道,高,宽])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            pred = model([x.to(device)])[0]    # 拿第一张（也是唯一一张）的预测

        vis = bgr.copy()                       # 在副本上画，别污染原图数据

        #绿框
        gt_boxes, _ = val_ds._parse_xml(stem)
        for (x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        #红框 + 类别分数
        pb = pred["boxes"].cpu().numpy()     # 第 i 个框的坐标 [N,4]
        pl = pred["labels"].cpu().numpy()    # 第 i 个框的类别 [N]
        ps = pred["scores"].cpu().numpy()    # 第 i 个框的置信度 [N]
        for (x1, y1, x2, y2), lab, sc in zip(pb, pl, ps):
            if sc < score_thr:
                continue
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(vis, f"{VOC_CLASSES[lab-1]} {sc:.2f}", (int(x1), int(y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        save_path = os.path.join(OUT, f"pred_{stem}.jpg")
        cv2.imwrite(save_path, vis)
        print(f"已保存: {save_path}")   


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    val_ds,loader = build_val_loader(n=QUICK_N)
    #先验证 preds 结构（拿一个 batch 看看，对不上就说明哪步错了）
    images, _ = next(iter(loader))
    preds = model([img.to(device) for img in images])
    p =preds[0]
    print("=== preds 结构 ===")
    print("boxes ", tuple(p["boxes"].shape))         # [N,4]  N = 这张图检出的框数
    print("labels", tuple(p["labels"].shape))        # [N]
    print("scores", tuple(p["scores"].shape),
          "| 单调递减?", bool(torch.all(p["scores"][:-1] >= p["scores"][1:])))  # 应 True

    #算map
    res = evaluate_map(model,loader,device)
    print("mAP@0.5      =", round(res["map_50"].item(), 4))    # 主指标
    print("mAP@0.5:0.95 =", round(res["map"].item(), 4))
    # 用 res["classes"] 做映射，避免"哪个 AP 对哪一类"错位（不要盲目按顺序 zip）
    print("每类 AP:")
    for c, ap in zip(res["classes"].tolist(), res["map_per_class"].tolist()):
        print(f"  {VOC_CLASSES[c - 1]:12s} {ap:.3f}")

    #可视化
    visualize(model,val_ds,device,val_ds.stems[:5])

if __name__ == "__main__":
    main()
