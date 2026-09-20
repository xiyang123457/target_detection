import sys

# 把项目根目录塞进"导入搜索路径"。
sys.path.insert(0, r"D:\target_detection")
import os
import csv
import re
import json
import math
import time
from datetime import datetime
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
CKPT = os.path.join(RUNS, "fasterrcnn_resnet50_fpn_voc_head_epoch8.pth")

#先在小规模上跑通
QUICK_N =1000


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
    """跑完整个 loader，返回 (res_main, res_50, n)。

    res_main —— 默认阈值（IoU 0.5:0.95 共 10 个点）的结果，其中 res_main["map"] 就是 mAP@0.5:0.95
    res_50   —— iou_thresholds=[0.5] 的结果，专门用来取【每类的 AP@0.5】
    n        —— 实际评估的图片张数

    为什么开两个实例：class_metrics=True 给的 map_per_class 只对"被平均的那个口径"有效，
    默认实例的逐类值其实是 AP@0.5:0.95，不是 AP@0.5（以前这里一直标错了名）。
    想要逐类 AP@0.5 只能再开一个只算 0.5 的实例。
    两个实例共用同一份推理结果（update 只读取张量、不原地修改），所以不会多跑一遍模型。
    """
    from torchmetrics.detection import MeanAveragePrecision   # 局部导入，不装也不影响脚本其余部分

    # box_format="xyxy"  —— 我们的框是 [x1,y1,x2,y2] 绝对像素（和模型输出一致）
    # iou_type="bbox"    —— 只算框的 mAP（不涉及分割掩码）
    # class_metrics=True —— 额外返回【每类 AP】，必须开：长尾数据集的总体 mAP 会掩盖弱势类
    metric_main = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",class_metrics=True)
    metric_50   = MeanAveragePrecision(box_format="xyxy",iou_type="bbox",class_metrics=True,
                                       iou_thresholds=[0.5])

    n = 0
    with torch.no_grad():
        for images, targets in loader:
            preds = model([img.to(device) for img in images])

            # 指标在 CPU 上算：.cpu() 把张量搬回内存，避免 GPU 显存越堆越多。
            preds_cpu   = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric_main.update(preds_cpu, targets_cpu)
            metric_50.update(preds_cpu, targets_cpu)

            n+=len(images)
            print(f"\r已评估 {n} 张", end="")     # \r 原地刷新，不刷屏
    print()

    res_main = metric_main.compute()
    res_50   = metric_50.compute()
    metric_main.reset()      # 算完就清内部缓存，别让几千张图的预测框一直挂在内存里
    metric_50.reset()
    return res_main, res_50, n


# ---- 指标落盘：把算出来的数字写成文件，供写实验报告时取数、溯源 ----

# 默认实例里要保存哪些标量指标，以及落到文件里用什么名字。
# "map" 改名叫 "map_50_95"：直白写清它是 AP 在 IoU 0.5:0.95 上的平均，
# 免得以后又和 map_50 混起来（这正是之前把逐类 AP 标错口径的根源）。
SCALAR_KEYS = {
    "map":        "map_50_95",
    "map_50":     "map_50",
    "map_75":     "map_75",
    "map_small":  "map_small",
    "map_medium": "map_medium",
    "map_large":  "map_large",
    "mar_1":      "mar_1",
    "mar_10":     "mar_10",
    "mar_100":    "mar_100",
    "mar_small":  "mar_small",
    "mar_medium": "mar_medium",
    "mar_large":  "mar_large",
}


def _fmt(x,nd=4):
    """打印用：None（算不出来）就写 n/a，别让 f-string 把 None 崩掉。"""
    return "n/a" if x is None else f"{x:.{nd}f}"


def _as_list(v):
    """把 res 里的向量字段统一成 list。

    只评估到 1 个类时 torchmetrics 返回的是 0 维张量（.tolist() 得到标量而不是列表），
    这里统一兜一下，避免后面用 zip 归并时静默错位。
    """
    return v.tolist() if v.dim() > 0 else [v.item()]


def _num(v):
    """张量 -> JSON 友好的 python 标量。

    torchmetrics 用 -1 表示"该类/该指标在这个评估集上算不出来"，nan 同理。
    两种都落成 None：宁可在报告里写"算不出"，也不要留一个看起来像成绩的负数。
    """
    if v is None:
        return None
    x = float(v.item()) if hasattr(v,"item") else float(v)
    return x if (math.isfinite(x) and x >= 0) else None


def _versions():
    """记下库版本，方便报告里写清环境。取不到就跳过，不因为查版本把评估搞崩。"""
    out = {"torch": torch.__version__}
    for mod_name in ("torchvision","torchmetrics"):
        try:
            out[mod_name] = __import__(mod_name).__version__
        except Exception:
            pass
    return out


def parse_mode_epoch(ckpt_path):
    """从权重文件名里解析出【实验模式】和【轮次】，用于给产物文件命名。

    fasterrcnn_resnet50_fpn_voc_head_epoch8.pth   -> ("head", 8)
    fasterrcnn_resnet50_fpn_voc_epoch5.pth        -> ("unknown", 5)   # 旧命名没写模式，不猜
    完全对不上                                      -> ("unknown", None)
    """
    name = os.path.basename(ckpt_path)
    m = re.search(r"_voc_([A-Za-z]+)_epoch(\d+)\.pth$", name)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"epoch(\d+)\.pth$", name)
    return "unknown", (int(m.group(1)) if m else None)


def collect_per_class(res_main,res_50):
    """把两个口径的逐类 AP 归并成 {类名: {class_id, ap50, ap50_95}}。

    关键：用 res["classes"] 做映射，而不是按顺序盲目 zip ——
    torchmetrics 只返回【评估集里出现过的类】，一旦某类缺席，按顺序 zip 就会整列错位。
    """
    ap50_95 = dict(zip(_as_list(res_main["classes"]), _as_list(res_main["map_per_class"])))
    ap50    = dict(zip(_as_list(res_50["classes"]),   _as_list(res_50["map_per_class"])))

    per_class = {}
    # VOC_CLASSES 的下标 + 1 就是标签值，所以直接把 20 类全列出来，缺席的留 None
    for cid,name in enumerate(VOC_CLASSES,start=1):
        per_class[name] = {
            "class_id": cid,
            "ap50":     _num(ap50.get(cid)),
            "ap50_95":  _num(ap50_95.get(cid)),
        }
    return per_class


def save_metrics(overall,per_class,meta):
    """把指标落盘到 runs/metrics_{mode}_e{epoch}_n{张数}.json 与同名 .csv，返回两个路径。

    为什么存两份：
      json —— 全部指标 + 元信息，机器可读、换脚本换实验都不会错位，报告数字靠它溯源
      csv  —— 一行一类的逐类 AP，可直接粘进报告表格

    为什么文件名要带 epoch 和张数：
      只按 mode 命名时，同一模式不同 epoch 的结果会互相覆盖，
      而被覆盖掉的那个文件名里恰好是唯一的 epoch 线索 —— 事后追不回来；
      子集结果（n1000）若和全量结果（n5823）同名，报告里极容易把子集数字当成全量引用。
    """
    os.makedirs(RUNS,exist_ok=True)
    ep        = meta.get("epoch")
    stem      = f"metrics_{meta['mode']}_e{ep if ep is not None else 'NA'}_n{meta['n_images']}"
    json_path = os.path.join(RUNS, stem + ".json")
    csv_path  = os.path.join(RUNS, stem + ".csv")

    payload = {
        "meta": meta,
        "overall": overall,
        "per_class": per_class,
        "notes": (
            "overall.map_50_95 = mAP@0.5:0.95；overall.map_50 = mAP@0.5。"
            "per_class.ap50 = 该类 AP@0.5；per_class.ap50_95 = 该类 AP@0.5:0.95"
            "（即 torchmetrics 的 map_per_class）。"
            "值为 null 表示该类在评估集上没有可算的样本（或指标非有限）。"
            "引用数字前先看 meta.n_images / meta.subset：subset=true 说明只评估了前 n 张子集，不是全量。"
        ),
    }
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,ensure_ascii=False)

    # newline=""：Windows 下不加这个，csv 写出的一行行会被再翻译一次，表格里平白多出空行
    with open(csv_path,"w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id","class_name","ap50","ap50_95"])
        for name,v in per_class.items():
            writer.writerow([
                v["class_id"], name,
                "" if v["ap50"]    is None else f"{v['ap50']:.4f}",
                "" if v["ap50_95"] is None else f"{v['ap50_95']:.4f}",
            ])
    return json_path, csv_path

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
    #注意：必须包 no_grad。build_model 加载 COCO 权重后大部分参数 requires_grad=True，
    #  不加的话这次前向会建一整张计算图，而 preds/p 会一直活到 main() 返回 ——
    #  整轮评估期间那批中间激活都挂在显存里，8.5G 的卡上很容易 OOM。
    images, _ = next(iter(loader))
    with torch.no_grad():
        preds = model([img.to(device) for img in images])
        p = preds[0]
        print("=== preds 结构 ===")
        print("boxes ", tuple(p["boxes"].shape))         # [N,4]  N = 这张图检出的框数
        print("labels", tuple(p["labels"].shape))        # [N]
        print("scores", tuple(p["scores"].shape),
              "| 单调递减?", bool(torch.all(p["scores"][:-1] >= p["scores"][1:])))  # 应 True
    del preds, p, images        # 确认完就放掉：这批图 evaluate_map 还会再算一遍，没必要留着

    #算map（一次遍历同时拿到 AP@0.5:0.95 与 AP@0.5 两个口径）
    t0 = time.time()
    res_main, res_50, n_eval = evaluate_map(model,loader,device)
    elapsed = time.time() - t0

    overall   = {new: _num(res_main[old]) for old,new in SCALAR_KEYS.items()}
    per_class = collect_per_class(res_main,res_50)

    print("mAP@0.5      =", _fmt(overall["map_50"]))       # 主指标
    print("mAP@0.5:0.95 =", _fmt(overall["map_50_95"]))
    # 两个口径并排打印：以前只打一列还标成"每类 AP"，抄进报告很容易把 0.43 当成 AP@0.5
    print("每类 AP（左 = AP@0.5，右 = AP@0.5:0.95）:")
    for name,v in per_class.items():
        print(f"  {name:12s} {_fmt(v['ap50'],3)}  {_fmt(v['ap50_95'],3)}")

    #落盘：runs/metrics_{mode}.json + .csv，报告取数/溯源用
    mode, epoch = parse_mode_epoch(CKPT)
    meta = {
        "checkpoint":      CKPT,                    # 这批数字是哪份权重跑出来的，必须能追溯
        "checkpoint_file": os.path.basename(CKPT),
        "mode":            mode,                    # head / full；解析不出就是 unknown
        "epoch":           epoch,
        "split":           "val",
        "skip_difficult":  True,                    # 忽略 difficult=1 的框，与 VOC 官方口径一致
        "n_images":        n_eval,                  # 实际评估张数
        "n_images_total":  len(val_ds),             # 该划分总张数
        "subset":          bool(QUICK_N and n_eval < len(val_ds)),
        #                  ^ subset=True 表示只跑了前 n 张子集，报告里绝不能当成全量结果用
        "batch_size":      loader.batch_size,
        "device":          str(device),
        "elapsed_sec":     round(elapsed,1),
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versions":        _versions(),
    }
    json_path, csv_path = save_metrics(overall,per_class,meta)
    print("已保存:", json_path)
    print("已保存:", csv_path)

    #可视化
    visualize(model,val_ds,device,val_ds.stems[:5])

if __name__ == "__main__":
    main()
