import sys
import os
sys.path.insert(0, r"D:\target_detection") # 添加自定义包的搜索路径
import os
import json
import re
import time
import random
import argparse
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from scripts.VOC_dataset import VOCDataset, collate_fn
import torch
from torchvision.models.detection import(
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = r"D:\target_detection\data\VOCdevkit\VOC2012"
RUNS = r"D:\target_detection\runs"

# ---- 本次运行的默认配置（都可用命令行覆盖，例：python scripts/train.py --mode full --lr 0.005）----
# 训练范围（本次实验的核心变量）：
#   "head"       —— 只训新的 21 类分类头，backbone / FPN / RPN / box_head 全冻结（lr=0.001，9/20 已跑）
#   "head_lr5e3" —— 同上，但 lr=0.005，与 full 对齐，专门用来做【单变量对照】
#   "full"       —— 全量微调，即 9/19 跑的那套（lr=0.005）
# ⚠️ 模式名会写进 ckpt 与 loss 历史的文件名（..._voc_{mode}_epoch{N}.pth / loss_history_{mode}.json），
#    换一组对照必须换名字 —— 复用 "head" 会把 9/20 那批 lr=0.001 的权重和曲线直接覆盖掉。
DEFAULT_MODE   = "head_lr5e3"
DEFAULT_LR     = 0.005      # 与 full 那次一致：旧 head 用的 0.001 是混淆变量，本次对齐掉
DEFAULT_EPOCHS = 18         # 与 full 的 epoch 0-18 对齐
DEFAULT_BATCH  = 4
DEFAULT_SEED   = 42         # 固定后：新分类头初始化 / 每轮数据顺序 / 增强随机性都可复现
DEFAULT_RESUME = None       # None = 从 COCO 预训练权重重新开始（对照实验必须从头跑，不能热启动）

def parse_args():
    """把超参从"改代码"变成"改命令行"。

    为什么必须参数化：本轮要做多组对照（full / head_lr5e3 / 旧 head 参考），
    超参写死在文件里的话，每换一组就得改一次源码、跑完还得改回来，
    而报告里的"实验设置表"根本无从追溯当时到底用了哪组值。
    """
    ap = argparse.ArgumentParser(description="VOC 微调训练（head / head_lr5e3 / full）")
    ap.add_argument("--mode",       default=DEFAULT_MODE, help=f"训练范围（默认 {DEFAULT_MODE}）")
    ap.add_argument("--lr",         type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH, dest="batch_size")
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    ap.add_argument("--resume",     default=DEFAULT_RESUME,
                    help="接着某个权重继续训：填 ckpt 完整路径；None = 从 COCO 预训练权重重新开始")
    return ap.parse_args()


def set_seed(seed):
    """固定随机性。三处来源都得管：

    random / np.random —— albumentations 的增强（翻不翻转、缩放多少）走 numpy
    torch             —— 新分类头的随机初始化、以及 DataLoader 的 shuffle 顺序

    只在进程开头调一次；训练中途不要重置，否则每轮的数据顺序会重复。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"已固定随机种子 seed={seed}")


def save_config(args, start_epoch, n_train, n_total, n_images):
    """把本次运行的超参落盘到 runs/train_config_{mode}.json。

    为什么单独存一个文件而不是塞进 loss_history：
    loss_history 是 list[dict]，plot_loss 会逐条校验 epoch + 4 个 loss 字段，
    往里混进一条没有 loss 的记录会直接把画图脚本搞崩。
    报告里的"实验设置表"一律从这个文件取数，不手抄。
    """
    os.makedirs(RUNS, exist_ok=True)
    cfg = {
        "mode": args.mode,
        "lr": args.lr,
        "epochs": args.epochs,
        "epoch_from": start_epoch,
        "epoch_to": start_epoch + args.epochs - 1,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "resume": args.resume,
        "optimizer": "SGD(momentum=0.9, weight_decay=0.0005)",
        "min_size": 480,
        "max_size": 800,
        "train_images": n_images,
        "trainable_params": n_train,
        "total_params": n_total,
        "torch": torch.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(RUNS, f"train_config_{args.mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"配置已存 {path}")
    return path


def resume_start_epoch(ckpt_path):
    """从权重文件名里解析出它训到了第几轮，续训时接着往后编号。

    为什么要接着编号：如果续训又从 epoch0 开始存，
    会把已经存在的那批权重覆盖掉 —— 而那些正是要留着做对比的证据。
    """
    m = re.search(r"epoch(\d+)\.pth$", os.path.basename(ckpt_path))
    return int(m.group(1)) + 1 if m else 0

def build_model(num_classes=21,device="cuda"):
    #VOC 20类，背景1类

    #加载预训练权重
    model = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    #只换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features       #读取输入维度
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    #调小输入尺寸
    model.transform.min_size = (480,)   #这里要求传入元组
    model.transform.max_size = 800

    return model.to(device)

def set_trainable(model,mode):
    """按 mode 决定哪些参数参与训练，并打印可训练参数量。

    为什么必须显式设 requires_grad=False：
      torchvision 加载预训练权重时，只默认冻结了 backbone 的 conv1+layer1，
      其余约 99.5% 的参数都是 requires_grad=True。
      下面优化器里的 `if p.requires_grad` 看着像"只训放开的那部分"，
      但只要没有代码去把它设成 False，这个过滤条件就等于没过滤。
    """
    # 用 startswith("head") 而不是 == "head"：
    # 冻结范围只由"是不是只训头"决定，与 lr 无关 —— head_lr5e3 / head_lr1e3 都应走同一分支，
    # 否则每加一个 lr 档就要在这儿加一个 elif，漏一个就会静默变成全量微调（最难查的那类 bug）。
    if mode.startswith("head"):
        #先全部冻结，再只放开新分类头 —— 相当于在 COCO 特征上做线性探针
        for p in model.parameters():
            p.requires_grad = False
        for p in model.roi_heads.box_predictor.parameters():
            p.requires_grad = True
    elif mode == "full":
        #保持原样：torchvision 默认冻结 conv1+layer1，其余全部可训练
        pass
    else:
        raise ValueError(f"未知 TRAIN_MODE: {mode}（可选 'head*' / 'full'）")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"TRAIN_MODE={mode} | 可训练参数 {n_train:,} / {n_total:,}（{n_train / n_total:.2%}）")
    return model, n_train, n_total

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RUNS, exist_ok=True)
    set_seed(args.seed)      # 必须在建模型（新头随机初始化）和建 DataLoader 之前

    ds_train = VOCDataset(ROOT,split="train",train=True)
    # 给 DataLoader 一个独立的 generator：这样"每轮数据顺序"只由 seed 决定，
    # 不受前面任何随机操作（如新头初始化）消耗了多少全局随机数的影响 —— 否则换一次模型就换一次数据顺序。
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds_train,batch_size=args.batch_size,shuffle=True,num_workers=0,
                        collate_fn=collate_fn,pin_memory=True,generator=g)   # 有 GPU 时加速内存→显存拷贝
    print(f"训练集 {len(ds_train)} 张  |  device = {device}")

    #模型与优化器
    model = build_model(device=device)
    model, n_train, n_total = set_trainable(model,args.mode)     #必须在建优化器之前：优化器是按 requires_grad 过滤的

    #续训：把之前某轮的权重盖到当前模型上（结构一致，strict 加载即可）
    start_epoch = 0
    if args.resume:
        model.load_state_dict(torch.load(args.resume,map_location=device))
        start_epoch = resume_start_epoch(args.resume)
        print(f"续训：已加载 {args.resume}，从 epoch {start_epoch} 接着编号")
    else:
        print("从 COCO 预训练权重开始训练")

    save_config(args, start_epoch, n_train, n_total, len(ds_train))
    print(f"本次配置：mode={args.mode} | lr={args.lr} | epochs={args.epochs} | batch={args.batch_size} "
          f"| seed={args.seed} | epoch 编号 {start_epoch}→{start_epoch + args.epochs - 1}")

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr =args.lr,momentum=0.9 ,weight_decay=0.0005) #weight_decay就是l2正则化

    #训练模式
    model.train()
    KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    # loss_classifier：ROIHead目标类别分类损失
    # loss_box_reg：ROIHead检测框坐标回归损失
    # loss_objectness：RPN前景/背景二分类损失
    # loss_rpn_box_reg：RPN候选框坐标回归损失
    #loss 历史：续训时先把之前那段的点读进来，后面每轮追加后再整体写回，曲线保持连续
    hist_path = os.path.join(RUNS, f"loss_history_{args.mode}.json")
    history = []
    if args.resume and os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"已接上 {hist_path} 里已有的 {len(history)} 个点")

    #训练循环
    for epoch in range(start_epoch, start_epoch + args.epochs):
        running = {k: 0 for k in KEYS}
        n = 0           #本epoch 已处理的 step数
        t0 = time.time()

        for images, targets in loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #forward
            loss_dict = model(images,targets)
            loss = sum(loss_dict.values())      #总损失

            optimizer.zero_grad()       #清空上一轮梯度
            loss.backward()             #反向传播
            optimizer.step()            #更新参数

            # Step 3 判据：只打印【本次运行的第一个 step】的 4 个分量，用来确认模型搭对了
            #   全新开始：loss_classifier ≈ 3.0（= ln(21)，说明分类头是随机初始化的新头）
            #             loss_objectness < 0.1（说明 RPN 保留了预训练，没被误初始化）
            #   续训时：直接显示接续位置的 loss（不会再是 3.0，属正常）
            if epoch == start_epoch and n == 0:
                print("第一个step的loss")
                for k in KEYS:
                    print(f"  {k:18s} {loss_dict[k].item():.4f}")

            for k in KEYS:
                running[k] += loss_dict[k].item()
            n += 1

        #每个epoch收尾
        avg = {k: running[k] / n for k in KEYS}
        history.append({"epoch":epoch, **avg}) #**表示把所有的键值对拆开放入
        print(f"epoch {epoch} | " + "  ".join(f"{k}={avg[k]:.4f}" for k in KEYS)
              + f" | total={sum(avg.values()):.4f} | {time.time()-t0:.0f}s")

        #存checkpoint
        #   文件名带 mode：head / head_lr5e3 / full 三套实验的权重互不覆盖
        ckpt_path = os.path.join(RUNS, f"fasterrcnn_resnet50_fpn_voc_{args.mode}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        #每轮落一次 loss 历史：中途停掉也不会丢曲线，续训时靠它接上
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    #收尾
    print(f"训练完成，loss 历史已存 {hist_path}")


if __name__ == "__main__":
    train(parse_args())



  