import sys
import os
sys.path.insert(0, r"D:\target_detection") # 添加自定义包的搜索路径
import os
import json
import re
import time
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

NUM_EPOCHS = 10            #本次运行要训的轮数：从 epoch9 训到 epoch18（续训时从 checkpoint 的轮次往后接着编号）
BATCH_SIZE = 4
LR = 0.001

#训练范围（本次实验的核心变量）：
#   "head" —— 只训新的 21 类分类头，backbone / FPN / RPN / box_head 全部冻结
#   "full" —— 全量微调，即之前跑的那套
TRAIN_MODE = "head"

#接着某个权重继续训：填 checkpoint 的完整路径；None = 从 COCO 预训练权重重新开始
#   例：RESUME_CKPT = os.path.join(RUNS, "fasterrcnn_resnet50_fpn_voc_head_epoch2.pth")
#   注意：权重文件里只有模型参数，没有优化器动量、也没有已训轮数，
#        所以这是"热启动"（warm start），不是逐 bit 的断点续训。
#        好处是：可以跨模式用 —— 比如拿 full 训到一半的权重，冻结后只训头。
RESUME_CKPT = os.path.join(RUNS, "fasterrcnn_resnet50_fpn_voc_head_epoch8.pth")

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
    if mode == "head":
        #先全部冻结，再只放开新分类头 —— 相当于在 COCO 特征上做线性探针
        for p in model.parameters():
            p.requires_grad = False
        for p in model.roi_heads.box_predictor.parameters():
            p.requires_grad = True
    elif mode == "full":
        #保持原样：torchvision 默认冻结 conv1+layer1，其余全部可训练
        pass
    else:
        raise ValueError(f"未知 TRAIN_MODE: {mode}（可选 'head' / 'full'）")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"TRAIN_MODE={mode} | 可训练参数 {n_train:,} / {n_total:,}（{n_train / n_total:.2%}）")
    return model

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RUNS, exist_ok=True)
    ds_train = VOCDataset(ROOT,split="train",train=True)
    loader = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate_fn,pin_memory=True)     # 有 GPU 时加速内存→显存拷贝
    print(f"训练集 {len(ds_train)} 张  |  device = {device}")

    #模型与优化器
    model = build_model(device=device)
    model = set_trainable(model,TRAIN_MODE)     #必须在建优化器之前：优化器是按 requires_grad 过滤的

    #续训：把之前某轮的权重盖到当前模型上（结构一致，strict 加载即可）
    start_epoch = 0
    if RESUME_CKPT:
        model.load_state_dict(torch.load(RESUME_CKPT,map_location=device))
        start_epoch = resume_start_epoch(RESUME_CKPT)
        print(f"续训：已加载 {RESUME_CKPT}，从 epoch {start_epoch} 接着编号")
    else:
        print("从 COCO 预训练权重开始训练")

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr =LR,momentum=0.9 ,weight_decay=0.0005) #weight_decay就是l2正则化

    #训练模式
    model.train()
    KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    # loss_classifier：ROIHead目标类别分类损失
    # loss_box_reg：ROIHead检测框坐标回归损失
    # loss_objectness：RPN前景/背景二分类损失
    # loss_rpn_box_reg：RPN候选框坐标回归损失
    #loss 历史：续训时先把之前那段的点读进来，后面每轮追加后再整体写回，曲线保持连续
    hist_path = os.path.join(RUNS, f"loss_history_{TRAIN_MODE}.json")
    history = []
    if RESUME_CKPT and os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"已接上 {hist_path} 里已有的 {len(history)} 个点")

    #训练循环
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
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
        #   文件名带 TRAIN_MODE：head / full 两套实验的权重不互相覆盖
        ckpt_path = os.path.join(RUNS, f"fasterrcnn_resnet50_fpn_voc_{TRAIN_MODE}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        #每轮落一次 loss 历史：中途停掉也不会丢曲线，续训时靠它接上
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    #收尾
    print(f"训练完成，loss 历史已存 {hist_path}")


if __name__ == "__main__":
    train()



  