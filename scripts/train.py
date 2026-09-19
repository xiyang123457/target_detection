import sys
import os
sys.path.insert(0, r"D:\target_detection") # 添加自定义包的搜索路径
import os
import json
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

NUM_EPOCHS = 5
BATCH_SIZE = 4
LR = 0.005

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

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RUNS, exist_ok=True)
    ds_train = VOCDataset(ROOT,split="train",train=True)
    loader = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=collate_fn,pin_memory=True)     # 有 GPU 时加速内存→显存拷贝
    print(f"训练集 {len(ds_train)} 张  |  device = {device}")

    #模型与优化器
    model = build_model(device=device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr =LR,momentum=0.9 ,weight_decay=0.0005) #weight_decay就是l2正则化

    #训练模式
    model.train()
    KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    # loss_classifier：ROIHead目标类别分类损失
    # loss_box_reg：ROIHead检测框坐标回归损失
    # loss_objectness：RPN前景/背景二分类损失
    # loss_rpn_box_reg：RPN候选框坐标回归损失
    history = []

    #训练循环
    for epoch in range(NUM_EPOCHS):
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

            # Step 3 判据：只打印【第一个 step】的 4 个分量，用来确认模型搭对了
            #   loss_classifier ≈ 3.0（= ln(21)，说明分类头是随机初始化的新头）
            #   loss_objectness < 0.1（说明 RPN 保留了预训练，没被误初始化）
            if epoch == 0 and n == 0:
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
        torch.save(model.state_dict(),os.path.join(RUNS,f"fasterrcnn_resnet50_fpn_voc_epoch{epoch}.pth"))


    #收尾   把 loss 历史存 json（D6 画曲线用)
    with open(os.path.join(RUNS, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("训练完成，loss 历史已存 runs/loss_history.json")


if __name__ == "__main__":
    train()



  