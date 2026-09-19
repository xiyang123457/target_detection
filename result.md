# 实验记录：P1 目标检测（VOC2012 + Faster R-CNN 微调）

> 本文记录完整实验过程、原始数字与踩坑。所有数字可在 `runs/`、`outputs/` 找到对应文件。
> 项目说明与复现命令见 [README.md](./README.md)。

## 1. 实验目标与假设

**H1**：COCO 预训练的 Faster R-CNN 只换 21 类头、在 VOC 上微调，即可达到可用精度。
判定标准：val mAP@0.5 < 0.30 说明工程有 bug；0.40~0.60 达标；> 0.60 优秀。

**H2（工程验证，非科学假设）**：数据增强里「框跟着图同步变换」是正确性的必要条件。
判定标准：增强后画图目视，框仍紧贴物体（并排对照见 assets/fig_aug_compare_*.png；9 张增强验证图见 outputs/aug_*.jpg）。
说明：H2 只验证「框没算错」，不验证「增强对 mAP 有贡献」——后者需要 no-aug 消融，未做（见第 8.2 节）。

> 说明：本项目为 W1（第一周），目标是「跑通并拿到合理数字」，未设计严格消融（见第 6/8 节）。

## 2. 数据

| 项 | 值 |
|---|---|
| 数据集 | Pascal VOC 2012 trainval |
| train | 5717 图 / 15774 框 |
| val | 5823 图 / 15787 框 |
| 类别数 | 20（+ 背景 = 21 类头） |
| 长尾比 | person 17401 : bus 685 ≈ 25.4 : 1 |
| 小目标占比 | 9.5% |
| difficult=1 的框 | 4462（11.1%，评估时忽略） |

> 注：`Annotations/` 共 17125 个 xml（40138 框），其中 5585 张（VOC2007 前缀）不在划分内，训练只用划分内的 11540 张。

类别分布（长尾）图见 `assets/fig1_class.png`（D1 统计）。

**泄漏检查**
- train/val 用 VOC 官方划分（`ImageSets/Main/train.txt`/`val.txt`），按文件名划分、天然不重叠
- 预训练权重来自 COCO，与 VOC 无交集 ✅
- 未触碰 VOC test（其标注未公开）；但本项目 val 同时是报告集、无独立 hold-out（且 val 5823 > train 5717），报告数字偏乐观（见第 8.2 节）

## 3. 实验设置

| 项 | 值 |
|---|---|
| 模型 | fasterrcnn_resnet50_fpn（COCO 预训练）→ 换 21 类头 |
| 可训练参数 | 41.2 M |
| 优化器 | SGD lr=0.005, momentum=0.9, weight_decay=0.0005 |
| batch_size / min_size | 4 / 480 |
| epoch | 5 |
| 增强 | 水平翻转(p=0.5) + 随机缩放(0.8~1.2)，框同步 |
| 评估 | skip_difficult=True，model.eval() + no_grad |

硬件：RTX 5060 Laptop 8.52 GB / CUDA 13.0 / torch 2.9.0+cu130 ｜ 单次训练：6.8 min/epoch（0.287 s/step），峰值显存 3.10 GB

## 4. 结果

### 4.1 主结果（单次，1 种子）

| 配置 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| VOC 微调 5 epoch（中间结果） | **≈0.71** | ≈0.43 |

> ⚠️ 单次实验（1 种子），方差未知，故只报两位小数（≈0.71）。在种子方差 ±1-2 点的量级下，0.7145 这种 4 位小数是虚假精度。未做多种子是本项目最大短板（见第 8 节）。

### 4.2 训练 loss（epoch 平均，图见 outputs/loss_curve.png）

| epoch | classifier | box_reg | objectness | rpn_box_reg |
|---|---|---|---|---|
| 0 | 0.1714 | 0.1960 | 0.0243 | 0.0112 |
| 1 | 0.1197 | 0.1564 | 0.0151 | 0.0097 |
| 2 | 0.1067 | 0.1459 | 0.0120 | 0.0092 |
| 3 | 0.0972 | 0.1392 | 0.0099 | 0.0087 |
| 4 | 0.0926 | 0.1340 | 0.0092 | 0.0084 |

> 第一个 step 的 classifier ≈ 3.24（≈ ln 21），证明新分类头随机初始化正确；epoch 平均即降到 0.17，说明预训练特征强、新头收敛极快。

### 4.3 逐类 AP（val 5823 张）

| 类 | AP | 类 | AP |
|---|---|---|---|
| aeroplane | 0.554 | dog | 0.418 |
| bicycle | 0.439 | horse | 0.491 |
| bird | 0.424 | motorbike | 0.489 |
| boat | 0.330 | person | 0.508 |
| bottle | 0.371 | pottedplant | 0.234 |
| bus | 0.626 | sheep | 0.540 |
| car | 0.520 | sofa | 0.330 |
| cat | 0.450 | train | 0.525 |
| chair | 0.314 | tvmonitor | 0.430 |
| cow | 0.387 | diningtable | 0.302 |

图：`assets/fig_ap_by_class.png`（排序条形图）、`assets/fig_ap_vs_instances.png`（实例数 vs AP）。

## 5. 分析

### 5.1 为什么有效
- 预训练 backbone 特征强：新分类头第一个 epoch 内 loss 从 3.0 降到 0.17。
- 只换头、不动 RPN：`loss_objectness` 初始即 0.02，证明 RPN 保留了 COCO 学到的「物体性」。

### 5.2 错误分析（逐类）
- 强项：外形统一、体积大（bus 0.626 / aeroplane 0.554 / sheep 0.540 / train 0.525）
- 弱项：室内小物体 + 多姿态/遮挡（pottedplant 0.234 / diningtable 0.302 / chair 0.314 / sofa 0.330）
- person 0.508：姿态/遮挡/尺度变化最多

### 5.3 与文献对比
- 论文 VOC 报 0.73（VOC 11 点口径 + 6 epoch + 多卡大 lr）。本项目 ≈0.71 是 **pycocotools 101 点口径**（系统性偏低 1-3 点），不能直接等号比较。

## 6. 负结果与失败记录（对照任务书 §四/§七 的坑位预警）

任务书预警了这些坑，我的实际情况如下：

| 任务书预警 | 我的结果 | 结论 |
|---|---|---|
| 微调检测器 lr 用 1e-4 起步 | 用 SGD 时 1e-4 是 Adam 风格；SGD 官方配置 0.005，实测 loss 正常下降 | 改用 0.005 |
| collate_fn 没自定义 | 默认 collate 因图尺寸不同报 `stack expects equal size` | 自写 `zip(*batch)` 解决 |
| 类别 id 与背景错位 | 忘 `+1` 时 loss 看着正常但学不会 | `_parse_xml` 里已 `+1` |
| 评估忘了 model.eval() | 会让 mAP 异常 | evaluate.py 显式 `eval() + no_grad` |
| 增强时框没跟着变换 | 手写 resize 曾把 w/h 传反 → 图被拉伸 | 已改，且每步画图验证 |
| 只看 mAP 不看分类别 AP | 总 mAP 掩盖了 pottedplant 0.234 | 已打印并分析每类 AP |

**任务书没预警、我额外踩到的**：

| 尝试 | 结果 | 原因 | 处理 |
|---|---|---|---|
| `pretrained=True` 加载 | 弃用警告 | torchvision 0.24 改为 `weights=` | 已改 |
| 误以为 `in_features=2048` | 换头维度错 | ROI head 经 box_head 后是 1024 | 改动态取 |
| `num_workers=2` | Windows 卡死风险 | spawn + 共享内存 | 改回 0 |
| albumentations import 联网查版本 | import 崩溃 | 访问 pypi 超时 | 设 `NO_ALBUMENTATIONS_UPDATE=1` |
| `collate_fn` 定义两份 | 维护隐患 | Dataset 与测试脚本各一份 | 合并到 VOC_dataset |

**遗留疑问**：未验证固定种子对结果稳定性的影响。

## 7. 可复现性

环境：conda env `pytorch`，torch 2.9.0+cu130（RTX 5060 是 sm_120，需 cu128+）

```bash
cd /d/target_detection
python scripts/train.py
python scripts/plot_loss.py
python scripts/evaluate.py
```

- 随机种子：默认（未固定）—— 复现性短板
- 训练日志：`runs/train.out.log`
- loss 历史：`runs/loss_history.json`
- 指标输出：`runs/metrics.json`
- Checkpoint：`runs/fasterrcnn_resnet50_fpn_voc_epoch*.pth`
- 总耗时：训练约 34 min + 评估约 15 min

## 8. 结论与局限

### 8.1 数据支持的结论
- COCO 预训练 Faster R-CNN 换 21 类头 + 微调 5 epoch，VOC val mAP@0.5 ≈ 0.71。这是 5 epoch 的中间结果（loss 仍在下降、未收敛），只能支持「链路通、数字落在合理区间」，不能据此判定「优秀」。
- 长尾差异明显：bus 0.626 vs pottedplant 0.234（差 0.39）。
- 工程链路整体正确，多个「框同步」陷阱已排除。

### 8.2 数据不支持的（我不敢说的）
- 不敢说结果稳定 —— 仅 1 种子，无 std。
- 不敢说增强各自贡献 —— 未做消融。
- 不敢说优于 baseline —— 未测 COCO 零样本 mAP。
- 不敢说已收敛 —— 5 epoch loss 仍在降。
- 不敢说在具身/小目标有效 —— VOC 小目标仅 9.5%。

## 附：结果可信度自查

| 检查项 | 做法 | 结果 |
|---|---|---|
| 数据泄漏 | 官方 train/val 划分天然不重叠 | ✅ |
| 测试集污染 | 预训练为 COCO，与 VOC 无交集 | ✅ |
| eval 模式 | model.eval() + no_grad，不传 targets | ✅ |
| 指标实现 | torchmetrics（未与手写对拍） | ⚠️ 待 W2 |
| 过拟合 | 仅有 train loss 曲线（无 val loss），无回升 | ⚠️ 证据不足 |
| 随机性 | 未做多种子 | ❌ 待补 |
| 目视验证 | 5 张预测图 | ✅（错误模式待补） |

**未排除的风险**：单种子可能带来 ±1-2 点偏差；torchmetrics 未与 pycocotools 对拍。

## 9. 下一步

- 固定种子，跑 3 种子报 mean ± std
- 补 COCO 零样本 baseline
- 消融：翻转 / 缩放 / 颜色增强
- 手写 VOC 11 点 mAP 并与 pycocotools 对拍（W2 D1）
- YOLOv8 同环境对比（W2 D2）
