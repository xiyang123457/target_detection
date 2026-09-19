"""画训练 loss 曲线（4 个分量各一条），存 outputs/loss_curve.png。

数据来源：runs/loss_history.json（由 train.py 在每个 epoch 结束后 append，训练全部结束时写出）。
所以这个脚本要在【训练结束之后】运行。
"""
import sys
sys.path.insert(0, r"D:\target_detection")

import os
import json

import matplotlib
matplotlib.use("Agg")          # 无窗口后端：脚本只存图、不弹窗，避免远程/无 GUI 环境报错
import matplotlib.pyplot as plt

RUNS = r"D:\target_detection\runs"
OUT = r"D:\target_detection\outputs"
KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def main():
    with open(os.path.join(RUNS, "loss_history.json"), "r", encoding="utf-8") as f:
        hist = json.load(f)                    # list[dict]，每个 dict 形如 {"epoch":0, "loss_classifier":..., ...}

    epochs = [h["epoch"] for h in hist]

    # 4 个分量画 4 个子图：因为量级差很大（3.0 vs 0.006），合成一张图看不清
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.ravel(), KEYS):
        ax.plot(epochs, [h[k] for h in hist], marker="o")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("avg loss")
        ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    save_path = os.path.join(OUT, "loss_curve.png")
    plt.savefig(save_path, dpi=140)
    plt.close()
    print("已保存", save_path)

    # 顺便把每轮的 loss_classifier 打印出来，方便判断收敛趋势
    print("loss_classifier 逐 epoch：",
          [round(h["loss_classifier"], 4) for h in hist])


if __name__ == "__main__":
    main()
