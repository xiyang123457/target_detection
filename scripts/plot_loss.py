"""画训练 loss 曲线（4 个分量各一条），存 <out>/loss_curve_<mode>.png。

数据来源：runs/loss_history_<mode>.json。
train.py 每跑完一个 epoch 就整体重写一次这个文件，所以【训练中途也能跑】这个脚本，
图会反映到最近一个已完成的 epoch —— 不必等训练全部结束。

用法：
    python scripts/plot_loss.py                  # full；找不到时回退到改名前的 loss_history.json
    python scripts/plot_loss.py --mode head      # head，需要 runs/loss_history_head.json 存在
"""
import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")          # 无窗口后端：脚本只存图、不弹窗，避免远程/无 GUI 环境报错
import matplotlib.pyplot as plt

RUNS = r"D:\target_detection\runs"
OUT = r"D:\target_detection\outputs"
KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


def parse_args():
    ap = argparse.ArgumentParser(description="画训练 loss 曲线")
    ap.add_argument("--mode", default="full",
                    help="读哪套实验的历史：loss_history_<mode>.json（默认 full）")
    ap.add_argument("--runs", default=RUNS, help=f"历史文件所在目录（默认 {RUNS}）")
    ap.add_argument("--out", default=OUT, help=f"图片输出目录（默认 {OUT}）")
    return ap.parse_args()


def resolve_history(mode, runs_dir):
    """定位该 mode 的 loss 历史文件，返回路径；找不到就抛错并列出目录里有什么。

    【为什么回退只在 full 时开口】改名前的 loss_history.json（不分模式）内容就是 full 那套，
    所以 mode=="full" 时回退是安全的。若对任意 mode 都回退，
    一个手误（比如 --mode heda）就会静默画出另一套实验的曲线 ——
    图有标题、有数据点、形状正常，只有一行 print 能看出不对，非常容易看走眼。
    """
    path = os.path.join(runs_dir, f"loss_history_{mode}.json")
    if os.path.exists(path):
        return path

    if mode == "full":
        legacy = os.path.join(runs_dir, "loss_history.json")
        if os.path.exists(legacy):
            return legacy

    # 报错时把目录里现有的历史列出来，省得再猜是文件名不同还是训练没跑
    have = []
    if os.path.isdir(runs_dir):
        have = sorted(f for f in os.listdir(runs_dir) if f.startswith("loss_history"))
    raise FileNotFoundError(
        f"没找到 {path}\n"
        f"  先跑 train.py 里 TRAIN_MODE={mode} 的训练\n"
        f"  {runs_dir} 现有的历史文件：{have if have else '（一个都没有）'}"
    )


def load_history(path):
    """读历史并自检：非空、字段齐、按 epoch 升序。返回 (epochs, hist)。

    为什么要自检：这个脚本会静默地"有数据就画"，
    一旦字段名不对或文件接错，图照样出得来，只是内容毫无意义 ——
    所以宁可在这里崩，也别画出一张看着正常的错图。
    """
    with open(path, "r", encoding="utf-8") as f:
        hist = json.load(f)        # list[dict]，每个 dict 形如 {"epoch":0, "loss_classifier":..., ...}

    if not hist:
        raise ValueError(f"{path} 里没有任何 epoch 记录（空列表），没东西可画")

    for h in hist:
        missing = [k for k in ["epoch"] + KEYS if k not in h]
        if missing:
            raise KeyError(
                f"{path} 里 epoch={h.get('epoch')} 这条记录缺字段 {missing}；"
                f"实际有的字段是 {sorted(h)}"
            )

    hist = sorted(hist, key=lambda h: h["epoch"])   # 不依赖文件里的书写顺序
    return [h["epoch"] for h in hist], hist


def summarize(epochs, hist):
    """把 4 个分量按 epoch 打成一张表，并给出首→末降幅。

    图只能看趋势；写报告要的是具体数字（report.md 6.2 节的 loss 表就是这么来的）。
    """
    print(f"{'epoch':>5s}  " + "".join(f"{k:>18s}" for k in KEYS))
    for h in hist:
        print(f"{h['epoch']:>5d}  " + "".join(f"{h[k]:>18.4f}" for k in KEYS))

    print(f"\nepoch 范围: {epochs[0]} → {epochs[-1]}（共 {len(hist)} 个点）")
    for k in KEYS:
        first, last = hist[0][k], hist[-1][k]
        drop = (first - last) / first * 100 if first else 0.0
        print(f"  {k:18s} {first:.4f} → {last:.4f}（降 {drop:.1f}%）")


def main():
    args = parse_args()
    path = resolve_history(args.mode, args.runs)
    print(f"数据来源: {path}")
    if os.path.basename(path) == "loss_history.json":
        print("注意：用的是改名前的旧文件 loss_history.json（不分模式，内容即 full 那套）")

    epochs, hist = load_history(path)
    summarize(epochs, hist)

    # 4 个分量画 4 个子图：因为量级差很大（3.0 vs 0.006），合成一张图看不清
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, k in zip(axes.ravel(), KEYS):
        ax.plot(epochs, [h[k] for h in hist], marker="o")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        ax.set_ylabel("avg loss")
        ax.grid(alpha=0.3)

    # 图里必须写清是哪套实验、哪个 epoch 区间：
    # head 是接着 epoch8 续训的，横轴从 9 起，不标出来很容易被误读成"只画了后半段"。
    # 标题用英文：matplotlib 默认字体没有 CJK 字符，写中文会变成方块。
    fig.suptitle(f"{args.mode} | epoch {epochs[0]}-{epochs[-1]} | {len(hist)} points")
    plt.tight_layout(rect=(0, 0, 1, 0.95))     # 给 suptitle 留出顶部空间，别和子图标题压在一起

    os.makedirs(args.out, exist_ok=True)
    # 文件名带 mode：否则 --mode head 会直接覆盖 --mode full 的图，两套实验只能留一张
    save_path = os.path.join(args.out, f"loss_curve_{args.mode}.png")
    plt.savefig(save_path, dpi=140)
    plt.close()
    print("已保存", save_path)


if __name__ == "__main__":
    main()
