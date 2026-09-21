"""多口径指标拆解图：以零样本为基准，各配置在 6 个指标上的差值。

为什么需要这张图：
    主结果表只报一个 mAP@0.5，而它会掩盖"换头微调"的真实代价 ——
      head_lr5e3 的 mAP@0.5 比零样本高 1.9 点，但 mAP@0.75 低 4.7 点、
      mAP@0.5:0.95 低 2.9 点。
    只看单一指标会得出"微调有小幅增益"的结论；拆开看才知道它提升的是"检出"、
    牺牲的是"定位"：mAP@0.5 只要求 IoU>0.5（框大致套住物体即可），
    mAP@0.75 要求框相当贴合。两个指标的走向相反，直接说明模型的框变松了。
    这是"微调没有真正改善检测质量"最直接的可视化证据，而它在一张 mAP@0.5 表里看不见。

    ⚠️ 2026-09-21 修订：补跑 head(lr=1e-3) 全曲线后，上图结论需要收窄 ——
    "mAP@0.5 涨而 mAP@0.75 跌"是 lr=5e-3 这一档的【特有现象】，不是线性探针的
    普遍性质。lr=1e-3 收敛后（ep15）在 6 个指标上有 4 个为正。因此本图必须
    同时画出两档 lr，否则会把"学习率过高"误读成"线性探针有害"。
    为此增加 head ep0 一根柱子：它量化"重新初始化分类头"这一步的即时代价。

    ⚠️ 2026-09-21 二次修订：head_lr5e3 已补训至 ep18（13 个评估点）。
    实测它在**全部 13 个点上**都是 mAP@0.5 高于零样本、mAP@0.5:0.95 低于零样本，
    符号从未翻转 —— 该分歧不是单轮次的偶然。本图取它的 mAP@0.5 峰值 ep13 作代表，
    完整区间见报告 6.3 节的表。

数据来源：
    runs/metrics_*.json —— 全部从落盘文件读，脚本里不写死任何数字。

用法：
    python -m scripts.plot_metric_breakdown                 # 默认 n=1000
    python -m scripts.plot_metric_breakdown --n 0           # 全量（取数据里最大的 n）
    python -m scripts.plot_metric_breakdown --baseline zeroshot

产物：
    outputs/fig_metric_breakdown_n{n}.png

跑完应该看到（自检）：
    - 基准（零样本）的 6 个指标值会被打印出来，mAP@0.5 应约 0.7635
    - head lr=1e-3 ep0：6 个指标应全为负（重新初始化分类头的代价，mAP@0.75 约 -13.5）
    - head lr=1e-3 ep15：6 个指标的绝对值应都在 1.5 点内（收敛后≈零样本）
    - head lr=5e-3 ep13：mAP@0.5 约 +2.2 点、mAP@0.5:0.95 约 -1.4 点（方向相反）
    - full ep0 / ep18：6 个指标应全为负（ep0 的 small 例外，见报告 6.4 节）
"""
import os
import json
import glob
import argparse

import matplotlib
matplotlib.use("Agg")          # 无窗口后端：只存图不弹窗
import matplotlib.pyplot as plt
import numpy as np

RUNS = r"D:\target_detection\runs"
OUT = r"D:\target_detection\outputs"

# 要拆解的 6 个指标：字段名 -> 图上的显示名
# 为什么把 map_75 排在 map_50_95 前面：本图的核心是 map_50 与 map_75 的分歧，
#   两者必须相邻才能一眼比较；按"数值大小"排会把它们拆开。
METRICS = [
    ("map_50",     "mAP@0.5"),
    ("map_75",     "mAP@0.75"),
    ("map_50_95",  "mAP@0.5:0.95"),
    ("map_small",  "mAP small"),
    ("map_medium", "mAP medium"),
    ("map_large",  "mAP large"),
]

# 默认要对比的配置：(mode, epoch, 图例名)
# 取 ep0 与"峰值"两类点：峰值与起点都给，才不会被质疑"你故意挑了最差的 epoch"。
# 为什么 head 要出现两次（ep0 与 ep15）：
#   ep0  = 分类头刚被随机初始化、只训了 1 轮 → 量化"换头"这一步本身的代价；
#   ep15 = 收敛后的峰值状态 → 量化"换头 + 训练"的净收益。
#   两者之差就是训练真正赚回来的部分，少了 ep0 这一列就看不出这笔账。
# 为什么统一取各自 mAP@0.5 的峰值（head ep15 / head_lr5e3 ep9 / full ep0）：
#   与 compare_per_class.py 的 --pick best 口径一致，两张图才能互相印证。
DEFAULT_PICKS = [
    ("head",       0,  "head lr=1e-3 ep0 (reinit cost)"),
    ("head",       15, "head lr=1e-3 ep15 (peak)"),
    ("head_lr5e3", 13, "head lr=5e-3 ep13 (peak)"),
    ("full",       0,  "full ep0 (peak)"),
    ("full",       18, "full ep18 (last)"),
]

COLORS = ["#FF9F1C", "#2CA02C", "#9467BD", "#D62728", "#7F7F7F"]


# ==== 1. 参数 ====

def parse_args():
    ap = argparse.ArgumentParser(description="多口径指标拆解（相对零样本的差值）")
    ap.add_argument("--n", type=int, default=1000, help="评估规模；0 = 取最大的 n")
    ap.add_argument("--baseline", default="zeroshot", help="基准 mode（默认 zeroshot）")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    return ap.parse_args()


# ==== 2. 读数据 ====

def load_records(runs_dir):
    """读全部 metrics_*.json，返回 list[dict]，每条含 mode/epoch/n/overall。

    为什么要把 n 一起读进来：同一组实验可能有 n=1000 与 n=5823 两份记录，
    基准和对比组必须来自同一个 n，否则差值里混进了"评估规模"这个无关变量。
    """
    recs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "metrics_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta") or {}
        ov = d.get("overall") or {}
        if meta.get("mode") is None or meta.get("n_images") is None:
            continue
        recs.append({
            "mode": meta.get("mode"),
            "epoch": meta.get("epoch"),
            "n": meta.get("n_images"),
            "overall": ov,
            "file": os.path.basename(p),
        })
    if not recs:
        raise SystemExit(f"{runs_dir} 里没有可用的 metrics_*.json")
    return recs


def find(recs, mode, epoch, n):
    """按 (mode, epoch, n) 精确取一条记录；找不到返回 None 并说明原因。"""
    for r in recs:
        if r["mode"] == mode and r["epoch"] == epoch and r["n"] == n:
            return r
    return None


# ==== 3. 画图 ====

def plot(base, groups, n, out_dir):
    """分组柱状图：x 轴是 6 个指标，y 轴是"相对基准的差值（点）"。

    为什么画差值而不是绝对值：6 个指标的绝对值跨度是 0.16~0.76，
    画在一起时小指标（small/medium）的柱高差会被压扁到看不出来；
    差值图把它们拉到同一个量纲（点 = 0.01），谁在退化、退化多少一目了然。
    """
    labels = [disp for _, disp in METRICS]
    x = np.arange(len(METRICS))
    width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    for gi, (name, ov) in enumerate(groups):
        # 差值统一换算成"点"（1 点 = 0.01），和报告里 +1.9 点 / -4.7 点的说法一致
        vals = [(ov.get(k, 0.0) - base.get(k, 0.0)) * 100 for k, _ in METRICS]
        pos = x + (gi - (len(groups) - 1) / 2) * width
        bars = ax.bar(pos, vals, width * 0.92, label=name, color=COLORS[gi % len(COLORS)])
        for b, v in zip(bars, vals):
            # 数字标在柱顶/柱底外侧：柱内放不下，且容易被柱色吞掉
            # 5 组同屏时柱宽只剩 0.16，字号必须降到 6.2，否则相邻标签会叠在一起
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.25 if v >= 0 else -0.25),
                    f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.2)

    ax.axhline(0, color="#2C2C2A", lw=1.0)     # 0 线：以上是相对零样本有增益
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("delta vs zero-shot (points)")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.16)

    # 标题用英文：matplotlib 默认字体没有 CJK 字符，中文会渲染成方块。
    # 必须标 n：子集与全量的数值不同，不标迟早混着引用。
    # 副标题点明"要看哪两行"：5 组同屏，不指路的话读者会盯着最显眼的那根柱子看。
    ax.set_title(f"Per-metric delta vs zero-shot | val n={n}\n"
                 "green (head lr=1e-3 ep15): all |delta| <= 1.5 pt  |  "
                 "purple (head lr=5e-3 ep13): mAP@0.5 up, mAP@0.5:0.95 down  |  "
                 "orange (head ep0) = cost of re-initializing the head",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_metric_breakdown_n{n}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"\n已保存 {path}")
    return path


# ==== 4. 主流程 ====

def main():
    args = parse_args()
    print(f"读 {args.runs}")
    recs = load_records(args.runs)

    ns = sorted({r["n"] for r in recs})
    n_target = ns[-1] if args.n == 0 else args.n
    if n_target not in ns:
        raise SystemExit(f"没有 n={n_target} 的记录（现有 {ns}）")
    print(f"评估规模 n={n_target}")

    base_rec = find(recs, args.baseline, None, n_target)
    if base_rec is None:
        raise SystemExit(f"找不到基准 {args.baseline} 在 n={n_target} 下的记录")
    base = base_rec["overall"]
    print(f"基准 {args.baseline}：{os.path.basename(base_rec['file'])}")

    # 自检：把基准的各指标打出来，人工确认没接错文件
    print("  基准各指标：" + "  ".join(f"{disp}={base.get(k, float('nan')):.4f}"
                                       for k, disp in METRICS))

    groups = []
    for mode, ep, name in DEFAULT_PICKS:
        r = find(recs, mode, ep, n_target)
        if r is None:
            print(f"  跳过 {name}：找不到 mode={mode} epoch={ep} n={n_target} 的记录")
            continue
        groups.append((name, r["overall"]))
        deltas = "  ".join(f"{disp}={(r['overall'].get(k, 0) - base.get(k, 0)) * 100:+.1f}"
                           for k, disp in METRICS)
        print(f"  {name:<20} {deltas}")

    if not groups:
        raise SystemExit("一个对比组都没有，先跑 evaluate.py")

    plot(base, groups, n_target, args.out)


if __name__ == "__main__":
    main()
