"""汇总 runs/metrics_*.json：画 mAP-epoch 曲线 + 输出三路对比表。

为什么需要这个脚本：
本轮要评估十几个 checkpoint，数字散在十几个 json 里。靠翻日志比趋势、手抄进报告，
既慢又必然出错（这正是之前把 AP@0.5 与 AP@0.5:0.95 弄混的原因）。
这里统一从落盘 json 取数，报告里的每张表、每张图都由它生成。

用法：
    python scripts/plot_map_curve.py --n 1000                      # 子集曲线（趋势用）
    python scripts/plot_map_curve.py --n 0                         # 全量对比表（0 = 取最大 n）
    python scripts/plot_map_curve.py --n 1000 --modes full head_lr5e3

产物：
    outputs/map_curve_n{n}.png   曲线图
    runs/summary_n{n}.csv        汇总表（报告取数的唯一来源）
"""
import os
import json
import glob
import csv
import argparse

import matplotlib
matplotlib.use("Agg")          # 无窗口后端：只存图不弹窗
import matplotlib.pyplot as plt

RUNS = r"D:\target_detection\runs"
OUT  = r"D:\target_detection\outputs"


def parse_args():
    ap = argparse.ArgumentParser(description="汇总评测结果：mAP-epoch 曲线 + 对比表")
    ap.add_argument("--n", type=int, default=1000,
                    help="取哪个评估规模的結果（按 n_images 精确匹配）；0 = 全量（取最大的 n）")
    ap.add_argument("--modes", nargs="*", default=None, help="只看这几组实验（默认全部）")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    return ap.parse_args()


def load_rows(runs_dir):
    """读所有 metrics_*.json，抽出 (mode, epoch, n, mAP) 四元组。

    字段缺了就直接跳过并说明原因，不要静默丢 —— 少一个点，曲线上根本看不出来。
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "metrics_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta", {})
        ov   = d.get("overall", {})
        name = os.path.basename(p)
        mode, epoch, n = meta.get("mode"), meta.get("epoch"), meta.get("n_images")
        if mode is None or n is None:
            print(f"  跳过 {name}：缺 mode / n_images")
            continue
        if ov.get("map_50") is None and ov.get("map_50_95") is None:
            print(f"  跳过 {name}：map_50 与 map_50_95 都算不出来")
            continue
        rows.append({"mode": mode, "epoch": epoch, "n": n,
                     "map_50": ov.get("map_50"), "map_50_95": ov.get("map_50_95"),
                     "file": name})
    if not rows:
        raise SystemExit(f"{runs_dir} 里没有任何可用的 metrics_*.json，先跑 evaluate.py")
    return rows


def pick(rows, n_req, modes):
    """按评估规模（n）与模式筛选。

    n_req=0 表示"要全量"：取数据里出现过的最大 n，而不是写死 5823 ——
    万一以后换了划分或评估中断，写死的数字会筛出空表且看不出原因。
    """
    all_n = sorted({r["n"] for r in rows})
    n_target = max(all_n) if n_req == 0 else n_req
    if n_req != 0 and n_req not in all_n:
        print(f"警告：没有 n={n_req} 的结果，现有的 n 有 {all_n}")
    out = [r for r in rows if r["n"] == n_target and (not modes or r["mode"] in modes)]
    return out, n_target


def split_curve_vs_baseline(rows):
    """分成"有 epoch 的（画曲线）"和"零样本（画水平参考线）"两组。

    零样本没有轮次，画成折线没有意义；但它是判断"微调有没有用"的锚点，
    所以画成一条灰色水平虚线，让两条微调曲线直接和它比。
    """
    curve = sorted([r for r in rows if r["epoch"] is not None],
                   key=lambda r: (r["mode"], r["epoch"]))
    base  = [r for r in rows if r["epoch"] is None]
    return curve, base


def plot_curve(curve, base, n_target, out_dir):
    modes = []
    for r in curve:
        if r["mode"] not in modes:
            modes.append(r["mode"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    specs = [("map_50", "mAP@0.5"), ("map_50_95", "mAP@0.5:0.95")]
    for ax, (key, title) in zip(axes, specs):
        for mode in modes:
            pts = [r for r in curve if r["mode"] == mode and r[key] is not None]
            if not pts:
                continue
            ax.plot([r["epoch"] for r in pts], [r[key] for r in pts], marker="o", label=mode)
        for b in base:
            if b[key] is not None:
                ax.axhline(b[key], color="gray", ls="--", lw=1,
                           label=f"{b['mode']} ({b[key]:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # 标题用英文：matplotlib 默认字体没有 CJK 字符，写中文会变成方块
    # 必须标 n：子集曲线和全量曲线长得像但不是一个数，不标迟早混着引用
    fig.suptitle(f"mAP vs epoch | val n={n_target} | {len(curve)} points")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"map_curve_n{n_target}.png")
    plt.savefig(save_path, dpi=140)
    plt.close()
    print("已保存", save_path)


def save_summary(rows, n_target, runs_dir):
    """把筛选后的结果写成 runs/summary_n{n}.csv，并打印一份可直接粘进报告的 Markdown 表。"""
    ordered = sorted(rows, key=lambda r: (r["mode"], -1 if r["epoch"] is None else r["epoch"]))
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"summary_n{n_target}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "epoch", "n_images", "map_50", "map_50_95", "source"])
        for r in ordered:
            w.writerow([r["mode"], "" if r["epoch"] is None else r["epoch"], r["n"],
                        "" if r["map_50"] is None else f"{r['map_50']:.4f}",
                        "" if r["map_50_95"] is None else f"{r['map_50_95']:.4f}",
                        r["file"]])
    print("已保存", path)

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(f"\n===== 汇总表（val n={n_target}）=====")
    print(f"| 配置 | epoch | n | mAP@0.5 | mAP@0.5:0.95 |")
    print(f"|---|---|---|---|---|")
    for r in ordered:
        ep = "—" if r["epoch"] is None else r["epoch"]
        print(f"| {r['mode']} | {ep} | {r['n']} | {fmt(r['map_50'])} | {fmt(r['map_50_95'])} |")
    return path


def selfcheck(rows):
    """自检：同一 mode 下 epoch 是否连续、是否同一 n 里混了不同规模。

    曲线缺一两个点很难用眼睛发现，但会让"最高点在哪一轮"的结论悄悄跑偏。
    """
    by_mode = {}
    for r in rows:
        by_mode.setdefault((r["mode"], r["n"]), []).append(r["epoch"])
    for (mode, n), eps in sorted(by_mode.items()):
        eps = sorted(e for e in eps if e is not None)
        if not eps:
            print(f"  自检：{mode} (n={n}) 只有零样本/无轮次记录，不进曲线")
            continue
        missing = [e for e in range(eps[0], eps[-1] + 1) if e not in eps]
        print(f"  自检：{mode} (n={n}) epoch {eps[0]}→{eps[-1]}，共 {len(eps)} 点"
              + (f"，缺 {missing}" if missing else "，连续"))


def main():
    args = parse_args()
    rows = load_rows(args.runs)
    print(f"共读到 {len(rows)} 条 metrics 记录")

    sel, n_target = pick(rows, args.n, args.modes)
    if not sel:
        raise SystemExit(f"n={args.n}（解析为 {n_target}）+ modes={args.modes} 筛不出任何结果")
    print(f"筛出 {len(sel)} 条（n={n_target}）")

    selfcheck(sel)
    curve, base = split_curve_vs_baseline(sel)
    if curve:
        plot_curve(curve, base, n_target, args.out)
    else:
        print("没有带 epoch 的记录，跳过曲线（只有 baseline 时这是正常的）")
    save_summary(sel, n_target, args.runs)


if __name__ == "__main__":
    main()
