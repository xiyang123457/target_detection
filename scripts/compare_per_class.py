"""逐类 AP@0.5 三方对比：零样本 / 全参微调 / 只换头。

为什么需要这个脚本：
    plot_map_curve.py 只画【总体】mAP —— 它回答"谁更好"，
    但回答不了"好在哪、差在哪"。导师看完总表一定会追问这一句。
    本脚本把三个配置的 20 类 AP@0.5 按类名对齐，找出被微调"训坏"的类。

与 plot_analysis.py 的区别（重要）：
    那个脚本里的 20 个 AP 是【硬编码】的旧数据（full-5ep / 全量 5823）。
    本脚本全部从 runs/metrics_*.json 读，不写死任何数字 ——
    否则一旦换了评估规模（n=1000 vs 5823），表和图会悄悄对不上。

用法：
    python -m scripts.compare_per_class                       # 默认 n=1000，自动选每组的峰值 epoch
    python -m scripts.compare_per_class --n 5823              # 换评估规模
    python -m scripts.compare_per_class --pick last           # 改用"最后一个 epoch"而不是峰值
    python -m scripts.compare_per_class --modes zero full head_lr5e3

产物：
    runs/per_class_3way_n{n}.csv          逐类三方对照表
    outputs/fig_per_class_delta_n{n}.png  横轴排序条形图，退化类标红

跑完应该看到（自检）：
    - 终端打印每个来源实际用了哪个 epoch 的哪份 json
    - 每组的"逐类均值"应该≈该组 overall 的 map_50（差 <0.001 才算对齐没错）
    - 退化类数量（gain < 0 的类）
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

# 三个来源在脚本内部的代号 → 用于 CSV 列名和终端表头
# 为什么用代号而不是直接用 mode 名：head 有两组（lr=0.001 和 0.005），
#   报告里要按"角色"对比（零样本锚点 / 全参 / 只换头），不是按实验编号。
ROLE_DEFAULT = {
    "zero": "zeroshot",     # 零样本：不训练，直接拿 COCO 权重评 VOC
    "full": "full",         # 全参微调：41.2M 参数全放开
    "head": "head_lr5e3",   # 只换头：只训 box_predictor（0.26% 参数）
}

RED   = "#E24B4A"   # 退化（gain < 0）
BLUE  = "#378ADD"   # 提升（gain > 0）
GRAY  = "#B4B2A9"   # 缺数据占位


# ==== 1. 参数 ====

def parse_args():
    ap = argparse.ArgumentParser(description="逐类 AP@0.5 三方对比")
    ap.add_argument("--n", type=int, default=1000,
                    help="评估规模，按 n_images 精确匹配（默认 1000）")
    ap.add_argument("--pick", choices=["best", "last"], default="best",
                    help="每组有多轮时挑哪条：best=峰值 mAP（默认），last=最大 epoch")
    ap.add_argument("--modes", nargs=3, default=None, metavar=("ZERO", "FULL", "HEAD"),
                    help=f"三个角色的 mode 名，默认 {list(ROLE_DEFAULT.values())}")
    ap.add_argument("--x", default="auto",
                    help="横轴用哪个差值列：auto / gain_head / gain_full / delta_head_minus_full")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    return ap.parse_args()


# ==== 2. 读所有 metrics json，按 mode 分组 ====

def load_all(runs_dir):
    """扫 runs/metrics_*.json，返回 {mode: [记录, ...]}。

    每条记录只保留本脚本要用的字段 —— 整份 json 里有 meta/overall/per_class
    三层，全塞进来会让后面挑记录时看不出到底在比什么。
    """
    by_mode = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "metrics_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta", {})
        mode = meta.get("mode")
        n = meta.get("n_images")
        pc = d.get("per_class") or {}
        if mode is None or n is None:
            print(f"  跳过 {os.path.basename(p)}：缺 mode / n_images")
            continue
        if not pc:
            print(f"  跳过 {os.path.basename(p)}：per_class 为空（评估没跑完？）")
            continue
        by_mode.setdefault(mode, []).append({
            "mode":  mode,
            "epoch": meta.get("epoch"),
            "n":     n,
            "map_50": (d.get("overall") or {}).get("map_50"),
            "per_class": pc,
            "file":  os.path.basename(p),
        })
    return by_mode


def pick_record(records, n, how):
    """在某个 mode 的多轮记录里挑一条。

    参数 records: list[dict]，同一个 mode 下的所有评估记录
    参数 n: int，要哪个评估规模
    参数 how: str，"best"=mAP@0.5 最高（默认）/ "last"=epoch 最大
    返回 dict 或 None（该 mode 在这个 n 下没有记录）

    为什么默认挑"峰值"而不是"最后一轮"：
        全参微调实测是【单调下降】的（0.714→0.609）。如果硬报 epoch18，
        等于拿对手的最差状态来比，结论会被质疑"你故意挑了个差 epoch"。
        让每个配置都出现在自己的最好状态，比较才公平。
    """
    cand = [r for r in records if r["n"] == n]
    if not cand:
        return None
    if how == "last":
        return max(cand, key=lambda r: (-1 if r["epoch"] is None else r["epoch"]))
    # best：零样本没有 epoch，map_50 就是它的全部
    return max(cand, key=lambda r: (-1.0 if r["map_50"] is None else r["map_50"]))


# ==== 3. 按类名对齐 ====

def collect_names(recs):
    """取所有来源类名的并集，按 class_id 排序。

    为什么用并集而不是取某一个来源的键：
        万一某个来源漏了类（评估中断、某类没有 GT），取交集会静默少几行，
        而"少一行"在 20 行的表里很难被眼睛发现。
    """
    id_of = {}
    for rec in recs.values():
        if rec is None:
            continue
        for name, e in rec["per_class"].items():
            cid = e.get("class_id")
            if cid is not None:
                id_of.setdefault(name, cid)
    return [n for n, _ in sorted(id_of.items(), key=lambda kv: kv[1])]


def align_rows(recs, names):
    """按【类名】把三方 AP 对齐成一张表。

    参数 recs: {"zero": rec|None, "full": rec|None, "head": rec|None}
    返回 list[dict]，每行一个类

    为什么按类名而不是按位置 zip：
        per_class 是 dict（按类名索引），不同来源的键顺序不保证一致。
        按位置 zip 一旦顺序错位，会把 aeroplane 的分数配到 bicycle 上，
        而且不会报错 —— 这种错误比崩溃危险得多。
    """
    rows = []
    for name in names:
        row = {"name": name}
        for tag in ("zero", "full", "head"):
            rec = recs.get(tag)
            if rec is None:
                row[tag] = None
                continue
            e = rec["per_class"].get(name)
            if e is None:
                # 缺了就说出来，绝不填 0 —— 填 0 会让"没评到"看起来像"检不出"
                print(f"  警告：{rec['mode']} 的 per_class 里没有 {name}，记为 None")
                row[tag] = None
            else:
                row[tag] = e.get("ap50")
        row["class_id"] = next((r["per_class"][name].get("class_id")
                                for r in recs.values()
                                if r and name in r["per_class"]), None)
        rows.append(row)
    return rows


def add_deltas(rows):
    """算三列差值。任一列为 None 时结果也是 None，不做任何填补。"""
    for r in rows:
        z, f, h = r.get("zero"), r.get("full"), r.get("head")
        r["gain_full"]            = None if (z is None or f is None) else f - z
        r["gain_head"]            = None if (z is None or h is None) else h - z
        r["delta_head_minus_full"] = None if (f is None or h is None) else h - f
    return rows


# ==== 4. 自检 ====

def selfcheck(rows, recs):
    """三件事：逐类均值 vs overall、退化类计数、每个来源用的哪份文件。

    逐类均值这一步是关键：如果它和 overall 的 map_50 对不上，
    说明 per_class 的解析漏了类或错位了 —— 那种情况下整张表都不可信。
    """
    print("\n===== 自检 =====")
    for tag, rec in recs.items():
        if rec is None:
            print(f"  [{tag}] 无数据")
            continue
        vals = [r[tag] for r in rows if r.get(tag) is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        ov = rec["map_50"]
        flag = "✓" if (ov is not None and abs(mean - ov) < 1e-3) else "✗ 对不上，检查解析"
        print(f"  [{tag}] {rec['file']} | epoch={rec['epoch']} | n={rec['n']} "
              f"| 逐类均值={mean:.4f} vs overall={ov if ov is None else round(ov, 4)} {flag}")

    for key in ("gain_head", "gain_full", "delta_head_minus_full"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        neg = [r["name"] for r in rows if r.get(key) is not None and r[key] < 0]
        print(f"  [{key}] 均值={sum(vals)/len(vals):+.4f} | 退化(<0) {len(neg)}/{len(vals)} 类"
              + (f"：{', '.join(neg[:6])}{'...' if len(neg) > 6 else ''}" if neg else ""))


# ==== 5. 画图 ====

def plot_delta(rows, xkey, n, out_dir, recs):
    """横轴排序条形图：左端是"被训坏"的类，右端是"被训好"的类。

    为什么横着画：类名（pottedplant、diningtable）太长，竖着放会挤成一团。
    """
    data = [r for r in rows if r.get(xkey) is not None]
    if not data:
        print(f"  {xkey} 全为 None，跳过画图")
        return None
    data.sort(key=lambda r: r[xkey])
    names = [r["name"] for r in data]
    vals  = [r[xkey] for r in data]
    colors = [RED if v < 0 else BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    bars = ax.barh(names, vals, color=colors, height=0.7)
    ax.axvline(0, color="#2C2C2A", lw=0.8)      # 0 线：左右即"变好/变坏"的分界
    for b, v in zip(bars, vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)

    # 标题用英文：matplotlib 默认字体没有 CJK 字符，写中文会变成方块
    # 必须标 n 和 epoch：子集曲线与全量长得像但不是一个数，不标迟早混着引用
    tag = "gain = finetuned - zero-shot" if xkey.startswith("gain") else "head - full"
    ep  = " | ".join(f"{t}:ep{recs[t]['epoch']}" for t in ("zero", "full", "head")
                     if recs.get(t) and recs[t]["epoch"] is not None)
    ax.set_xlabel(tag)
    ax.set_title(f"Per-class AP@0.5 {tag} | val n={n} | {ep}")
    ax.grid(axis="x", alpha=0.3)
    ax.margins(x=0.12)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_per_class_delta_n{n}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"\n已保存 {path}（红色 = 微调后反而退化）")
    return path


# ==== 6. 落盘 CSV + 打印 Markdown ====

COLS = ["class_id", "name", "ap50_zero", "ap50_full", "ap50_head",
        "gain_full", "gain_head", "delta_head_minus_full"]


def save_csv(rows, n, runs_dir):
    ordered = sorted(rows, key=lambda r: (-1e9 if r.get("delta_head_minus_full") is None
                                          else r["delta_head_minus_full"]))
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"per_class_3way_n{n}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in ordered:
            w.writerow(["" if r.get(c) is None else
                        (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in COLS])
    print(f"已保存 {path}")
    return path


def print_md(rows, n):
    """打印可直接粘进报告的 Markdown 表（按 head−full 降序，最有信息量的一列排前面）。"""
    ordered = sorted(rows, key=lambda r: (-1e9 if r.get("delta_head_minus_full") is None
                                          else r["delta_head_minus_full"]))

    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    print(f"\n===== 逐类对照表（val n={n}，口径 AP@0.5）=====")
    print("| 类 | zero | full | head | head−full | head−zero |")
    print("|---|---|---|---|---|---|")
    for r in ordered:
        mark = " ⚠️" if (r.get("delta_head_minus_full") or 0) > 0 else ""
        print(f"| {r['name']}{mark} | {f(r.get('zero'))} | {f(r.get('full'))} | "
              f"{f(r.get('head'))} | {f(r.get('delta_head_minus_full'))} | {f(r.get('gain_head'))} |")
    print("\n注：⚠️ = 只换头比全参微调更高（全参微调在该类上相对更差）")


# ==== 7. 主流程 ====

def main():
    args = parse_args()
    roles = args.modes if args.modes else list(ROLE_DEFAULT.values())
    tags  = ["zero", "full", "head"]
    want  = dict(zip(tags, roles))

    print(f"读 {args.runs}")
    by_mode = load_all(args.runs)
    print(f"发现 mode：{ {m: len(v) for m, v in by_mode.items()} }")

    # ==== 挑每个角色用哪条记录 ====
    recs = {}
    for tag in tags:
        mode = want[tag]
        if mode not in by_mode:
            print(f"  [{tag}] 找不到 mode={mode} 的结果，本列留空")
            recs[tag] = None
            continue
        recs[tag] = pick_record(by_mode[mode], args.n, args.pick)
        if recs[tag] is None:
            print(f"  [{tag}] mode={mode} 没有 n={args.n} 的结果")

    if all(r is None for r in recs.values()):
        raise SystemExit("三个角色都没数据，先跑 evaluate.py / baseline_zero_shot.py")

    # ==== 对齐 + 算差值 + 自检 ====
    names = collect_names(recs)
    rows  = add_deltas(align_rows(recs, names))
    print(f"共 {len(rows)} 个类")
    selfcheck(rows, recs)

    # ==== 横轴列：默认"只换头相对零样本的增益"，没有零样本就退回 head−full ====
    xkey = args.x
    if xkey == "auto":
        xkey = "gain_head" if recs["zero"] is not None and recs["head"] is not None \
               else "delta_head_minus_full"
    print(f"\n横轴用 {xkey}")

    plot_delta(rows, xkey, args.n, args.out, recs)
    save_csv(rows, args.n, args.runs)
    print_md(rows, args.n)


if __name__ == "__main__":
    main()
