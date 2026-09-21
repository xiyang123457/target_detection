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
    python -m scripts.compare_per_class --modes zeroshot full head_lr5e3   # 换成 lr 消融那一路

产物：
    runs/per_class_3way_n{n}.csv          逐类三方对照表
    outputs/fig_per_class_delta_n{n}.png  逐类增益条形图（并排画 full−zero 与 head−zero）

跑完应该看到（自检）：
    - 终端打印每个来源实际用了哪个 epoch 的哪份 json
    - 每组的"逐类均值"应该≈该组 overall 的 map_50（差 <0.001 才算对齐没错）
    - 退化类数量（gain < 0 的类）
    - CSV 的三个 AP 列非空行数都应为 20（否则列名映射坏了，见 CSV_KEY 处的说明）
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
# 2026-09-21 变更：head 角色由 head_lr5e3 改为 head（lr=1e-3）。
#   原因：补跑 head(lr=1e-3) 全曲线后，它在 6 个指标中有 5 个优于 head_lr5e3
#   （仅 mAP@0.5 低 0.7 点，但 mAP@0.75 高 4.9 点）——它才是"只换头"这一策略的
#   正确代表。head_lr5e3 降级为学习率消融，其逐类结果可用
#   `--modes zeroshot full head_lr5e3` 单独生成。
ROLE_DEFAULT = {
    "zero": "zeroshot",     # 零样本：不训练，直接拿 COCO 权重评 VOC
    "full": "full",         # 全参微调：41.2M 参数全放开
    "head": "head",         # 只换头：只训 box_predictor（0.26% 参数），lr=1e-3
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
                    help="画哪几条序列：auto（默认，画 full−zero 与 head−zero 两条）/ "
                         "gain_head / gain_full / delta_head_minus_full（单条）")
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
        全参微调实测是【总体下降】的（0.714→0.609，中间有 ±2 点波动，非严格单调）。
        如果硬报 epoch18，等于拿对手的最差状态来比，结论会被质疑"你故意挑了个差 epoch"。
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

def plot_delta(rows, xkeys, n, out_dir, recs):
    """横轴排序条形图：同一类上并排画两条 —— 全参微调的增益 与 只换头的增益。

    为什么横着画：类名（pottedplant、diningtable）太长，竖着放会挤成一团。

    为什么改成"双序列"而不是单序列（2026-09-21）：
        原实现只画 --x 指定的那一列，默认是 gain_head。但报告 6.5 节要论证的
        两件事分别需要两条序列 —— "全参微调的退化是全局的"（18/20 类为负）
        和"只换头的增益接近于零且逐类有正有负"。只画一条时，另一条的结论
        在图上没有依据，正文与图会对不上。两条并排后，"全参整体左移、只换头
        贴着 0 线"这个对比一眼可见。

    参数 xkeys: list[(key, 显示名, 颜色)]，按画图顺序
    """
    series = [(k, lab, col) for k, lab, col in xkeys]
    data = [r for r in rows if any(r.get(k) is not None for k, _, _ in series)]
    if not data:
        print("  两条序列全为 None，跳过画图")
        return None

    # 排序键取第一条序列（full），它是本图的主线：让"退化最狠的类"排在底部
    data.sort(key=lambda r: (1e9 if r.get(series[0][0]) is None else r[series[0][0]]))
    names = [r["name"] for r in data]
    m = len(series)
    height = 0.8 / m

    fig, ax = plt.subplots(figsize=(10, 8))
    for si, (key, lab, col) in enumerate(series):
        vals = [r.get(key) for r in data]
        # None 用 0 占位，但下标要偏移，否则缺数据的类上两根柱子会重叠
        plot_vals = [0.0 if v is None else v for v in vals]
        pos = [i + (si - (m - 1) / 2) * height for i in range(len(data))]
        bars = ax.barh(pos, plot_vals, height=height * 0.92, color=col, label=lab)
        for b, v in zip(bars, vals):
            if v is None:
                continue      # 缺数据不标数字：标 0 会被读成"该类增益为零"
            ax.text(v + (0.004 if v >= 0 else -0.004), b.get_y() + b.get_height() / 2,
                    f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.axvline(0, color="#2C2C2A", lw=0.8)      # 0 线：左右即"变好/变坏"的分界

    # 标题用英文：matplotlib 默认字体没有 CJK 字符，写中文会变成方块
    # 必须标 n 和 epoch：子集曲线与全量长得像但不是一个数，不标迟早混着引用
    ep = " | ".join(f"{t}:ep{recs[t]['epoch']}" for t in ("zero", "full", "head")
                    if recs.get(t) and recs[t]["epoch"] is not None)
    ax.set_xlabel("AP@0.5 gain vs zero-shot (absolute AP)")
    ax.set_title(f"Per-class AP@0.5 gain vs zero-shot | val n={n} | {ep}")
    ax.grid(axis="x", alpha=0.3)
    ax.margins(x=0.14)
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_per_class_delta_n{n}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"\n已保存 {path}")
    return path


# ==== 6. 落盘 CSV + 打印 Markdown ====

COLS = ["class_id", "name", "ap50_zero", "ap50_full", "ap50_head",
        "gain_full", "gain_head", "delta_head_minus_full"]

# CSV 列名 -> rows 里实际使用的键名
# 变量 CSV_KEY：dict[str, str]，把"给 Excel 看的列名"映射到"代码内部的角色名"
#   例：{"ap50_zero": "zero", "ap50_full": "full", "ap50_head": "head"}
# 为什么需要这层映射：列名带 ap50_ 前缀是为了让 CSV 自解释（导进 Excel 能看出这列是
#   AP@0.5 而不是 AP@0.5:0.95），但 rows 里的键是 zero/full/head ——
#   plot_delta 与 print_md 都按角色名取值，改键名会连带改坏两处。
# 坑：直接拿列名当键用，前三列会【永远写空且不抛异常】。2026-09-21 之前的
#   per_class_3way_n1000.csv 就是这个状态：gain 列有数、三个 AP 列全空白，
#   导进 Excel 会被误读成"这三个配置没有逐类 AP"。已在本函数末尾加自检兜住。
CSV_KEY = {"ap50_zero": "zero", "ap50_full": "full", "ap50_head": "head"}


def _cell(row, col):
    """取 row 中某一列的值并格式化成字符串。
    # 方法签名 -> str
    #   作用：统一 CSV 单元格的取值与格式化
    #   关键参数：col 是 COLS 里的列名，经 CSV_KEY 映射后才去 row 里取
    #   坑：None 必须写成空串，不能写 0 或 "n/a" ——
    #       写 0 会被下游当成"该类 AP 为零"（真实含义是"没评到"）；
    #       写 "n/a" 会把这列变成字符串列，Excel 里没法按数值排序。
    """
    v = row.get(CSV_KEY.get(col, col))
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else v


def save_csv(rows, n, runs_dir):
    ordered = sorted(rows, key=lambda r: (-1e9 if r.get("delta_head_minus_full") is None
                                          else r["delta_head_minus_full"]))
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"per_class_3way_n{n}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in ordered:
            w.writerow([_cell(r, c) for c in COLS])

    # 自检：三个 AP 列各自至少要有一行非空。
    # 为什么非查不可：列名映射错了，CSV 照样写得出来、不报任何错，
    #   只是那几列静默全空 —— 唯一能发现它的就是这一行检查。
    filled = {c: sum(1 for r in ordered if _cell(r, c) != "")
              for c in ("ap50_zero", "ap50_full", "ap50_head")}
    flag = "✓" if all(v > 0 for v in filled.values()) else "✗ 有整列为空，检查 CSV_KEY 映射"
    print(f"已保存 {path}（AP 列非空行数：{filled}）{flag}")
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

    # ==== 画哪几条序列 ====
    # 默认（auto）画两条：全参微调相对零样本、只换头相对零样本 ——
    #   报告 6.5 节的两个论点各需要一条（"全参退化是全局的" / "只换头增益≈0"），
    #   只画一条会让正文与图对不上。
    # 没有零样本时退回"只换头 − 全参"这一条（此时没有共同锚点，两者之差仍有意义）。
    if args.x == "auto":
        if all(recs[t] is not None for t in ("zero", "full", "head")):
            series = [("gain_full", "full - zero-shot", RED),
                      ("gain_head", "head - zero-shot", BLUE)]
        else:
            series = [("delta_head_minus_full", "head - full", BLUE)]
    else:
        series = [(args.x, args.x, BLUE)]
    print("\n画序列：" + "  ".join(k for k, _, _ in series))

    plot_delta(rows, series, args.n, args.out, recs)
    save_csv(rows, args.n, args.runs)
    print_md(rows, args.n)


if __name__ == "__main__":
    main()
