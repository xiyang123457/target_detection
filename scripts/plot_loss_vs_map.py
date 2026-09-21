"""训练目标 vs 泛化性能的背离图：train loss_classifier 与 val mAP@0.5 同图对照。

为什么需要这张图：
    只看到"mAP 随训练下降"，无法区分两种完全不同的原因 ——
      (a) 训练不充分：loss 还没降下去，模型根本没学好；
      (b) 泛化 gap：训练集上越学越准，验证集上越来越差（灾难性遗忘）。
    把 train loss 与 val mAP 画在一起，方向是否一致就是判据：
      full      ：loss 降 57.7%，mAP 降 10.4 点  → 方向相反，属 (b)
      head_lr5e3：loss 降 34.1%，mAP 升 1.8 点   → 方向相同，属正常收敛
    两条曲线放在同一张图上，"全参微调是遗忘而不是没学好"这个论断才有直接证据，
    否则只能靠推理（推理是：若只是没学好，train loss 不该降）。

数据来源（全部从落盘文件读，脚本里不写死任何数字）：
    runs/loss_history{,_head_lr5e3}.json    每 epoch 的 4 个 loss 分量均值
    runs/metrics_{mode}_e{ep}_n{n}.json     每个评估点的 mAP

用法：
    python -m scripts.plot_loss_vs_map                          # 默认 n=1000
    python -m scripts.plot_loss_vs_map --n 0                    # 全量（取数据里最大的 n）
    python -m scripts.plot_loss_vs_map --modes full head_lr5e3

产物：
    outputs/fig_loss_vs_map_n{n}.png

跑完应该看到（自检）：
    - full      ：loss 0.1688 → 0.0715（降 57.7%），mAP 0.7137 → 0.6093（降 10.4 点）
    - head_lr5e3：loss 0.1453 → 0.0957（降 34.1%），mAP 0.7606 → 0.7784（升 1.8 点）
    - 每个 mode 实际用了哪个 loss 文件、哪些 epoch 的 mAP，终端会逐条打印
    - 若某组"有 mAP 但缺 loss 历史"，脚本明确报出来，绝不静默少画一条线

⚠️ 上面这两行是【人工核对用的期望值】，不是脚本的数据来源 —— 图里的每个数字都从 runs/*.json 现读。
   换了 n、换了 mode 或重训之后，这两行会过期；此时以终端打印的实测值为准，并顺手把这里更新掉。
   （本文件早期就曾因为漏更新而留下一组过期数字，见 report.md 页脚的修订记录。）
"""
import os
import re
import json
import glob
import argparse

import matplotlib
matplotlib.use("Agg")          # 无窗口后端：只存图不弹窗，避免无 GUI 环境报错
import matplotlib.pyplot as plt

RUNS = r"D:\target_detection\runs"
OUT = r"D:\target_detection\outputs"

# 要画的 loss 分量。只用 classifier：
#   它是"模型对类别的判别能力"最直接的指标，也是全参微调里降幅最大的分量。
#   另三个分量（box_reg / objectness / rpn_box_reg）留着，改 LOSS_KEY 即可换。
LOSS_KEY = "loss_classifier"

# 每个 mode 的配色。键就是 mode 名，与 runs/ 里的文件名后缀一致。
COLOR = {"full": "#1F77B4", "head_lr5e3": "#2CA02C"}
FALLBACK_COLOR = "#9467BD"     # 未登记的 mode 用紫色，避免和上面两个撞色


# ==== 1. 参数 ====

def parse_args():
    ap = argparse.ArgumentParser(description="train loss 与 val mAP 背离图")
    ap.add_argument("--n", type=int, default=1000,
                    help="取哪个评估规模（按 n_images 精确匹配）；0 = 取数据里最大的 n")
    ap.add_argument("--modes", nargs="*", default=["full", "head_lr5e3"],
                    help="画哪几组实验（默认 full 与 head_lr5e3）")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    return ap.parse_args()


# ==== 2. 读 loss 历史 ====

def load_loss(mode, runs_dir):
    """读某组的 loss 历史，返回 {epoch: loss_value}；文件不存在时返回 None。

    为什么返回 None 而不是抛异常：
        head 那组是用 --resume 续训的，它的 loss_history 可能只有后半段
        （实测 loss_history_head.json 只有 epoch 9-18）。缺文件/缺段都属于
        "预期内的情况"，脚本应当照常出图并明确告知，而不是直接崩掉。
    """
    path = os.path.join(runs_dir, f"loss_history_{mode}.json")
    if not os.path.exists(path) and mode == "full":
        # 改名前的旧文件（不分模式），内容就是 full 那套，回退是安全的
        legacy = os.path.join(runs_dir, "loss_history.json")
        if os.path.exists(legacy):
            path = legacy
    if not os.path.exists(path):
        print(f"  [{mode}] 没有 loss 历史（找过 {path}），该组的 loss 线不画")
        return None

    with open(path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    if not hist:
        print(f"  [{mode}] {os.path.basename(path)} 是空列表，该组的 loss 线不画")
        return None

    out = {}
    for h in hist:
        if "epoch" in h and LOSS_KEY in h:
            out[h["epoch"]] = h[LOSS_KEY]
    eps = sorted(out)
    print(f"  [{mode}] loss 来自 {os.path.basename(path)}，epoch {eps[0]}-{eps[-1]}（{len(eps)} 点）")
    return out


# ==== 3. 读 mAP ====

def load_map(mode, n_req, runs_dir):
    """读某组所有 metrics_*.json，返回 ({epoch: map_50}, n_used)。

    为什么按 meta["mode"] 精确匹配，而不是按文件名猜：
        head 与 head_lr5e3 前缀重叠，用 startswith 会把两组混进一条曲线 ——
        而且混完曲线形状依然正常，只有数值悄悄变了，肉眼发现不了。
    为什么要先按 n 分组再挑：同一组实验可能同时有 n=1000 与 n=5823 的评估记录，
        混着画会把"子集"和"全量"接成一条不存在的曲线。
    """
    # by_n：dict[int, dict[int, float]]，n_images -> {epoch -> map_50}
    by_n = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "metrics_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta") or {}
        if meta.get("mode") != mode:
            continue
        n = meta.get("n_images")
        ep = meta.get("epoch")
        m50 = (d.get("overall") or {}).get("map_50")
        # 零样本的 epoch 落盘为 null —— 它不进曲线，只作水平参考线
        if n is None or ep is None or m50 is None:
            continue
        by_n.setdefault(n, {})[ep] = m50

    if not by_n:
        print(f"  [{mode}] 没有可用的 metrics 记录（含 epoch 的）")
        return {}, None

    ns = sorted(by_n)
    n_target = ns[-1] if n_req == 0 else n_req
    if n_target not in by_n:
        print(f"  [{mode}] 没有 n={n_target} 的记录（现有 {ns}），该组不画")
        return {}, None

    keep = by_n[n_target]
    print(f"  [{mode}] mAP 来自 {len(keep)} 个评估点，epoch {sorted(keep)}，n={n_target}")
    return keep, n_target


def load_zero_shot(n_req, runs_dir):
    """读零样本 mAP@0.5，作为水平参考线。找不到就返回 None。"""
    best = None
    for p in sorted(glob.glob(os.path.join(runs_dir, "metrics_zeroshot_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("meta") or {}
        n = meta.get("n_images")
        m50 = (d.get("overall") or {}).get("map_50")
        if m50 is None:
            continue
        if n_req == 0 or n == n_req:
            best = m50
            print(f"  零样本参考线 mAP@0.5 = {m50:.4f}（来自 {os.path.basename(p)}，n={n}）")
            break
    if best is None:
        print(f"  没找到 n={n_req} 的零样本记录，图上不画参考线")
    return best


# ==== 4. 画图 ====

def plot(pack, n_target, out_dir):
    """双 y 轴：左轴 train loss（实线），右轴 val mAP@0.5（虚线 + 方块）。

    为什么用双轴而不是两个子图：
        两个子图只能让人分别看两条曲线，得自己在脑子里对齐 epoch；
        叠在同一张图上，full 的 loss 一路向下、mAP 也一路向下（方向相反 = 背离），
        而 head_lr5e3 的 loss 向下、mAP 向上（方向相同 = 正常收敛）——
        这个对照一眼就能看出来。
    为什么 loss 线用实线、mAP 线用虚线 + 方块：
        两条线在同一个视觉区域里，颜色只能区分 mode、区分不了"这是哪个量"，
        必须再加一层线型编码。
    """
    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    ax2 = ax1.twinx()          # 共享 x 轴、独立 y 轴

    xmax = 0
    for mode, d in pack.items():
        if mode.startswith("__"):
            continue          # __zero__ 是标量参考线，不是 {loss, map} 结构，不能进这个循环
        color = COLOR.get(mode, FALLBACK_COLOR)

        loss = d["loss"]
        if loss:
            eps = sorted(loss)
            ax1.plot(eps, [loss[e] for e in eps], "-o", color=color, lw=1.8, ms=4,
                     label=f"{mode}  train {LOSS_KEY} (left)")
            xmax = max(xmax, eps[-1])

        m50 = d["map"]
        if m50:
            eps = sorted(m50)
            ax2.plot(eps, [m50[e] for e in eps], "--s", color=color, lw=1.8, ms=6,
                     label=f"{mode}  val mAP@0.5 (right)")
            xmax = max(xmax, eps[-1])

    if pack.get("__zero__") is not None:
        ax2.axhline(pack["__zero__"], color="gray", ls=":", lw=1.2,
                    label=f"zero-shot mAP@0.5 ({pack['__zero__']:.3f})")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel(f"train {LOSS_KEY}  (solid)")
    ax2.set_ylabel("val mAP@0.5  (dashed)")
    ax1.set_xlim(-0.5, xmax + 0.5)
    ax1.grid(alpha=0.3)

    # 副标题里的降幅全部现算，绝不写死 ——
    # 一旦写死（例如手敲"57.7%"），换 n、换 mode 或重训之后图与数字就会对不上，
    # 而图本身看着完全正常，属于最难发现的那类错误。
    notes = []
    for mode, d in pack.items():
        if mode.startswith("__"):
            continue
        lz, mz = sorted(d["loss"]), sorted(d["map"])
        txt = f"{mode}:"
        if lz:
            a, b = d["loss"][lz[0]], d["loss"][lz[-1]]
            txt += f" loss {(b - a) / a * 100:+.1f}%"
        if mz:
            a, b = d["map"][mz[0]], d["map"][mz[-1]]
            txt += f" / mAP {(b - a) * 100:+.1f}pt"
        notes.append(txt)

    # 标题与图例用英文：matplotlib 默认字体没有 CJK 字符，中文会渲染成方块。
    # 必须标 n：n=1000 子集与全量 5823 的曲线形状相似但数值不同，不标迟早混着引用。
    ax1.set_title(f"Training objective vs generalization | val n={n_target}\n"
                  + "    ".join(notes), fontsize=10)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_loss_vs_map_n{n_target}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"\n已保存 {path}")
    return path


# ==== 5. 自检 ====

def selfcheck(pack, n_target):
    """打印每组的首末值与该变量的变化，供人工核对图上的数字对不对。

    为什么把降幅算出来打印：图只能看趋势，"降了多少"必须落到数字才能写进报告；
    而且这一步能顺手抓出"loss 与 mAP 用了不同 n"这类口径错配。
    """
    print(f"\n===== 自检（n={n_target}）=====")
    for mode, d in pack.items():
        if mode.startswith("__"):
            continue
        loss, m50 = d["loss"], d["map"]
        if loss:
            eps = sorted(loss)
            a, b = loss[eps[0]], loss[eps[-1]]
            print(f"  [{mode}] {LOSS_KEY}: {a:.4f} → {b:.4f}"
                  f"（{'+' if b >= a else '-'}{abs(b - a) / a * 100:.1f}%）")
        if m50:
            eps = sorted(m50)
            a, b = m50[eps[0]], m50[eps[-1]]
            print(f"  [{mode}] val mAP@0.5: {a:.4f} → {b:.4f}"
                  f"（{'+' if b >= a else '-'}{abs(b - a) * 100:.1f} 点）")


# ==== 6. 主流程 ====

def main():
    args = parse_args()
    print(f"读 {args.runs}")

    pack = {}
    n_used = None
    for mode in args.modes:
        loss = load_loss(mode, args.runs)
        m50, n_used = load_map(mode, args.n, args.runs)
        if loss is None and not m50:
            print(f"  [{mode}] loss 与 mAP 都没有，跳过该组")
            continue
        pack[mode] = {"loss": loss or {}, "map": m50}

    if not pack:
        raise SystemExit("一组都凑不齐，先跑 train.py 与 evaluate.py")

    # n_used 是 load_map 解析后的实际评估规模（--n 0 时它等于数据里最大的 n）。
    # 零样本参考线必须按同一个 n 去找，否则参考线和曲线不同口径，比出来的差值没意义。
    n_target = n_used
    pack["__zero__"] = load_zero_shot(n_target, args.runs)

    selfcheck(pack, n_target)
    plot(pack, n_target, args.out)


if __name__ == "__main__":
    main()
