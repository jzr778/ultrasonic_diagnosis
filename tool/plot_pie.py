#!/usr/bin/env python3
"""从文本文件读取 name,count 数据并绘制饼图。

数据文件格式（每行 name,count）：
  other_obstacle,933
  wheel_stop,347
  ...

用法：
  python tool/plot_pie.py                          # 默认读 tool/output/bingtu.txt
  python tool/plot_pie.py -i data.txt -o pie.png   # 自定义输入输出
  python tool/plot_pie.py --title "分布统计"        # 自定义标题
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_CJK_FONT: fm.FontProperties | None = None
for _p in _CJK_FONT_CANDIDATES:
    if os.path.isfile(_p):
        _CJK_FONT = fm.FontProperties(fname=_p)
        break


def parse_data(path: str) -> tuple[list[str], list[int]]:
    labels, values = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            labels.append(parts[0].strip())
            values.append(int(parts[1].strip()))
    return labels, values


def plot_pie(labels, values, title="", output=""):
    colors = plt.cm.Set2.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * sum(values)))})",
        colors=colors,
        startangle=140,
        pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(9)
        if _CJK_FONT:
            t.set_fontproperties(_CJK_FONT)
    for t in texts:
        t.set_fontsize(10)
        if _CJK_FONT:
            t.set_fontproperties(_CJK_FONT)

    if title:
        kw = {"fontsize": 14, "fontweight": "bold", "pad": 16}
        if _CJK_FONT:
            kw["fontproperties"] = _CJK_FONT
        ax.set_title(title, **kw)

    fig.tight_layout()
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"已保存: {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    default_input = os.path.join(os.path.dirname(__file__), "output", "bingtu.txt")
    default_output = os.path.splitext(default_input)[0] + ".png"

    parser = argparse.ArgumentParser(description="从 name,count 文本绘制饼图")
    parser.add_argument("-i", "--input", default=default_input, help="数据文件路径")
    parser.add_argument("-o", "--output", default=default_output, help="输出图片路径（留空则弹窗显示）")
    parser.add_argument("--title", default="", help="图表标题")
    args = parser.parse_args()

    labels, values = parse_data(args.input)
    if not labels:
        print(f"未读到有效数据: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"读取 {len(labels)} 条: {dict(zip(labels, values))}")
    plot_pie(labels, values, title=args.title, output=args.output)


if __name__ == "__main__":
    main()
