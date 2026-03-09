import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

def clean_graph_name(x):
    x = str(x).replace(".tsv", "")
    mapping = {
        "string_graph_experimental": "STRING-experimantal",
        "fmppi_unsup_pll_M20_m10": "PLL-M20-m10",
        "fmppi_unsup_pll_M10_m10": "PLL-M10-m10",
        "fmppi_unsup_perturb_M40_m20": "Perturb",
        "gated_mix_pll": "Gated_mix_PLL-M20-m10",
        "reweight_pll": "Reweight_PLL-M20-m10",
        "fm_sim_knn_M500_m20_row_softmax": "kNN-rowsoftmax",
        "fm_sim_knn_M500_m20_cos_pos": "kNN-cos",
    }
    return mapping.get(x, x)

def load_df(path, sort_by):
    df = pd.read_csv(path, sep="\t")
    need = [
        "graph", "auprc", "recall@k", "precision@k", "enrich_factor@k"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")

    df = df.copy()
    df["label"] = df["graph"].map(clean_graph_name)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    return df

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)

def plot_grouped_main(df, outprefix):
    metrics = ["auprc", "precision@k", "recall@k"]
    metric_labels = ["AUPRC", "Precision@k", "Recall@k"]

    x = list(range(len(df)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for j, (m, lab) in enumerate(zip(metrics, metric_labels)):
        xpos = [i + (j - 1) * width for i in x]
        vals = df[m].values
        ax.bar(xpos, vals, width=width, label=lab, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Graph")
    ax.set_ylim(0, max(df[metrics].max()) * 1.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_ax(ax)
    plt.tight_layout()

    plt.savefig(outprefix + "_main_grouped.png", dpi=400, bbox_inches="tight")
    plt.savefig(outprefix + "_main_grouped.pdf", bbox_inches="tight")
    plt.close()

def plot_ef(df, outprefix):
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    vals = df["enrich_factor@k"].values
    ax.bar(df["label"], vals, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Enrichment factor@k")
    ax.set_xlabel("Graph")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="x", rotation=30)
    style_ax(ax)
    plt.tight_layout()

    plt.savefig(outprefix + "_ef.png", dpi=400, bbox_inches="tight")
    plt.savefig(outprefix + "_ef.pdf", bbox_inches="tight")
    plt.close()

def plot_top_methods(df, outprefix, topn=6):
    sub = df.head(topn).copy()

    metrics = ["auprc", "precision@k", "recall@k"]
    metric_labels = ["AUPRC", "Precision@k", "Recall@k"]

    x = list(range(len(sub)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 4.4))

    for j, (m, lab) in enumerate(zip(metrics, metric_labels)):
        xpos = [i + (j - 1) * width for i in x]
        vals = sub[m].values
        ax.bar(xpos, vals, width=width, label=lab, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["label"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Top methods")
    ax.set_ylim(0, max(sub[metrics].max()) * 1.2)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    style_ax(ax)
    plt.tight_layout()

    plt.savefig(outprefix + "_top_methods.png", dpi=400, bbox_inches="tight")
    plt.savefig(outprefix + "_top_methods.pdf", bbox_inches="tight")
    plt.close()

def plot_horizontal_precision(df, outprefix):
    sub = df.sort_values("precision@k", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(sub["label"], sub["precision@k"], edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Precision@k")
    ax.set_ylabel("Graph")
    ax.set_xlim(0, max(sub["precision@k"]) * 1.15)
    style_ax(ax)
    plt.tight_layout()

    plt.savefig(outprefix + "_precision_horizontal.png", dpi=400, bbox_inches="tight")
    plt.savefig(outprefix + "_precision_horizontal.pdf", bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="summary.tsv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--sort-by", default="precision@k")
    ap.add_argument("--topn", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outprefix = os.path.join(args.outdir, "graph_metrics")

    df = load_df(args.infile, args.sort_by)

    plot_grouped_main(df, outprefix)
    plot_ef(df, outprefix)
    plot_top_methods(df, outprefix, topn=args.topn)
    plot_horizontal_precision(df, outprefix)

    print("saved figures:")
    print(outprefix + "_main_grouped.png")
    print(outprefix + "_main_grouped.pdf")
    print(outprefix + "_ef.png")
    print(outprefix + "_ef.pdf")
    print(outprefix + "_top_methods.png")
    print(outprefix + "_top_methods.pdf")
    print(outprefix + "_precision_horizontal.png")
    print(outprefix + "_precision_horizontal.pdf")

if __name__ == "__main__":
    main()