"""Figures, drawn only from files in results/. All are 7.2 inches wide, the text width of the write up, so the
fonts print at their real size in the PDF."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import FIGURES, RESULTS  # noqa: E402

LUAD, LUSC, GREY = "#3b6ea8", "#d9822b", "#8c8c8c"
WIDTH = 7.2
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
METHODS = [("L2 logistic (ours)", "L2 model (ours)", LUSC), ("9 rank rules (ours)", "9 gene pair rules (ours)", "#f0b27a"),
           ("Hamaneh 2022 8-gene", "Hamaneh 2022, 8 genes", "#7f9cc0"), ("Li 2019 KRT5>AGR2", "Li 2019, KRT5 > AGR2", GREY)]
SHORT_COHORT = {"TCGA (CV, institution-grouped)": "TCGA\n(cross\nvalidated)", "CPTAC-3": "CPTAC-3", "Uppsala RNA-seq (GSE81089)": "Uppsala\nRNA seq",
                "GSE30219 (France)": "France", "GSE37745 (Sweden)": "Sweden", "GSE50081 (Canada)": "Canada", "GSE43580 (SBV IMPROVER)": "SBV\nIMPROVER"}


def _external_panel(ax, bench):
    cohorts = list(dict.fromkeys(bench.cohort))
    lowest = 1.0
    for k, (meth, label, col) in enumerate(METHODS):
        v = [bench[(bench.cohort == ch) & (bench.method == meth)].accuracy.mean() for ch in cohorts]
        ax.plot(np.arange(len(cohorts)) + (k - 1.5) * 0.13, v, "o", color=col, ms=4, label=label)
        lowest = min(lowest, np.nanmin(v))
    ax.set_xticks(np.arange(len(cohorts)), [SHORT_COHORT.get(ch, ch) for ch in cohorts], fontsize=6)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(-0.5, len(cohorts) - 0.5)
    ax.grid(axis="y", lw=0.3, alpha=0.6)
    ax.set_ylabel("accuracy")
    ax.axvline(2.5, color=GREY, lw=0.6, ls=":")
    ax.text(1, lowest - 0.025, "RNA seq", ha="center", va="top", fontsize=6, color=GREY)
    ax.text(4.5, lowest - 0.025, "microarrays (no L2 model)", ha="center", va="top", fontsize=6, color=GREY)
    ax.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.3, columnspacing=1)


def _dilution_panel(ax, conf):
    dil = pd.DataFrame(conf["T6_tumour_content"]["in_silico_dilution"])
    ax.plot(dil.tumour_fraction * 100, dil.lusc_recall, "o-", color=LUSC, ms=3, label="LUSC called LUSC")
    ax.plot(dil.tumour_fraction * 100, dil.luad_recall, "o-", color=LUAD, ms=3, label="LUAD called LUAD")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([100, 50, 30, 20, 10, 5], ["100", "50", "30", "20", "10", "5"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("% tumour in the sample (rest: normal lung)")
    ax.set_ylabel("share called correctly")
    ax.legend(frameon=False, loc="lower left")


def fig1():
    wf = pd.read_csv(RESULTS / "cohort_waterfall.tsv", sep="\t")
    bench = pd.read_csv(RESULTS / "external_benchmark.tsv", sep="\t")
    m = json.load(open(RESULTS / "model.json"))
    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 0.55], height_ratios=[1, 1.05])
    ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    short = ["All RNA seq samples", "Primary tumour only", "No FFPE", "No pathology exclusions", "No treatment for an earlier cancer",
             "No treatment before surgery", "No recurrence or off spec", "One sample per patient"]
    yy = np.arange(len(wf))[::-1]
    ax0.barh(yy, wf.LUAD, color=LUAD, label="LUAD")
    ax0.barh(yy, wf.LUSC, left=wf.LUAD, color=LUSC, label="LUSC")
    for i, t in zip(yy, wf.total):
        ax0.text(t + 15, i, str(t), va="center", fontsize=6)
    ax0.set_yticks(yy, short[: len(wf)])
    ax0.set_xlim(0, 1450)
    ax0.set_xlabel("tumours")
    ax0.set_title("A  Which samples were used", loc="left", fontweight="bold")
    ax0.legend(frameon=False, loc="lower right")
    ms = m["mean_sd"]
    vals = [ms["random"]["auroc"], ms["institution"]["auroc"]]
    ax1.bar([0, 1], [v[0] for v in vals], yerr=[v[1] for v in vals], color=[GREY, LUSC], capsize=3, width=0.6)
    ax1.set_xticks([0, 1], ["random\nsplit", "hospitals\nheld out"])
    ax1.set_ylim(0.95, 1.0)
    ax1.set_ylabel("AUROC (mean and SD, 5 seeds)")
    ax1.set_title("B  Fair split", loc="left", fontweight="bold")
    _external_panel(ax2, bench)
    ax2.set_title("C  New patients, other countries and platforms (model frozen)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_data_model_validation.png", dpi=220)
    plt.close(fig)


def fig2():
    top = pd.read_csv(RESULTS / "top20_genes.tsv", sep="\t")
    pair = pd.read_csv(RESULTS / "lipid_paired_tumour_normal.tsv", sep="\t")
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.3), gridspec_kw={"width_ratios": [1, 1.05]})
    t = top.iloc[::-1]
    shown = [100 * v if side == "LUSC" else -100 * v for v, side in zip(t.share_of_gap, t.side)]
    ax[0].barh(np.arange(len(t)), shown, color=[LUSC if s == "LUSC" else LUAD for s in t.side])
    ax[0].set_yticks(np.arange(len(t)), [f"{g} ({f:.0%})" for g, f in zip(t.gene, t.bootstrap_top20_freq)], fontsize=6)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlabel("% of the LUSC vs LUAD score gap\n(right: higher in LUSC, left: higher in LUAD)")
    ax[0].set_title("A  The 20 genes that drive the model\n(how often each stays in the top 20 over 50 resamples)", loc="left", fontweight="bold")
    genes = ["CERS3", "ABCA12", "ELOVL4", "ALOXE3", "ALOX12B", "PNPLA1", "CYP4F22", "ABHD5", "FA2H", "UGT8"]
    x = np.arange(len(genes))
    for k, (lab, col) in enumerate((("LUSC", LUSC), ("LUAD", LUAD))):
        sub = pair[pair.tumour == lab].set_index("gene").reindex(genes)
        ax[1].bar(x + (k - 0.5) * 0.38, sub.median_log2fc, 0.38, color=col, label=f"{lab} vs own normal")
        for xi, (fc, pv) in enumerate(zip(sub.median_log2fc, sub.bonferroni_p)):
            if pv < 0.05:
                ax[1].text(xi + (k - 0.5) * 0.38, fc + (0.08 if fc >= 0 else -0.3), "*", ha="center", fontsize=7)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(7.5, color=GREY, lw=0.6, ls=":")
    ax[1].text(3.5, 4.55, "skin barrier set (Akiyama 2017)", ha="center", fontsize=6, color=GREY)
    ax[1].text(8.5, 4.55, "Lemay 2019", ha="center", fontsize=6, color=GREY)
    ax[1].set_ylim(-1.45, 4.8)
    ax[1].set_xticks(x, genes, rotation=45, ha="right")
    ax[1].set_ylabel("median log2 change vs the\nsame patient's normal lung")
    ax[1].set_title("B  Skin barrier lipid genes, chosen in advance\n(* significant after Bonferroni correction)", loc="left", fontweight="bold")
    ax[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 0.9))
    fig.tight_layout(w_pad=2)
    fig.savefig(FIGURES / "fig2_genes_lipid.png", dpi=220)
    plt.close(fig)


def fig3():
    conf = json.load(open(RESULTS / "confounders.json"))
    st = pd.DataFrame(conf["T3_strata"])
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    yy = np.arange(len(st))[::-1]
    ax[0].scatter(st.auroc, yy, color=LUSC, s=12, zorder=3)
    ax[0].set_yticks(yy, [f"{s} (n={n})" for s, n in zip(st.stratum, st.n)], fontsize=5.8)
    ax[0].axvline(0.95, color=GREY, ls=":", lw=0.8)
    ax[0].set_xlim(0.9, 1.0)
    ax[0].set_xlabel("AUROC within the subgroup")
    ax[0].set_title("A  High accuracy in every subgroup", loc="left", fontweight="bold")
    _dilution_panel(ax[1], conf)
    ax[1].set_title("B  Less tumour: calls drift to LUAD", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_confounders.png", dpi=220)
    plt.close(fig)


def fig_writeup():
    """The one compact figure in the three page write up: external validation and the tumour content test."""
    bench = pd.read_csv(RESULTS / "external_benchmark.tsv", sep="\t")
    conf = json.load(open(RESULTS / "confounders.json"))
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.05), gridspec_kw={"width_ratios": [1.75, 1]})
    _external_panel(ax[0], bench)
    ax[0].set_title("A  Accuracy on each cohort (model frozen)", loc="left", fontweight="bold")
    _dilution_panel(ax[1], conf)
    ax[1].set_title("B  Less tumour: calls drift to LUAD", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(FIGURES / "writeup_figure.png", dpi=250)
    plt.close(fig)


def run():
    fig1()
    fig2()
    fig3()
    fig_writeup()


if __name__ == "__main__":
    run()
