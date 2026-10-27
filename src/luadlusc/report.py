"""Fill report/writeup_template.md with numbers from results/, write report/writeup.md and render report/writeup.pdf.

The build fails if a placeholder has no value, if a top 10 gene has no checked note, if a gene named in the
prose is no longer where the prose says it is, or if the PDF is longer than three pages. That way the write-up
can never disagree with the code that produced the results.
"""

import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from .config import AKIYAMA_CLE_GENES, RAW, REPORT, RESULTS
from .gene_notes import NOTES
from .render_pdf import render

# Genes the template names in its own words, and the side the words put them on. If a rerun moves one of them,
# the build stops so the prose can be rewritten instead of going stale.
PROSE_GENES = {"KRT5": "LUSC", "KRT6A": "LUSC", "KRT14": "LUSC", "DSG3": "LUSC", "DSC3": "LUSC", "BNC1": "LUSC",
               "MUC21": "LUAD", "MUC3A": "LUAD", "HNF1B": "LUAD", "ELFN2": "LUAD", "BCL2L15": "LUAD"}


def f3(x):
    return f"{x:.3f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def pm(pair, d=3):
    return f"{pair[0]:.{d}f} ± {pair[1]:.{d}f}"


def pfmt(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def check_prose_genes(top):
    side = dict(zip(top.gene, top.side))
    wrong = {g: s for g, s in PROSE_GENES.items() if side.get(g) != s}
    if wrong:
        raise SystemExit(f"the write-up names genes that are no longer in the top 20 on that side: {wrong}; update the template")
    if not {"ELFN2", "BCL2L15"} <= set(top[top.side == "LUAD"].gene.head(3)):
        raise SystemExit("the summary calls the top LUAD genes 'glandular genes plus two new leads'; update the template")


def label_noise_sentences(t5):
    """Plain sentences for the DNA check, worded by what the two Fisher tests actually show."""
    def stat(r):
        return f"odds ratio {r['odds_ratio']:.1f}, p = {pfmt(r['fisher_p'])}"

    # (label, called, test result) for each direction
    sides = [("LUSC", "LUAD", t5["LUSC-labelled"]), ("LUAD", "LUSC", t5["LUAD-labelled"])]
    sig = [s for s in sides if s[2]["fisher_p"] < 0.05]
    weak = [s for s in sides if s[2]["fisher_p"] >= 0.05]
    if len(sig) == 2:
        return ("It does: " + " and ".join(f'"{lab}" tumours called {called} carry {called} changes more often ({stat(r)})' for lab, called, r in sides)
                + " than the tumours the model agrees with.",
                "Some of its rare mistakes look like label problems, because those tumours carry the other subtype's DNA changes more often than expected.")
    if len(sig) == 1:
        (lab, called, r), (lab2, called2, r2) = sig[0], weak[0]
        return (f'Partly: "{lab}" tumours called {called} carry {called} changes more often than the tumours the model agrees with ({stat(r)}); '
                f'for "{lab2}" tumours called {called2} the trend is the same but not significant ({stat(r2)}).',
                f"Some of its rare mistakes may be label problems: tumours labelled {lab} but called {called} carry {called} DNA changes more often than expected.")
    return ("Not clearly: the trend points that way, but neither test is significant (" + "; ".join(stat(r) for _, _, r in sides) + ").",
            "Its rare mistakes are not clearly explained by DNA changes.")


def build_values():
    m = json.load(open(RESULTS / "model.json"))
    gj = json.load(open(RESULTS / "genes.json"))
    cf = json.load(open(RESULTS / "confounders.json"))
    ex = json.load(open(RESULTS / "external.json"))
    wf = pd.read_csv(RESULTS / "cohort_waterfall.tsv", sep="\t")
    c = pd.read_csv(RESULTS / "cohort.tsv", sep="\t", index_col=0)
    top = pd.read_csv(RESULTS / "top20_genes.tsv", sep="\t")
    bench = pd.read_csv(RESULTS / "external_benchmark.tsv", sep="\t")
    check_prose_genes(top)
    v = {"date": dt.date.today().strftime("%d %B %Y").lstrip("0")}

    names = ["All RNA seq samples in Xena", "Primary tumour only (type 01)", "Remove FFPE samples", "Remove working group pathology exclusions",
             "Remove 'do not use' (treated for an earlier cancer)", "Remove tumours treated before surgery", "Remove recurrence or out of specification",
             "One sample per patient"]
    rows = ["| Step | LUAD | LUSC | Total |", "|---|---|---|---|"] + [f"| {n} | {a} | {b} | {t} |" for n, a, b, t in zip(names, wf.LUAD, wf.LUSC, wf.total)]
    v["waterfall_table"] = "\n".join(rows)
    d = dict(zip(range(len(wf)), wf.total))
    v["n_awg"], v["n_dnu"], v["n_neo"], v["n_recur"] = d[2] - d[3], d[3] - d[4], d[4] - d[5], d[5] - d[6]
    v["n_final"], v["n_luad"], v["n_lusc"] = len(c), int((c.label == 0).sum()), int((c.label == 1).sum())
    v["n_inst"] = c.institution.nunique()
    v["n_inst_shared"] = int(c.groupby("institution").label.nunique().gt(1).sum())
    v["n_tss"] = c.tss.nunique()
    v["n_plates"] = c.plate.nunique()
    v["n_plates_shared"] = int(c.groupby("plate").label.nunique().gt(1).sum())
    v["n_double"] = int((c.n_rna_aliquots > 1).sum())
    v["n_genes"] = f"{m['n_genes_protein_coding']:,}"
    v["tss_accuracy"] = pct(cf["T1_site_batch"]["tss_code_only_accuracy"])

    ms = m["mean_sd"]
    v["inst_auroc"] = f3(ms["institution"]["auroc"][0])
    v["rand_auroc"] = f3(ms["random"]["auroc"][0])
    v["delta_auroc"] = f"{m['delta_auroc_random_minus_institution']:+.4f}"
    ci = m["auroc_ci95_institution"]
    rr = m["rank_rules_cv"]
    perf = ["| Split (mean ± SD over 5 seeds) | AUROC | AUPRC | Accuracy | Balanced accuracy | Calibration error (ECE) |", "|---|---|---|---|---|---|"]
    for k, lab in (("institution", "Grouped by institution (used)"), ("random", "Random (for comparison)")):
        perf.append(f"| {lab} | {pm(ms[k]['auroc'])} | {pm(ms[k]['auprc'])} | {pm(ms[k]['accuracy'])} | {pm(ms[k]['balanced_accuracy'])} | {pm(ms[k]['ece'])} |")
    perf.append(f"| Seed 42, grouped by institution, 95% bootstrap CI | {f3(m['institution']['auroc'])} ({f3(ci[0])} to {f3(ci[1])}) | {f3(m['institution']['auprc'])} | {f3(m['institution']['accuracy'])} | {f3(m['institution']['balanced_accuracy'])} | {f3(m['institution']['ece'])} |")
    perf.append(f"| Nine gene pair rules (learned inside the same folds) | {f3(rr['auroc'])} | {f3(rr['auprc'])} | {f3(rr['accuracy'])} | {f3(rr['balanced_accuracy'])} | n/a |")
    v["performance_table"] = "\n".join(perf)
    v["rules_auroc"] = f3(rr["auroc"])
    nn = m["nearest_neighbour_audit"]
    v["nn_same_random"] = pct(nn["random"]["frac_nearest_from_same_institution"], 1)
    v["nn_same_inst"] = pct(nn["institution"]["frac_nearest_from_same_institution"], 1)
    can = m["random_label_canary"]
    v["canary_ours"] = f3(can["selection inside folds (our pipeline)"])
    v["canary_leaky"] = f3(can["selection before split (leaky)"])
    v["trust90"] = pct([t for t in m["trustworthiness"] if t["coverage"] == 0.9][0]["accuracy"], 1)

    def row(cohort, method):
        r = bench[(bench.cohort == cohort) & (bench.method == method)]
        assert len(r) == 1, (cohort, method)
        return r.iloc[0]

    cp = row("CPTAC-3", "L2 logistic (ours)")
    up = row("Uppsala RNA-seq (GSE81089)", "L2 logistic (ours)")
    arrays = [ch for ch in bench.cohort.unique() if ch.startswith("GSE") and "81089" not in ch]
    acc_rules = [row(ch, "9 rank rules (ours)").accuracy for ch in arrays]
    asian = ex["cptac3_ancestry_l2"]["Asian"]
    k = ex["korea"]
    v["external_sentence"] = (
        f"On CPTAC-3 ({ex['cptac3_n']} patients, same GDC pipeline) the model reaches AUROC {f3(cp.auroc)} and accuracy {pct(cp.accuracy, 1)}; "
        f"in its {ex['cptac3_n_asian']} Asian patients, AUROC {f3(asian['auroc'])} (TCGA has only {ex['tcga_n_asian']} Asian patients in this cohort). "
        f"On Uppsala RNA seq ({int(up.n)} patients, histology reviewed again by a pathologist in 2013) it reaches AUROC {f3(up.auroc)}, but accuracy only {pct(up.accuracy, 1)}: "
        f"the ranking transfers while the 0.5 cut point drifts on a new pipeline. The nine gene pair rules, which compare genes inside one sample, do not drift: "
        f"accuracy {pct(row('Uppsala RNA-seq (GSE81089)', '9 rank rules (ours)').accuracy, 1)} there and {pct(min(acc_rules), 1)} to {pct(max(acc_rules), 1)} on four microarray cohorts "
        f"(France, Sweden, Canada and the SBV IMPROVER set). Of {k['n_luad']} Korean adenocarcinomas (GSE40419), {pct(k['luad_called_luad_l2'], 1)} were called LUAD, "
        f"including {pct(k['never_smoker_luad_called_luad_l2'])} of {k['never_smoker_luad_n']} never smokers.")
    n_ext = sum(int(row(ch, "9 rank rules (ours)").n) for ch in ["CPTAC-3"] + arrays)
    v["n_ext_patients"] = f"{n_ext:,}"
    v["n_asian"] = ex["cptac3_n_asian"]
    bm = ex["benchmark_mean"]
    v["bench_rules"] = pct(bm["9 rank rules (ours)"]["accuracy"], 1)
    v["bench_hamaneh"] = pct(bm["Hamaneh 2022 8-gene"]["accuracy"], 1)
    v["bench_li"] = pct(bm["Li 2019 KRT5>AGR2"]["accuracy"], 1)

    t = ["| Rank | Gene | Higher in | Share of gap | Kept | AUROC, TCGA | AUROC, CPTAC-3 | log2 change (LUSC vs LUAD) |", "|---|---|---|---|---|---|---|---|"]
    for r in top.itertuples():
        t.append(f"| {r.rank} | {r.gene} | {r.side} | {pct(r.share_of_gap, 2)} | {pct(r.bootstrap_top20_freq)} | {f3(r.single_gene_auroc)} | "
                 f"{f3(r.cptac3_auroc) if pd.notna(r.cptac3_auroc) else 'n/a'} | {r.log2fc_lusc_vs_luad:+.2f} |")
    v["top20_table"] = "\n".join(t)
    checks = []
    if (top.de_q < 1e-10).all():
        checks.append("All 20 also differ strongly in a plain t test (q < 1e-10).")
    cp_ok = top.cptac3_auroc.dropna()
    if len(cp_ok):
        checks.append(f"On their own, they separate the subtypes in CPTAC-3 too (single gene AUROC {f3(cp_ok.min())} to {f3(cp_ok.max())}).")
    v["top20_checks"] = " ".join(checks)
    low = gj["by_weight_top10_low_auroc"]
    if len(low) < 2:
        raise SystemExit("the write-up says the raw weight top 10 holds genes that barely differ between the types; update the template")
    v["weight_examples"] = f"{', '.join(low[:-1])} and {low[-1]}" if len(low) <= 3 else f"{', '.join(low[:3])} and others"
    v["kept_median"] = pct(gj["top20_bootstrap_freq_median"])
    v["weight_kept"] = pct(gj["by_weight_bootstrap_freq_median"])
    v["n_model_genes"] = f"{gj['n_model_genes']:,}"
    v["top20_share"] = pct(gj["top20_share_of_gap"], 1)
    cpa = dict(zip(top.gene, top.cptac3_auroc))
    v["elfn2_cp"], v["bcl2l15_cp"] = f3(cpa["ELFN2"]), f3(cpa["BCL2L15"])
    notes = []
    for r in top.head(10).itertuples():
        if r.gene not in NOTES:
            raise KeyError(f"no checked note for top 10 gene {r.gene}; add it to gene_notes.py")
        what, marker = NOTES[r.gene]
        notes.append(f"- **{r.rank}. {r.gene}, higher in {r.side}.** {what} {marker}")
    v["top10_notes"] = "\n".join(notes)
    v["top_lusc_genes"] = ", ".join(top[top.side == "LUSC"].gene.head(3))
    v["top_luad_genes"] = ", ".join(top[top.side == "LUAD"].gene.head(3))
    first, second = top.iloc[0], top.iloc[1]
    if first.side != second.side:
        v["top2_sentence"] = f"The top two genes are {first.gene} (higher in {first.side}) and {second.gene} (higher in {second.side}), one for each type."
        v["stain_what"] = "both"
    else:
        other = top[top.side != first.side].iloc[0]
        v["top2_sentence"] = (f"The top two genes, {first.gene} and {second.gene}, are both higher in {first.side}, so I would add the top {other.side} gene, "
                              f"{other.gene}, to give each type a positive stain.")
        v["stain_what"] = "all three"
    stained = {first.gene, second.gene} if first.side != second.side else {first.gene, second.gene, other.gene}
    if "KRT5" in stained:
        v["top2_sentence"] += " KRT5 is already stained in practice as part of CK5/6 (PMID 25120805), so it doubles as a check that the method works."

    gained = gj["lipid_lusc_gained"]
    v["lipid_gained"] = len(gained)
    v["lipid_not_gained"] = " or ".join(g for g in AKIYAMA_CLE_GENES if g not in gained)
    v["cers3_fold"] = f"{2 ** gj['cers3_lusc_log2fc']:.0f}"
    la = list(gj["lipid_module_auroc"].values())
    v["lipid_range"] = f"{f3(min(la))} to {f3(max(la))}"

    t2 = cf["T2_covariates_only"]
    v["cov_auroc"] = f3(t2["covariate_model"]["auroc"])
    v["sex_auroc"] = f3(t2["single_covariate_auroc"]["male"])
    v["pack_auroc"] = f3(t2["single_covariate_auroc"]["pack_years"])
    if not (c[c.label == 1].sex.eq("male").mean() > c[c.label == 0].sex.eq("male").mean() and c[c.label == 1].pack_years.median() > c[c.label == 0].pack_years.median()):
        raise SystemExit("the write-up says LUSC patients are more often men and smoked more; update the template")
    v["depth_luad"] = f"{c[c.label == 0].library_size.median() / 1e6:.0f}"
    v["depth_lusc"] = f"{c[c.label == 1].library_size.median() / 1e6:.0f}"
    st = pd.DataFrame(cf["T3_strata"])
    v["strata_min"] = f3(st.auroc.min())
    v["n_strata"] = len(st)
    neo = cf["T4_treatment_and_exclusions"]["scored"]["neoadjuvant-treated"]
    v["neo_n"], v["neo_acc"] = neo["n"], pct(neo["accuracy_vs_label"])
    v["cv_acc"] = pct(1 - cf["T4_treatment_and_exclusions"]["cv_error_rate"], 1)
    t5 = cf["T5_label_noise_dna"]
    v["n_disputed"] = t5["n_disputed"]
    v["label_noise_sentence"], v["short_label_sentence"] = label_noise_sentences(t5)
    t6 = cf["T6_tumour_content"]
    v["lowpur_lusc"] = pct(t6["real_low_purity"]["lusc_recall"])
    v["lowpur_luad"] = pct(t6["real_low_purity"]["luad_recall"])
    d10 = [r for r in t6["in_silico_dilution"] if r["tumour_fraction"] == 0.1][0]
    v["dil10_lusc"], v["dil10_luad"] = pct(d10["lusc_recall"]), pct(d10["luad_recall"])
    v["korea_normal"] = pct(k["normal_lung_called_luad_l2"])
    # values used only by the README
    ps = m["mean_sd"]["institution"]
    v["inst_auroc_pm"], v["inst_acc_pm"] = pm(ps["auroc"]), pm(ps["accuracy"])
    v["cp_n"], v["cp_auroc"], v["cp_acc"] = int(cp.n), f3(cp.auroc), pct(cp.accuracy, 1)
    v["asian_auroc"] = f3(asian["auroc"])
    v["up_n"], v["up_auroc"] = int(up.n), f3(up.auroc)
    v["up_rules_acc"] = pct(row("Uppsala RNA-seq (GSE81089)", "9 rank rules (ours)").accuracy, 1)
    v["arrays_n"] = sum(int(row(ch, "9 rank rules (ours)").n) for ch in arrays)
    v["arrays_range"] = f"{pct(min(acc_rules), 1)} to {pct(max(acc_rules), 1)}"
    v["korea_n"], v["korea_luad"] = k["n_luad"], pct(k["luad_called_luad_l2"], 1)
    return v


def write_example():
    """Six CPTAC-3 tumours (3 LUAD, 3 LUSC; never used in training) as a ready input for `predict`."""
    from . import io

    x, y, sel = io.cptac3()
    genes = json.load(open(RESULTS / "final_model.json"))["genes"]
    pick = list(sel.index[y == 0][:3]) + list(sel.index[y == 1][:3])
    out = x.loc[x.index.intersection(genes), pick].round(4)
    (REPORT.parent / "examples").mkdir(exist_ok=True)
    out.to_csv(REPORT.parent / "examples" / "cptac3_six_tumours.tsv", sep="\t")
    pd.Series(np.where(y[np.isin(sel.index, pick)] == 1, "LUSC", "LUAD"), index=[p for p in sel.index if p in pick]).reindex(pick).to_csv(
        REPORT.parent / "examples" / "cptac3_six_tumours_true_labels.tsv", sep="\t", header=["histology"])


def run():
    v = build_values()
    tmpl = (REPORT / "writeup_template.md").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{(\w+)\}", tmpl)) - set(v))
    if missing:
        raise KeyError(f"no value for placeholders: {missing}")
    text = tmpl.format_map(v)
    (REPORT / "writeup.md").write_text(text, encoding="utf-8")
    pages = render(text, REPORT / "writeup.pdf", "Telling LUAD from LUSC with TCGA RNA seq", base_dir=REPORT)
    json.dump(v, open(RESULTS / "report_values.json", "w"), indent=1, default=str)
    rt = (REPORT / "README_template.md").read_text(encoding="utf-8")
    miss = sorted(set(re.findall(r"\{(\w+)\}", rt)) - set(v))
    if miss:
        raise KeyError(f"README placeholders without a value: {miss}")
    (REPORT.parent / "README.md").write_text(rt.format_map(v), encoding="utf-8")
    if (RAW / "external" / "cptac3" / "cptac3_tumour_tpm.tsv.gz").exists():
        write_example()
    else:
        print("CPTAC-3 data not downloaded; kept the example files already in examples/")
    print(f"writeup.pdf: {pages} pages")
    if pages > 3:
        raise SystemExit(f"write-up is {pages} pages; the limit is 3")
    return pages


if __name__ == "__main__":
    run()
