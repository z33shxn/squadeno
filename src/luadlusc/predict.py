"""Score new tumour samples with the frozen model trained on TCGA.

Input: a table with genes (HGNC symbols) as rows and samples as columns, holding TPM or log2(TPM+1)
values from RNA seq. TPM is detected automatically (any value above 50) and converted to log2(TPM+1).
Output: one row per sample with P(LUSC) from the main model, the share of the 9 rank rules voting LUSC,
the share of model genes found, the call, and a note when the result needs a closer look.

When fewer than 80% of the model genes are present, the main model is not trusted: missing genes are set to
their training mean, so with many of them every score slides toward the model's baseline (about 0.62).
The call then comes from the rank rules, which compare genes inside each sample and need only 18 genes.

Research use only. This is not a medical device and must not guide the care of any patient.
"""

import json
import sys

import numpy as np
import pandas as pd

from .config import RESULTS
from .models import predict_from_json, predict_rank_rules

MIN_GENE_SHARE = 0.8


def _input_warnings(x_log):
    """Plain warnings when the input does not look like whole transcriptome log2(TPM+1)."""
    notes = []
    sums = np.power(2.0, x_log).sub(1).sum(axis=0)
    if sums.median() > 1.5e6:
        notes.append("column sums are far above 1 million TPM: the input may be counts or FPKM, not TPM")
    if (x_log.min(axis=0) > 1).mean() > 0.5:
        notes.append("most samples have no zero values: the input may be microarray data; use the rank rules column")
    return notes


def score(table: pd.DataFrame, model_path=RESULTS / "final_model.json", rules_path=RESULTS / "rank_rules.json") -> pd.DataFrame:
    x = table.apply(pd.to_numeric, errors="coerce")
    n_dup = int(x.index.duplicated().sum())
    if n_dup:  # keep the row with the highest mean for a repeated gene symbol
        x = x.iloc[np.argsort(-x.mean(axis=1).values, kind="stable")]
        x = x[~x.index.duplicated(keep="first")]
    if x.max().max() > 50:
        x = np.log2(x.clip(lower=0) + 1)
    model = json.load(open(model_path))
    rules = [(r["lusc_gene"], r["luad_gene"], r["training_score"]) for r in json.load(open(rules_path))]
    present = sum(g in x.index for g in model["genes"]) / len(model["genes"])
    p = predict_from_json(model, x)
    votes, n_rules = predict_rank_rules(x, rules)
    out = pd.DataFrame({"p_lusc_model": p.round(4), "rules_voting_lusc": votes.round(3)}, index=x.columns)
    out["model_genes_present"] = round(present, 3)
    if present >= MIN_GENE_SHARE:
        out["call"] = np.where(out.p_lusc_model > 0.5, "LUSC", "LUAD")
        disagree = (out.p_lusc_model > 0.5) != (out.rules_voting_lusc > 0.5)
        unsure = (out.p_lusc_model > 0.2) & (out.p_lusc_model < 0.8)
        out["check"] = np.where(disagree | unsure, "review: model unsure or rules disagree", "")
    else:
        out["call"] = np.where(out.rules_voting_lusc > 0.5, "LUSC", np.where(out.rules_voting_lusc < 0.5, "LUAD", "unclear"))
        out["check"] = f"few model genes found ({present:.0%}): call made by the rank rules"
    out.attrs.update(model_genes_present=present, rules_used=n_rules, duplicates_dropped=n_dup, warnings=_input_warnings(x))
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        raise SystemExit("usage: python -m luadlusc predict <expression.tsv or .csv> [output.tsv]")
    sep = "," if argv[0].lower().endswith(".csv") else "\t"
    table = pd.read_csv(argv[0], sep=sep, index_col=0)
    res = score(table)
    print(f"model genes present: {res.attrs['model_genes_present']:.0%}; rank rules usable: {res.attrs['rules_used']}/9")
    if res.attrs["duplicates_dropped"]:
        print(f"note: {res.attrs['duplicates_dropped']} repeated gene symbols were merged (kept the row with the highest mean)")
    if res.attrs["model_genes_present"] < MIN_GENE_SHARE:
        print(f"warning: fewer than {MIN_GENE_SHARE:.0%} of the model genes are present, so the call comes from the rank rules")
    for w in res.attrs["warnings"]:
        print("warning:", w)
    if len(argv) > 1:
        res.to_csv(argv[1], sep="\t")
        print("wrote", argv[1])
    else:
        print(res.to_string())


if __name__ == "__main__":
    main()
