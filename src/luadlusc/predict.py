"""Score new tumour samples with the frozen model trained on TCGA.

Input: a table with genes (HGNC symbols) as rows and samples as columns, holding TPM or log2(TPM+1)
values from RNA seq. TPM is detected automatically (any value above 50) and converted to log2(TPM+1).
Output: one row per sample with P(LUSC) from the main model, the share of the 9 rank rules voting LUSC,
the call, and a flag when the two disagree or the main model is unsure (P between 0.2 and 0.8).

Research use only. This is not a medical device and must not guide the care of any patient.
"""

import json
import sys

import numpy as np
import pandas as pd

from .config import RESULTS
from .models import predict_from_json, predict_rank_rules


def score(table: pd.DataFrame, model_path=RESULTS / "final_model.json", rules_path=RESULTS / "rank_rules.json") -> pd.DataFrame:
    x = table.apply(pd.to_numeric, errors="coerce")
    if x.max().max() > 50:
        x = np.log2(x.clip(lower=0) + 1)
    model = json.load(open(model_path))
    rules = [(r["lusc_gene"], r["luad_gene"], r["training_score"]) for r in json.load(open(rules_path))]
    present = sum(g in x.index for g in model["genes"]) / len(model["genes"])
    p = predict_from_json(model, x)
    votes, n_rules = predict_rank_rules(x, rules)
    out = pd.DataFrame({"p_lusc_model": p.round(4), "rules_voting_lusc": votes.round(3)}, index=x.columns)
    out["call"] = np.where(out.p_lusc_model > 0.5, "LUSC", "LUAD")
    disagree = (out.p_lusc_model > 0.5) != (out.rules_voting_lusc > 0.5)
    unsure = (out.p_lusc_model > 0.2) & (out.p_lusc_model < 0.8)
    out["check"] = np.where(disagree | unsure, "review: model unsure or rules disagree", "")
    out.attrs["model_genes_present"] = present
    out.attrs["rules_used"] = n_rules
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        raise SystemExit("usage: python -m luadlusc predict <expression.tsv or .csv> [output.tsv]")
    sep = "," if argv[0].lower().endswith(".csv") else "\t"
    table = pd.read_csv(argv[0], sep=sep, index_col=0)
    res = score(table)
    share = res.attrs["model_genes_present"]
    print(f"model genes present: {share:.0%}; rank rules usable: {res.attrs['rules_used']}/9")
    if share < 0.8:
        print("warning: fewer than 80% of the model genes are present; prefer the rank rules column")
    if len(argv) > 1:
        res.to_csv(argv[1], sep="\t")
        print("wrote", argv[1])
    else:
        print(res.to_string())


if __name__ == "__main__":
    main()
