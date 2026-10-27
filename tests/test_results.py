"""Checks on the real outputs. Skipped unless the pipeline has been run (results/ present)."""

import json

import pandas as pd
import pytest

from luadlusc.config import RESULTS

pytestmark = pytest.mark.skipif(not (RESULTS / "cohort.tsv").exists(), reason="run `python -m luadlusc all` first")


@pytest.fixture(scope="module")
def cohort():
    return pd.read_csv(RESULTS / "cohort.tsv", sep="\t", index_col=0)


def test_cohort_rules(cohort):
    assert cohort.patient.is_unique
    assert (cohort.index.str[13:15] == "01").all()
    assert not cohort.is_ffpe.any()
    ex = pd.read_csv(RESULTS / "excluded_samples.tsv", sep="\t", index_col=0)
    removed_patients = set(ex[ex.reason.str.contains("AWG|Do_not_use|neoadjuvant|recurrence")].patient)
    assert not set(cohort.patient) & removed_patients


def test_waterfall_adds_up(cohort):
    wf = pd.read_csv(RESULTS / "cohort_waterfall.tsv", sep="\t")
    assert (wf.total == wf.LUAD + wf.LUSC).all()
    assert wf.total.is_monotonic_decreasing
    assert wf.total.iloc[-1] == len(cohort)


def test_site_code_is_never_shared(cohort):
    assert cohort.groupby("tss").label.nunique().max() == 1


def test_cv_folds_separate_institutions(cohort):
    pred = pd.read_csv(RESULTS / "cv_predictions.tsv", sep="\t", index_col=0).reindex(cohort.index)
    per_inst = pred.groupby("institution").fold.nunique()
    assert (per_inst == 1).all(), "an institution appears in more than one test fold"


def test_model_is_well_above_chance_and_null_is_at_chance():
    m = json.load(open(RESULTS / "model.json"))
    assert m["institution"]["auroc"] > 0.95
    canary = m["random_label_canary"]["selection inside folds (our pipeline)"]
    assert 0.4 < canary < 0.6


def test_report_numbers_come_from_results():
    md = (RESULTS.parent / "report" / "writeup.md")
    if not md.exists():
        pytest.skip("report not built")
    text = md.read_text(encoding="utf-8")
    assert "{" not in text and "}" not in text, "unfilled placeholder in the write-up"
