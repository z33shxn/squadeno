"""Command line: `python -m luadlusc <step>`.

Steps (run in order by `all`):
  download    fetch every input from its public source (skips files already present)
  verify      compare local inputs with the SHA-256 manifest used for the published results
  cohort      part 1: which samples are used, and the table of removed samples
  model       part 2: nested cross validation grouped by institution, split comparison, leakage audit, final model
  genes       part 3: the top 20 genes, their stability, differential expression, the lipid gene set
  confounders part 4: six tests of reasons other than tumour biology (T1 to T6)
  external    new cohorts with the model frozen, and the comparison with published classifiers
  figures     the figures in figures/
  report      fill report/writeup.md and README.md from results/ and render report/writeup.pdf

Score your own samples (genes as rows, samples as columns, TPM or log2(TPM+1)):
  python -m luadlusc predict my_samples.tsv [scores.tsv]
"""

import argparse
import time

STEPS = ("cohort", "model", "genes", "confounders", "external", "figures", "report")


def _run(step):
    t = time.time()
    if step == "download":
        from .download import fetch_all
        fetch_all()
    elif step == "verify":
        from .download import verify_checksums
        verify_checksums()
    elif step == "cohort":
        from .cohort import build_cohort
        _, _, wf, _ = build_cohort()
        print(wf.to_string(index=False))
    elif step == "model":
        from .analysis_model import run
        run()
    elif step == "genes":
        from .analysis_genes import run
        run()
    elif step == "confounders":
        from .analysis_confounders import run
        run()
    elif step == "external":
        from .analysis_external import run
        run()
    elif step == "figures":
        from .figures import run
        run()
    elif step == "report":
        from .report import run
        run()
    else:
        raise SystemExit(f"unknown step {step}")
    print(f"[{step}] done in {time.time() - t:.0f} s", flush=True)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        from .predict import main as predict_main
        predict_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(prog="luadlusc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("all", "download", "verify") + STEPS)
    a = ap.parse_args()
    if a.step == "all":
        for s in ("download",) + STEPS:
            _run(s)
    else:
        _run(a.step)


if __name__ == "__main__":
    main()
