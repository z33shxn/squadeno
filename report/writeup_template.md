# Telling LUAD from LUSC with TCGA RNA seq, and what the model learned

<small>Mohammed Zeeshan, {date}. Code, data list and every result: github.com/z33shxn/tcga-luad-lusc-classifier. All numbers below are filled in by the code from the results files.</small>

**In short.** A simple linear model tells LUAD from LUSC with AUROC {inst_auroc} when it is tested on hospitals it never saw. Frozen, it reaches AUROC {cp_auroc} on {cp_n} new CPTAC-3 patients, {n_asian} of them Asian, and nine gene pair rules learned with it hold up on {n_ext_patients} new patients in five independent cohorts from other countries and platforms. Its strongest genes are the squamous, skin like programme that pathologists already stain for in LUSC ({top_lusc_genes}) and glandular genes plus two new leads in LUAD ({top_luad_genes}). Hospital, sex, smoking, RNA quality and tumour purity do not explain it. {short_label_sentence}

## 1. Data: which samples and why

I used the UCSC Xena GDC hub files the email suggested (STAR gene expression as TPM, and the phenotype tables). I checked that Xena's numbers equal the GDC original files: for two tumours I downloaded from GDC, all 60,660 genes matched to machine precision. Every rule below was applied to the TCGA barcode (for example TCGA-55-7726-01A: site 55, patient 7726, sample type 01 = primary tumour, vial A) or to an official TCGA quality table. Each removed sample and its reason is listed in the repository.

{waterfall_table}

Why each step. Normal tissue (type 11) and recurrences (02) are not the question. FFPE samples have damaged RNA and those patients also have a frozen sample. On review, the TCGA working group called {n_awg} LUAD cases as "NSCLC, not otherwise specified" or "favour large cell neuroendocrine", so their label is not reliable. PanCanAtlas marks {n_dnu} patients "do not use" because of treatment for an earlier cancer. GDC notes show {n_neo} tumours were treated before surgery, which changes gene expression, and {n_recur} cases were a recurrence or had results out of specification. The final cohort is **{n_final} tumours ({n_luad} LUAD, {n_lusc} LUSC), one per patient**, from {n_inst} institutions. For {n_double} tumours that were sequenced twice, Xena stores the mean of the two runs.

## 2. Model and a split I can defend

**Model.** L2 penalised logistic regression on log2(TPM+1) of {n_genes} protein coding genes. The gene filter (expressed in over 20% of training tumours) and the scaling are fitted inside each training fold. Classes are weighted equally. The penalty is chosen by an inner cross validation from four values, and the whole procedure is repeated with five seeds (42, 123, 456, 789, 1337).

**Split.** Five folds grouped by institution: a hospital is either fully in training or fully in testing, and each patient appears once. Why this suits TCGA: the site code in the barcode was issued separately for each disease, so none of the {n_tss} codes is shared by LUAD and LUSC, and the code alone predicts the label {tss_accuracy} of the time. Sequencing plates are almost as pure ({n_plates_shared} of {n_plates} contain both types). Institutions, however, are shared ({n_inst_shared} of {n_inst} sent both types). Grouping by institution therefore asks the fair question: does the model work at a hospital it has never seen?

{performance_table}

Leakage audit, in the style of a residual similarity check. With a random split, the closest training tumour to a test tumour came from the same hospital {nn_same_random} of the time; with my split, {nn_same_inst}. On shuffled labels my pipeline scores AUROC {canary_ours} (chance), while the common mistake of picking genes before splitting scores {canary_leaky}. If the least confident 10% are set aside, accuracy rises to {trust90}.

**External tests (model frozen, nothing refitted).** {external_sentence} Against two published classifiers on the same seven cohorts, my nine gene pair rules average accuracy {bench_rules} versus {bench_hamaneh} (Hamaneh 2022, 8 genes, PMID 35722224) and {bench_li} (Li 2019, KRT5 > AGR2, PMID 31752667). So the model matches the best published signature; the value of this work is in the checks, not a new record.

## 3. Biology: the 20 genes that drive the classification

For a linear model, the gap between its mean score for LUSC and for LUAD splits exactly into one term per gene: the gene's weight times its difference in mean between the two types (in standard deviations). I rank genes by that term. Ranking by raw weight would instead put {weight_examples} in the top 10, although they barely differ between the types (single gene AUROC below 0.6); such genes only adjust single samples. "Kept" is how often a gene stays in the top 20 when the model is refitted on 50 resamples (median {kept_median}, against {weight_kept} for the raw weight list). The model spreads its weight over {n_model_genes} genes, so these 20 carry only {top20_share} of the gap, yet a few genes are enough: the nine gene pair rules reach AUROC {rules_auroc}. {top20_checks}

{top20_table}

**The top 10, what they do, and whether they make sense as markers.**

{top10_notes}

**What the model learned.** The LUSC side is the programme of layered, skin like epithelium: keratins (KRT5, KRT6A, KRT14), desmosome glue (DSG3, DSC3) and BNC1, a basal cell factor that p63 switches on (PMID 21741828). The LUAD side is glandular: mucins (MUC21, MUC3A) and HNF1B, a gene that LUSC switches off by DNA methylation (PMID 29169318, 32117905). This is the split that the standard stain panel captures (p40 and CK5/6 against TTF-1 and Napsin A, PMID 25120805). ELFN2 and BCL2L15 are the exceptions: I found no study of either in lung cancer subtyping, yet both separate the types in CPTAC-3 too (AUROC {elfn2_cp} and {bcl2l15_cp}). They are new leads, not established markers.

**A lipid view, chosen in advance.** Akiyama 2017 lists the eight genes that build the lipid envelope of the skin barrier (PMID 28623042). I tested exactly that list. Against each patient's own normal lung, LUSC raised {lipid_gained} of the 8 (CERS3 about {cers3_fold} fold), but not {lipid_not_gained}. The 8 gene score separates the subtypes in all seven cohorts (AUROC {lipid_range}). LUAD instead raised FA2H, matching Lemay 2019, who measured both this gene and the lipids it makes (PMID 31409741). These are gene levels, not lipid levels; they predict more of the very long chain ceramides of the skin barrier in LUSC, which a lipid lab can measure.

## 4. Could it be something other than tumour biology?

- **Hospital batch.** If the model had learned hospitals, scores would drop when hospitals are held out. They did not: random split {rand_auroc}, split by institution {inst_auroc} (difference {delta_auroc}).
- **Non biological variables.** A model built only from sex, age, pack years, two tumour purity measures (DNA based ABSOLUTE and the pathologist's tumour nuclei count), RNA integrity, sequencing depth and tumour location reaches AUROC {cov_auroc}, far below the expression model. Much of that is a real difference between the patients: here LUSC patients are more often men (sex alone, AUROC {sex_auroc}) and smoked more (pack years, {pack_auroc}). Sequencing depth also differs by type (median {depth_luad} vs {depth_lusc} million reads), which is why the model uses TPM, not raw counts.
- **Within every subgroup.** AUROC stays at or above {strata_min} in all {n_strata} subgroups: sex, never and ever smokers, each purity tertile, pathologist tumour nuclei below and above 60%, low and high RNA integrity, low and high depth, central and peripheral tumours, early and late stage, and hospitals that sent both types.
- **Treatment before surgery.** Scored by the frozen model, the {neo_n} pre treated tumours were labelled correctly only {neo_acc} of the time, against {cv_acc} in the cohort. This supports removing them.
- **Wrong labels.** For {n_disputed} tumours the model and the label disagree. If the label is the problem, their DNA should carry the other subtype's changes (for LUAD, changes such as KRAS, EGFR or STK11 mutation or NKX2-1 gain, PMID 25079552; for LUSC, SOX2, FGFR1 or TP63 gain or PTEN, KMT2D or NFE2L2 mutation, PMID 22960745). {label_noise_sentence} TCGA's own rules allow up to 5% adenocarcinoma inside a squamous sample (GDC lung quality control form).
- **Little tumour in the sample.** Normal lung looks glandular, so dilution pushes calls toward LUAD. In real tumours with purity below 0.25, LUSC was called correctly {lowpur_lusc} of the time against {lowpur_luad} for LUAD. When I mixed test tumours with held out normal lung down to 10% tumour, LUSC recall fell to {dil10_lusc} while LUAD stayed at {dil10_luad}. In Korean data, {korea_normal} of normal lungs were called LUAD.

![Accuracy on each cohort, and the tumour content test](../figures/writeup_figure.png)
<small>**Figure 1.** A: accuracy with the model frozen. The L2 model needs RNA seq, so the four microarray cohorts are scored by the gene pair rules and the two published classifiers. B: test tumours mixed in silico with held out normal lung; as the tumour share falls, LUSC calls drift to LUAD.</small>

## 5. Clinic and bench

**Why the subtype matters.** Pathologists looking at H&E slides alone agree only moderately on squamous versus non squamous (kappa 0.55, PMID 22583114), and in small biopsies many tumours stay "not otherwise specified" until stains are added (stains resolved 73% with 86% accuracy, PMID 20195168). The label changes treatment. Pemetrexed helped adenocarcinoma (12.6 vs 10.9 months) but did worse than gemcitabine in squamous cancer (9.4 vs 10.8 months; PMID 18506025). Bevacizumab caused major lung bleeding linked to squamous histology, so squamous patients were excluded from its pivotal trial (PMID 15169807, 17167137). Immunotherapy combinations are tested separately: pembrolizumab with pemetrexed for non squamous disease (PMID 29658856) and with paclitaxel for squamous disease (PMID 30280635). Guidelines ask for EGFR, ALK, ROS1 and other gene tests in adenocarcinoma (PMID 29355391); in Asia, about 80% of adenocarcinomas carry a targetable change (PMID 39615406). Calling a Korean adenocarcinoma squamous can therefore hide a targeted drug, and calling a squamous cancer adenocarcinoma can lead to pemetrexed or bevacizumab.

**A wet lab test of the top markers.** {top2_sentence} I would stain {stain_what} by immunohistochemistry on tissue microarrays of resected lung cancers (two or three cores per tumour; core sampling is reliable for lineage stains in NSCLC, PMID 24486059), with p40 and TTF-1 on serial sections as the reference (p40 is 100% sensitive and 98% specific for squamous carcinoma, PMID 22056955). Tissue would come from a hospital archive such as Gachon University Gil Medical Center, whose pathology department has published stain and FISH work on resected NSCLC (PMID 22207554), under ethics approval. I would enrich for the hard cases: TTF-1 negative adenocarcinomas (about 28% of adenocarcinomas, PMID 12023581), solid and poorly differentiated tumours, and cases first called "not otherwise specified". Two pathologists, blind to the RNA result and to each other, would give H scores to tumour cells only, ignoring normal airway and immune cells. I would report sensitivity, specificity and AUROC against the final diagnosis, agreement as Cohen's kappa, and McNemar's test against p40 and TTF-1, with the sample size set for the expected sensitivity (PMID 24582925). Matched frozen tissue could add a measurement of very long chain ceramides by liquid chromatography mass spectrometry, the kind of ceramide work this lab already does (PMID 35513574).

**Limits.** The labels come from one pathology review per tumour, so part of the measured error is label error. Most TCGA and European patients are white; the Asian checks rest on CPTAC-3 and one Korean cohort. I tested resected tumours, not small biopsies, and low tumour content pushes calls toward LUAD, so biopsies need their own test before any clinical use.

**Reproduce.** `pip install -e .` then `python -m luadlusc all` downloads the public data, reruns every analysis and rebuilds this document.

## AI use

I used an AI assistant (Claude, Anthropic) throughout this project: to draft and debug the code, to propose and run analyses that I then reviewed, to search PubMed and check that each PMID exists and supports its claim, and to edit the English. I directed the work, made the analysis decisions, reviewed the results and the claims, and take responsibility for the content.
