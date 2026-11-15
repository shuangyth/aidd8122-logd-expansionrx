"""Build the presentation (.pptx) from saved results; convert to PDF with LibreOffice.
Usage: python build_slides.py [repo_url] [revision]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES, FIG, P2 = ROOT / "results", ROOT / "figures", ROOT / "results" / "phase2"
REPO = sys.argv[1] if len(sys.argv) > 1 else "github.com/<USER>/<REPO>"
REV = sys.argv[2] if len(sys.argv) > 2 else "<commit>"
NAME_CN = os.environ.get("AIDD_STUDENT_NAME", "Student")
SID = os.environ.get("AIDD_STUDENT_ID", "P00000")
OUT = HERE / "AIDD8122_Final_Project_Slides.pptx"

# ----------------------------------------------------------------- numbers
summ = pd.read_csv(RES / "metrics_test_summary.csv").set_index("model")
mae = {m: summ.loc[m, "MAE_mean"] for m in summ.index}
rel = {m: 100 * (mae["M0"] - mae[m]) / mae["M0"] for m in mae}
pdiff = pd.read_csv(RES / "paired_diff_M3_minus_M2.csv").MAE_M3_minus_M2
boot = json.loads((RES / "exploratory_bootstrap_ci.json").read_text())
chg = json.loads((RES / "exploratory_charged_subset.json").read_text())
sens = json.loads((RES / "sensitivity_flat_group_split.json").read_text())
p2 = json.loads((P2 / "phase2_summary.json").read_text())
a1, a2, a3, a4 = p2["act1"], p2["act2"], p2["act3"], p2["act4"]
prot, opt, rimp = a2["protocols"], a2["optimism_ratio_vs_test"], a2["rel_improvement_vs_M0_pct"]
t20 = pd.read_csv(P2 / "act3_all_configs_cv_vs_test.csv")
quint = pd.read_csv(P2 / "act1_train_logd_by_id_quintile.csv")["mean"].tolist()
smd = pd.read_csv(P2 / "act1_descriptor_smd.csv").set_index("descriptor")
rc = json.loads((P2 / "act4_risk_coverage_curves.json").read_text())
PM = a3["primary_test_mae_mean"]
mlp = json.loads((P2 / "mlp_summary.json").read_text())
M8 = mlp["test_mae"]["M8"]
bsp = a3["bootstrap"]


def T(f, s, c):
    return float(t20[(t20.family == f) & (t20.scheme == s)][c].iloc[0])


max_rec = max(abs(T(f, s, "test_mae") - T(f, "none", "test_mae")) for f in ("M3", "M4", "M5", "M6", "M7") for s in ("w1000", "w2500"))

# ----------------------------------------------------------- name as picture
name_png = HERE / "_name.png"
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 120)
bbox = font.getbbox(NAME_CN)
im = Image.new("RGBA", (bbox[2] + 20, bbox[3] + 30), (255, 255, 255, 0))
ImageDraw.Draw(im).text((10, 5), NAME_CN, font=font, fill=(255, 255, 255, 255))
im.save(name_png)

# --------------------------------------------------------------------- deck
NAVY, ACC, GREY, DARK = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0xD6, 0x27, 0x28), RGBColor(0x59, 0x59, 0x59), RGBColor(0x22, 0x22, 0x22)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
page = [0]


def tb(slide, text, left, top, width, height, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
    return box


def bullets(slide, items, left, top, width, height, size=18, gap=6):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        bold = None
        if isinstance(it, tuple):
            bold, it = it
        r0 = p.add_run()
        r0.text = "• "
        r0.font.size, r0.font.color.rgb, r0.font.name = Pt(size), ACC, "Calibri"
        if bold:
            r1 = p.add_run()
            r1.text = bold + " "
            r1.font.size, r1.font.bold, r1.font.color.rgb, r1.font.name = Pt(size), True, NAVY, "Calibri"
        r2 = p.add_run()
        r2.text = it
        r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size), DARK, "Calibri"
    return box


def new_slide(title, notes=None, backup=False):
    s = prs.slides.add_slide(BLANK)
    page[0] += 1
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY if not backup else GREY
    bar.line.fill.background()
    tb(s, title, 0.55, 0.3, 12.2, 0.9, size=30, bold=True, color=NAVY)
    tb(s, f"AIDD8122 · {SID}" + ("  ·  backup" if backup else ""), 0.55, 7.05, 6, 0.35, size=11, color=GREY)
    tb(s, str(page[0]), 12.2, 7.05, 0.6, 0.35, size=11, color=GREY, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def pic(slide, path, left, top, width=None, height=None):
    kw = {}
    if width:
        kw["width"] = Inches(width)
    if height:
        kw["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def callout(slide, text, left, top, width, height, size=18):
    box = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xEA, 0xF1, 0xF8)
    box.line.color.rgb = NAVY
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size, r.font.color.rgb, r.font.name, r.font.bold = Pt(size), NAVY, "Calibri", True


# 1 Title -----------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
page[0] += 1
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb(s, "Chasing a Moving Target", 0.8, 1.6, 11.5, 1.2, size=48, bold=True, color=RGBColor(255, 255, 255))
tb(s, "Temporal drift, validation protocols and failure detection for LogD models\non the OpenADMET–ExpansionRx lead-optimisation data",
   0.8, 2.9, 11.5, 1.4, size=24, color=RGBColor(0xDD, 0xE6, 0xF0))
tb(s, "AIDD8122 – Advanced Topics in AI & Drug Discovery  ·  Final project presentation", 0.8, 5.0, 11.5, 0.5, size=16,
   color=RGBColor(0xDD, 0xE6, 0xF0))
tb(s, f"{SID}", 0.8, 5.55, 1.6, 0.5, size=18, bold=True, color=RGBColor(255, 255, 255))
pic(s, name_png, 2.35, 5.6, height=0.38)
tb(s, f"Code: {REPO}  ({REV})", 0.8, 6.3, 11.5, 0.4, size=13, color=RGBColor(0xDD, 0xE6, 0xF0))
s.notes_slide.notes_text_frame.text = ("[0:30] Introduce the question: simple LogD models, trained on early compounds of a real "
                                       "lead-optimisation campaign, are asked to predict later compounds. What happens, and can the model tell when it is wrong?")

# 2 Question & data --------------------------------------------------------
s = new_slide("The question and the data", notes=(
    "[1:00] LogD guides optimisation. The data come from one campaign; the official split is early vs late. We use only SMILES-derived "
    "features and CPU models so that everything is reproducible in minutes. Be explicit: test labels are public, so this is a retrospective study."))
bullets(s, [
    ("Why LogD:", "lipophilicity at a given pH – linked to solubility, permeability, clearance; measured for every new analogue."),
    ("Question:", "a model trained on early compounds – how well does it predict later ones, and can it tell when not to be trusted?"),
    ("Data:", f"OpenADMET–ExpansionRx (CC BY 4.0): 5,039 early training / 2,270 later test molecules, official split."),
    ("Constraints:", "SMILES-derived features only, CPU only, no external data. Test labels are public → retrospective study."),
    ("Plan:", "Phase 1 baselines → Phase 2: drift → validation protocols → stronger models → failure detection."),
], 0.6, 1.5, 12.2, 5.2, size=22, gap=14)

# 3 Phase 1 design ---------------------------------------------------------
s = new_slide("Phase 1: four lightweight baselines, frozen protocol", notes=(
    "[1:00] Show the workflow. Stress the integrity measures: hashes, canonical SMILES identity, zero overlap, parameters frozen on an internal "
    "validation split, test scored once. M0-M3 are deliberately simple."))
pic(s, FIG / "fig1_workflow.png", 0.6, 1.4, width=12.1)
bullets(s, [
    ("M0", "median predictor     "), ("M1", "ridge, 8 RDKit descriptors"), ("M2", "random forest, 8 descriptors"),
    ("M3", "random forest, 8 descriptors + Morgan fingerprint (r=2, 1024 bit)"),
    ("Checks:", "no parse failures, no duplicates, no train–test overlap; predictions re-scored and re-fitted (differences < 1e-9)."),
], 0.8, 3.7, 11.8, 3.2, size=20, gap=10)

# 4 Phase 1 results --------------------------------------------------------
s = new_slide("Baselines: non-linearity helps, the fingerprint helps more", notes=(
    f"[1:00] Test MAE {mae['M0']:.3f} → {mae['M1']:.3f} → {mae['M2']:.3f} → {mae['M3']:.3f}. M3 is better than M2 in all three seeds; "
    "the seeds only capture training randomness, so the molecule-level bootstrap interval is the better uncertainty statement."))
pic(s, FIG / "fig3_mae_comparison.png", 0.6, 1.4, width=6.6)
bullets(s, [
    (f"M1 ridge:", f"MAE {mae['M1']:.3f} (−{rel['M1']:.1f}% vs median)"),
    (f"M2 RF:", f"{mae['M2']:.3f} (−{rel['M2']:.1f}%)"),
    (f"M3 RF + fingerprint:", f"{mae['M3']:.3f} (−{rel['M3']:.1f}%)"),
    ("M3 vs M2:", f"lower in 3/3 seeds ({pdiff.min():+.3f} to {pdiff.max():+.3f}); molecule bootstrap "
                  f"{boot['MAE_M3_minus_M2']['point']:+.3f} [{boot['MAE_M3_minus_M2']['ci95'][0]:+.3f}, {boot['MAE_M3_minus_M2']['ci95'][1]:+.3f}]"),
], 7.4, 1.6, 5.5, 4.5, size=19, gap=12)
callout(s, "Gain is clear but modest – and the story starts here.", 0.8, 5.9, 6.4, 0.7, size=18)

# 5 Drift ------------------------------------------------------------------
s = new_slide("Act 1 · The campaign drifts", notes=(
    "[1:00] LogD fell across the training period and partly recovered in the test period. Later molecules are bigger and more complex while the calculated "
    "logP is unchanged. A classifier tells train from test at AUROC 0.98. Say plainly: molecule number is a chronological proxy, supported by the contiguous split but not confirmed by the data provider."))
pic(s, FIG / "fig7_drift.png", 0.5, 1.3, width=8.4)
bullets(s, [
    ("LogD:", f"mean {quint[0]:.2f} → {quint[4]:.2f} across the training period; test ≈ 2.0–2.3"),
    ("Size/complexity:", f"mol. weight {smd.loc['MolWt', 'train_mean']:.0f} → {smd.loc['MolWt', 'test_mean']:.0f}; rings "
                         f"{smd.loc['NumRings', 'train_mean']:.1f} → {smd.loc['NumRings', 'test_mean']:.1f}"),
    ("Calculated logP:", f"unchanged ({smd.loc['MolLogP', 'train_mean']:.2f} vs {smd.loc['MolLogP', 'test_mean']:.2f})"),
    ("Train vs test classifier:", f"AUROC {a1['adversarial_auroc']['D8']:.2f} (8 descriptors), {a1['adversarial_auroc']['FULL']:.2f} (all)"),
], 9.0, 1.4, 4.0, 5.4, size=17, gap=10)
tb(s, "Molecule number = chronological proxy (train and test ID ranges are contiguous).", 0.6, 4.95, 8.2, 0.6, size=14, color=GREY)
callout(s, "Later compounds are bigger and more complex – and easy to tell apart.", 0.6, 6.05, 12.1, 0.7, size=20)

# 6 Validation protocols ----------------------------------------------------
s = new_slide("Act 2 · Which validation can you trust?", notes=(
    "[1:00] Random 80/20 splits look great (0.387) but the test is 0.623. Time-ordered CV on the training pool gets the learned models' absolute error right within about 3%. "
    "Caveats: it overstates the gain over the baseline, and the folds differ a lot. The stereoisomer twins explain only a small part of the random-split optimism."))
pic(s, FIG / "fig8_protocols.png", 0.5, 1.3, width=8.6)
bullets(s, [
    ("Random 80/20:", f"M3 MAE {prot['random_80_20']['M3']:.3f} vs test {prot['official_test']['M3']:.3f} – {100 * (1 - opt['random_80_20']['M3']):.0f}% optimistic; "
                      f"gain over M0 {rimp['random_80_20']['M3']:.0f}% vs {rimp['official_test']['M3']:.0f}% on test"),
    ("Time-ordered CV:", f"M3 {prot['temporal_cv_mean']['M3']:.3f} – within about {100 * abs(1 - opt['temporal_cv_mean']['M3']):.1f}% of test"),
    ("Stereoisomer twins:", f"{sens['primary_val_molecules_with_flat_twin_in_fit']} of {sens['primary_n_val']:,} validation molecules; 0 in test"),
    ("Caveat:", "CV overstates the gain over the baseline; folds vary widely"),
], 9.2, 1.4, 3.9, 5.5, size=17, gap=10)
callout(s, "Random validation flatters the model; time-ordered CV is closer to reality.", 0.6, 6.05, 12.1, 0.7, size=20)

# 7 Models + recency ---------------------------------------------------------
s = new_slide("Act 3 · Stronger models: boosting, then a small neural network", notes=(
    f"[1:00] Everything was selected on time-ordered CV of the training pool. The rule picked gradient boosting (test {PM:.3f}). A small MLP ensemble was added afterwards, under its own written protocol, "
    f"and did better ({M8:.3f}); averaging it with boosting gives {mlp['test_mae']['M9']:.3f}. Recency weighting did not matter (backup). CV ranks the models correctly but exaggerates the absolute error gain. "
    "Caveat: only a small scikit-learn network and a four-configuration grid; graph networks and pretrained models were not tried."))
pic(s, FIG / "fig11_mlp_comparison.png", 0.5, 1.4, width=7.9)
b8 = mlp["bootstrap"]["M8_minus_primary"]
bullets(s, [
    ("Boosting (chosen by CV):", f"test MAE {PM:.3f}, −{100 * (mae['M0'] - PM) / mae['M0']:.0f}% vs median"),
    ("Small MLP (3 networks, added later):", f"{M8:.3f}, −{100 * (mae['M0'] - M8) / mae['M0']:.0f}%; Δ vs primary boosting {b8['point']:+.3f} [{b8['ci95'][0]:+.3f}, {b8['ci95'][1]:+.3f}]"),
    ("Mean of boosting + MLP:", f"{mlp['test_mae']['M9']:.3f}"),
    ("Recency weighting:", f"no reliable benefit (effect ≤ {max_rec:.3f}, see backup)"),
    ("CV vs test:", f"right order, too optimistic (MLP {mlp['cv_mae']['M8']:.2f} → {M8:.2f})"),
], 8.6, 1.4, 4.5, 5.0, size=16, gap=10)
callout(s, "A small neural network beat gradient boosting – and validation still flattered both.", 0.6, 6.05, 12.1, 0.7, size=20)

# 8 Failure -----------------------------------------------------------------
s = new_slide("Act 4a · Where it fails: permanently charged compounds", notes=(
    "[1:00] All five largest errors are close analogues with a permanent positive charge. Measured LogD is -1 to -1.8; the model predicts about 2.5. Only 3 charged molecules in training. "
    "They are 0.5% of the test set and move the average MAE by only 0.015, but they dominate the tail. State that the explanation is an observation, not a tested mechanism."))
pic(s, FIG / "table3_failure_structures.png", 0.4, 1.3, width=7.4)
c3 = chg["M3"]
bullets(s, [
    ("Top-5 errors:", "same N-methyl-isoquinolinium cation series; measured −1.8 to −1.0, predicted 2.3–2.8"),
    ("Coverage:", f"{chg['train_charged']} charged molecules in {chg['train_total']:,} training vs {chg['test_charged']} in {chg['test_total']:,} test"),
    ("Tail:", f"all {chg['test_charged']} are among the 50 largest errors; M3 MAE {c3['MAE_charged']:.2f} vs {c3['MAE_uncharged']:.3f} for the rest"),
    ("Average effect:", f"only +{c3['MAE_all'] - c3['MAE_uncharged']:.3f} MAE – the mean hides them"),
], 8.0, 1.4, 5.0, 4.6, size=17, gap=10)
tb(s, "Neutral descriptors carry no charge information (Crippen logP of these molecules is moderate).", 0.5, 5.3, 7.2, 0.6, size=14, color=GREY)
callout(s, "Rare (0.5% of the test set) – but they dominate the tail of the errors.", 0.6, 6.05, 12.1, 0.7, size=20)

# 9 Can we tell? ------------------------------------------------------------
s = new_slide("Act 4b · Can the model tell when not to be trusted?", notes=(
    "[1:00] Tree spread and similarity carry modest information (AUROC 0.64-0.67 for M3; 0.6-0.7 range). Crucially the charged molecules are not flagged: they rank around the 34th-41st percentile. A simple charge rule does flag them, "
    "but it was designed after seeing the test errors and has no training-side evidence, so it is a hypothesis, not a validated safeguard."))
pic(s, FIG / "fig10_risk_coverage.png", 0.4, 1.3, width=8.6)
bullets(s, [
    ("Tree spread / similarity:", f"AUROC {a4['oof_auroc_tree_sd']:.2f} / {a4['oof_auroc_dissimilarity']:.2f} on train (out-of-fold), "
                                  f"{a4['test_auroc_M3_tree_sd']:.2f} / {a4['test_auroc_M3_dissimilarity']:.2f} on test"),
    ("Reject 20% by spread:", f"MAE {rc['primary_tree SD']['0']:.3f} → {rc['primary_tree SD']['20']:.3f} (primary)"),
    ("Charged molecules:", f"percentile {100 * a4['charged_rank_pct_tree_sd_median']:.0f} (spread), {100 * a4['charged_rank_pct_dissimilarity_median']:.0f} (dissimilarity) → not flagged"),
    ("Charge rule:", "flags all 11 (precision 1.0, cost 0.5%) – but designed post hoc: a hypothesis"),
], 9.2, 1.4, 3.9, 5.5, size=16, gap=10)
callout(s, "Charged analogues are confidently wrong – generic uncertainty does not catch them.", 0.6, 6.05, 12.1, 0.7, size=20)

# 10 Conclusions ---------------------------------------------------------------
s = new_slide("Conclusions and limits", notes=(
    "[1:00] Summarise the five answers, then the limitations. End on the message: how far cheap models go, how much the answer depends on validation, and where they break."))
bullets(s, [
    ("Baselines:", f"ridge −{rel['M1']:.1f}%, RF −{rel['M2']:.1f}%, RF+fingerprint −{rel['M3']:.1f}% in MAE vs median."),
    ("Drift:", "later molecules are larger/more complex, easy to tell apart (AUROC 0.98)."),
    ("Validation:", "random splits mislead; time-ordered CV matches the absolute error but not the gain over baseline."),
    ("Models:", f"gradient boosting reaches {PM:.3f} (−{100 * (mae['M0'] - PM) / mae['M0']:.0f}%); recency weighting does not help."),
    ("Failures:", "charged analogues are missed and not flagged by forest spread or similarity."),
    ("Limits:", "one campaign; molecule number is a proxy; test inspected in Phase 1; test labels public; charge explanation = hypothesis."),
], 0.6, 1.4, 12.2, 5.5, size=21, gap=12)

# ------------------------------------------------------------------ backup
s = new_slide("Backup · Model family × recency weighting (Phase 2 grid)", backup=True)
pic(s, FIG / "fig9_models_recency.png", 0.5, 1.4, width=9.0)
bullets(s, [
    ("Selection:", "temporal CV of the training pool only; primary = lowest CV MAE of 20 configurations (boosting, half-life 2,500)."),
    ("Recency:", f"test effects within ±{max_rec:.3f}; latest-50% only is worse for all five families."),
    ("Multiplicity:", "20 configurations were scored on the test set (once each)."),
], 9.7, 1.5, 3.3, 5, size=15, gap=10)

s = new_slide("Backup · LogD vs logP, and why keep the official split", backup=True)
bullets(s, [
    ("LogD ≠ logP:", "LogD is measured at a stated pH and includes ionisation; Crippen logP is a calculated neutral-molecule value. Assay pH is not stated in the sources consulted."),
    ("Official split kept:", "later compounds differ chemically; re-splitting would hide the question. Random validation is used only for Phase-1 parameter choice."),
    ("Label scope:", "ML-ready LogD column has no censored (< or >) values; conclusions apply to the retained range."),
], 0.6, 1.5, 12.2, 5, size=20, gap=14)

s = new_slide("Backup · Leakage and identity checks", backup=True)
bullets(s, [
    ("Identity:", "RDKit canonical isomeric SMILES; salts, tautomers and charge states not normalised."),
    ("Audit:", "0 parse failures, 0 duplicates, 0 train–test overlap; records without LogD excluded (287 train, 12 test)."),
    ("Stereoisomers:", f"counted as different molecules → {sens['primary_val_molecules_with_flat_twin_in_fit']} validation molecules have a flat-structure twin in the fit set; "
                       "regrouping did not change the selected parameters."),
    ("Seeds:", "RF seeds 11/22/33 fixed in advance; they capture training randomness only."),
], 0.6, 1.5, 12.2, 5, size=20, gap=14)

s = new_slide("Backup · Fairness of M2 vs M3 and test-set discipline", backup=True)
bullets(s, [
    ("M2 vs M3:", "same algorithm and settings, but max_features='sqrt' picks ~2 of 8 vs ~32 of 1,032 features per split → a practical comparison, not an isolated fingerprint effect."),
    ("Test discipline:", "Phase-1 parameters frozen before test scoring. Phase 2 protocol written before running; every choice made on the training pool; test scored once for 20 pre-listed configurations (multiplicity caveat). The neural network was added afterwards under its own written protocol."),
    ("Not blind:", "Phase 1 inspected the test set before Phase 2 was designed; test labels are public."),
    ("Reproducibility:", f"Phase 2 run twice with identical results; code and results at {REPO}."),
], 0.6, 1.5, 12.2, 5, size=20, gap=14)

s = new_slide("Backup · Time proxy and the charged-molecule claim", backup=True)
bullets(s, [
    ("Time proxy:", "molecule numbers are contiguous across the split (train 10,647–20,100; test 20,101–27,239). Not confirmed by the data provider."),
    ("Charged claim:", "based on 11 test molecules; no causal test; the charge rule has no training-side evidence (only 3 charged training molecules)."),
    ("Future:", "charge/ionisation descriptors evaluated on data with charged compounds; deep and pretrained models; retraining as new data arrive; calibrated uncertainty."),
], 0.6, 1.5, 12.2, 5, size=20, gap=14)

prs.save(OUT)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE), str(OUT)], check=True, capture_output=True)
print("slides:", len(prs.slides), "->", OUT.name)
