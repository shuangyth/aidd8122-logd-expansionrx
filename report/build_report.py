"""Build the final report (.docx) from the MPU template, filling numbers from results/.

Usage: python build_report.py   -> report/AIDD8122_Final_Project_Report.docx
(then convert to PDF with LibreOffice: soffice --headless --convert-to pdf ...)
"""
import copy
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES, FIG = ROOT / "results", ROOT / "figures"
TEMPLATE = HERE / "template" / "AIDD8122_Final_Project_Report_Template.docx"
OUT = HERE / "AIDD8122_Final_Project_Report.docx"

STUDENT_ID = os.environ.get("AIDD_STUDENT_ID", "P00000")
STUDENT_NAME = os.environ.get("AIDD_STUDENT_NAME", "Student")
SUBMIT_DATE = "16/11/2026"
TITLE = ("Chasing a Moving Target: Temporal Drift, Validation Protocols and Failure Detection for "
         "LogD Models on the OpenADMET–ExpansionRx Lead-Optimisation Data")
REPO_URL = (sys.argv[1] if len(sys.argv) > 1 else "https://github.com/<USER>/<REPO>")
REPO_REV = (sys.argv[2] if len(sys.argv) > 2 else "<commit-or-tag>")

# ---------------------------------------------------------------- load results
audit = json.loads((RES / "audit.json").read_text())
prov = json.loads((RES / "data_provenance.json").read_text())
env = json.loads((RES / "environment_and_run.json").read_text())
frozen = json.loads((RES / "frozen_params.json").read_text())
chg = json.loads((RES / "exploratory_charged_subset.json").read_text())
boot = json.loads((RES / "exploratory_bootstrap_ci.json").read_text())
sim = json.loads((RES / "exploratory_similarity_summary.json").read_text())
sens = json.loads((RES / "sensitivity_flat_group_split.json").read_text())
simbins = pd.read_csv(RES / "exploratory_similarity_bins.csv")
val = pd.read_csv(RES / "metrics_validation.csv")
runs = pd.read_csv(RES / "metrics_test_per_run.csv", keep_default_na=False)
runs["seed"] = runs.seed.astype(str)
summ = pd.read_csv(RES / "metrics_test_summary.csv").set_index("model")
pdiff = pd.read_csv(RES / "paired_diff_M3_minus_M2.csv")
fail = pd.read_csv(RES / "table3_failure_cases.csv")

S = summ
mae = {m: S.loc[m, "MAE_mean"] for m in S.index}
sd = {m: S.loc[m, "MAE_std"] for m in S.index}
rel = {m: 100 * (mae["M0"] - mae[m]) / mae["M0"] for m in mae}
tr, te = audit["train_file"], audit["test_file"]
ls = audit["label_stats"]
vmae = {r.model: r.MAE for r in val.itertuples() if r.model in ("M0", "M3")}
vm1 = val[val.model == "M1"].MAE.min()
vm2 = val[(val.model == "M2")].set_index("param").MAE
ci = boot


def f3(x):
    s_ = f"{x:.3f}"
    return "0.000" if s_ == "-0.000" else s_


def cip(k):
    d = ci[k]
    return f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def seed_mae(m, s):
    return runs[(runs.model == m) & (runs.seed == str(s))].MAE.iloc[0]


# --------------------------------------------------------------------- helpers
doc = Document(str(TEMPLATE))
BODY_PT = 11


class Cursor:
    el = None


cur = Cursor()


def put(el):
    cur.el.addnext(el)
    cur.el = el


URL_RE = re.compile(r"(https?://[^\s)]+[^\s).,;])")
TOKEN_RE = re.compile(r"(\*\*.+?\*\*|https?://[^\s)]+[^\s).,;])")


def add_hyperlink(p, url, text, size=BODY_PT):
    rid = doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                             is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "0563C1")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rpr.append(c)
    rpr.append(u)
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    h.append(r)
    p._p.append(h)


def fill(p, text, size=BODY_PT, bold=False, italic=False):
    """Mini-markup: **bold**, _italic_, bare URLs become hyperlinks."""
    for tok in TOKEN_RE.split(text):
        if not tok:
            continue
        if tok.startswith("http"):
            add_hyperlink(p, tok, tok, size)
            continue
        b, i = bold, italic
        if tok.startswith("**") and tok.endswith("**"):
            tok, b = tok[2:-2], True
        r = p.add_run(tok)
        r.font.size = Pt(size)
        r.bold, r.italic = b, i


def para(text, align="justify", size=BODY_PT, after=6, before=0, line=1.15, bold=False,
         italic=False, indent=None, keep=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after, pf.space_before, pf.line_spacing = Pt(after), Pt(before), line
    if keep:
        pf.keep_with_next = True
    if indent is not None:
        pf.left_indent = Cm(indent)
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    fill(p, text, size, bold, italic)
    put(p._p)
    return p


def bullet(text):
    p = para("• " + text, indent=0.6, after=3)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    return p


BOOKMARKS = {}
_bm_id = [100]


def heading(text, level=1, bm=None):
    p = doc.add_paragraph(style="Heading 1" if level == 1 else "Heading 2")
    p.paragraph_format.space_before = Pt(14 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    if level == 2:
        r.font.size = Pt(12)
    if bm:
        s = OxmlElement("w:bookmarkStart")
        s.set(qn("w:id"), str(_bm_id[0]))
        s.set(qn("w:name"), bm)
        e = OxmlElement("w:bookmarkEnd")
        e.set(qn("w:id"), str(_bm_id[0]))
        _bm_id[0] += 1
        p._p.insert(list(p._p).index(r._r), s)
        p._p.append(e)
    put(p._p)
    return p


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tcPr.append(sh)


def borders(tbl, style="single"):
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), style)
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "808080")
        b.append(e)
    tblPr.append(b)


def table(rows, widths, header=True, size=9, align=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t)
    for row_ in t.rows[:-1]:           # keep each table on one page
        for c_ in row_.cells:
            c_.paragraphs[0].paragraph_format.keep_with_next = True
        trPr = row_._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    for i, row in enumerate(rows):
        for j, txt in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            a = (align[j] if align else ("left" if j == 0 else "center"))
            p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER}[a]
            fill(p, str(txt), size=size, bold=(header and i == 0))
            if header and i == 0:
                shade(c, "E7EEF6")
    for j, w_ in enumerate(widths):
        t.columns[j].width = Cm(w_)
    put(t._tbl)
    sp = para("", after=0, before=0, size=4, line=1.0)      # small spacer after each table
    return t


def caption(text, before=3, after=10):
    return para(text, align="center", size=9.5, after=after, before=before, italic=False)


def tcaption(text):
    return para(text, align="center", size=9.5, after=3, before=8, keep=True)


def picture(path, width_cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    put(p._p)


def picture_row(items):
    """items: [(path, width_cm), ...] side by side in a borderless table."""
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, (path, w) in zip(t.rows[0].cells, items):
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(w))
    put(t._tbl)


# ------------------------------------------------------------------ cover page
tbl = doc.tables[0]


def set_cell(cell, text, bold=None, size=None):
    p = cell.paragraphs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    if p.runs:
        r = p.runs[0]
        r.text = text
    else:
        r = p.add_run(text)
    if bold is not None:
        r.bold = bold
    if size:
        r.font.size = Pt(size)


title_cell = tbl.cell(0, 0)
for p in title_cell.paragraphs[1:]:
    p._p.getparent().remove(p._p)
set_cell(title_cell, TITLE, bold=True, size=13)
title_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
# the second row of the title block holds blank paragraphs; centre the first run vertically
tbl.cell(3, 1).paragraphs[0].runs[0].text = STUDENT_ID
for r in tbl.cell(3, 1).paragraphs[0].runs[1:]:
    r.text = ""
set_cell(tbl.cell(4, 1), STUDENT_NAME)
_nr = tbl.cell(4, 1).paragraphs[0].runs[0]._r
_rf = _nr.get_or_add_rPr().get_or_add_rFonts()
for _a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
    _rf.set(qn(_a), "Arial Unicode MS")
_rf.set(qn("w:hint"), "eastAsia")
_nr.get_or_add_rPr().append(OxmlElement("w:lang"))
_nr.rPr[-1].set(qn("w:eastAsia"), "zh-CN")
# the four empty paragraphs after the cover table push a blank page before the TOC: drop them
_t = tbl._tbl
for _el in list(_t.itersiblings())[:4]:
    if _el.tag == qn("w:p") and not "".join(_el.itertext()).strip() and "type=\"page\"" not in _el.xml:
        _el.getparent().remove(_el)
set_cell(tbl.cell(9, 1), SUBMIT_DATE)

# ------------------------------------------------------ locate anchors, clean up
paras = doc.paragraphs
ref_idx = max(i for i, p in enumerate(paras) if p.text.strip() == "References" and p.style.name == "Heading 1")
ref_head = paras[ref_idx]
anchor = paras[ref_idx - 2]            # empty Heading-1 paragraph that carries the section break
assert anchor._p.find(".//" + qn("w:sectPr")) is not None, "anchor must hold the section break"
body = doc.element.body
for el in list(ref_head._p.itersiblings()):
    if el.tag != qn("w:sectPr"):
        body.remove(el)                # drop the Mendeley placeholder bibliography
cur.el = anchor._p

# TOC bookmarks: make every TOC entry point to a real heading (template had a duplicate target)
toc_paras = [p for p in doc.paragraphs if p.style.name == "toc 1"]
names = []
for p in toc_paras:
    xml = p._p.xml
    m = re.search(r"_Toc\d+", xml)
    names.append(m.group(0) if m else None)
if names[5] == names[4]:               # Conclusions pointed at the Results bookmark in the template
    new = "_Toc89984199"
    for t in toc_paras[5]._p.iter(qn("w:instrText")):
        t.text = t.text.replace(names[4], new)
    names[5] = new


# ----------------------------------------------------------------------- BODY
# ---- Phase-2 data (all numbers come from results/phase2)
P2 = RES / "phase2"
p2 = json.loads((P2 / "phase2_summary.json").read_text())
a1, a2, a3, a4 = p2["act1"], p2["act2"], p2["act3"], p2["act4"]
t20 = pd.read_csv(P2 / "act3_all_configs_cv_vs_test.csv")
smd = pd.read_csv(P2 / "act1_descriptor_smd.csv").set_index("descriptor")
quint = pd.read_csv(P2 / "act1_train_logd_by_id_quintile.csv")
tert = pd.read_csv(P2 / "act1_test_logd_by_id_tertile.csv")
rc = json.loads((P2 / "act4_risk_coverage_curves.json").read_text())
mlp = json.loads((P2 / "mlp_summary.json").read_text())
mlpcv = pd.read_csv(P2 / "mlp_temporal_cv.csv")
prot, rimp, opt = a2["protocols"], a2["rel_improvement_vs_M0_pct"], a2["optimism_ratio_vs_test"]
cvfold = a2["temporal_cv_per_fold"]
nnf = a2["nn_similarity_median_per_fold"]
bsp = a3["bootstrap"]
prim = a3["primary"]
PM = a3["primary_test_mae_mean"]
from rdkit import Chem  # noqa: E402

_pp = pd.read_csv(P2 / "act3_primary_test_predictions.csv")
_ch = _pp.canonical_smiles.map(lambda s: Chem.GetFormalCharge(Chem.MolFromSmiles(s)) != 0)
prim_chg_mae = float((_pp.y_true - _pp.primary_seed11)[_ch].abs().mean())
FAMNAME = {"M3": "RF, 8 descriptors + Morgan bits", "M4": "RF, full RDKit descriptors",
           "M5": "RF, full descriptors + Morgan bits", "M6": "Gradient boosting, full descriptors + Morgan counts",
           "M7": "Mean of M5 and M6",
           "M8": "Multilayer perceptron, full descriptors + Morgan counts", "M9": "Mean of M6 and M8"}


def T(fam, sc, col):
    return float(t20[(t20.family == fam) & (t20.scheme == sc)][col].iloc[0])


def pc(x):
    return f"{x:.0f}%"


# 1 Introduction ------------------------------------------------------------
heading("Introduction", 1, names[0])
para("Lipophilicity is one of the most influential physicochemical properties in drug discovery. "
     "The distribution coefficient (LogD) measures the partitioning of a compound between an organic and an "
     "aqueous phase at a stated pH, so it includes the effect of ionisation, and it is associated with "
     "solubility, permeability, metabolic clearance and promiscuity [1,2]. During lead optimisation, medicinal "
     "chemists synthesise successive generations of analogues and measure LogD experimentally. A model trained "
     "on early compounds that anticipates the LogD of later compounds could help to prioritise synthesis, "
     "provided that it remains accurate as the chemistry moves on.")
para("Whether this holds is a question of temporal generalisation, and the way a model is validated decides "
     "whether the answer is believable. Random cross-validation places close analogues on both sides of a "
     "split and can look optimistic, whereas a time-ordered evaluation asks the question that matters in "
     "practice. An industrial study of 120 internal prospective data sets stresses the value of evaluating ADME "
     "models over time [3]. The OpenADMET–ExpansionRx challenge released measurements from one lead-optimisation "
     "campaign, with early-stage compounds for training and later-stage compounds for testing [4,5], which "
     "provides a public temporal split.")
para("This project uses that split not as a leaderboard exercise but to follow one question: **what happens to "
     "simple LogD models as a campaign moves on, and can we tell when not to trust them?** Phase 1 compares four "
     "lightweight baselines (a median predictor, ridge regression and two random forests). Phase 2 follows "
     "what Phase 1 revealed: it measures how the compounds and labels drift, tests which validation protocol "
     "predicts the later-stage error, asks whether stronger models and recency weighting help, and examines "
     "whether the models can signal their own failures. The work is a controlled method comparison and a "
     "retrospective evaluation, not a new algorithm and not a prospective blind test, because the test labels "
     "are already public.")
para(f"In brief: the baseline forests reduce the test MAE of the median predictor by {rel['M2']:.0f}–{rel['M3']:.0f}%; "
     f"later-stage molecules are larger and more complex and are separable from early ones with an AUROC of "
     f"{a1['adversarial_auroc']['FULL']:.2f}; a random 80/20 validation overstates accuracy by about "
     f"{100 * (1 - opt['random_80_20']['M3']):.0f}% whereas a time-ordered validation is within about "
     f"{100 * abs(1 - opt['temporal_cv_mean']['M3']):.1f}% for the same model; a gradient-boosting model and a small neural network reduce the "
     f"test MAE by {100 * (mae['M0'] - PM) / mae['M0']:.0f}% and {100 * (mae['M0'] - mlp['test_mae']['M8']) / mae['M0']:.0f}%; and the worst failures, permanently charged compounds, "
     "are neither flagged by the models’ internal uncertainty nor by similarity to the training data. "
     f"All code, splits and results are available at {REPO_URL} (version {REPO_REV}).")

# 2 Literature review -------------------------------------------------------
heading("Literature Review", 1, names[1])
heading("Lipophilicity and LogD in drug discovery", 2)
para("Reviews of the role of lipophilicity in drug discovery describe how logP and LogD relate to potency, "
     "solubility, permeability, metabolism and off-target effects, which is why lipophilicity is tracked "
     "throughout optimisation [1]. Property-based guidelines such as the ‘rule of five’ connect molecular "
     "weight, calculated lipophilicity and hydrogen-bond donor and acceptor counts to solubility and "
     "permeability [2]. These simple properties motivate the descriptor set used here. LogD differs from logP "
     "because it refers to a specific pH and so depends on the ionisation state of the molecule, which turns out "
     "to matter for the failure cases analysed in Section 5.")
heading("Machine learning for ADME prediction and temporal evaluation", 2)
para("Fang et al. collected 120 internal prospective data sets over 20 months for six in vitro ADME endpoints "
     "and compared several algorithms and molecular representations [3]. Gradient-boosted decision trees and "
     "deep-learning models consistently outperformed random forest over time, models retrained on a fixed "
     "schedule performed better, more frequent retraining generally increased accuracy, and hyperparameter tuning "
     "improved the prospective predictions only marginally [3]. Three lessons apply here: time-ordered evaluation "
     "is the appropriate test for a model meant for ongoing projects, a random forest is a reasonable but not "
     "necessarily the strongest model, and keeping models up to date matters. LogD was not among the six "
     "endpoints of that study, so its findings are context rather than a prediction for this data set.")
heading("The OpenADMET–ExpansionRx challenge and data", 2)
para("The challenge was organised by OpenADMET with Expansion Therapeutics and Hugging Face. It ran from "
     "27 October 2025 to 19 January 2026 and asked participants to predict nine ADMET endpoints, including LogD, "
     "for later-stage molecules of a lead-optimisation campaign from earlier-stage data [4,6]. Each endpoint was "
     "scored by MAE and the overall leaderboard used a macro-averaged relative absolute error [4]. Submissions "
     "could use external data and pretraining [6], so the restricted setting in this project (official data only, "
     "SMILES-derived features, CPU) is not comparable to the leaderboard. The training and test files are "
     "released on Hugging Face under CC BY 4.0 [5]. The challenge page describes LogD as lipophilicity at a "
     "specific pH but, in the pages consulted, gives no assay protocol [4].")
heading("Molecular representations and learners", 2)
para("Descriptor-based models use computed physicochemical quantities. The Wildman–Crippen atom-contribution "
     "method gives a calculated logP [7], and topological polar surface area (TPSA) is a fast fragment-based "
     "estimate of polarity [8]. Extended-connectivity (Morgan) fingerprints encode circular substructures as "
     "bit or count vectors [9]. Random forests average many decorrelated decision trees and need little tuning "
     "[10]; ridge regression is a linear model with an L2 penalty [11]; gradient boosting builds an additive "
     "ensemble of shallow trees [12]; a multilayer perceptron is a small feed-forward neural network. All features were computed with RDKit [13] and the models were fitted with "
     "scikit-learn, including its histogram-based gradient-boosting regressor and multilayer perceptron [14].")
heading("Gap addressed by this project", 2)
para("Prior work shows that temporal evaluation matters, that algorithm choice and retraining schedule and frequency "
     "change prospective accuracy, and that hyperparameter tuning helps only marginally [3]. What a small, fully reproducible study on the public ExpansionRx split can "
     "add is a quantified chain from drift, through the reliability of different validation protocols, to model "
     "choice and failure detection, with an explicit record of which decisions were made before the test set was "
     "scored. It does not claim to beat the challenge leaderboard.")

# 3 RQ & objectives ---------------------------------------------------------
heading("Research Questions and Objectives", 1, names[2])
para("**RQ1 (baselines).** Under the official early-to-later-stage split, how well do eight physicochemical "
     "descriptors with a linear or a non-linear model predict LogD compared with a median baseline, and does a "
     "Morgan fingerprint add to the random forest?", after=4)
para("**RQ2 (drift).** How do later-stage compounds and their LogD labels differ from earlier ones?", after=4)
para("**RQ3 (evaluation).** Which internal validation protocol (random, structure-grouped random or time-ordered) "
     "best predicts the error on the later-stage test set?", after=4)
para("**RQ4 (models and recency).** Do richer descriptors, gradient boosting, ensembling, a small neural network or recency weighting of the "
     "training data lower the later-stage error, when chosen with the most trustworthy validation protocol?", after=4)
para("**RQ5 (knowing when not to trust).** Where do the models fail, and can tree-ensemble uncertainty, similarity "
     "to the training data or a simple charge flag identify the failures?")
para("Motivation: RQ1 establishes what a very cheap model can deliver. RQ2 and RQ3 ask why that answer changes with "
     "the way it is measured, and RQ4 and RQ5 ask what a practitioner could do about it. The objectives are:", after=3)
bullet("O1. Build an auditable, leakage-checked data set with fixed identities and splits (Sections 4.1–4.4).")
bullet("O2. Compare pre-specified baseline configurations M0–M3 with a frozen protocol (Sections 4.5, 5.1).")
bullet("O3. Quantify drift and the optimism of different validation protocols (Sections 5.2–5.3).")
bullet("O4. Select stronger models and recency schemes by time-ordered validation and score a pre-listed set of configurations once each on the test set "
       "(Sections 4.7, 5.4).")
bullet("O5. Characterise failures and test whether they can be flagged (Sections 5.5–5.6).")
bullet("O6. Provide a reproducible code package (Section 4.8).")

# 4 Methods ------------------------------------------------------------------
heading("Methods and Materials", 1, names[3])
heading("Data and label", 2)
para(f"The official training and test files of the OpenADMET–ExpansionRx challenge were downloaded from "
     f"Hugging Face [5] and verified by SHA256 (train {prov['files']['expansion_data_train.csv']['sha256'][:12]}…, "
     f"test {prov['files']['expansion_data_test.csv']['sha256'][:12]}…); the dataset revision was "
     f"{prov['hf_revision_sha'][:8]} (last modified {prov['hf_last_modified'][:10]}). The label is the "
     "‘LogD’ column of the ML-ready files, already on a log scale. Records without a finite LogD value "
     "were excluded and no value was imputed. The ML-ready LogD column contains only plain numeric values (no ‘<’ or ‘>’ "
     "entries), so censored measurements are not part of this study. The assay pH and protocol were not stated "
     "in the sources consulted, so no pH value is assumed. The early-to-later ordering follows the publisher’s "
     "description [4]. The files carry no dates, but the molecule numbers (‘E-xxxxx’) are contiguous across the split: "
     "training molecules with a LogD value span 10,647–20,100 and test molecules 20,101–27,239 with no overlap (54 unlabelled "
     "training records carry lower numbers). The molecule number is "
     "therefore used as a chronological proxy in Phase 2; the data provider has not confirmed that numbers follow "
     "synthesis or measurement order.")
heading("Molecule identity and cleaning", 2)
para("SMILES were stripped of white space and parsed with RDKit. The molecule identity key is the RDKit "
     "canonical isomeric SMILES. Salts, tautomers and charge states are not normalised, so the key does not "
     "detect duplicates that differ only in those respects. Within a file, records with the same key would be "
     "merged by the median LogD; across files, any shared key would be removed from the training side. The "
     "audit (Table 1) found no parse failures, no duplicate identities and no train–test overlap, so the "
     "only exclusions were records without a LogD value. Because stereochemistry is part of the key, "
     "stereoisomers count as different molecules; Section 5.3 shows what this means for random validation.")
tcaption("Table 1. Data audit and final set sizes (counts of records or molecules).")
table([["Item", "Train file", "Test file"],
       ["Raw records", f"{tr['raw_records']:,}", f"{te['raw_records']:,}"],
       ["Excluded: missing LogD", f"{tr['excluded_missing_label']}", f"{te['excluded_missing_label']}"],
       ["RDKit parse failures", f"{tr['rdkit_parse_failures']}", f"{te['rdkit_parse_failures']}"],
       ["Records merged as duplicates", f"{tr['extra_records_merged_by_canonical_smiles']}",
        f"{te['extra_records_merged_by_canonical_smiles']}"],
       ["Train molecules also in test (removed)", f"{tr['removed_for_overlap_with_test']}", "–"],
       ["Final molecules", f"{tr['final_train_pool']:,}", f"{te['final_test']:,}"],
       ["Phase-1 random split of train pool (fit / validation)",
        f"{audit['split']['fit']:,} / {audit['split']['val']:,}", "–"]],
      [7.2, 3.6, 3.6])
heading("Features", 2)
para("Eight descriptors were computed from the canonical SMILES with RDKit: molecular weight, Crippen logP, "
     "TPSA, number of hydrogen-bond donors, number of acceptors, number of rotatable bonds, number of rings "
     "and fraction of sp3 carbons. The Morgan fingerprint uses radius 2 and 1,024 bits (bit vectors; count vectors "
     f"in Phase 2; RDKit {env['rdkit']}; the similarity analyses use a separate 2,048-bit Morgan fingerprint). Phase 2 also uses the full RDKit descriptor list "
     f"({p2['n_full_desc']} descriptors that were finite and non-constant on train and test). The Crippen logP is a "
     "calculated value, not LogD. No molecule name, other ADMET endpoint or label-derived feature was used as a "
     "model input.")
heading("Splits and evaluation protocol (Phase 1)", 2)
para("The official test set was kept untouched until the Phase-1 protocol was frozen. For parameter selection, "
     "the cleaned training pool was divided 80/20 with scikit-learn’s GroupShuffleSplit (groups = identity "
     "key, random_state 42). All configurations share the same fit, validation and test molecules. After selecting "
     "parameters by validation MAE, models were retrained on the entire cleaned training pool and scored once "
     "on the test set. The three data sets were verified to be disjoint by identity key.")
heading("Baseline models and parameter selection", 2)
tcaption("Table 2. Model configurations (M0–M3 in Phase 1; M4–M7 in Phase 2; M8–M9 in the later neural-network comparison).")
table([["ID", "Model and input", "Parameters chosen on"],
       ["M0", "Median predictor", "–"],
       ["M1", "StandardScaler + ridge, 8 descriptors", "random split: alpha ∈ {0.1, 1, 10, 100}"],
       ["M2", "Random forest, 8 descriptors", "random split: min_samples_leaf ∈ {1, 2, 5}"],
       ["M3", "Random forest, 8 descriptors + Morgan bits", "same as M2 (not tuned separately)"],
       ["M4", "Random forest, full RDKit descriptors", "temporal CV: leaf {1, 2, 5} × max_features {sqrt, 0.33}"],
       ["M5", "Random forest, full descriptors + Morgan bits", "temporal CV: same grid"],
       ["M6", "Histogram gradient boosting, full descriptors + Morgan counts",
        "temporal CV: learning rate {0.03, 0.1} × max leaf nodes {15, 31}"],
       ["M7", "Mean of the M5 and M6 predictions", "from M5, M6"],
       ["M8", "Multilayer perceptron (3-network mean), full descriptors + Morgan counts", "temporal CV: hidden layers {(256), (256, 128)} × alpha {1e-3, 0.1}"],
       ["M9", "Mean of the M6 and M8 predictions", "from M6, M8"]],
      [1.2, 7.4, 7.8], align=["left", "left", "left"])
para("Random forests used 300 trees, unlimited depth and parallel jobs; M2 and M3 used max_features = ‘sqrt’, "
     "selected by the lowest validation MAE (ties towards the larger alpha or leaf size). The scaler of M1 is "
     "learnt only from the data being fitted. M2 and M3 were run with model seeds 11, 22 and 33, fixed in advance. "
     "Because ‘sqrt’ selects about 2 of 8 candidate features per split for M2 but about 32 of 1,032 for M3, the "
     "M2–M3 contrast is a practical comparison of two configurations of one algorithm rather than an isolated effect "
     "of the fingerprint.", before=6)
heading("Metrics and statistics", 2)
para("The primary metric is MAE in LogD units, which is also the challenge’s per-endpoint metric [4]; "
     "RMSE and R² are secondary. For models with random components, seed means and sample standard deviations "
     "(n = 3) are reported. As an exploratory check of sampling uncertainty over test molecules, paired bootstraps "
     "(2,000 resamples of test molecules, seed 0) were applied to single-seed runs for the Phase-1 and Phase-2 models and to the three-network mean (M8) and the average of M6 and that mean (M9) for the neural-network comparison, always against the seed-11 primary model. Percentages of improvement are "
     "relative to the median predictor on the same data. Analyses added after seeing Phase-1 test errors (similarity, "
     "charged molecules, bootstrap, the stereoisomer sensitivity analysis, all of Phase 2 and the neural-network comparison) are labelled "
     "exploratory or post hoc.")
heading("Phase 2 design", 2)
para("The Phase-2 protocol was written to a file (PHASE2_PROTOCOL.md) before its code was run. Because Phase 1 had "
     "already inspected the test set, Phase 2 is partly informed by test observations; every choice was therefore "
     "made on the training pool only; the 20 pre-listed configurations were each scored once on the test set (seed 11), and the "
     "pre-selected primary model was additionally refit with three seeds and compared with M3 and M0 by paired bootstrap.", after=3)
bullet("**Drift.** LogD by molecule-number quintile; standardised mean differences of the eight descriptors; "
       "adversarial validation, i.e. a random forest separating train from test molecules (5-fold CV AUROC; "
       "molecule numbers are not used as features).")
bullet("**Validation protocols.** Rolling-origin temporal cross-validation: the training pool is ordered by "
       "molecule number and cut into six equal blocks; fold k fits on blocks 1..k and validates on block k+1 "
       "(k = 2..5, four folds). M0–M3 with the Phase-1 settings are scored under four protocols (random 80/20, "
       "random grouped by flat structure, temporal CV, official test).")
bullet("**Models and recency.** M3 (Phase-1 settings, not re-tuned) and M4–M7 (Table 2), i.e. five families, each under four "
       "recency schemes: no weighting, exponentially decaying weights with a half-life of 1,000 or 2,500 records (age counted "
       "in records within the fit set), or only the latest 50% of the fit set. For M4–M7, parameters are chosen per family by "
       "mean temporal-CV MAE without weighting and then reused. The primary Phase-2 model is the (family, scheme) with the "
       "lowest temporal-CV MAE among all 20 (5 families × 4 schemes); all 20 are also scored once on the test set as secondary results.")
bullet("**Neural network (added after Phase 2 was run).** Under a separate written protocol (PHASE2B_MLP_PROTOCOL.md): a "
       "scikit-learn multilayer perceptron on the M6 features (descriptors standardised and clipped, counts log-transformed; Adam, "
       "early stopping), four configurations chosen on the same temporal CV, each the mean of three networks (seeds 11, 22, 33); "
       "M9 averages M6 and M8. This is a secondary comparison and does not replace the primary model.")
bullet("**Uncertainty signals.** Computed with the Phase-1 M3 forest (seed 11): tree-prediction standard deviation "
       "and 1 − nearest-neighbour Tanimoto similarity (Morgan, radius 2, 2,048 bits) to the training pool, plus a "
       "rule flag for a non-zero net formal charge. Evaluated by AUROC for absolute error ≥ 1.0 LogD (fixed in "
       "advance), risk–coverage curves, and out-of-fold temporal-CV predictions of the training pool. No threshold "
       "is derived from test results.")
heading("Implementation, reproducibility and code availability", 2)
para(f"The pipeline is Python (Python {env['python'].split()[0]}, RDKit {env['rdkit']}, "
     f"scikit-learn {env['sklearn']}, pandas {env['pandas']}, NumPy {env['numpy']}) and runs on a laptop CPU "
     f"(Phase 1 in about {env['total_seconds']:.0f} s, Phase 2 in about 10–12 min and the neural-network comparison in about 8 min). "
     "Phase 1 includes self-checks: metrics recomputed from the saved predictions, identical test molecules and "
     "order for every model, and a repeat fit of M1, M2 and M3 (seed 11) whose predictions differ by less than "
     "1e-9. Two early executions of Phase 1 stopped in the self-check stage (a seed label read back as a missing value, "
     "and an exact-equality test that was too strict for floating-point arithmetic); both were fixed without "
     "changing any modelling choice. Phase 2 was run twice with numerically identical results (differences below 1e-7). Code, configuration, cleaned "
     f"splits and results are at {REPO_URL} (version {REPO_REV}). Raw CSVs are re-downloaded and verified by hash.",
     after=6)
picture(FIG / "fig1_workflow.png", 16.0)
caption("Figure 1. Workflow of the Phase-1 study; Phase 2 reuses the cleaned splits and adds temporal validation.")

# 5 Results ------------------------------------------------------------------
heading("Results and Discussion", 1, names[4])
heading("Baselines on the later-stage test set (RQ1)", 2)
tcaption("Table 3. Phase-1 test-set performance (n = 2,270). RF rows: mean ± sample SD over seeds 11, 22, 33.")
rows = [["Model", "MAE", "RMSE", "R²", "MAE change vs M0"]]
for m, lab in (("M0", "M0 median"), ("M1", "M1 Ridge, 8 descriptors"),
               ("M2", "M2 RF, 8 descriptors"), ("M3", "M3 RF, 8 descriptors + Morgan")):
    if m in ("M0", "M1"):
        rows.append([lab, f3(mae[m]), f3(S.loc[m, "RMSE_mean"]), f3(S.loc[m, "R2_mean"]),
                     "–" if m == "M0" else f"−{rel[m]:.1f}%"])
    else:
        rows.append([lab, f"{mae[m]:.3f} ± {sd[m]:.3f}", f"{S.loc[m, 'RMSE_mean']:.3f} ± {S.loc[m, 'RMSE_std']:.3f}",
                     f"{S.loc[m, 'R2_mean']:.3f} ± {S.loc[m, 'R2_std']:.3f}", f"−{rel[m]:.1f}%"])
table(rows, [5.6, 2.9, 2.9, 2.9, 2.7])
para(f"Ridge regression on the eight descriptors improves MAE by only {rel['M1']:.1f}% (R² = {S.loc['M1', 'R2_mean']:.2f}), "
     "which suggests that the relationship between these descriptors and measured LogD in this series is largely "
     f"non-linear or that the descriptors are insufficient. The random forest on the same descriptors reduces MAE by "
     f"{rel['M2']:.1f}% (R² = {S.loc['M2', 'R2_mean']:.2f}). The paired bootstrap over test molecules gives MAE "
     f"differences M1 − M0 of {cip('MAE_M1_minus_M0')} and M2 − M1 of {cip('MAE_M2_minus_M1')} (point estimate and 95% "
     "interval), so both improvements are distinguishable from zero for this test sample.", before=6)
d = pdiff.set_index("seed").MAE_M3_minus_M2
para(f"Adding the fingerprint (M3) lowered the test MAE for every seed ({seed_mae('M3', 11):.3f} vs "
     f"{seed_mae('M2', 11):.3f}, {seed_mae('M3', 22):.3f} vs {seed_mae('M2', 22):.3f}, {seed_mae('M3', 33):.3f} vs "
     f"{seed_mae('M2', 33):.3f}); the paired differences are {d[11]:+.3f}, {d[22]:+.3f} and {d[33]:+.3f}, and the bootstrap "
     f"over test molecules (seed 11) gives {cip('MAE_M3_minus_M2')}. The seed standard deviations "
     f"({sd['M2']:.3f}, {sd['M3']:.3f}) cover training randomness only. Whether the gain comes from the fingerprint "
     "information or from the different feature sampling of the forest was not separated (Section 4.5). "
     "In answer to RQ1, descriptors with a non-linear model give a clear but modest gain, descriptors with a linear "
     "model a very small one, and the fingerprint a further gain.")
picture(FIG / "fig3_mae_comparison.png", 8.6)
caption("Figure 2. Phase-1 test MAE of the four configurations; black dots are the individual random-forest seeds.")

heading("The campaign drifts (RQ2)", 2)
q = quint["mean"].tolist()
t3 = tert["mean"].tolist()
para(f"The training molecules were ordered by molecule number and cut into five equal groups. Their mean LogD falls from "
     f"{q[0]:.2f} and {q[1]:.2f} in the first two groups to {q[3]:.2f} and {q[4]:.2f} in the last two, a decline of about "
     f"{q[1] - q[4]:.1f} LogD units, and then moves back to {t3[0]:.2f}, {t3[1]:.2f} and {t3[2]:.2f} across the three "
     "tertiles of the test set (Figure 3, left). The rolling median shows that measured LogD falls with molecule "
     "number for a long stretch and that the later compounds do not simply continue the trend. LogD values "
     "also show local clustering among neighbouring molecule numbers (visual impression in Figure 3, not quantified).")
para(f"The later compounds are also structurally different. Test molecules are larger (mean molecular weight "
     f"{smd.loc['MolWt', 'test_mean']:.0f} vs {smd.loc['MolWt', 'train_mean']:.0f}; standardised mean difference "
     f"{smd.loc['MolWt', 'SMD']:.2f}), have more rings ({smd.loc['NumRings', 'test_mean']:.1f} vs "
     f"{smd.loc['NumRings', 'train_mean']:.1f}; {smd.loc['NumRings', 'SMD']:.2f}) and more sp3 carbons "
     f"({smd.loc['FractionCSP3', 'SMD']:.2f}), whereas the calculated logP is almost unchanged "
     f"({smd.loc['MolLogP', 'test_mean']:.2f} vs {smd.loc['MolLogP', 'train_mean']:.2f}; {smd.loc['MolLogP', 'SMD']:.2f}) and so "
     f"is TPSA ({smd.loc['TPSA', 'SMD']:.2f}). A random forest trained to tell training from test molecules reaches a "
     f"cross-validated AUROC of {a1['adversarial_auroc']['D8']:.2f} with the eight descriptors and "
     f"{a1['adversarial_auroc']['FULL']:.2f} with the full descriptor list; the most important features are "
     "surface-area-partitioned logP and complexity descriptors (SlogP_VSA8, BCUT2D_LOGPHI, BertzCT). Late-stage "
     "molecules are therefore easy to recognise as different. The picture is of a series growing in size and "
     "complexity while the calculated lipophilicity stays about the same, which is an observation about this "
     "data set and not evidence about the programme’s intent.")
picture(FIG / "fig7_drift.png", 16.0)
caption("Figure 3. Left: LogD against molecule number (chronological proxy) with a rolling median; the dashed line is the "
        "official train–test boundary. Right: top features of the forest separating train from test.")

heading("Evaluation protocols disagree (RQ3)", 2)
tcaption("Table 4. MAE of M0–M3 under four evaluation protocols. Optimism = MAE in the protocol / MAE on the official test.")
rows = [["Protocol", "M0", "M1", "M2", "M3", "M3 gain vs M0", "M3 optimism"]]
for key, lab in (("random_80_20", "Random 80/20"), ("flat_grouped_80_20", "Random, flat-grouped"),
                 ("temporal_cv_mean", "Temporal CV (4 folds)"), ("official_test", "Official test")):
    rows.append([lab] + [f3(prot[key][m]) for m in ("M0", "M1", "M2", "M3")]
                + [pc(rimp[key]["M3"]), f"{opt[key]['M3']:.2f}"])
table(rows, [4.2, 1.7, 1.7, 1.7, 1.7, 2.8, 2.6])
para(f"The ranking M0 > M1 > M2 > M3 is the same under all four protocols, but the size of the apparent success is not. "
     f"A random 80/20 split gives M3 a validation MAE of {prot['random_80_20']['M3']:.3f}, about "
     f"{100 * (1 - opt['random_80_20']['M3']):.0f}% below its test MAE, and a relative gain over M0 of "
     f"{rimp['random_80_20']['M3']:.0f}% that is more than double the {rimp['official_test']['M3']:.0f}% obtained on the test "
     "set. Part of this comes from stereoisomer twins that straddle the random split: "
     f"{sens['primary_val_molecules_with_flat_twin_in_fit']} of the {sens['primary_n_val']:,} validation molecules "
     f"({100 * sens['primary_val_molecules_with_flat_twin_in_fit'] / sens['primary_n_val']:.1f}%) have a stereoisomer with the "
     "same flat structure in the fit set, whereas none of the test molecules has a flat twin in the training pool. "
     "In a post hoc sensitivity analysis the split was regrouped by flat structure; the selected parameters did not change "
     f"and the validation MAE of M3 rose only to {prot['flat_grouped_80_20']['M3']:.3f}, so the twins explain a small part "
     "of the optimism.")
para(f"Temporal cross-validation does much better. For the learned models its mean MAE is within a few per cent of the "
     f"test MAE (M1 {prot['temporal_cv_mean']['M1']:.3f} vs {prot['official_test']['M1']:.3f}, M2 "
     f"{prot['temporal_cv_mean']['M2']:.3f} vs {prot['official_test']['M2']:.3f}, M3 {prot['temporal_cv_mean']['M3']:.3f} "
     f"vs {prot['official_test']['M3']:.3f}), and the validation blocks resemble the test set in how far they are from "
     f"the fit data (median nearest-neighbour Tanimoto {min(nnf):.2f}–{max(nnf):.2f} against {sim['test_nn_sim_median']:.2f} "
     "for the test set). It is less reliable for the baseline: the median predictor’s MAE is "
     f"{prot['temporal_cv_mean']['M0']:.3f} in temporal CV but {prot['official_test']['M0']:.3f} on the test set, so the "
     f"relative gain of M3 is overstated ({rimp['temporal_cv_mean']['M3']:.0f}% vs {rimp['official_test']['M3']:.0f}%). "
     f"The four folds differ a great deal (M0 from {min(cvfold['M0']):.2f} to {max(cvfold['M0']):.2f}; M3 from "
     f"{min(cvfold['M3']):.2f} to {max(cvfold['M3']):.2f}; Figure 4, right), which is why a single time-ordered split "
     "would be a noisy guide, and why the agreement of the mean with the test MAE should be read within that fold-to-fold spread. "
     "In answer to RQ3, on this data set time-ordered validation reproduces the absolute error of the learned "
     "models much better than random validation does, but it does not remove uncertainty about the gain over a "
     "baseline.")
picture(FIG / "fig8_protocols.png", 16.0)
caption("Figure 4. Left: MAE of M0–M3 under four protocols. Right: MAE of M0 and M3 in each temporal-CV fold; dotted lines mark "
        "the official-test MAE.")

heading("Stronger models and recency weighting (RQ4)", 2)
ch = a3["chosen_params"]
para(f"All choices in this section were made on temporal CV of the training pool. Among the 20 (family, scheme) "
     f"configurations the lowest CV MAE ({prim['cv_mae']:.3f}) belongs to M6 (gradient boosting) with half-life-2,500 "
     "weighting, which is therefore the primary Phase-2 model. Its test MAE is "
     f"{PM:.3f}, i.e. {100 * (mae['M0'] - PM) / mae['M0']:.0f}% below the median predictor and "
     f"{100 * (mae['M3'] - PM) / mae['M3']:.0f}% below M3 (three-seed mean {mae['M3']:.3f}; the bootstrap below uses the seed-11 run, 0.620). The test result does not vary with the "
     "seed because the boosting implementation has no random component here. The paired bootstrap over test molecules gives "
     f"primary − M3 = {bsp['primary_minus_M3_phase1']['point']:+.3f} [{bsp['primary_minus_M3_phase1']['ci95'][0]:+.3f}, "
     f"{bsp['primary_minus_M3_phase1']['ci95'][1]:+.3f}] and primary − M0 = {bsp['primary_minus_M0']['point']:+.3f} "
     f"[{bsp['primary_minus_M0']['ci95'][0]:+.3f}, {bsp['primary_minus_M0']['ci95'][1]:+.3f}].")
tcaption("Table 5. Model families: temporal-CV MAE (train pool) and test MAE, without weighting and with half-life-2,500 weighting (M8 and M9 were added later and are not weighted).")
rows = [["ID", "Model", "CV MAE", "Test MAE", "CV (w2500)", "Test (w2500)"]]
for fam in ("M3", "M4", "M5", "M6", "M7"):
    rows.append([fam, FAMNAME[fam], f3(T(fam, "none", "cv_mae")), f3(T(fam, "none", "test_mae")),
                 f3(T(fam, "w2500", "cv_mae")), f3(T(fam, "w2500", "test_mae"))])
rows.append(["M8", FAMNAME["M8"], f3(mlp["cv_mae"]["M8"]), f3(mlp["test_mae"]["M8"]), "–", "–"])
rows.append(["M9", FAMNAME["M9"], f3(mlp["cv_mae"]["M9"]), f3(mlp["test_mae"]["M9"]), "–", "–"])
table(rows, [1.0, 6.9, 1.9, 1.9, 2.2, 2.4], align=["left", "left", "center", "center", "center", "center"])
cv_gap = T("M5", "none", "cv_mae") - T("M6", "none", "cv_mae")
te_gap = T("M5", "none", "test_mae") - T("M6", "none", "test_mae")
best_te = t20.sort_values("test_mae").iloc[0]
para(f"Three observations follow (Figure 5). First, CV and the test set agree on the broad ordering of families (boosting M6 and the ensemble M7 best, then M5, then M3 and M4; "
     f"the order of M6 and M7 and of M3 and M4 is swapped between CV and test), and over "
     f"all 20 configurations the Spearman correlation between CV MAE and test MAE is {a3['cv_test_spearman_over_20_configs'][0]:.2f}, so "
     f"temporal CV is a useful guide to model choice. Second, it exaggerates the advantage of boosting: M6 beats M5 by "
     f"{cv_gap:.3f} in CV but by {te_gap:.3f} on the test set. Richer descriptors help the random forest "
     f"(M4 {T('M4', 'none', 'test_mae'):.3f} vs M3 {T('M3', 'none', 'test_mae'):.3f}), and the fingerprint helps once more "
     f"(M5 {T('M5', 'none', 'test_mae'):.3f}). Third, recency weighting gave no reliable benefit: on the test set the "
     f"weighted variants differ from the unweighted ones by at most {t20[t20.scheme.isin(['w1000', 'w2500'])].apply(lambda r: abs(r.test_mae - T(r.family, 'none', 'test_mae')), axis=1).max():.3f}, "
     "in either direction, and on the test set using only the latest half of the data was worse for all five families. "
     "The selection rule picked half-life 2,500 for M6 by a margin of "
     f"{T('M6', 'none', 'cv_mae') - T('M6', 'w2500', 'cv_mae'):.4f} in CV MAE (against the unweighted M6), a negligible margin; the weighted M6 is "
     f"{T('M6', 'w2500', 'test_mae') - T('M6', 'none', 'test_mae'):.3f} worse on the test set, so the weighting should not be "
     f"credited with the improvement. The lowest test MAE among the 20 pre-listed (Phase-2) configurations is that of the unweighted ensemble "
     f"({best_te.family}, {best_te.scheme}: {best_te.test_mae:.3f}); it was not the primary model and is reported "
     "only as a post hoc observation, since 20 configurations were scored before M8 and M9 were added (25 configurations in all were scored on the test set: M0–M2 in Phase 1, the 20 Phase-2 configurations, which include M3, and M8 and M9; M8 and M9 score lower, see below). The recency finding does not reproduce the "
     "benefit of frequent retraining reported in [3]; retraining on newly measured compounds is a different "
     "intervention from reweighting a fixed training set, so this is not a contradiction.")
_b8, _b9 = mlp["bootstrap"]["M8_minus_primary"], mlp["bootstrap"]["M9_minus_primary"]
para(f"A small neural network (M8, Table 5) was added after Phase 2 had been run, under a separate written protocol, as a secondary "
     f"comparison; it does not replace the primary model, which was fixed by the Phase-2 rule. Its architecture was chosen on temporal CV "
     f"(hidden layers {tuple(mlp['selected']['hidden'])}, L2 alpha {mlp['selected']['alpha']}), the four configurations differed little in CV "
     f"({mlpcv.cv_mae.min():.3f}–{mlpcv.cv_mae.max():.3f}), and the final model averages three networks. On the test set M8 reaches "
     f"{mlp['test_mae']['M8']:.3f} (RMSE {mlp['test_rmse']['M8']:.3f}), lower than the primary model ({PM:.3f}) and than unweighted "
     f"gradient boosting ({mlp['test_mae']['M6_none']:.3f}); the paired bootstrap gives M8 − primary = {_b8['point']:+.3f} "
     f"[{_b8['ci95'][0]:+.3f}, {_b8['ci95'][1]:+.3f}], and the mean of M6 and M8 (M9) reaches {mlp['test_mae']['M9']:.3f} "
     f"(RMSE {mlp['test_rmse']['M9']:.3f}; M9 − primary = {_b9['point']:+.3f} [{_b9['ci95'][0]:+.3f}, {_b9['ci95'][1]:+.3f}]). "
     f"The three single networks give {mlp['M8_single_mean']:.3f} ± {mlp['M8_single_sd']:.3f} (seeds 11, 22, 33), so averaging "
     f"three networks accounts for part of the result. Temporal CV again ranks M8 ahead of M6 but exaggerates the gap ("
     f"{mlp['cv_mae']['M6_none'] - mlp['cv_mae']['M8']:.3f} in CV, {mlp['test_mae']['M6_none'] - mlp['test_mae']['M8']:.3f} on the test set) and is too optimistic in absolute "
     f"terms ({mlp['cv_mae']['M8']:.3f} in CV against {mlp['test_mae']['M8']:.3f} on the test set); M9 is slightly better than M8 on the test set although worse in CV. Because M8 and M9 were added after the "
     "Phase-2 results were seen, the architecture grid has four configurations and only four CV folds were used, this is evidence that "
     "a small feed-forward network is competitive with, and here better than, gradient boosting on these features, not a general "
     "statement about deep learning; graph neural networks and pretrained models were not evaluated.")
picture(FIG / "fig9_models_recency.png", 15.5)
caption("Figure 5. Model family × recency scheme. Left: temporal-CV MAE on the training pool (used for selection). "
        "Right: test MAE (each configuration scored once, seed 11).")

heading("Where the models fail: permanently charged compounds", 2)
picture(FIG / "fig4_scatter_M1_M3.png", 12.0)
caption("Figure 6. Predicted versus experimental test LogD for M1 and M3 (seed 11); dashed line y = x.")
para("The ridge model spreads predictions widely around the diagonal with no tight trend. The M3 predictions follow the "
     "trend better but are compressed: they stay roughly between 0 and 4 while the experimental values span about −2 to 5, "
     "a common pattern for tree ensembles that average training labels.")
tcaption("Table 6. The five largest M3 test errors (seed 11, fixed in advance).")
rows = [["Molecule", "Exp. LogD", "Pred. LogD", "Abs. error", "MolWt", "Crippen logP", "TPSA"]]
for r in fail.itertuples():
    rows.append([r.name, f"{r.y_true:.2f}", f"{r.y_pred:.2f}", f"{r.abs_err:.2f}", f"{r.MolWt:.0f}",
                 f"{r.MolLogP:.2f}", f"{r.TPSA:.1f}"])
table(rows, [2.6, 2.2, 2.2, 2.2, 1.9, 2.7, 1.9])
picture(FIG / "table3_failure_structures.png", 13.0)
caption("Figure 7. Structures of the five largest M3 errors (experimental and predicted LogD shown).")
c3 = chg["M3"]
para(f"All five molecules are close analogues sharing an N-methyl-isoquinolinium core, which carries a permanent "
     f"positive charge. Their experimental LogD is between −1.8 and −1.0, but M3 predicts 2.3 to 2.8, "
     f"errors of about 3.8 to 4.2 units. The Crippen logP of these molecules is moderate (about 2.9 to 3.7), "
     "so the neutral-molecule descriptors give no sign of the very low measured lipophilicity. Only "
     f"{chg['train_charged']} of {chg['train_total']:,} training molecules have a non-zero net formal charge, against "
     f"{chg['test_charged']} of {chg['test_total']:,} test molecules. All {chg['test_charged']} charged test molecules are among "
     f"the 50 largest absolute errors, and the M3 MAE on them is {c3['MAE_charged']:.2f}, against {c3['MAE_uncharged']:.3f} "
     f"for all other molecules (overall {c3['MAE_all']:.3f}). They are only "
     f"{100 * chg['test_charged'] / chg['test_total']:.1f}% of the test set and raise the average MAE by about "
     f"{c3['MAE_all'] - c3['MAE_uncharged']:.3f}, but they dominate the tail of the error distribution. The primary "
     f"Phase-2 model is better on them but still far off (MAE {prim_chg_mae:.2f}). This pattern is consistent with a "
     "late-stage chemotype that is nearly absent from the early data and not encoded by the features; it was not tested "
     "causally, and the charged-molecule analysis was designed after seeing the errors.")

heading("Can the models tell when not to trust themselves? (RQ5)", 2)
tcaption(f"Table 7. AUROC for identifying absolute errors ≥ {a4['threshold_abs_error']:.1f} LogD.")
table([["Signal", "Test, M3 errors", "Test, primary errors", "Train pool, out-of-fold M3"],
       ["Tree-prediction SD (M3 forest)", f3(a4["test_auroc_M3_tree_sd"]), f3(a4["test_auroc_primary_tree_sd"]),
        f3(a4["oof_auroc_tree_sd"])],
       ["1 − nearest-neighbour similarity", f3(a4["test_auroc_M3_dissimilarity"]), f3(a4["test_auroc_primary_dissimilarity"]),
        f3(a4["oof_auroc_dissimilarity"])],
       ["Net-charge flag (11 molecules)", f3(a4["test_auroc_M3_charged_flag"]), f3(a4["test_auroc_primary_charged_flag"]), "n/a (3 molecules)"]],
      [5.8, 3.4, 3.6, 3.8])
para(f"Errors of at least 1 LogD unit are common: {100 * a4['test_big_error_rate_M3']:.0f}% of test molecules for M3, "
     f"{100 * a4['test_big_error_rate_primary']:.0f}% for the primary model and {100 * a4['oof_big_error_rate']:.0f}% in the "
     f"out-of-fold training predictions. Both generic signals carry some information. Tree-prediction spread separates large errors "
     f"with an AUROC of {a4['oof_auroc_tree_sd']:.2f} on the training pool (no test data) and {a4['test_auroc_M3_tree_sd']:.2f} on "
     f"the test set, and similarity gives {a4['oof_auroc_dissimilarity']:.2f} and {a4['test_auroc_M3_dissimilarity']:.2f}; against the "
     f"primary model’s errors they reach {a4['test_auroc_primary_tree_sd']:.2f} and {a4['test_auroc_primary_dissimilarity']:.2f}. "
     "The effect is modest. Rejecting the 20% of test molecules with the largest tree spread lowers the MAE of the "
     f"primary model from {rc['primary_tree SD']['0']:.3f} to {rc['primary_tree SD']['20']:.3f} and that of M3 from "
     f"{rc['M3_tree SD']['0']:.3f} to {rc['M3_tree SD']['20']:.3f}, far from the oracle that rejects the true worst errors "
     "(Figure 8).")
para(f"The charged compounds are the clearest failure of both signals. The 11 charged test molecules lie at a median percentile of "
     f"only {100 * a4['charged_rank_pct_tree_sd_median']:.0f} of the tree spread and {100 * a4['charged_rank_pct_dissimilarity_median']:.0f} of the "
     "dissimilarity ranking, so the models are not unusually uncertain about them, and the molecules look more similar to the training "
     "data than the median test molecule. A plausible reading, not tested here, is that a small structural change (making the "
     "core a cation) has a large effect on LogD, a property cliff that neither fingerprint similarity nor forest variance can "
     "see. The simple rule flag does identify them (every flagged molecule is a large error; precision "
     f"{a4['test_charged_flag_precision_M3']:.1f}) at a cost of {100 * a4['n_charged_test'] / p2['n_test']:.1f}% of the molecules, and "
     f"rejecting them lowers the M3 MAE from {a4['test_mae_all_M3']:.3f} to {a4['test_mae_without_charged_M3']:.3f}. This is, however, a "
     "rule designed after seeing the test errors, and the training pool contains too few charged molecules "
     f"({a4['charged_in_oof_training_pool']}) for any validation on the training side; it should be read as a hypothesis for "
     "future data, not as a validated safeguard.")
picture(FIG / "fig10_risk_coverage.png", 15.5)
caption("Figure 8. MAE of the retained test molecules after rejecting the highest-scoring q% by tree spread or dissimilarity, for M3 "
        "(left) and the primary model (right); dotted: oracle, dashed: MAE after rejecting only the 11 charged molecules.")

heading("Comparison with prior work, strengths, limitations and future directions", 2)
para("The challenge leaderboard cannot be used as a direct benchmark, because participants could use external "
     "data and pretraining [6] and the scoring used a different aggregate metric [4]. The prospective ADME study "
     "reports that gradient boosting and deep learning outperformed random forest over time on other "
     "endpoints [3]; the gradient-boosting model and the small neural network here also beat the forests, which is consistent with "
     "that report; graph neural networks and pretrained models were not evaluated. Strengths of this study are the auditable protocol, the "
     "written Phase-2 protocol, identical evaluation molecules for all models, Phase-2 model selection on time-ordered validation "
     "of the training pool only (the Phase-1 baselines were tuned on a random split), paired comparisons at seed and molecule level, and self-checked results. Limitations are:", after=3)
bullet("One campaign and one split: results do not show generalisation to other projects or assays.")
bullet("The molecule number is a chronological proxy supported, not confirmed, by the contiguous split (the order within the training pool is unverified), and identity is by canonical "
       "isomeric SMILES without salt, tautomer or charge normalisation.")
bullet("The ML-ready data exclude censored measurements; the LogD assay pH is not stated in the sources consulted.")
bullet("The test set was inspected in Phase 1 before Phase 2 was designed; Phase 2 limits snooping by selecting on the training "
       "pool and scoring each pre-listed configuration once (the primary model with three seeds), but it is not a fully blind evaluation. Twenty-five configurations were scored on the test set in all: M0–M2 (Phase 1), the 20 pre-listed Phase-2 configurations (which include M3) and the two later neural-network models (M8, M9).")
bullet("Four temporal-CV folds overlap in training data and differ widely; the gradient-boosting model is deterministic, so seeds "
       "do not measure its variability; uncertainty is taken from the random forest only.")
bullet("The charged-molecule explanation and the charge flag come from 11 test molecules and are hypotheses.")
bullet("The neural network is a small scikit-learn multilayer perceptron with a four-configuration grid, added after the Phase-2 "
       "results were seen; it is not evidence about graph neural networks or pretrained models.")
bullet("Test labels are public, so this is a retrospective evaluation, not a prospective blind test.")
para("Future work could add charge or ionisation-state descriptors and evaluate them on data that contain charged compounds, compare "
     "graph neural networks and pretrained models, test retraining as new data arrive (as examined in [3]), add conformal or other calibrated "
     "uncertainty estimates, and verify the time proxy with the data provider.", before=3)

# 6 Conclusions --------------------------------------------------------------
heading("Conclusions", 1, names[5])
para(f"On the OpenADMET–ExpansionRx early-to-later-stage split ({tr['final_train_pool']:,} training and {te['final_test']:,} test "
     "molecules), this project built an auditable, reproducible chain of evidence from baselines to drift, validation "
     "protocols, stronger models and failure detection. Answers to the research questions are:", after=3)
bullet(f"RQ1: Ridge regression on eight descriptors reduced test MAE by {rel['M1']:.1f}% relative to the median predictor; a random forest "
       f"on the same descriptors by {rel['M2']:.1f}%, and adding a Morgan fingerprint by {rel['M3']:.1f}% ({mae['M3']:.3f}), lower than "
       "M2 in all three seeds.")
bullet(f"RQ2: Later-stage molecules are larger and more complex with unchanged calculated logP and are separable from earlier ones "
       f"(AUROC {a1['adversarial_auroc']['FULL']:.2f}); mean LogD fell during the campaign and recovered somewhat in the test period.")
bullet(f"RQ3: Random validation overstated accuracy (M3 validation MAE {prot['random_80_20']['M3']:.2f} vs test "
       f"{prot['official_test']['M3']:.2f}); time-ordered validation matched the test MAE of the learned models within about "
       f"{100 * max(abs(1 - opt['temporal_cv_mean'][m]) for m in ('M1', 'M2', 'M3')):.1f}% (M3 the largest) but overstated the gain over the baseline.")
bullet(f"RQ4: Gradient boosting on full descriptors and fingerprint counts, selected by temporal CV, lowered the test MAE to {PM:.3f} "
       f"({100 * (mae['M0'] - PM) / mae['M0']:.0f}% below the baseline). A small neural network ensemble, added later under its own written "
       f"protocol, did better ({mlp['test_mae']['M8']:.3f}, {100 * (mae['M0'] - mlp['test_mae']['M8']) / mae['M0']:.0f}% below; "
       f"M8 − primary {mlp['bootstrap']['M8_minus_primary']['point']:+.3f} [{mlp['bootstrap']['M8_minus_primary']['ci95'][0]:+.3f}, "
       f"{mlp['bootstrap']['M8_minus_primary']['ci95'][1]:+.3f}]). Recency weighting gave no reliable benefit, and temporal CV exaggerated "
       "the advantage of both model families.")
bullet("RQ5: The largest errors are permanently charged analogues that are almost absent from the training data. Tree spread and "
       "similarity carry modest information about errors in general but do not flag these molecules; a simple charge rule does, "
       "as a hypothesis for future data.")
para("The results do not show that these models are adequate for decision-making, only how far such cheap models go on this split, "
     "how much the answer depends on how it is validated, and where they break. The principal limitations are the single data source, "
     "public test labels, the test-set inspection in Phase 1 and the exploratory nature of the failure analyses. All objectives "
     "O1–O6 were addressed within these limits.", before=3)

# 7 References ---------------------------------------------------------------
cur.el = ref_head._p
REFS = [
    "Waring MJ. Lipophilicity in drug discovery. Expert Opin Drug Discov. 2010;5:235–248. doi:10.1517/17460441003605098",
    "Lipinski CA, Lombardo F, Dominy BW, Feeney PJ. Experimental and computational approaches to estimate solubility and "
    "permeability in drug discovery and development settings. Adv Drug Deliv Rev. 1997;23:3–25. doi:10.1016/S0169-409X(96)00423-1",
    "Fang C, Wang Y, Grater R, Kapadnis S, Black C, Trapa P, Sciabola S. Prospective validation of machine learning "
    "algorithms for absorption, distribution, metabolism, and excretion prediction: an industrial perspective. "
    "J Chem Inf Model. 2023;63:3263–3274. doi:10.1021/acs.jcim.3c00160",
    "MacDermott-Opeskin H, Castellanos M. OpenADMET–ExpansionRx blind challenge. OpenADMET; 2025. "
    "https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/ (accessed 15 November 2026).",
    "OpenADMET. openadmet-expansionrx-challenge-data (Hugging Face dataset, CC BY 4.0). "
    f"https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data (revision {prov['hf_revision_sha'][:8]}; accessed 15 November 2026).",
    "OpenADMET. ExpansionRx-Challenge-Tutorial (GitHub repository). https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial (accessed 15 November 2026).",
    "Wildman SA, Crippen GM. Prediction of physicochemical parameters by atomic contributions. "
    "J Chem Inf Comput Sci. 1999;39:868–873. doi:10.1021/ci990307l",
    "Ertl P, Rohde B, Selzer P. Fast calculation of molecular polar surface area as a sum of fragment-based "
    "contributions and its application to the prediction of drug transport properties. J Med Chem. 2000;43:3714–3717. doi:10.1021/jm000942e",
    "Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50:742–754. doi:10.1021/ci100050t",
    "Breiman L. Random forests. Mach Learn. 2001;45:5–32. doi:10.1023/A:1010933404324",
    "Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. Technometrics. "
    "1970;12:55–67. doi:10.1080/00401706.1970.10488634",
    "Friedman JH. Greedy function approximation: a gradient boosting machine. Ann Stat. 2001;29:1189–1232. doi:10.1214/aos/1013203451",
    f"Landrum G. RDKit: open-source cheminformatics software, version {env['rdkit']}. https://www.rdkit.org",
    "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825–2830.",
]
for i, r in enumerate(REFS, 1):
    p = para(f"[{i}] " + r, align="left", size=10, after=4, indent=0.9, line=1.1)
    p.paragraph_format.first_line_indent = Cm(-0.9)

import json as _json
_pg = HERE / "_pages.json"
if _pg.exists():
    info = _json.loads(_pg.read_text())
    for p_, n_ in zip(toc_paras, info["toc"]):
        ts = [t for t in p_._p.iter(qn("w:t")) if t.text and t.text.strip().isdigit()]
        ts[-1].text = str(n_)
    ft = doc.sections[1].footer._element
    instr = list(ft.iter(qn("w:instrText")))
    for it in instr:
        if "SECTIONPAGES" in it.text:
            r_ = it.getparent()
            nxt = [t for t in r_.itersiblings() if t.find(qn("w:t")) is not None]
            if nxt:
                nxt[0].find(qn("w:t")).text = str(info["section_pages"])
doc.core_properties.title = TITLE
doc.core_properties.author = STUDENT_NAME
doc.save(OUT)
print("saved", OUT)
