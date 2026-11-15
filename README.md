# Chasing a moving target: LogD models on the OpenADMET-ExpansionRx temporal split

AIDD8122 final project. Single endpoint (LogD), SMILES-derived features only, CPU only.

| ID | Model | Input |
|---|---|---|
| M0 | median predictor | none |
| M1 | StandardScaler + Ridge | 8 RDKit descriptors |
| M2 | Random Forest | 8 descriptors |
| M3 | Random Forest | 8 descriptors + Morgan (r=2, 1024 bit) |

## Reproduce
Use Python 3.13.5 and the versions pinned in `requirements.txt` (recorded on macOS arm64).
Run the commands below with that environment's Python; the system Python may be a different version.
These scripts overwrite outputs, so reproduce in a separate clone to preserve the submitted results.
Network access is needed to download the raw CSVs; their expected SHA256 values are in `config.json`.
```bash
python -m pip install -r requirements.txt
python -I run_project.py      # download, audit, validate, final test run, self-checks (~20 s)
python -I make_figures.py     # figures/ and results tables from saved predictions
python -I sensitivity_flat_groups.py   # post hoc stereoisomer sensitivity analysis (results/)
```
Protocol: official train/test split kept; identity = RDKit canonical isomeric SMILES; internal 80/20
GroupShuffleSplit (seed 42) for parameter selection only; parameters frozen, then retrained on the full
cleaned train pool; M2/M3 run with seeds 11, 22, 33. Primary metric MAE; also RMSE, R2.

## Layout
`config.json` settings | `run_project.py` pipeline | `make_figures.py` figures | `report/` report build (template-based) |
`splits/` cleaned data and splits | `results/` metrics, predictions, logs, checks | `figures/` report figures |
`slides/` presentation build

## Phase 2 (written protocol: `PHASE2_PROTOCOL.md`)
`python phase2.py` (about 10 min): drift analysis, temporal cross-validation (molecule number as a chronological proxy),
stronger models (full RDKit descriptors, gradient boosting, ensemble) with recency weighting selected on temporal CV only,
and uncertainty signals / risk-coverage analysis. Outputs: `results/phase2/`, `figures/fig7..fig10`.

## Phase 2b: MLP comparison (written protocol: `PHASE2B_MLP_PROTOCOL.md`)
After Phase 1 and Phase 2, run `python -I phase2b_mlp.py` (about 8 min).
Outputs: `results/phase2/mlp_*` and `figures/fig11_mlp_comparison.png`.
Runtimes are approximate and depend on hardware. Phase 2 and Phase 2b saved outputs are included;
rebuilding them is separate from the lightweight Phase-1 reproduction check.

## Report and slides (optional, local artifacts)
The report requires the original course template at
`report/template/AIDD8122_Final_Project_Report_Template.docx`, which is excluded from the repository.
Both builders need LibreOffice (`soffice`), the pinned document/image packages, and the macOS font
`/System/Library/Fonts/Supplemental/Arial Unicode.ttf`; the report also uses Poppler (`pdftotext`).
Paths and fonts may need adaptation on other systems. Generated DOCX/PDF/PPTX files are not committed.
Personal details are read from environment variables, defaulting to `P00000` and `Student`.
After publication, rebuild with your own values:

```bash
export AIDD_STUDENT_ID="YOUR_STUDENT_ID"
export AIDD_STUDENT_NAME="YOUR_NAME"
python -I report/make_report.py "https://github.com/shuangyth/aidd8122-logd-expansionrx" "v1.0-submission"
python -I slides/build_slides.py "github.com/shuangyth/aidd8122-logd-expansionrx" "v1.0-submission"
```

Outputs: `report/AIDD8122_Final_Project_Report.docx` / `.pdf` and
`slides/AIDD8122_Final_Project_Slides.pptx` / `.pdf`.

## Data and code licences
Code is licensed under the MIT License; see `LICENSE`.
The OpenADMET-ExpansionRx dataset and the derived cleaned/split data are attributed separately
under CC BY 4.0; see `DATA_SOURCES.md` for the source links and
`results/data_provenance.json` for full file hashes and the recorded dataset revision.
Raw data are downloaded by the pipeline and excluded from Git.

## Limitations
Single research programme; time order per publisher description; labels are public (retrospective);
ML-ready file excludes out-of-range measurements; seeds capture training randomness only.
Data: see `DATA_SOURCES.md` (CC BY 4.0).
