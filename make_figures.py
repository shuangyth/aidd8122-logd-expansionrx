"""Build report figures/tables from saved results (no model refitting).

Usage: python make_figures.py   (after run_project.py)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent
RES, SPL, FIG = ROOT / "results", ROOT / "splits", ROOT / "figures"
FIG.mkdir(exist_ok=True)
CFG = json.loads((ROOT / "config.json").read_text())
LAB = CFG["label"]
COL = {"M0": "#8c8c8c", "M1": "#4c78a8", "M2": "#f58518", "M3": "#54a24b"}
NAME = {"M0": "M0 median", "M1": "M1 Ridge (8 desc.)", "M2": "M2 RF (8 desc.)",
        "M3": "M3 RF (8 desc.+Morgan)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

P = pd.read_csv(RES / "predictions_test.csv", keep_default_na=False)
P["seed"] = P["seed"].astype(str)
P["y_true"] = P.y_true.astype(float)
P["y_pred"] = P.y_pred.astype(float)
M = pd.read_csv(RES / "metrics_test_per_run.csv", keep_default_na=False)
M["seed"] = M["seed"].astype(str)
train = pd.read_csv(SPL / "train_full_clean.csv")
test = pd.read_csv(SPL / "test_clean.csv")


def fig1_workflow():
    steps = ["1. Official CSVs\ntrain / test\nSHA256 pinned",
             "2. Audit\nRDKit canonical\nSMILES, leakage\ncheck",
             "3. Internal split\n80/20 grouped\nseed 42",
             "4. Features\n8 descriptors\n+ Morgan\nr=2, 1024 bit",
             "5. Models M0-M3\nvalidation-based\nparameter freeze",
             "6. Final test\nretrain on full\ntrain pool, score\nlater-stage test"]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(0, 6 * 2.0)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(steps):
        x = i * 2.0 + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.08), 1.6, 0.84, boxstyle="round,pad=0.02",
                                    fc="#eaf1f8", ec="#4c78a8"))
        ax.text(x + 0.8, 0.5, s, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.98, 0.5), xytext=(x + 1.68, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    fig.savefig(FIG / "fig1_workflow.png")
    plt.close(fig)


def fig2_labels():
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.linspace(-2.5, 5.5, 41)
    ax.hist(train[LAB], bins=bins, density=True, alpha=0.55, color="#4c78a8",
            label=f"Train (n={len(train)})")
    ax.hist(test[LAB], bins=bins, density=True, alpha=0.55, color="#f58518",
            label=f"Test (n={len(test)})")
    ax.set_xlabel("Experimental LogD")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig2_label_distribution.png")
    plt.close(fig)


def fig3_mae():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, m in enumerate(["M0", "M1", "M2", "M3"]):
        d = M[M.model == m]
        ax.bar(i, d.MAE.mean(), color=COL[m], alpha=0.75, width=0.6)
        if len(d) > 1:
            ax.scatter([i] * len(d), d.MAE, color="k", s=14, zorder=3)
        ax.text(i, d.MAE.mean() + 0.015, f"{d.MAE.mean():.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["M0\nmedian", "M1\nRidge", "M2\nRF", "M3\nRF+FP"])
    ax.set_ylabel("Test MAE (LogD units)")
    ax.set_ylim(0, 0.95)
    fig.savefig(FIG / "fig3_mae_comparison.png")
    plt.close(fig)


def fig4_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)
    for ax, m in zip(axes, ["M1", "M3"]):
        s = "det" if m == "M1" else "11"
        d = P[(P.model == m) & (P.seed == s)]
        ax.scatter(d.y_true, d.y_pred, s=5, alpha=0.35, color=COL[m])
        lim = [-2.5, 5.5]
        ax.plot(lim, lim, "k--", lw=0.8)
        mae = np.mean(np.abs(d.y_true - d.y_pred))
        ax.set_title(f"{NAME[m]}" + (" (seed 11)" if m == "M3" else "") + f"\nMAE={mae:.3f}",
                     fontsize=9)
        ax.set_xlabel("Experimental LogD")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted LogD")
    fig.savefig(FIG / "fig4_scatter_M1_M3.png")
    plt.close(fig)


def table3_failures():
    d = P[(P.model == "M3") & (P.seed == "11")].copy()
    d["abs_err"] = (d.y_true - d.y_pred).abs()
    top = d.sort_values("abs_err", ascending=False).head(5).reset_index(drop=True)
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    rows = []
    for _, r in top.iterrows():
        m = Chem.MolFromSmiles(r.canonical_smiles)
        rows.append({"name": r["name"], "y_true": r.y_true, "y_pred": r.y_pred,
                     "abs_err": r.abs_err, "MolWt": Descriptors.MolWt(m),
                     "MolLogP": Crippen.MolLogP(m), "TPSA": rdMolDescriptors.CalcTPSA(m),
                     "smiles": r.canonical_smiles})
    T = pd.DataFrame(rows)
    T.to_csv(RES / "table3_failure_cases.csv", index=False)
    mols = [Chem.MolFromSmiles(s) for s in T.smiles]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, m, (_, r) in zip(axes.ravel(), mols, T.iterrows()):
        ax.imshow(Draw.MolToImage(m, size=(520, 400)))
        ax.set_title(f"{r['name']}\nexp {r.y_true:.2f} | pred {r.y_pred:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "table3_failure_structures.png")
    plt.close(fig)


def fig5_similarity():
    """Exploratory: test absolute error vs. nearest-neighbour Tanimoto to the train pool."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tr_fp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.canonical_smiles]
    te = P[(P.model == "M3") & (P.seed == "11")].reset_index(drop=True)
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(
        gen.GetFingerprint(Chem.MolFromSmiles(s)), tr_fp)) for s in te.canonical_smiles])
    te["nn_sim"] = nn
    te["abs_err"] = (te.y_true - te.y_pred).abs()
    # same analysis for validation-vs-fit to explain the optimistic internal split
    fit = pd.read_csv(SPL / "train_fit.csv")
    val = pd.read_csv(SPL / "train_val.csv")
    fit_fp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in fit.canonical_smiles]
    val_nn = np.array([max(DataStructs.BulkTanimotoSimilarity(
        gen.GetFingerprint(Chem.MolFromSmiles(s)), fit_fp)) for s in val.canonical_smiles])
    te["bin"] = pd.qcut(te.nn_sim, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    B = te.groupby("bin", observed=True).agg(n=("abs_err", "size"), sim_min=("nn_sim", "min"),
                                             sim_max=("nn_sim", "max"), MAE=("abs_err", "mean"))
    B.to_csv(RES / "exploratory_similarity_bins.csv")
    summ = {"test_nn_sim_median": float(np.median(nn)), "val_nn_sim_median": float(np.median(val_nn)),
            "val_frac_nn_sim_ge_0.999": float((val_nn >= 0.999).mean()), "val_n_nn_sim_ge_0.999": int((val_nn >= 0.999).sum()),
            "val_n": int(len(val_nn)), "test_frac_nn_sim_ge_0.999": float((nn >= 0.999).mean()),
            "spearman_sim_vs_abs_err": float(te[["nn_sim", "abs_err"]].corr("spearman").iloc[0, 1])}
    (RES / "exploratory_similarity_summary.json").write_text(json.dumps(summ, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    axes[0].hist(val_nn, bins=30, alpha=0.55, density=True, color="#4c78a8",
                 label="Validation -> fit")
    axes[0].hist(nn, bins=30, alpha=0.55, density=True, color="#f58518",
                 label="Test -> train pool")
    axes[0].set_xlabel("Nearest-neighbour Tanimoto (Morgan r=2, 2048)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(range(4), B.MAE, color="#54a24b", alpha=0.75)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(B.index, fontsize=8)
    axes[1].set_ylabel("M3 test MAE (seed 11)")
    axes[1].set_xlabel("Similarity quartile of test molecules")
    fig.savefig(FIG / "fig5_similarity_exploratory.png")
    plt.close(fig)
    print(B, summ)


def charged_subset():
    """Exploratory: error on permanently charged molecules (net formal charge != 0)."""
    def q(smi):
        return Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
    out = {"train_charged": int((train.canonical_smiles.map(q) != 0).sum()),
           "train_total": int(len(train)),
           "test_charged": int((test.canonical_smiles.map(q) != 0).sum()),
           "test_total": int(len(test))}
    tq = test.set_index("canonical_smiles")[LAB]
    for m, s in (("M1", "det"), ("M2", "11"), ("M3", "11")):
        d = P[(P.model == m) & (P.seed == s)].copy()
        d["ch"] = d.canonical_smiles.map(q) != 0
        d["ae"] = (d.y_true - d.y_pred).abs()
        out[m] = {"MAE_all": float(d.ae.mean()), "MAE_charged": float(d[d.ch].ae.mean()),
                  "MAE_uncharged": float(d[~d.ch].ae.mean()),
                  "median_true_logD_charged": float(d[d.ch].y_true.median()),
                  "n_top50_errors_charged": int(d.sort_values("ae", ascending=False).head(50).ch.sum())}
    (RES / "exploratory_charged_subset.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def bootstrap_ci(n_boot=2000, seed=0):
    """Exploratory: paired bootstrap over test molecules for MAE differences (seed-11 / deterministic runs)."""
    sel = {"M0": "det", "M1": "det", "M2": "11", "M3": "11"}
    ae = {}
    for m, s in sel.items():
        d = P[(P.model == m) & (P.seed == s)]
        ae[m] = (d.y_true - d.y_pred).abs().values
    n = len(ae["M0"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {"n_boot": n_boot, "rng_seed": seed, "n_test": n}
    for a, b in (("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M3", "M0")):
        diff = (ae[a] - ae[b])
        boots = diff[idx].mean(axis=1)
        out[f"MAE_{a}_minus_{b}"] = {"point": float(diff.mean()),
                                     "ci95": [float(np.percentile(boots, 2.5)),
                                              float(np.percentile(boots, 97.5))]}
    (RES / "exploratory_bootstrap_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    fig1_workflow()
    fig2_labels()
    fig3_mae()
    fig4_scatter()
    table3_failures()
    fig5_similarity()
    charged_subset()
    bootstrap_ci()
    print("figures written to", FIG)
