"""Phase 2: drift, evaluation protocols, stronger models + recency, uncertainty flags.
Protocol: PHASE2_PROTOCOL.md (written before this script was run). Usage: python phase2.py  (after run_project.py)
Outputs: results/phase2/*, figures/fig7..fig11.
"""
import importlib.util
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", ROOT / "run_project.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
CFG, SPL, RES, LAB = rp.CFG, rp.SPL, rp.RES, rp.CFG["label"]
OUT = RES / "phase2"
OUT.mkdir(exist_ok=True)
FIG = ROOT / "figures"
NJ = 8
T0 = time.time()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ data
def idnum(df):
    return df["Molecule Name"].str.extract(r"(\d+)")[0].astype(int)


train = pd.read_csv(SPL / "train_full_clean.csv")
test = pd.read_csv(SPL / "test_clean.csv")
train["id"], test["id"] = idnum(train), idnum(test)
train = train.sort_values("id", kind="stable").reset_index(drop=True)     # chronological proxy order
ytr, yte = train[LAB].values, test[LAB].values
N = len(train)

gen_c = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def feats(df):
    mols = [Chem.MolFromSmiles(s) for s in df.canonical_smiles]
    d8, _ = rp.featurize(df)
    bits = np.array([gen_c.GetFingerprintAsNumPy(m) for m in mols], dtype=float)
    cnts = np.array([gen_c.GetCountFingerprintAsNumPy(m) for m in mols], dtype=float)
    names = [n for n, _ in Descriptors.descList]
    def safe(f, m):
        try:
            return float(f(m))
        except Exception:  # noqa: BLE001
            return float("nan")
    full = np.array([[safe(f, m) for _, f in Descriptors.descList] for m in mols], dtype=float)
    return d8, bits, cnts, full, names


log("computing features")
d8_tr, b_tr, c_tr, f_tr, names = feats(train)
d8_te, b_te, c_te, f_te, _ = feats(test)
ok = np.isfinite(f_tr).all(0) & np.isfinite(f_te).all(0) & (f_tr.std(0) > 0)
full_names = [n for n, k in zip(names, ok) if k]
f_tr, f_te = f_tr[:, ok], f_te[:, ok]
log(f"full descriptors kept: {ok.sum()} of {len(ok)}")
F = {"D8": (d8_tr, d8_te), "DF": (np.hstack([d8_tr, b_tr]), np.hstack([d8_te, b_te])),
     "FULL": (f_tr, f_te), "FULLFP": (np.hstack([f_tr, b_tr]), np.hstack([f_te, b_te])),
     "FULLCNT": (np.hstack([f_tr, c_tr]), np.hstack([f_te, c_te]))}
summary = {"n_train": N, "n_test": len(test), "n_full_desc": int(ok.sum())}

# ----------------------------------------------------------------- Act 1
log("Act 1: drift")
q = pd.qcut(np.arange(N), 5, labels=False)
tq = train.groupby(q)[LAB].agg(["mean", "std", "count"])
tq["id_min"] = train.groupby(q).id.min()
tq["id_max"] = train.groupby(q).id.max()
tq.to_csv(OUT / "act1_train_logd_by_id_quintile.csv")
te_sorted = test.sort_values("id").reset_index(drop=True)
t3 = te_sorted.groupby(pd.qcut(np.arange(len(te_sorted)), 3, labels=False))[LAB].agg(["mean", "std", "count"])
t3.to_csv(OUT / "act1_test_logd_by_id_tertile.csv")
smd = []
for j, n in enumerate(rp.DESC_NAMES):
    a, b = d8_tr[:, j], d8_te[:, j]
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd.append({"descriptor": n, "train_mean": a.mean(), "test_mean": b.mean(), "SMD": (b.mean() - a.mean()) / s})
pd.DataFrame(smd).to_csv(OUT / "act1_descriptor_smd.csv", index=False)

# adversarial validation: can a forest tell train from test? (IDs never used)
adv = {}
yadv = np.r_[np.zeros(N), np.ones(len(test))]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for tag, (a, b) in (("D8", F["D8"]), ("FULL", F["FULL"]), ("FULLFP", F["FULLFP"])):
    X = np.vstack([a, b])
    p = cross_val_predict(RandomForestClassifier(300, min_samples_leaf=3, n_jobs=NJ, random_state=0), X, yadv,
                          cv=skf, method="predict_proba")[:, 1]
    adv[tag] = float(roc_auc_score(yadv, p))
rf_adv = RandomForestClassifier(300, min_samples_leaf=3, n_jobs=NJ, random_state=0).fit(np.vstack(F["FULL"]), yadv)
imp = pd.Series(rf_adv.feature_importances_, index=full_names).sort_values(ascending=False).head(10)
imp.to_csv(OUT / "act1_adversarial_top_features.csv", header=["importance"])
summary["act1"] = {"adversarial_auroc": adv, "top_features": imp.index.tolist()[:5],
                   "quintile_means": tq["mean"].round(3).tolist(), "test_tertile_means": t3["mean"].round(3).tolist()}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
allid = pd.concat([train[["id", LAB]], test[["id", LAB]]]).sort_values("id")
ax[0].scatter(allid.id, allid[LAB], s=2, alpha=0.2, color="#8c8c8c")
roll = allid.set_index("id")[LAB].rolling(400, center=True).median()
ax[0].plot(roll.index, roll.values, color="#d62728", lw=2, label="rolling median (400)")
ax[0].axvline(20100.5, color="k", ls="--", lw=1)
ax[0].text(20300, 4.6, "official\ntest", fontsize=8)
ax[0].set_xlabel("Molecule number (chronological proxy)")
ax[0].set_ylabel("Experimental LogD")
ax[0].legend(frameon=False, loc="lower left", fontsize=8)
imp[::-1].plot.barh(ax=ax[1], color="#4c78a8")
ax[1].set_xlabel(f"Importance (AUROC {adv['FULL']:.2f})")
ax[1].tick_params(axis="y", labelsize=7)
fig.savefig(FIG / "fig7_drift.png")
plt.close(fig)

# ----------------------------------------------------------------- model helpers
blocks = np.array_split(np.arange(N), 6)
FOLDS = [(np.concatenate(blocks[:k]), blocks[k]) for k in range(2, 6)]      # fit blocks 1..k, validate block k+1
HL = {"w1000": 1000, "w2500": 2500}
SCHEMES = ["none", "w1000", "w2500", "last50"]
FAMILY_FEATS = {"M3": "DF", "M4": "FULL", "M5": "FULLFP", "M6": "FULLCNT"}


def scheme_idx(fit_idx, scheme):
    fit_idx = np.sort(fit_idx)
    n = len(fit_idx)
    if scheme == "last50":
        return fit_idx[n // 2:], None
    if scheme in HL:
        return fit_idx, 0.5 ** ((n - 1 - np.arange(n)) / HL[scheme])
    return fit_idx, None


def make(fam, params, seed):
    if fam == "M6":
        return HistGradientBoostingRegressor(learning_rate=params["lr"], max_leaf_nodes=params["leaves"], max_iter=400,
                                             l2_regularization=1.0, early_stopping=False, random_state=seed)
    mf = params["mf"]
    return RandomForestRegressor(n_estimators=300, max_features=mf, min_samples_leaf=params["leaf"], n_jobs=NJ,
                                 random_state=seed)


def fit_pred(fam, params, Xa, ya, Xb, fit_idx, scheme, seed=11, return_model=False):
    idx, w = scheme_idx(fit_idx, scheme)
    m = make(fam, params, seed).fit(Xa[idx], ya[idx], sample_weight=w)
    p = m.predict(Xb)
    return (p, m) if return_model else p


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ----------------------------------------------------------------- Act 2
log("Act 2: protocols")
P1 = {"leaf": 1, "mf": "sqrt"}          # Phase-1 frozen M2/M3 settings
cvm = {"M0": [], "M1": [], "M2": [], "M3": []}
nn_fold = []
gen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp2 = [gen2.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.canonical_smiles]
for fi, (fit, val) in enumerate(FOLDS):
    ya, yb = ytr[fit], ytr[val]
    cvm["M0"].append(mae(yb, np.full(len(yb), np.median(ya))))
    cvm["M1"].append(mae(yb, rp.make_model("M1", alpha=0.1).fit(d8_tr[fit], ya).predict(d8_tr[val])))
    cvm["M2"].append(mae(yb, rp.make_model("M2", 11, leaf=1).fit(d8_tr[fit], ya).predict(d8_tr[val])))
    cvm["M3"].append(mae(yb, rp.make_model("M3", 11, leaf=1).fit(F["DF"][0][fit], ya).predict(F["DF"][0][val])))
    fit_fps = [fp2[i] for i in fit]
    nn_fold.append(float(np.median([max(DataStructs.BulkTanimotoSimilarity(fp2[i], fit_fps)) for i in val])))
    log(f"  fold {fi + 1}: " + ", ".join(f"{k}={v[-1]:.3f}" for k, v in cvm.items()))
val_rand = pd.read_csv(RES / "metrics_validation.csv")
sens = json.loads((RES / "sensitivity_flat_group_split.json").read_text())
tsum = pd.read_csv(RES / "metrics_test_summary.csv").set_index("model")
prot = {"random_80_20": {"M0": float(val_rand[val_rand.model == "M0"].MAE.iloc[0]),
                         "M1": float(val_rand[(val_rand.model == "M1")].MAE.min()),
                         "M2": float(val_rand[(val_rand.model == "M2")].MAE.min()),
                         "M3": float(val_rand[val_rand.model == "M3"].MAE.iloc[0])},
        "flat_grouped_80_20": {"M0": sens["val_M0_MAE"], "M1": min(r["MAE"] for r in sens["val_ridge"]),
                               "M2": min(r["MAE"] for r in sens["val_rf_M2"]), "M3": sens["val_M3_at_best_leaf"]["MAE"]},
        "temporal_cv_mean": {k: float(np.mean(v)) for k, v in cvm.items()},
        "official_test": {k: float(tsum.loc[k, "MAE_mean"]) for k in ("M0", "M1", "M2", "M3")}}
act2 = {"protocols": prot, "temporal_cv_per_fold": cvm, "nn_similarity_median_per_fold": nn_fold,
        "optimism_ratio_vs_test": {p: {k: prot[p][k] / prot["official_test"][k] for k in prot[p]} for p in prot},
        "rel_improvement_vs_M0_pct": {p: {k: 100 * (prot[p]["M0"] - prot[p][k]) / prot[p]["M0"] for k in prot[p]} for p in prot}}
(OUT / "act2_protocols.json").write_text(json.dumps(act2, indent=2))
summary["act2"] = act2

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
labs = {"random_80_20": "random\n80/20", "flat_grouped_80_20": "random,\nflat-grouped", "temporal_cv_mean": "temporal\nCV (mean)",
        "official_test": "official\ntest"}
cols = {"M0": "#8c8c8c", "M1": "#4c78a8", "M2": "#f58518", "M3": "#54a24b"}
for i, k in enumerate(["M0", "M1", "M2", "M3"]):
    ax[0].bar(np.arange(4) + (i - 1.5) * 0.2, [prot[p][k] for p in labs], 0.2, color=cols[k], label=k)
ax[0].set_xticks(range(4))
ax[0].set_xticklabels(labs.values(), fontsize=8)
ax[0].set_ylabel("MAE (LogD units)")
ax[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
ax[0].set_ylim(0, 1.15)
for k in ("M0", "M3"):
    ax[1].plot(range(1, 5), cvm[k], "o-", color=cols[k], label=k)
    ax[1].axhline(prot["official_test"][k], color=cols[k], ls=":", lw=1)
ax[1].set_xlabel("Temporal-CV fold (later = more recent)")
ax[1].set_ylabel("MAE")
ax[1].legend(frameon=False, fontsize=8)
ax[1].text(1.0, prot["official_test"]["M0"] + 0.02, "dotted: official test", fontsize=7)
fig.savefig(FIG / "fig8_protocols.png")
plt.close(fig)

# ----------------------------------------------------------------- Act 3
log("Act 3: models and recency (temporal CV)")
GRIDS = {"M4": [{"leaf": l, "mf": m} for l in (1, 2, 5) for m in ("sqrt", 0.33)],
         "M5": [{"leaf": l, "mf": m} for l in (1, 2, 5) for m in ("sqrt", 0.33)],
         "M6": [{"lr": a, "leaves": b} for a in (0.03, 0.1) for b in (15, 31)]}
cv_rows, cvpred = [], {}


def cv_config(fam, params, scheme):
    X = F[FAMILY_FEATS[fam]][0]
    maes = []
    for fi, (fit, val) in enumerate(FOLDS):
        p = fit_pred(fam, params, X, ytr, X[val], fit, scheme)
        cvpred[(fam, json.dumps(params, sort_keys=True), scheme, fi)] = p
        maes.append(mae(ytr[val], p))
    return maes


chosen = {"M3": P1}
for fam, grid in GRIDS.items():
    best = None
    for prm in grid:
        m = cv_config(fam, prm, "none")
        cv_rows.append({"family": fam, "params": json.dumps(prm), "scheme": "none", "cv_mae": np.mean(m), "folds": m})
        log(f"  {fam} {prm}: {np.mean(m):.4f}")
        if best is None or np.mean(m) < best[0]:
            best = (np.mean(m), prm)
    chosen[fam] = best[1]
m = cv_config("M3", P1, "none")
cv_rows.append({"family": "M3", "params": json.dumps(P1), "scheme": "none", "cv_mae": np.mean(m), "folds": m})
for fam in ("M3", "M4", "M5", "M6"):
    for sc in SCHEMES[1:]:
        m = cv_config(fam, chosen[fam], sc)
        cv_rows.append({"family": fam, "params": json.dumps(chosen[fam]), "scheme": sc, "cv_mae": np.mean(m), "folds": m})
        log(f"  {fam} {sc}: {np.mean(m):.4f}")
# M7 = mean of M5 and M6 for every scheme (CV)
for sc in SCHEMES:
    m = []
    for fi, (fit, val) in enumerate(FOLDS):
        p5 = cvpred[("M5", json.dumps(chosen["M5"], sort_keys=True), sc, fi)]
        p6 = cvpred[("M6", json.dumps(chosen["M6"], sort_keys=True), sc, fi)]
        m.append(mae(ytr[val], (p5 + p6) / 2))
    cv_rows.append({"family": "M7", "params": "mean(M5,M6)", "scheme": sc, "cv_mae": np.mean(m), "folds": m})
cvdf = pd.DataFrame(cv_rows)
cvdf.to_csv(OUT / "act3_temporal_cv.csv", index=False)
cfg = cvdf[cvdf.params.isin([json.dumps(chosen[f]) for f in chosen] + ["mean(M5,M6)"]) | (cvdf.family == "M3")]
cfg = cfg[~((cfg.family.isin(["M4", "M5", "M6"])) & (cfg.params != cfg.family.map(lambda f: json.dumps(chosen.get(f)))))]
cfg = cfg.drop_duplicates(["family", "scheme"]).reset_index(drop=True)
assert len(cfg) == 20, len(cfg)
primary = cfg.sort_values("cv_mae").iloc[0]
summary["act3"] = {"chosen_params": chosen, "primary": {"family": primary.family, "scheme": primary.scheme,
                                                         "cv_mae": float(primary.cv_mae)}}
log(f"primary by temporal CV: {primary.family} / {primary.scheme} ({primary.cv_mae:.4f})")

# --- test: score all 20 configurations once (seed 11)
log("Act 3: test scoring")
allidx = np.arange(N)
test_pred = {}


def test_fit(fam, scheme, seed):
    X, Xt = F[FAMILY_FEATS[fam]]
    return fit_pred(fam, chosen[fam], X, ytr, Xt, allidx, scheme, seed)


rows = []
for sc in SCHEMES:
    for fam in ("M3", "M4", "M5", "M6"):
        test_pred[(fam, sc)] = test_fit(fam, sc, 11)
    test_pred[("M7", sc)] = (test_pred[("M5", sc)] + test_pred[("M6", sc)]) / 2
for fam in ("M3", "M4", "M5", "M6", "M7"):
    for sc in SCHEMES:
        c = float(cfg[(cfg.family == fam) & (cfg.scheme == sc)].cv_mae.iloc[0])
        rows.append({"family": fam, "scheme": sc, "cv_mae": c, "test_mae": mae(yte, test_pred[(fam, sc)]),
                     "test_rmse": float(np.sqrt(np.mean((yte - test_pred[(fam, sc)]) ** 2)))})
t20 = pd.DataFrame(rows)
t20.to_csv(OUT / "act3_all_configs_cv_vs_test.csv", index=False)
rho = spearmanr(t20.cv_mae, t20.test_mae)
summary["act3"]["cv_test_spearman_over_20_configs"] = [float(rho[0]), float(rho[1])]

# --- primary with 3 seeds
pf, ps = primary.family, primary.scheme
seeds = CFG["final_seeds"]
prim_runs = []
for s in seeds:
    if pf == "M7":
        p = (test_fit("M5", ps, s) + test_fit("M6", ps, s)) / 2
    else:
        p = test_fit(pf, ps, s)
    prim_runs.append(p)
prim_mae = [mae(yte, p) for p in prim_runs]
summary["act3"]["primary_test_mae_seeds"] = prim_mae
summary["act3"]["primary_test_mae_mean"] = float(np.mean(prim_mae))
summary["act3"]["primary_test_mae_sd"] = float(np.std(prim_mae, ddof=1))
pred_prim = prim_runs[0]
pd.DataFrame({"canonical_smiles": test.canonical_smiles, "y_true": yte, "primary_seed11": prim_runs[0]}).to_csv(
    OUT / "act3_primary_test_predictions.csv", index=False)
# bootstrap vs M3 (Phase 1, seed 11) and vs M0
P1pred = pd.read_csv(RES / "predictions_test.csv", keep_default_na=False)
P1pred["seed"] = P1pred.seed.astype(str)


def p1(model, seed):
    d = P1pred[(P1pred.model == model) & (P1pred.seed == seed)].set_index("canonical_smiles")
    return d.loc[test.canonical_smiles, "y_pred"].astype(float).values


ae = {"primary": np.abs(yte - pred_prim), "M3_phase1": np.abs(yte - p1("M3", "11")), "M0": np.abs(yte - p1("M0", "det"))}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(yte), size=(2000, len(yte)))
bs = {}
for a, b in (("primary", "M3_phase1"), ("primary", "M0")):
    d = ae[a] - ae[b]
    bb = d[idx].mean(1)
    bs[f"{a}_minus_{b}"] = {"point": float(d.mean()), "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
summary["act3"]["bootstrap"] = bs

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, col, ttl in zip(ax, ("cv_mae", "test_mae"), ("Temporal CV (train pool only)", "Official test (scored once)")):
    pv = t20.pivot(index="family", columns="scheme", values=col).loc[["M3", "M4", "M5", "M6", "M7"], SCHEMES]
    im = a.imshow(pv.values, cmap="viridis_r", aspect="auto")
    a.set_xticks(range(4))
    a.set_xticklabels(["none", "half-life\n1000", "half-life\n2500", "last 50%"], fontsize=8)
    a.set_yticks(range(5))
    a.set_yticklabels(["M3 RF 8d+FP", "M4 RF full-d", "M5 RF full-d+FP", "M6 HGB full-d+cnt", "M7 mean(M5,M6)"], fontsize=8)
    for i in range(5):
        for j in range(4):
            a.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    a.set_title(ttl, fontsize=9)
fig.savefig(FIG / "fig9_models_recency.png")
plt.close(fig)

# ----------------------------------------------------------------- Act 4
log("Act 4: uncertainty flags")


def tree_sd(model, X):
    X = np.asarray(X, dtype=np.float32)
    return np.std(np.stack([t.predict(X) for t in model.estimators_]), axis=0)


def charged(smiles):
    return np.array([Chem.GetFormalCharge(Chem.MolFromSmiles(s)) != 0 for s in smiles])


# test-side scores from the Phase-1 M3 forest (seed 11)
X3, X3t = F["DF"]
m3 = rp.make_model("M3", 11, leaf=1).fit(X3, ytr)
sd_te = tree_sd(m3, X3t)
tr_fp = [gen2.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.canonical_smiles]
nn_te = np.array([max(DataStructs.BulkTanimotoSimilarity(gen2.GetFingerprint(Chem.MolFromSmiles(s)), tr_fp))
                  for s in test.canonical_smiles])
chg_te = charged(test.canonical_smiles)
scores_te = {"tree_sd": sd_te, "dissimilarity": 1 - nn_te}
err_te = {"M3": np.abs(yte - p1("M3", "11")), "primary": np.abs(yte - pred_prim)}

# out-of-fold scores on the training pool (no test data)
oof = {"err": [], "tree_sd": [], "dissimilarity": []}
for fit, val in FOLDS:
    mm = rp.make_model("M3", 11, leaf=1).fit(X3[fit], ytr[fit])
    pv = mm.predict(X3[val])
    oof["err"].append(np.abs(ytr[val] - pv))
    oof["tree_sd"].append(tree_sd(mm, X3[val]))
    ffp = [tr_fp[i] for i in fit]
    oof["dissimilarity"].append(np.array([1 - max(DataStructs.BulkTanimotoSimilarity(tr_fp[i], ffp)) for i in val]))
oof = {k: np.concatenate(v) for k, v in oof.items()}
THR = 1.0
act4 = {"threshold_abs_error": THR, "n_charged_test": int(chg_te.sum())}
for name, e in err_te.items():
    big = (e >= THR).astype(int)
    act4[f"test_big_error_rate_{name}"] = float(big.mean())
    for sname, sc in scores_te.items():
        act4[f"test_auroc_{name}_{sname}"] = float(roc_auc_score(big, sc))
        act4[f"test_spearman_{name}_{sname}"] = float(spearmanr(sc, e)[0])
    act4[f"test_auroc_{name}_charged_flag"] = float(roc_auc_score(big, chg_te.astype(int)))
    act4[f"test_charged_flag_precision_{name}"] = float(big[chg_te].mean())
    act4[f"test_mae_all_{name}"] = float(e.mean())
    act4[f"test_mae_without_charged_{name}"] = float(e[~chg_te].mean())
bigo = (oof["err"] >= THR).astype(int)
act4["oof_big_error_rate"] = float(bigo.mean())
for sname in ("tree_sd", "dissimilarity"):
    act4[f"oof_auroc_{sname}"] = float(roc_auc_score(bigo, oof[sname]))
for sname, sc in scores_te.items():
    pct = pd.Series(sc).rank(pct=True).values
    act4[f"charged_rank_pct_{sname}_median"] = float(np.median(pct[chg_te]))
act4["charged_in_oof_training_pool"] = int(charged(train.canonical_smiles).sum())
(OUT / "act4_uncertainty.json").write_text(json.dumps(act4, indent=2))
summary["act4"] = act4

# risk-coverage on the test set (M3 errors and primary errors)
qs = np.arange(0, 31, 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
rc = {}
for name, a in zip(("M3", "primary"), ax):
    e = err_te[name]
    for sname, sc, col in (("tree SD", sd_te, "#4c78a8"), ("1 - NN similarity", 1 - nn_te, "#f58518")):
        order = np.argsort(-sc)
        curve = [e[order[int(len(e) * qq / 100):]].mean() for qq in qs]
        a.plot(qs, curve, color=col, label=sname)
        rc[f"{name}_{sname}"] = dict(zip(qs.tolist(), np.round(curve, 4).tolist()))
    order = np.argsort(-e)
    a.plot(qs, [e[order[int(len(e) * qq / 100):]].mean() for qq in qs], "k:", label="oracle")
    keep = ~chg_te
    a.axhline(e[keep].mean(), color="#d62728", ls="--", lw=1, label="reject 11 charged (0.5%)")
    a.set_xlabel("% of test molecules rejected (highest score first)")
    a.set_ylabel("MAE of retained molecules")
    a.set_title(f"{name} errors", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
fig.savefig(FIG / "fig10_risk_coverage.png")
plt.close(fig)
(OUT / "act4_risk_coverage_curves.json").write_text(json.dumps(rc))

# ---------------------------------------------------------------- done
summary["runtime_seconds"] = round(time.time() - T0, 1)
(OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=float))
log("done")
