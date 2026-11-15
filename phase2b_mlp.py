"""Phase 2b: small neural network (MLP) under the Phase-2 protocol. See PHASE2B_MLP_PROTOCOL.md.
Usage: python phase2b_mlp.py   (after run_project.py and phase2.py)
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
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", ROOT / "run_project.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
RES, SPL, LAB = rp.RES, rp.SPL, rp.CFG["label"]
OUT = RES / "phase2"
FIG = ROOT / "figures"
T0 = time.time()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


train = pd.read_csv(SPL / "train_full_clean.csv")
test = pd.read_csv(SPL / "test_clean.csv")
train["id"] = train["Molecule Name"].str.extract(r"(\d+)")[0].astype(int)
train = train.sort_values("id", kind="stable").reset_index(drop=True)
ytr, yte = train[LAB].values, test[LAB].values
N = len(train)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def feats(df):
    mols = [Chem.MolFromSmiles(s) for s in df.canonical_smiles]
    cnt = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=float)

    def safe(f, m):
        try:
            return float(f(m))
        except Exception:  # noqa: BLE001
            return float("nan")
    full = np.array([[safe(f, m) for _, f in Descriptors.descList] for m in mols], dtype=float)
    return full, cnt


log("features")
f_tr, c_tr = feats(train)
f_te, c_te = feats(test)
ok = np.isfinite(f_tr).all(0) & np.isfinite(f_te).all(0) & (f_tr.std(0) > 0)
f_tr, f_te = f_tr[:, ok], f_te[:, ok]
nd = f_tr.shape[1]
X_tr, X_te = np.hstack([f_tr, c_tr]), np.hstack([f_te, c_te])
log(f"descriptors kept {nd}; feature matrix {X_tr.shape}")

blocks = np.array_split(np.arange(N), 6)
FOLDS = [(np.concatenate(blocks[:k]), blocks[k]) for k in range(2, 6)]


def make_mlp(hidden, alpha, seed):
    pre = ColumnTransformer([
        ("desc", Pipeline([("sc", StandardScaler()), ("clip", FunctionTransformer(lambda x: np.clip(x, -5, 5)))]), list(range(nd))),
        ("cnt", FunctionTransformer(np.log1p), list(range(nd, X_tr.shape[1]))),
    ])
    mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, solver="adam", learning_rate_init=1e-3, batch_size=128,
                       max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                       activation="relu", random_state=seed)
    return make_pipeline(pre, mlp)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


SEEDS = [11, 22, 33]
GRID = [((256,), 1e-3), ((256,), 1e-1), ((256, 128), 1e-3), ((256, 128), 1e-1)]
rows, cvp = [], {}
for hid, al in GRID:
    folds = []
    for fi, (fit, val) in enumerate(FOLDS):
        ps = [make_mlp(hid, al, s).fit(X_tr[fit], ytr[fit]).predict(X_tr[val]) for s in SEEDS]
        p = np.mean(ps, axis=0)
        cvp[(str(hid), al, fi)] = p
        folds.append(mae(ytr[val], p))
    rows.append({"hidden": str(hid), "alpha": al, "cv_mae": float(np.mean(folds)), "folds": folds})
    log(f"  hidden={hid} alpha={al}: CV MAE {np.mean(folds):.4f}  folds {np.round(folds, 3).tolist()}")
cv = pd.DataFrame(rows)
cv.to_csv(OUT / "mlp_temporal_cv.csv", index=False)
best = cv.sort_values("cv_mae").iloc[0]
hid_b = eval_hid = tuple(json.loads(best.hidden.replace("(", "[").replace(",)", "]").replace(")", "]")))
al_b = float(best.alpha)
log(f"selected: hidden={hid_b}, alpha={al_b}, CV MAE {best.cv_mae:.4f}")

# M6 (Phase-2 hyper-parameters, no recency weighting) -- CV and test, for M9 and for reference
p2 = json.loads((OUT / "phase2_summary.json").read_text())
ch6 = p2["act3"]["chosen_params"]["M6"]


def m6():
    return HistGradientBoostingRegressor(learning_rate=ch6["lr"], max_leaf_nodes=ch6["leaves"], max_iter=400,
                                         l2_regularization=1.0, early_stopping=False, random_state=11)


cv_m6, cv_m9 = [], []
for fi, (fit, val) in enumerate(FOLDS):
    p6 = m6().fit(X_tr[fit], ytr[fit]).predict(X_tr[val])
    p8 = cvp[(str(hid_b), al_b, fi)]
    cv_m6.append(mae(ytr[val], p6))
    cv_m9.append(mae(ytr[val], (p6 + p8) / 2))
log(f"CV: M6 none {np.mean(cv_m6):.4f}, M8 {best.cv_mae:.4f}, M9 mean(M6,M8) {np.mean(cv_m9):.4f}")

# ---- test set (scored once)
single = [make_mlp(hid_b, al_b, s).fit(X_tr, ytr).predict(X_te) for s in SEEDS]
p8 = np.mean(single, axis=0)
p6 = m6().fit(X_tr, ytr).predict(X_te)
p9 = (p6 + p8) / 2
prim = pd.read_csv(OUT / "act3_primary_test_predictions.csv")
assert (prim.canonical_smiles.values == test.canonical_smiles.values).all()
pp = prim.primary_seed11.values
res = {"selected": {"hidden": list(hid_b), "alpha": al_b},
       "cv_mae": {"M8": float(best.cv_mae), "M6_none": float(np.mean(cv_m6)), "M9": float(np.mean(cv_m9))},
       "test_mae": {"M8": mae(yte, p8), "M9": mae(yte, p9), "M6_none": mae(yte, p6), "primary_M6_w2500": mae(yte, pp)},
       "test_rmse": {k: float(np.sqrt(np.mean((yte - v) ** 2))) for k, v in (("M8", p8), ("M9", p9), ("M6_none", p6))},
       "M8_single_network_test_mae": [mae(yte, p) for p in single]}
res["M8_single_mean"] = float(np.mean(res["M8_single_network_test_mae"]))
res["M8_single_sd"] = float(np.std(res["M8_single_network_test_mae"], ddof=1))
rng = np.random.default_rng(0)
idx = rng.integers(0, len(yte), size=(2000, len(yte)))
bs = {}
for name, p in (("M8", p8), ("M9", p9)):
    d = np.abs(yte - p) - np.abs(yte - pp)
    b = d[idx].mean(1)
    bs[f"{name}_minus_primary"] = {"point": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
res["bootstrap"] = bs
res["runtime_seconds"] = round(time.time() - T0, 1)
(OUT / "mlp_summary.json").write_text(json.dumps(res, indent=2))
pd.DataFrame({"canonical_smiles": test.canonical_smiles, "y_true": yte, "M8": p8, "M9": p9}).to_csv(OUT / "mlp_test_predictions.csv", index=False)
log(f"test MAE: {res['test_mae']}")

# ---- figure
fig, ax = plt.subplots(figsize=(6.2, 3.3))
t20 = pd.read_csv(OUT / "act3_all_configs_cv_vs_test.csv")
get = lambda f, c: float(t20[(t20.family == f) & (t20.scheme == "none")][c].iloc[0])  # noqa: E731
items = [("M3 RF 8d+FP", get("M3", "cv_mae"), get("M3", "test_mae"), "#54a24b"),
         ("M5 RF full+FP", get("M5", "cv_mae"), get("M5", "test_mae"), "#54a24b"),
         ("M8 MLP", res["cv_mae"]["M8"], res["test_mae"]["M8"], "#9467bd"),
         ("M6 GBDT", res["cv_mae"]["M6_none"], res["test_mae"]["M6_none"], "#f58518"),
         ("M9 mean(M6,M8)", res["cv_mae"]["M9"], res["test_mae"]["M9"], "#8c564b")]
for i, (n, c, t, col) in enumerate(items):
    ax.barh(i, t, color=col, alpha=0.8)
    ax.plot(c, i, "k|", ms=16, mew=2)
    ax.text(t + 0.005, i, f"{t:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([n for n, *_ in items], fontsize=9)
ax.set_xlabel("MAE (bar: official test; black tick: temporal CV)")
ax.set_xlim(0.4, 0.68)
ax.invert_yaxis()
fig.savefig(FIG / "fig11_mlp_comparison.png")
plt.close(fig)
log("done")
