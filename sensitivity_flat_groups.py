"""Post-hoc sensitivity analysis (NOT the primary protocol).

The primary identity key keeps stereochemistry, so stereoisomer twins can fall on both sides of the internal
random split. Here the internal split groups molecules by the NON-isomeric canonical SMILES instead, the
validation grid is repeated, and (if the selected parameters differ) the test set is re-scored with them.
Primary results are not replaced. Usage: python sensitivity_flat_groups.py   (after run_project.py)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rp", ROOT / "run_project.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

CFG, SPL, RES, LAB = rp.CFG, rp.SPL, rp.RES, rp.CFG["label"]
train = pd.read_csv(SPL / "train_full_clean.csv")
test = pd.read_csv(SPL / "test_clean.csv")
flat = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)  # noqa: E731
train["flat"] = train.canonical_smiles.map(flat)
test["flat"] = test.canonical_smiles.map(flat)

gss = GroupShuffleSplit(n_splits=1, test_size=CFG["split"]["test_size"], random_state=CFG["split"]["random_state"])
fi, vi = next(gss.split(train, groups=train.flat))
fit, val = train.iloc[fi].reset_index(drop=True), train.iloc[vi].reset_index(drop=True)
assert not (set(fit.flat) & set(val.flat))
_, Xf = rp.featurize(fit)
_, Xv = rp.featurize(val)
nd = len(rp.DESC_NAMES)
yf, yv = fit[LAB].values, val[LAB].values

out = {"n_fit": len(fit), "n_val": len(val),
       "test_molecules_with_flat_twin_in_train": int(test.flat.isin(set(train.flat)).sum()),
       "primary_val_molecules_with_flat_twin_in_fit": None}
prim_fit = pd.read_csv(SPL / "train_fit.csv").canonical_smiles.map(flat)
prim_val = pd.read_csv(SPL / "train_val.csv").canonical_smiles.map(flat)
out["primary_val_molecules_with_flat_twin_in_fit"] = int(prim_val.isin(set(prim_fit)).sum())
out["primary_n_val"] = int(len(prim_val))

grid = {"M0": rp.metrics(yv, rp.make_model("M0").fit(Xf[:, :nd], yf).predict(Xv[:, :nd]))["MAE"]}
ridge, rf = [], []
for a in CFG["ridge_alphas"]:
    m = rp.metrics(yv, rp.make_model("M1", alpha=a).fit(Xf[:, :nd], yf).predict(Xv[:, :nd]))
    ridge.append({"alpha": a, **m})
for leaf in CFG["rf"]["leaf_grid"]:
    m = rp.metrics(yv, rp.make_model("M2", CFG["rf"]["val_seed"], leaf=leaf).fit(Xf[:, :nd], yf).predict(Xv[:, :nd]))
    rf.append({"leaf": leaf, **m})
best_a, best_leaf = rp.pick(ridge, "alpha")["alpha"], rp.pick(rf, "leaf")["leaf"]
m3 = rp.metrics(yv, rp.make_model("M3", CFG["rf"]["val_seed"], leaf=best_leaf).fit(Xf, yf).predict(Xv))
out.update({"val_M0_MAE": grid["M0"], "val_ridge": ridge, "val_rf_M2": rf, "val_M3_at_best_leaf": m3,
            "selected": {"alpha": best_a, "leaf": best_leaf}})

# re-score the test set with these parameters (seed 11 for the single-seed comparison, plus 3-seed means)
_, Xtr = rp.featurize(train)
_, Xte = rp.featurize(test)
ytr, yte = train[LAB].values, test[LAB].values
res = {"M1": rp.metrics(yte, rp.make_model("M1", alpha=best_a).fit(Xtr[:, :nd], ytr).predict(Xte[:, :nd]))["MAE"]}
for mid, cols in (("M2", slice(0, nd)), ("M3", slice(None))):
    maes = [rp.metrics(yte, rp.make_model(mid, s, leaf=best_leaf).fit(Xtr[:, cols], ytr).predict(Xte[:, cols]))["MAE"]
            for s in CFG["final_seeds"]]
    res[mid] = {"per_seed": maes, "mean": float(np.mean(maes))}
out["test_MAE_with_selected_params"] = res
(RES / "sensitivity_flat_group_split.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
