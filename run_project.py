"""AIDD8122 project: lightweight LogD models on the OpenADMET-ExpansionRx temporal split.

Usage:  python run_project.py            (download -> audit -> validate -> final test run)
Outputs: data/, splits/, results/ (figures are made by make_figures.py).
"""
import hashlib
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
RAW, SPL, RES = ROOT / "data" / "raw", ROOT / "splits", ROOT / "results"
DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
              "NumRotatableBonds", "NumRings", "FractionCSP3"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- download
def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    info = {"downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for key in ("train_url", "test_url"):
        url = CFG[key]
        dest = RAW / url.rsplit("/", 1)[1]
        if not dest.exists():
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        digest = sha256(dest)
        expected = CFG["expected_sha256"][dest.name]
        info["files"][dest.name] = {"sha256": digest, "bytes": dest.stat().st_size,
                                    "matches_expected": digest == expected}
        if digest != expected:
            raise SystemExit(f"SHA256 mismatch for {dest.name}: {digest}; re-audit before freezing.")
    try:  # record the Hugging Face revision of the dataset (best effort)
        api = "https://huggingface.co/api/datasets/" + CFG["dataset"]
        meta = json.loads(urllib.request.urlopen(api, timeout=30).read())
        info["hf_revision_sha"] = meta.get("sha")
        info["hf_last_modified"] = meta.get("lastModified")
    except Exception as e:  # noqa: BLE001
        info["hf_revision_sha"] = f"unavailable ({e})"
    (RES / "data_provenance.json").write_text(json.dumps(info, indent=2))
    return info


# -------------------------------------------------------------------- audit
def canon(smi):
    m = Chem.MolFromSmiles(smi.strip())
    return None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)


def clean_one(path, label, smi_col, name_col):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    y = pd.to_numeric(raw[label], errors="coerce")
    ok = np.isfinite(y)
    df = raw.loc[ok].copy()
    df[label] = y[ok]
    n_valid = len(df)
    df["canonical_smiles"] = df[smi_col].astype(str).map(canon)
    n_fail = int(df["canonical_smiles"].isna().sum())
    df = df.dropna(subset=["canonical_smiles"])
    g = df.groupby("canonical_smiles")
    n_dups = len(df) - g.ngroups
    merged = g.agg(**{label: (label, "median"), "n_records": (label, "size"),
                      name_col: (name_col, "first"), "label_min": (label, "min"),
                      "label_max": (label, "max")}).reset_index()
    stats = {"raw_records": n_raw, "valid_label": n_valid,
             "excluded_missing_label": n_raw - n_valid, "rdkit_parse_failures": n_fail,
             "extra_records_merged_by_canonical_smiles": int(n_dups),
             "unique_molecules": len(merged)}
    return merged, stats


def audit():
    lab, smi, nm = CFG["label"], CFG["smiles_col"], CFG["name_col"]
    train, s_tr = clean_one(RAW / "expansion_data_train.csv", lab, smi, nm)
    test, s_te = clean_one(RAW / "expansion_data_test.csv", lab, smi, nm)
    overlap = sorted(set(train.canonical_smiles) & set(test.canonical_smiles))
    s_tr["removed_for_overlap_with_test"] = len(overlap)
    if overlap:  # remove from training side, keep test side (plan 5.1 step 7)
        train = train[~train.canonical_smiles.isin(overlap)].reset_index(drop=True)
        pd.Series(overlap, name="canonical_smiles").to_csv(SPL / "removed_overlap.csv", index=False)
    s_tr["final_train_pool"] = len(train)
    s_te["final_test"] = len(test)

    gss = GroupShuffleSplit(n_splits=1, test_size=CFG["split"]["test_size"],
                            random_state=CFG["split"]["random_state"])
    fit_idx, val_idx = next(gss.split(train, groups=train.canonical_smiles))
    fit, val = train.iloc[fit_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
    for d, tag in ((fit, "train_fit"), (val, "train_val")):
        d.to_csv(SPL / f"{tag}.csv", index=False)
    train.to_csv(SPL / "train_full_clean.csv", index=False)
    test.to_csv(SPL / "test_clean.csv", index=False)

    sets = {"fit": set(fit.canonical_smiles), "val": set(val.canonical_smiles),
            "test": set(test.canonical_smiles)}
    assert not (sets["fit"] & sets["val"]) and not (sets["fit"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "identity leakage between splits"
    audit_d = {"train_file": s_tr, "test_file": s_te,
               "split": {"fit": len(fit), "val": len(val), "test": len(test)},
               "label_stats": {k: v[lab].describe().round(3).to_dict()
                               for k, v in (("train_full", train), ("fit", fit),
                                            ("val", val), ("test", test))}}
    (RES / "audit.json").write_text(json.dumps(audit_d, indent=2))
    log(f"audit: fit={len(fit)} val={len(val)} test={len(test)} overlap_removed={len(overlap)}")
    return fit, val, train, test


# ----------------------------------------------------------------- features
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=CFG["morgan"]["radius"], fpSize=CFG["morgan"]["fpSize"])


def featurize(df):
    desc, fps = [], []
    for s in df.canonical_smiles:
        m = Chem.MolFromSmiles(s)
        desc.append([Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                     Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                     Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                     rdMolDescriptors.CalcFractionCSP3(m)])
        fps.append(_MORGAN.GetFingerprintAsNumPy(m))
    D = np.asarray(desc, dtype=float)
    F = np.asarray(fps, dtype=float)
    assert np.isfinite(D).all() and np.isfinite(F).all(), "non-finite features"
    return D, np.hstack([D, F])


# ------------------------------------------------------------------- models
def make_model(mid, seed=None, alpha=None, leaf=None):
    rf = CFG["rf"]
    if mid == "M0":
        return DummyRegressor(strategy="median")
    if mid == "M1":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return RandomForestRegressor(n_estimators=rf["n_estimators"], max_features=rf["max_features"],
                                 max_depth=rf["max_depth"], min_samples_leaf=leaf,
                                 n_jobs=rf["n_jobs"], random_state=seed)


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p))}


def pick(rows, key_param):
    """Lowest validation MAE; ties -> larger parameter value (plan 5.4)."""
    return sorted(rows, key=lambda r: (round(r["MAE"], 12), -r[key_param]))[0]


def validate(Xf, Xv, yf, yv):
    rows = []
    m0 = make_model("M0").fit(Xf["D"], yf)
    rows.append({"model": "M0", "param": "-", **metrics(yv, m0.predict(Xv["D"]))})
    ridge = []
    for a in CFG["ridge_alphas"]:
        r = {"model": "M1", "alpha": a,
             **metrics(yv, make_model("M1", alpha=a).fit(Xf["D"], yf).predict(Xv["D"]))}
        ridge.append(r)
        rows.append({**r, "param": f"alpha={a}"})
    best_a = pick(ridge, "alpha")["alpha"]
    rf = []
    for leaf in CFG["rf"]["leaf_grid"]:
        r = {"model": "M2", "leaf": leaf, **metrics(
            yv, make_model("M2", CFG["rf"]["val_seed"], leaf=leaf).fit(Xf["D"], yf).predict(Xv["D"]))}
        rf.append(r)
        rows.append({**r, "param": f"min_samples_leaf={leaf}"})
    best_leaf = pick(rf, "leaf")["leaf"]
    m3 = make_model("M3", CFG["rf"]["val_seed"], leaf=best_leaf).fit(Xf["DF"], yf)
    rows.append({"model": "M3", "param": f"min_samples_leaf={best_leaf}",
                 **metrics(yv, m3.predict(Xv["DF"]))})
    pd.DataFrame(rows)[["model", "param", "MAE", "RMSE", "R2"]].to_csv(
        RES / "metrics_validation.csv", index=False)
    params = {"M1_alpha": best_a, "M2_leaf": best_leaf, "M3_leaf": best_leaf,
              "selection_rule": "lowest validation MAE; ties -> larger value"}
    (RES / "frozen_params.json").write_text(json.dumps(params, indent=2))
    log(f"frozen params: {params}")
    return params


def run_final(train, test, params):
    lab = CFG["label"]
    _, Xtr_DF = featurize(train)
    Xtr_D = Xtr_DF[:, :len(DESC_NAMES)]
    Xte_D, Xte_DF = featurize(test)
    ytr, yte = train[lab].values, test[lab].values
    assert len(Xte_DF) == len(yte) == len(test)
    preds, mrows, timings = [], [], {}

    def run(mid, seed, model, Xa, Xb):
        t = time.time()
        p = model.fit(Xa, ytr).predict(Xb)
        timings[f"{mid}_seed{seed}"] = round(time.time() - t, 2)
        mrows.append({"model": mid, "seed": seed, **metrics(yte, p)})
        preds.append(pd.DataFrame({"model": mid, "seed": seed, "canonical_smiles": test.canonical_smiles,
                                   "name": test[CFG["name_col"]], "y_true": yte, "y_pred": p}))
        log(f"{mid} seed={seed}: {mrows[-1]}")
        return p

    run("M0", "det", make_model("M0"), Xtr_D, Xte_D)
    run("M1", "det", make_model("M1", alpha=params["M1_alpha"]), Xtr_D, Xte_D)
    for s in CFG["final_seeds"]:
        run("M2", s, make_model("M2", s, leaf=params["M2_leaf"]), Xtr_D, Xte_D)
        run("M3", s, make_model("M3", s, leaf=params["M3_leaf"]), Xtr_DF, Xte_DF)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(RES / "predictions_test.csv", index=False)
    M = pd.DataFrame(mrows)
    M.to_csv(RES / "metrics_test_per_run.csv", index=False)
    # summary: mean and sample std (ddof=1) over seeds; deterministic runs have std = NaN
    S = M.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S.to_csv(RES / "metrics_test_summary.csv")
    a = M[M.model == "M2"].set_index("seed").MAE
    b = M[M.model == "M3"].set_index("seed").MAE
    diff = (b - a).rename("MAE_M3_minus_M2").reset_index()
    diff.to_csv(RES / "paired_diff_M3_minus_M2.csv", index=False)
    return P, M, timings, (Xtr_D, Xtr_DF, Xte_D, Xte_DF, ytr, yte)


# -------------------------------------------------------------- verification
def verify(P, M, params, data):
    Xtr_D, Xtr_DF, Xte_D, Xte_DF, ytr, yte = data
    checks = {}
    Pd = pd.read_csv(RES / "predictions_test.csv")
    Pd["seed"] = Pd["seed"].astype(str)
    maxdiff = 0.0
    for _, r in M.iterrows():
        q = Pd[(Pd.model == r.model) & (Pd.seed == str(r.seed))]
        m = metrics(q.y_true.values, q.y_pred.values)
        maxdiff = max(maxdiff, *(abs(m[k] - r[k]) for k in m))
    checks["metrics_recompute_max_abs_diff"] = maxdiff
    orders = {tuple(g.canonical_smiles) for _, g in Pd.groupby(["model", "seed"])}
    checks["distinct_test_id_orders_across_runs"] = len(orders)
    checks["n_test"] = int(len(yte))
    rep = {}
    for mid, Xa, Xb, mdl in (("M1", Xtr_D, Xte_D, make_model("M1", alpha=params["M1_alpha"])),
                             ("M2", Xtr_D, Xte_D, make_model("M2", 11, leaf=params["M2_leaf"])),
                             ("M3", Xtr_DF, Xte_DF, make_model("M3", 11, leaf=params["M3_leaf"]))):
        p2 = mdl.fit(Xa, ytr).predict(Xb)
        seed = "det" if mid == "M1" else "11"
        p1 = Pd[(Pd.model == mid) & (Pd.seed == seed)].y_pred.values
        rep[mid] = float(np.max(np.abs(p1 - p2)))
    checks["rerun_max_abs_pred_diff"] = rep
    (RES / "verification.json").write_text(json.dumps(checks, indent=2))
    assert maxdiff < 1e-9 and len(orders) == 1 and all(v < 1e-9 for v in rep.values()), checks
    log(f"verification passed: {checks}")


def main():
    for d in (RAW, SPL, RES):
        d.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    download()
    fit, val, train, test = audit()
    lab = CFG["label"]
    _, Xf_DF = featurize(fit)
    _, Xv_DF = featurize(val)
    nd = len(DESC_NAMES)
    Xf = {"D": Xf_DF[:, :nd], "DF": Xf_DF}
    Xv = {"D": Xv_DF[:, :nd], "DF": Xv_DF}
    params = validate(Xf, Xv, fit[lab].values, val[lab].values)
    P, M, timings, data = run_final(train, test, params)
    verify(P, M, params, data)
    env = {"python": sys.version, "platform": platform.platform(), "rdkit": rdBase.rdkitVersion,
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "command": " ".join(sys.argv), "started_utc": start,
           "finished_utc": datetime.now(timezone.utc).isoformat(),
           "total_seconds": round(time.time() - T0, 1), "fit_seconds": timings,
           "descriptors": DESC_NAMES, "config": CFG}
    (RES / "environment_and_run.json").write_text(json.dumps(env, indent=2))
    log("done")


if __name__ == "__main__":
    main()
