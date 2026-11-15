# Phase 2 protocol (written before the Phase-2 code was run)

Story: *Chasing a moving target - how LogD models decay across a lead-optimisation campaign, and whether we can
tell when not to trust them.*

## Status of the test set
Phase 1 (M0-M3, failure analysis, similarity, charged-molecule analysis) already looked at the official test set.
Phase 2 is therefore **partly informed by test inspection**. To limit snooping: every choice below is made on the
training pool only (temporal cross-validation), and the test set is scored once for a pre-listed set of configurations.
Everything is reported, including configurations that do not help.

## Time proxy
`Molecule Name` numbers (E-xxxxx) are used as a chronological proxy. Evidence: train 10647-20100 and test 20101-27239
are contiguous with no overlap, so the official split is a cut on this number. It is NOT confirmed by the data provider
that numbers equal synthesis/measurement order; the report calls it a proxy.

## Act 1 - the campaign drifts
- LogD vs ID, rolling median; mean/SD per ID quintile (train) and per tertile (test).
- Standardised mean difference of the 8 descriptors, train vs test.
- Adversarial validation: random forest separating train from test (5-fold CV AUROC) on (a) 8 descriptors,
  (b) full RDKit descriptors; importances of (b). IDs are never used as features.

## Act 2 - evaluation protocols
- Rolling-origin temporal CV on the training pool: order by ID, 6 equal blocks, folds fit on blocks 1..k, validate on
  block k+1 for k = 2..5 (4 folds).
- Compare M0-M3 (Phase-1 frozen parameters) under: random 80/20 (Phase 1), flat-structure-grouped random (sensitivity),
  temporal CV (mean of folds), official test. Quantity: MAE, relative improvement over M0, optimism ratio
  MAE_protocol / MAE_test. Also median nearest-neighbour similarity of each validation block to its fit set.

## Act 3 - stronger models and recency (all selected on temporal CV only)
Families (features -> model):
- M3  RF, 8 descriptors + Morgan bits (Phase-1 parameters, not retuned)
- M4  RF, full RDKit descriptors. Grid: min_samples_leaf {1,2,5} x max_features {sqrt, 0.33}
- M5  RF, full descriptors + Morgan bits (1024). Same grid
- M6  HistGradientBoosting, full descriptors + Morgan counts (1024). Grid: learning_rate {0.03, 0.1} x max_leaf_nodes {15, 31},
      max_iter 400, l2 1.0, no early stopping
- M7  mean of the selected M5 and M6 predictions
Recency schemes: none; exponential weights with half-life 1000 or 2500 records (age counted in records within the fit
set, ordered by ID); last 50% of the fit set only. Per family, parameters are chosen with scheme "none" by mean temporal-CV MAE,
then reused for the other schemes.
Primary Phase-2 model = the (family, scheme) with the lowest temporal-CV MAE among all 20. It is scored on the test set with
model seeds 11, 22, 33. All 20 configurations are also scored once (seed 11) as secondary results, with a multiplicity warning.
Uncertainty about primary vs M3 on the test set: paired bootstrap over molecules (2000 resamples, seed 0).

## Act 4 - knowing when not to trust the model
Signals (computed with the Phase-1 M3 forest, seed 11, and Morgan-2048 Tanimoto): (i) tree-prediction SD,
(ii) 1 - nearest-neighbour Tanimoto to the training pool, (iii) rule flag net formal charge != 0.
Evaluation: AUROC for identifying absolute error >= 1.0 LogD (fixed in advance); risk-coverage curves (MAE after rejecting the
top q% of each score, q = 0..30); where the charged molecules rank. The same AUROCs are computed on out-of-fold
temporal-CV predictions of the training pool (no test data). No threshold is derived from test results. Errors of both M3
and the primary model are examined.
Limitation known in advance: only 3 training molecules are charged, so the charged flag has no training-side evidence.

## Not done by design
No external data, pretrained models, GNNs, multitask learning, other endpoints, or hand-written charge corrections
derived from test observations.
