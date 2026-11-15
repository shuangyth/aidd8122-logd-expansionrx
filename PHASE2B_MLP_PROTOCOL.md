# Phase 2b protocol: a small neural network (written before the code was run)

Added after Phase 2 had already been run and its results seen (primary model = M6 gradient boosting, half-life-2,500
weighting; test MAE 0.548). The purpose is to answer "does a small neural network help?" with the same protocol.
It is a **secondary comparison and does not replace the primary model**, which was fixed by the Phase-2 rule.

## Models
- M8: scikit-learn `MLPRegressor` on the M6 feature set (full RDKit descriptors + Morgan counts, 1,024).
  Preprocessing, fitted on the fit data of each fold only: descriptors standardised then clipped to [-5, 5];
  Morgan counts log1p-transformed.
  Fixed: Adam, learning rate 1e-3, batch size 128, max_iter 300, early stopping on a 10% internal validation split
  (patience 20), activation ReLU.
  Grid (temporal CV): hidden layers {(256,), (256, 128)} x L2 alpha {1e-3, 1e-1}  (4 configurations).
  Each configuration is an average of three networks (seeds 11, 22, 33).
- M9: mean of the M6 prediction (Phase-2 hyper-parameters, no recency weighting) and the M8 prediction.

## Protocol
Same four temporal-CV folds as Phase 2 (training pool ordered by molecule number, six blocks). M8 is selected by mean
temporal-CV MAE only. M8 and M9 are scored once on the test set (single-network MAEs of the three seeds are reported too, as
spread). No recency weighting is applied to M8. Paired bootstrap over test molecules (2,000 resamples, seed 0) for
M8 - primary and M9 - primary.

## Not done
No architecture search beyond the 4-configuration grid, no graph neural networks, no pretrained models, no external data.
