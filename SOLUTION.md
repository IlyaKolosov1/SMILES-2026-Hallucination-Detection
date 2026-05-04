# SOLUTION.md

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 solution.py
```

Artifacts:
- `results.json`
- `predictions.csv`

Current run constraints:
- `solution.py` is configured to require GPU (`cuda` or `mps`) and refuses CPU execution.
- Random seed is fixed (`42`) for reproducibility.

## Final Pipeline (Current)

- `aggregation.py`: fixed tail pooling from last layers + fixed geometric features.
- `probe.py`: fixed `StandardScaler -> PCA -> LogisticRegression` pipeline.
- `splitting.py`: stratified k-fold split with validation split per fold.
- Threshold tuning: decision threshold is selected by **validation accuracy**.

## Experiment Log

### Hypothesis 1

Hypothesis: tune decision threshold by **accuracy** (instead of F1) in `fit_hyperparameters`.

| Run | Change | Train Acc | Val Acc | Test Acc | Test F1 | Test AUROC | Extract time |
|---|---|---:|---:|---:|---:|---:|---:|
| Before H1 | Threshold by F1 | 70.06% | 70.19% | 70.19% | 82.49% | 74.28% | 86.2s |
| After H1 | Threshold by accuracy | 91.68% | 72.12% | 73.08% | 82.05% | 73.71% | 83.9s |
| Delta | After - Before | +21.62 pp | +1.93 pp | +2.89 pp | -0.44 pp | -0.57 pp | -2.3s |

Conclusion:
- Successful for target metric (accuracy): `70.19% -> 73.08%`.
- Overfitting remained high (`train >> val/test`).

### Hypothesis 2

Hypothesis: replace MLP-style probe with LogisticRegression.

| Configuration | Test Accuracy | Test AUROC | Conclusion |
|---|---:|---:|---|
| Baseline (majority class) | 70.19% | - | Reference |
| Hypothesis 1 (threshold by accuracy) | 73.08% | 73.71% | Improvement |
| Hypothesis 2 (LogReg, early version) | 70.19% | 74.15% | No accuracy gain |

Conclusion:
- Accuracy regressed to baseline level in that setup.
- AUROC stayed competitive, but primary metric is accuracy.

### Hypothesis 3

Hypothesis: regularized MLP to reduce overfitting while keeping accuracy.

| Configuration | Train Acc | Val Acc | Test Acc | Test AUROC | Conclusion |
|---|---:|---:|---:|---:|---|
| Baseline | 70.19% | 70.19% | 70.19% | - | Reference |
| Hypothesis 1 (threshold by accuracy) | 91.68% | 72.12% | 73.08% | 73.71% | Better test accuracy, strong overfitting |
| Hypothesis 2 (LogReg, early version) | 70.06% | 70.19% | 70.19% | 74.15% | No accuracy gain |
| Hypothesis 3 (regularized MLP) | 81.08% | 74.04% | 73.08% | 70.30% | Same test accuracy, better stability |

Conclusion:
- Same test accuracy as H1, lower overfitting, stronger val accuracy.

### Current Run (Final repository state)

Latest run (fixed hardcoded pipeline):

| Metric | Value |
|---|---:|
| Baseline Accuracy | 70.10% |
| Probe Train Accuracy | 76.08% |
| Probe Val Accuracy | 77.35% |
| Probe Test Accuracy | 73.73% |
| Probe Test F1 | 82.79% |
| Probe Test AUROC | 74.12% |
| Extract time | 85.4s |

Interpretation:
- Accuracy improvement over baseline: `+3.62 pp`.
- Overfitting gap is moderate and smaller than in H1 peak (`91.68% train`).
- This is the best documented accuracy in the current repository state.

## Imbalance Handling: Done vs Not Done

Done:
- Stratified split (`stratify=y`) in `splitting.py`.
- Threshold tuning by validation accuracy (aligned with target metric).

Not done yet:
- `pos_weight` sweep (e.g. `0.25`, `0.43`, `0.6`, `1.0`) for BCE-based probes.
- Threshold tie-break by minority recall.
- Explicit resampling (oversample minority / undersample majority).

## What Was Finally Chosen

Chosen for final repository:
- Fixed feature extraction + geometric features.
- Fixed PCA+LogReg probe.
- Accuracy-oriented threshold tuning.
- GPU-only execution guard for safe runtime.

Reason:
- Best trade-off between simplicity, reproducibility, and test accuracy in current runs.

## Next Experiments (Priority for Accuracy)

1. `pos_weight` sweep + accuracy threshold tuning (minimal code changes, high expected value).
2. Layer sweep in `aggregation.py` (last layer vs last 2/4 layers).
3. INSIDE-inspired feature clipping + covariance/eigen features in `aggregation.py`.
4. Lightweight self-consistency risk score (if compute budget allows).

## Related Work Used for Planning

- INSIDE (ICLR 2024): internal-state features, EigenScore, clipping
  https://arxiv.org/abs/2402.03744
- MIND (Findings ACL 2024): unsupervised real-time hallucination detection from internal states
  https://aclanthology.org/2024.findings-acl.854/
- SelfCheckGPT (EMNLP 2023): self-consistency for hallucination detection
  https://aclanthology.org/2023.emnlp-main.557/
- Semantic Entropy (Nature 2024): semantic-level uncertainty
  https://www.nature.com/articles/s41586-024-07421-0
- Neural Probe-Based Hallucination Detection (arXiv 2025)
  https://arxiv.org/abs/2512.20949

## Short Final Conclusion

- Main successful idea: optimize threshold for accuracy.
- Main risk observed: overfitting in higher-capacity probe variants.
- Current final state: simple fixed pipeline with `73.73%` test accuracy and `+3.62 pp` over baseline.
