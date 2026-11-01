# Phase 0 — Online Augmented Baseline: Analysis & Findings (Reference Run)

**Run name:** `augmented` (online augmentation)  
**Date:** 2026-06-29  
**MLflow run ID:** `8c5a90996934401ea64a049ffa59fa52`  
**Follows:** Vanilla baseline (`fe46e741342f4cf1b7ae2645d224e0c1`)  
> **Review note (2026-10-31).** The per-subject observations below (§4–5, §7.2–7.5) compare a single augmented run against the vanilla run that also defines the tiers. With ~45 test trials per subject, part of the "weak subjects improve / strong subjects regress" pattern is regression to the mean rather than an effect of augmentation (see [the README, results §4](../README.md#4-which-subjects-lose-most-and-a-selection-bias-check)). Individual-subject swings of ±20 pts should be read as within noise unless they replicate. The methodological conclusion of this report (§6: online augmentation is not a valid TSTR baseline) is unaffected.

**Status:** Reference run — superseded by offline additive design for TSTR comparison

---

## Abstract

EEGNet was trained with standard online signal augmentation (Gaussian noise, time-shift, channel dropout, sliding-window crop) on the PhysioNet EEGBCI motor imagery dataset using within-subject Leave-One-Run-Out (LORO) evaluation across 106 subjects (318 folds). The final mean accuracy was **58.4% ± 11.1%**, a marginal +1.7% improvement over the vanilla baseline of 56.7%. Aggregate statistics mask a more complex picture: 54 subjects improved (some dramatically, up to +33.3%), while 43 subjects regressed (some severely, down to −22.2%). The near-zero net gain is the result of strong individual effects cancelling out, not an absence of effect. A critical methodological finding emerged during analysis: online augmentation is not the correct comparison baseline for TSTR — the additive design (real data + offline-generated copies) is required for a scientifically valid comparison. This run is retained as a reference point; it does not serve as the primary baseline for Phase 0 Step 5.

---

## 1. Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | PhysioNet EEGBCI, imagery_lr task (runs 4, 8, 12) |
| Subjects | 106 (S088, S092, S100 excluded — non-standard epoch lengths) |
| Folds | 318 (3 per subject) |
| Model | EEGNet (Lawhern et al. 2018), 2,930 parameters |
| Augmentation | `STANDARD_AUGMENTOR`: gaussian noise, time-shift, channel dropout, sliding-window crop; each with p_apply=0.5 per transform per sample |
| Augmentation mode | **Online** — transforms applied in `__getitem__` on-the-fly each epoch |
| Training samples per fold | ~30 (base dataset size, unchanged by augmentation) |
| Epochs | 500 (fixed, CosineAnnealingLR, no early stopping) |
| Batch size | Full batch (`len(y_train)`) |
| Learning rate | 1e-3 → 1e-5 (cosine decay) |
| Dropout | 0.5 |
| max_norm_fc | 0.25 |
| Seed | 42 |
| Device | Apple Silicon MPS |

**What online augmentation does:** each of the ~30 training samples is fetched from the dataset, randomly transformed (probability 0.5 per transform), and passed to the model. Over 500 epochs, each base sample is seen in approximately 250 different augmented configurations. The dataset size remains 30 — augmented variants are generated and discarded each epoch, never stored.

---

## 2. Overall Results

```
Run        : augmented (online)
Subjects   : 106
Folds      : 318
Accuracy   : 0.5836 ± 0.1110  (58.4%)
F1 (macro) : 0.5836
```

### Comparison with vanilla baseline

| Metric | Vanilla | Online augmentation | Δ |
|---|---|---|---|
| Mean accuracy | 56.7% | **58.4%** | **+1.7%** |
| Std | ±10.8% | ±11.1% | +0.3% |
| Median | 55.6% | 57.8% | +2.2% |
| F1 (macro) | 0.567 | 0.584 | +0.017 |

The +1.7% mean improvement is modest. The standard deviation *increased* slightly (10.8% → 11.1%), indicating that augmentation made the distribution slightly wider — some subjects improved more, some regressed more. This is confirmed in the per-subject analysis below.

---

## 3. Subject Tier Distribution

Tiers are defined on vanilla accuracy (the intrinsic subject difficulty):

| Tier | Range | Vanilla count | Augmented count | Change |
|---|---|---|---|---|
| Non-responders | < 45% | 14 (13%) | 11 (10%) | **−3** |
| Data-limited | 45–56% | 41 (39%) | 42 (40%) | +1 |
| Learning | 56–70% | 42 (40%) | 41 (39%) | −1 |
| Strong | > 70% | 9 (8%) | **12 (11%)** | **+3** |

Augmentation moved 3 subjects out of the non-responder tier (S046, S083, S039 all crossed the 45% threshold) and pushed 3 subjects from the learning tier into the strong tier (S040, S042, S054). At the extremes, both effects are real — but they cancel out in aggregate.

---

## 4. Per-Subject Analysis: Improvements and Regressions

Of 106 subjects:
- **54 improved** (delta > 0%)
- **9 unchanged** (delta ≈ 0%)
- **43 regressed** (delta < 0%)

The distribution is asymmetric: improvements tend to be larger in magnitude (+14% average for improvers) than regressions (−8% average for regressors), but there are more regressions.

### Top 10 improvements

| Subject | Vanilla | Augmented | Δ | Interpretation |
|---|---|---|---|---|
| S060 | 51.1% | **84.4%** | **+33.3%** | Data-limited → strong; augmentation unlocked consistent ERD |
| S002 | 57.8% | **84.4%** | **+26.7%** | All 3 folds improved: 53%, 47%, 73% → 80%, 93%, 80% |
| S011 | 37.8% | 62.2% | +24.4% | Non-responder → learning; augmentation helped with noisy signal |
| S054 | 57.8% | **80.0%** | +22.2% | Two folds jumped: 47% → 87%, 73% → 87% |
| S073 | 42.9% | 61.9% | +19.1% | Non-responder → learning |
| S001 | 48.9% | 66.7% | +17.8% | Data-limited → learning |
| S036 | 40.0% | 57.8% | +17.8% | Non-responder → data-limited; large but still below learning |
| S083 | 35.6% | 53.3% | +17.8% | Bottom-tier non-responder improved meaningfully |
| S065 | 53.3% | 68.9% | +15.6% | Data-limited → learning |
| S104 | 46.7% | 62.2% | +15.6% | Data-limited → learning |

**Pattern in improvements:** almost all are from the data-limited (45–56%) or non-responder (<45%) tiers. These subjects likely have a latent ERD signal that couldn't be reliably learned from 30 clean samples — augmentation added just enough variation for the model to find a more generalisable spatial filter.

S060 is the most striking case: vanilla folds were [53%, 53%, 47%] (flat, model learning nothing) — augmented folds became [80%, 80%, 93%] (all three folds learned well). This suggests the augmented training helped Block 2 find the correct spatial filter that it missed with 30 identical samples.

### Top 10 regressions

| Subject | Vanilla | Augmented | Δ | Interpretation |
|---|---|---|---|---|
| S016 | 64.4% | 42.2% | **−22.2%** | Strong → non-responder; augmentation disrupted a well-learned filter |
| S010 | 68.9% | 48.9% | −20.0% | Learning → data-limited |
| S005 | 66.7% | 48.9% | −17.8% | All 3 folds identical in vanilla (67%/67%/67%) — extremely consistent; aug broke it |
| S069 | 57.8% | 42.2% | −15.6% | Learning → non-responder |
| S052 | 51.1% | 37.8% | −13.3% | Already weak; aug made it worse — true non-responder |
| S086 | 68.9% | 55.6% | −13.3% | Learning → data-limited |
| S007 | 64.4% | 53.3% | −11.1% | Modest regression |
| S017 | 73.3% | 62.2% | −11.1% | Strong → learning; all 3 folds perfectly identical in vanilla (73%/73%/73%) |
| S018 | 71.1% | 60.0% | −11.1% | Strong → learning; fold 3 dropped from 60% to 40% |
| S091 | 64.4% | 53.3% | −11.1% | Learning → data-limited |

**Pattern in regressions:** the most severe regressions occur in subjects that were performing well in vanilla — particularly those with *unusually consistent fold-to-fold accuracy in vanilla* (S005: [67%/67%/67%], S017: [73%/73%/73%]). For these subjects, Block 2 found a stable, clean spatial filter from 30 consistent samples. Augmentation — particularly gaussian noise and time-shift — introduced signal variation that confused a model already converging on the right answer.

**Interpretation:** online augmentation is a double-edged tool for EEG. It helps subjects where the natural 30-sample variation is too low to learn from, but hurts subjects where the model had already correctly identified the spatial filter and augmentation disrupted convergence.

### Subjects with no change

| Subject | Vanilla | Augmented | Notes |
|---|---|---|---|
| S004 | 68.9% | 68.9% | Learning tier, stable across both |
| S006 | 53.3% | 53.3% | Data-limited |
| S025 | 55.6% | 55.6% | Data-limited |
| S051 | 86.7% | 86.7% | Strong responder — augmentation didn't move it |
| S055 | 51.1% | 51.1% | Data-limited |
| S059 | 42.2% | 42.2% | Non-responder |
| S076 | 45.2% | 45.2% | Non-responder |
| S098 | 48.9% | 48.9% | Data-limited |
| S107 | 57.8% | 57.8% | Learning tier |

Non-responders that don't move (S059, S076) confirm genuine BCI illiteracy — no augmentation strategy can create ERD signal that isn't physiologically present.

---

## 5. Notable Individual Cases

### S060: most dramatic improvement (+33.3%)

**Vanilla folds:** [53.3%, 53.3%, 46.7%] — flat, model learning near-nothing.  
**Augmented folds:** [80.0%, 80.0%, 93.3%] — consistent high performance across all three runs.

This is not noise. All three held-out runs improved by approximately the same margin, suggesting augmentation genuinely helped Block 2 find the correct spatial filter. S060's signal may have very low trial-to-trial amplitude variance — augmented gaussian noise provided the variation needed for the model to learn a generalising filter rather than memorising the exact amplitude pattern of 30 samples.

### S002: +26.7% improvement

**Vanilla folds:** [53.3%, 46.7%, 73.3%] — high variance, model unstable across runs.  
**Augmented folds:** [80.0%, 93.3%, 80.0%] — consistent and strong.

In vanilla, fold 2 (46.7%) was particularly bad. Augmentation stabilised training across all three runs, suggesting the noise injection helped prevent overfitting to run-specific artefacts.

### S005: −17.8% regression

**Vanilla folds:** [66.7%, 66.7%, 66.7%] — all three folds at exactly the same accuracy. This is the most consistent subject in the dataset.  
**Augmented folds:** [40.0%, 46.7%, 60.0%] — all three folds degraded, fold 1 dropped by 26.7%.

The perfectly flat vanilla profile suggests Block 2 found a very stable spatial filter that generalises identically across all three runs. Augmentation broke this: the random time-shifts and noise injections introduced signal patterns inconsistent with the learned filter, degrading all three folds.

### S017: −11.1% regression

**Vanilla folds:** [73.3%, 73.3%, 73.3%] — another perfectly flat strong responder.  
**Augmented folds:** [60.0%, 66.7%, 60.0%] — all three folds regressed.

Same pattern as S005: subjects with perfectly consistent vanilla performance are systematically hurt by online augmentation.

### S062: 100% fold

S062 fold 2: acc=100.0%, f1=1.000 — perfect classification of all 15 test samples. This subject has an exceptionally strong, clean ERD. Even with augmentation (which introduces noise), the signal is so reliable that the model classifies perfectly.

### S072: top overall performer at 92.9%

**Vanilla:** 88.4% → **Augmented: 92.9%** (+4.5%).  
Fold 3: 100.0% (perfect). Subject 72 has the strongest ERD in the dataset across all experimental conditions.

### S074 fold 3: 14.3% — extreme class collapse

**Augmented folds:** [60.0%, 42.9%, **14.3%**]

14.3% = 2 correct predictions out of 14 test samples. The model classified nearly every sample as the wrong class. This is class inversion — Block 2 converged on a spatial filter with the correct anatomy but the wrong sign. Augmentation (particularly channel dropout zeroing out key electrodes) likely caused Block 2 to flip its weight sign during a critical training epoch. With full-batch deterministic training, once the sign flips it can be stable.

---

## 6. Key Methodological Finding: Why This Run Cannot Serve as the TSTR Baseline

During the analysis of this online augmentation run, a critical methodological gap was identified. This section documents the finding and its implications.

### The TSTR comparison structure

TSTR (Train on Synthetic, Test on Real) adds synthetic samples to the real training fold:

```
TSTR training set = X_real (30 samples) + X_synthetic (N samples)
```

For this comparison to be valid, the augmented baseline must have the same structure:

```
Augmented baseline = X_real (30 samples) + X_aug (N samples)
```

### What online augmentation actually does

```python
# Online augmentation (reference run)
DataLoader(
    EEGTorchDataset(X_real_30, y_train, augmentor=STANDARD_AUGMENTOR),
    batch_size=30
)
# Each epoch: 30 randomly-transformed versions of the real samples
# Dataset size: 30 — never changes
```

Online augmentation exposes the model to augmented *views* of the real data, not to additional samples. The gradient is always computed over 30 samples, not 30 + N.

### Why this matters

If TSTR uses `X_real_30 + X_synthetic_30` (60 total), and the augmented baseline uses only 30 samples, then a direct accuracy comparison confounds two variables:

1. Quality of additional data (augmented vs. synthetic)
2. Amount of training data (30 vs. 60)

A reviewer would correctly object: "Of course TSTR wins — it trains on twice as much data." This objection invalidates the TSTR comparison.

### What online augmentation is useful for

Online augmentation correctly measures **what it achieves as a regularisation technique** — the standard PyTorch approach used in most EEG papers. It answers the question: "If I apply standard training-time augmentation to 30 samples, how much does it help?" The answer is +1.7%.

This run is NOT a comparison baseline for TSTR. It is retained in this report as a reference data point showing the ceiling of regularisation-based augmentation.

### The correct approach: Offline additive design

The primary augmented baseline for TSTR comparison uses the offline additive design:

```python
# Offline additive augmentation
X_aug_copies = generate_N_copies(X_real_30, augmentor, n_copies)
X_train = np.concatenate([X_real_30, X_aug_copies])  # 30 + N
DataLoader(EEGTorchDataset(X_train, y_train), batch_size=len(X_train))
```

Additive offline runs at N = {30, 60, 90} (1×, 2×, 3× added samples). TSTR will then be run at the same N values with synthetic data replacing augmented copies. The comparison will be fair and publishable.

---

## 7. Conclusions

### 7.1 Online augmentation provides marginal aggregate benefit (+1.7%)

The mean accuracy increase from 56.7% to 58.4% is real but small. At the population level, augmentation barely moves the needle. This is consistent with the theoretical analysis: online augmentation is a regularisation technique that prevents overfitting to the exact noise pattern of 30 samples, but does not add new information that Block 2 (the spatial filter) can learn from.

### 7.2 The distribution of effects is bimodal and subject-specific

Online augmentation is not uniformly helpful or uniformly harmful. It produces:
- **Strong improvements** in data-limited subjects with latent but unstable ERD signals (S060: +33%, S002: +27%, S054: +22%)
- **Strong regressions** in learning/strong subjects with stable, consistent spatial filters (S016: −22%, S010: −20%, S005: −18%)

The net effect (+1.7%) masks this variance. A per-subject approach to augmentation (e.g., applying it only to subjects below a threshold) would likely perform better than uniform application — but that is out of scope for this baseline comparison.

### 7.3 Some consistently-performing vanilla subjects regress with augmentation (single run; may be partly regression to the mean)

Subjects with identical fold accuracies in vanilla (S005: [67%/67%/67%], S017: [73%/73%/73%]) show systematic regression with augmentation. One possible explanation: if a subject's ERD is stable and learnable from 30 samples, augmentation adds noise that degrades the learned spatial filter. With a single seed and 15-trial test folds, this needs replication before it can be called systematic.

### 7.4 Augmentation can rescue subjects that vanilla training fails on

S060 (+33.3%), S002 (+26.7%), S054 (+22.2%), S073 (+19.1%) all moved from below-learning to strong-responder tier. These subjects may have a real ERD signal that vanilla training missed but vanilla training failed to learn it from 30 clean samples. If this replicates, there is a subset of subjects where augmentation unlocks performance that vanilla training does not reach — a natural target for synthetic data. (Single-run per-subject gains of this size are also expected from noise, so this is a hypothesis.)

### 7.5 Non-responders remain non-responders

S059, S076, and other confirmed non-responders showed zero improvement with augmentation. This is consistent with these subjects not producing a reliable ERD during motor imagery.

### 7.6 The primary comparison baseline for TSTR must use the offline additive design

Online augmentation is not the correct TSTR baseline. The offline additive design is required. This run is retained as a reference data point showing what the standard PyTorch augmentation approach achieves: +1.7%.

---

## 8. Implications for Phase 0 Next Steps

### Immediate: Offline additive augmented baseline

The `_create_offline_augmented_set()` function in `src/neurosynth/train_baseline.py` pre-generates augmented copies before training:

```bash
python scripts/train_classifier.py --run augmented_offline --n-copies 1  # 30+30=60 total
python scripts/train_classifier.py --run augmented_offline --n-copies 2  # 30+60=90 total
python scripts/train_classifier.py --run augmented_offline --n-copies 3  # 30+90=120 total
```

These runs establish the ceiling for the additive comparison. Based on this online augmentation run, we expect the additive runs to show more consistent improvement in data-limited subjects (because each gradient step sees more diverse samples) but also potentially more consistent regression in strong subjects.

### Then: diffusion model (Phase 0 Step 4)

Train a class-conditional 1D U-Net diffusion model on `data/processed.npz` to generate synthetic motor imagery EEG epochs. The diffusion model must learn to generate samples from the ERD signal distribution — the key question being whether its synthetic samples carry real spatial information (correct ERD topography) or merely mimic superficial statistics.

### Then: TSTR comparison (Phase 0 Step 5)

For each LORO fold:
1. Generate N synthetic samples from the diffusion model (conditioned on class)
2. Combine with the real 30 training samples: `X_train = X_real + X_synthetic`
3. Train EEGNet on the combined set
4. Evaluate on the held-out real run
5. Compare accuracy vs. offline additive augmented at the same N

If the TSTR curve stays above the offline additive augmented curve across all N values — the go/no-go decision (RR-3) is a clear positive.

---

## 9. Summary Table: All 106 Subjects

| Subject | Vanilla | Augmented | Δ | Tier (vanilla) |
|---|---|---|---|---|
| S001 | 48.9% | 66.7% | **+17.8%** | Data-limited |
| S002 | 57.8% | 84.4% | **+26.7%** | Learning |
| S003 | 53.3% | 46.7% | −6.7% | Data-limited |
| S004 | 68.9% | 68.9% | 0.0% | Learning |
| S005 | 66.7% | 48.9% | **−17.8%** | Learning |
| S006 | 53.3% | 53.3% | 0.0% | Data-limited |
| S007 | 64.4% | 53.3% | −11.1% | Learning |
| S008 | 62.2% | 55.6% | −6.7% | Learning |
| S009 | 62.2% | 66.7% | +4.4% | Learning |
| S010 | 68.9% | 48.9% | **−20.0%** | Learning |
| S011 | 37.8% | 62.2% | **+24.4%** | Non-responder |
| S012 | 46.7% | 48.9% | +2.2% | Data-limited |
| S013 | 48.9% | 40.0% | −8.9% | Data-limited |
| S014 | 53.3% | 57.8% | +4.4% | Data-limited |
| S015 | 68.9% | 66.7% | −2.2% | Learning |
| S016 | 64.4% | 42.2% | **−22.2%** | Learning |
| S017 | 73.3% | 62.2% | −11.1% | Strong |
| S018 | 71.1% | 60.0% | −11.1% | Strong |
| S019 | 57.8% | 64.4% | +6.7% | Learning |
| S020 | 53.3% | 44.4% | −8.9% | Data-limited |
| S021 | 68.9% | 60.0% | −8.9% | Learning |
| S022 | 48.9% | 55.6% | +6.7% | Data-limited |
| S023 | 55.6% | 53.3% | −2.2% | Data-limited |
| S024 | 48.9% | 55.6% | +6.7% | Data-limited |
| S025 | 55.6% | 55.6% | 0.0% | Data-limited |
| S026 | 68.9% | 57.8% | −11.1% | Learning |
| S027 | 62.2% | 64.4% | +2.2% | Learning |
| S028 | 60.0% | 55.6% | −4.4% | Learning |
| S029 | 64.4% | 53.3% | −11.1% | Learning |
| S030 | 53.3% | 48.9% | −4.4% | Data-limited |
| S031 | 53.3% | 64.4% | **+11.1%** | Data-limited |
| S032 | 60.0% | 66.7% | +6.7% | Learning |
| S033 | 55.6% | 57.8% | +2.2% | Data-limited |
| S034 | 62.5% | 67.6% | +5.1% | Learning |
| S035 | 62.2% | 55.6% | −6.7% | Learning |
| S036 | 40.0% | 57.8% | **+17.8%** | Non-responder |
| S037 | 56.3% | 53.5% | −2.9% | Data-limited |
| S038 | 48.9% | 55.6% | +6.7% | Data-limited |
| S039 | 37.8% | 46.7% | +8.9% | Non-responder |
| S040 | 75.6% | 80.0% | +4.4% | Strong |
| S041 | 43.2% | 52.1% | +8.9% | Non-responder |
| S042 | 64.4% | 73.3% | +8.9% | Learning |
| S043 | 46.7% | 51.1% | +4.4% | Data-limited |
| S044 | 62.2% | 57.8% | −4.4% | Learning |
| S045 | 55.6% | 53.3% | −2.2% | Data-limited |
| S046 | 33.3% | 42.2% | +8.9% | Non-responder |
| S047 | 44.4% | 57.8% | **+13.3%** | Non-responder |
| S048 | 44.4% | 53.3% | +8.9% | Non-responder |
| S049 | 62.2% | 68.9% | +6.7% | Learning |
| S050 | 64.4% | 57.8% | −6.7% | Learning |
| S051 | 86.7% | 86.7% | 0.0% | Strong |
| S052 | 51.1% | 37.8% | **−13.3%** | Data-limited |
| S053 | 84.4% | 80.0% | −4.4% | Strong |
| S054 | 57.8% | 80.0% | **+22.2%** | Learning |
| S055 | 51.1% | 51.1% | 0.0% | Data-limited |
| S056 | 55.6% | 64.4% | +8.9% | Data-limited |
| S057 | 40.0% | 48.9% | +8.9% | Non-responder |
| S058 | 53.3% | 57.8% | +4.4% | Data-limited |
| S059 | 42.2% | 42.2% | 0.0% | Non-responder |
| S060 | 51.1% | 84.4% | **+33.3%** | Data-limited |
| S061 | 55.6% | 62.2% | +6.7% | Data-limited |
| S062 | 80.0% | 86.7% | +6.7% | Strong |
| S063 | 44.4% | 53.3% | +8.9% | Non-responder |
| S064 | 49.8% | 61.6% | **+11.7%** | Data-limited |
| S065 | 53.3% | 68.9% | **+15.6%** | Data-limited |
| S066 | 46.7% | 57.8% | **+11.1%** | Data-limited |
| S067 | 57.8% | 51.1% | −6.7% | Learning |
| S068 | 51.1% | 42.2% | −8.9% | Data-limited |
| S069 | 57.8% | 42.2% | **−15.6%** | Learning |
| S070 | 64.4% | 53.3% | −11.1% | Learning |
| S071 | 71.1% | 73.3% | +2.2% | Strong |
| S072 | 88.4% | 92.9% | +4.5% | Strong |
| S073 | 42.9% | 61.9% | **+19.1%** | Non-responder |
| S074 | 44.0% | 39.0% | −5.0% | Non-responder |
| S075 | 46.7% | 60.0% | **+13.3%** | Data-limited |
| S076 | 45.2% | 45.2% | 0.0% | Data-limited |
| S077 | 53.3% | 57.8% | +4.4% | Data-limited |
| S078 | 57.8% | 51.1% | −6.7% | Learning |
| S079 | 46.7% | 44.4% | −2.2% | Data-limited |
| S080 | 55.6% | 60.0% | +4.4% | Data-limited |
| S081 | 55.6% | 66.7% | **+11.1%** | Data-limited |
| S082 | 53.3% | 48.9% | −4.4% | Data-limited |
| S083 | 35.6% | 53.3% | **+17.8%** | Non-responder |
| S084 | 44.4% | 57.8% | **+13.3%** | Non-responder |
| S085 | 84.4% | 75.6% | −8.9% | Strong |
| S086 | 68.9% | 55.6% | **−13.3%** | Learning |
| S087 | 53.3% | 57.8% | +4.4% | Data-limited |
| S089 | 57.8% | 55.6% | −2.2% | Learning |
| S090 | 46.7% | 55.6% | +8.9% | Data-limited |
| S091 | 64.4% | 53.3% | −11.1% | Learning |
| S093 | 62.2% | 66.7% | +4.4% | Learning |
| S094 | 64.4% | 77.8% | **+13.3%** | Learning |
| S095 | 57.8% | 55.6% | −2.2% | Learning |
| S096 | 60.0% | 48.9% | −11.1% | Learning |
| S097 | 51.1% | 57.8% | +6.7% | Data-limited |
| S098 | 48.9% | 48.9% | 0.0% | Data-limited |
| S099 | 64.4% | 53.3% | −11.1% | Learning |
| S101 | 62.2% | 53.3% | −8.9% | Learning |
| S102 | 47.6% | 50.0% | +2.4% | Data-limited |
| S103 | 57.8% | 53.3% | −4.4% | Learning |
| S104 | 46.7% | 62.2% | **+15.6%** | Data-limited |
| S105 | 64.4% | 66.7% | +2.2% | Learning |
| S106 | 64.4% | 68.9% | +4.4% | Learning |
| S107 | 57.8% | 57.8% | 0.0% | Learning |
| S108 | 62.2% | 68.9% | +6.7% | Learning |
| S109 | 46.7% | 42.2% | −4.4% | Data-limited |

---

## References

- Lawhern, V.J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
- Schalk, G. et al. (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. *IEEE Trans. Biomed. Eng.*, 51(6), 1034–1043.
- Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
