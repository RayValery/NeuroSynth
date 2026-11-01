"""
Tests for the invariants every reported number depends on:

1. Leave-one-run-out splits never put a test-run epoch in training and never mix subjects.
2. Synthetic data can only be drawn from a generator that never saw the fold's test run
   (and, for subject-conditional banks, only from the fold's own subject).
3. Augmented and synthetic arms add the same number of samples with the same class balance,
   so the comparison is sample-matched.

Runs on CPU in a few seconds with tiny synthetic arrays — no dataset or checkpoint needed:
    pytest tests/
"""
from __future__ import annotations

import numpy as np
import pytest

from neurosynth.augmentations import STANDARD_AUGMENTOR
from neurosynth.dataset import within_subject_loro_split
from neurosynth.train_baseline import (
    SyntheticBank,
    _create_offline_augmented_set,
    _load_synthetic_set,
)

RUNS = (4, 8, 12)
C, T = 4, 32


def _toy_dataset(subjects=(1, 2, 3), per_run=6, seed=0):
    """Toy (X, y, subject_ids, run_ids). Each epoch carries its global index in X[:, 0, 0]."""
    rng = np.random.default_rng(seed)
    X, y, sid, rid = [], [], [], []
    for s in subjects:
        for r in RUNS:
            for k in range(per_run):
                X.append(rng.standard_normal((C, T)).astype(np.float32))
                y.append(k % 2)
                sid.append(s)
                rid.append(r)
    X = np.stack(X)
    X[:, 0, 0] = np.arange(len(X))  # tag
    return X, np.array(y, dtype=np.int64), np.array(sid), np.array(rid)


def _bank(held_out_run, subject_id=None, n_per_class=64, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((2 * n_per_class, C, T)).astype(np.float32)
    y = np.repeat([0, 1], n_per_class).astype(np.int64)
    return SyntheticBank(held_out_run=held_out_run, X=X, y=y, subject_id=subject_id)


# ── 1. LORO split ──────────────────────────────────────────────────────────────

def test_loro_produces_one_fold_per_subject_and_run():
    X, y, sid, rid = _toy_dataset()
    splits = within_subject_loro_split(X, y, sid, rid)
    assert len(splits) == 3 * len(RUNS)
    assert {(s["subject"], s["test_run"]) for s in splits} == {(s, r) for s in (1, 2, 3) for r in RUNS}


def test_loro_test_run_never_in_training_and_subjects_never_mix():
    X, y, sid, rid = _toy_dataset()
    for split in within_subject_loro_split(X, y, sid, rid):
        train_idx = split["X_train"][:, 0, 0].astype(int)
        test_idx = split["X_test"][:, 0, 0].astype(int)
        assert set(train_idx).isdisjoint(test_idx)
        assert np.all(rid[test_idx] == split["test_run"])
        assert np.all(rid[train_idx] != split["test_run"])
        assert np.all(sid[train_idx] == split["subject"])
        assert np.all(sid[test_idx] == split["subject"])
        # train ∪ test is exactly this subject's data
        assert set(train_idx) | set(test_idx) == set(np.where(sid == split["subject"])[0])


# ── 2. Generator routing guards ────────────────────────────────────────────────

def test_synthetic_set_rejects_generator_that_saw_the_test_run():
    X_real = np.zeros((10, C, T), dtype=np.float32)
    y_real = np.array([0, 1] * 5, dtype=np.int64)
    with pytest.raises(AssertionError, match="Leakage guard"):
        _load_synthetic_set(X_real, y_real, test_run=8, n_copies=1, bank=_bank(held_out_run=4), seed=0)


def test_subject_conditional_set_rejects_other_subjects_bank():
    X_real = np.zeros((10, C, T), dtype=np.float32)
    y_real = np.array([0, 1] * 5, dtype=np.int64)
    bank = _bank(held_out_run=8, subject_id=2)
    with pytest.raises(AssertionError, match="Two-key routing guard"):
        _load_synthetic_set(X_real, y_real, test_run=8, n_copies=1, bank=bank, seed=0, expected_subject=3)


def test_synthetic_set_accepts_matching_bank():
    X_real = np.zeros((10, C, T), dtype=np.float32)
    y_real = np.array([0, 1] * 5, dtype=np.int64)
    bank = _bank(held_out_run=8, subject_id=3)
    X, y = _load_synthetic_set(X_real, y_real, test_run=8, n_copies=2, bank=bank, seed=0, expected_subject=3)
    assert len(y) == 30


# ── 3. Sample-matched comparison ──────────────────────────────────────────────

@pytest.mark.parametrize("n_copies", [1, 2, 3])
def test_augmented_and_synthetic_arms_are_sample_matched(n_copies):
    rng = np.random.default_rng(0)
    X_real = rng.standard_normal((30, C, T)).astype(np.float32)
    y_real = np.array([0] * 16 + [1] * 14, dtype=np.int64)  # slightly unbalanced, as in real folds

    X_aug, y_aug = _create_offline_augmented_set(X_real, y_real, n_copies, STANDARD_AUGMENTOR, seed=0)
    X_syn, y_syn = _load_synthetic_set(X_real, y_real, test_run=4, n_copies=n_copies,
                                       bank=_bank(held_out_run=4), seed=0)

    assert len(y_aug) == len(y_syn) == 30 * (1 + n_copies)
    assert np.array_equal(np.bincount(y_aug), np.bincount(y_syn))
    # Real samples are prepended unchanged in both arms
    np.testing.assert_array_equal(X_aug[:30], X_real)
    np.testing.assert_array_equal(X_syn[:30], X_real)
    np.testing.assert_array_equal(y_syn[:30], y_real)


def test_synthetic_draw_is_deterministic_and_without_replacement():
    X_real = np.zeros((30, C, T), dtype=np.float32)
    y_real = np.array([0, 1] * 15, dtype=np.int64)
    bank = _bank(held_out_run=12)
    a, _ = _load_synthetic_set(X_real, y_real, test_run=12, n_copies=3, bank=bank, seed=7)
    b, _ = _load_synthetic_set(X_real, y_real, test_run=12, n_copies=3, bank=bank, seed=7)
    np.testing.assert_array_equal(a, b)
    syn = a[30:].reshape(len(a) - 30, -1)
    assert len(np.unique(syn, axis=0)) == len(syn)  # no bank epoch used twice in a fold
