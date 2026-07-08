"""
Train EEGNet on EEGBCI using within-subject Leave-One-Run-Out evaluation.

Four run modes establish the go/no-go comparison baselines:
  vanilla            — real data only, no augmentation
  augmented          — online augmentation (regularisation only, reference run)
  augmented_offline  — additive offline augmentation: X_real + N×aug_copies
  tstr               — additive synthetic augmentation: X_real + N synthetic epochs
                       from the run-holdout diffusion model that never saw the
                       fold's test run (Train-on-Synthetic, Test-on-Real gate)

For each subject, EEGNet is trained independently on 2 of 3 runs and tested
on the held-out run. Accuracy is averaged across 3 folds then across subjects.

Usage:
    python eval/train_baseline.py --data data/processed.npz --run vanilla
    python eval/train_baseline.py --data data/processed.npz --run augmented
    python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 1
    python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 2
    python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 3
    python eval/train_baseline.py --data data/processed.npz --run tstr --n-copies 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from data.eegbci_pipeline import EEGDataset
from eval.augmentations import STANDARD_AUGMENTOR
from eval.dataset import EEGTorchDataset, within_subject_loro_split
from eval.eegnet import EEGNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_offline_augmented_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_copies: int,
    augmentor,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-generate n_copies augmented versions of each training sample.

    Returns X_combined (n*(1+n_copies), C, T) and y_combined of the same length.
    Real samples are always prepended unchanged so Block 2 anchors on clean signal.
    Class balance is preserved: all samples (including copies) are generated.
    """
    rng = np.random.default_rng(seed)
    aug_X: list[np.ndarray] = []
    aug_y: list[int] = []
    for _ in range(n_copies):
        perm = rng.permutation(len(X_train))
        for idx in perm:
            x = torch.tensor(X_train[idx], dtype=torch.float32)
            aug_X.append(augmentor(x).numpy())
            aug_y.append(int(y_train[idx]))
    X_combined = np.concatenate([X_train, np.array(aug_X, dtype=np.float32)])
    y_combined = np.concatenate([y_train, np.array(aug_y, dtype=np.int64)])
    return X_combined, y_combined


@dataclass
class SyntheticBank:
    """A class-balanced bank of synthetic epochs generated once per diffusion model.

    Keyed in the banks dict by ``held_out_run`` — the run the source model never
    saw — which is exactly the fold ``test_run`` it is allowed to augment.
    """
    held_out_run: int      # 4, 8, or 12 — the run the source model never saw
    X: np.ndarray          # (bank_size, 64, 801) float32, z-scored
    y: np.ndarray          # (bank_size,) int64 in {0, 1}, class-balanced


def _build_synthetic_banks(
    checkpoint_dir: Path,
    device: torch.device,
    bank_size: int = 2048,
    seed: int = 42,
    runs: tuple[int, ...] = (4, 8, 12),
) -> dict[int, SyntheticBank]:
    """
    Load each ``diffusion_holdout_run{R}.pt`` once and generate a class-balanced
    bank of ``bank_size`` epochs. Returns ``{R: SyntheticBank(held_out_run=R, X, y)}``.

    The models are subject-agnostic, so a fold's synthetic samples depend only on
    its test run (which selects the model), not on its subject — hence one bank per
    model rather than 50-step DDIM sampling inside each of the 318 folds.

    Guards (must not be removed):
      * each loaded model's ``held_out_run`` must match its key ``R``;
      * generated ``X.std()`` must sit near 1.0 (real z-scored data has std=1.0) —
        a std ~700 means the DDIM sampler regressed to the pre-fix explosion (e.g.
        ``clip_denoised`` fell back to 0), so we abort loudly *before* spending any
        experiment compute on a garbage bank.

    A missing checkpoint aborts with the exact Step 4 command to produce/pull it.
    """
    from eval.diffusion import EEGDiffusionModel

    banks: dict[int, SyntheticBank] = {}
    for R in runs:
        ckpt = checkpoint_dir / f"diffusion_holdout_run{R}.pt"
        if not ckpt.exists():
            sys.exit(
                f"Diffusion checkpoint not found: {ckpt}\n"
                "Produce it with Step 4, e.g.:\n"
                f"    modal run --detach modal_train_diffusion.py --runs {R}\n"
                f"    modal volume get neurosynth-checkpoints diffusion_holdout_run{R}.pt "
                f"{checkpoint_dir}/"
            )
        model = EEGDiffusionModel.load(ckpt, device=device)
        assert model.held_out_run == R, (
            f"Checkpoint {ckpt.name} reports held_out_run={model.held_out_run}, "
            f"expected {R} — wrong file for this key."
        )
        X = model.generate(bank_size, subject_id=0, seed=seed + R)
        y = model.labels(bank_size)

        std = float(X.std())
        if not (0.8 < std < 1.3):
            sys.exit(
                f"Sampler-health guard FAILED for diffusion_holdout_run{R}: generated "
                f"std={std:.3f} (expected ~1.0 for z-scored data). A std of a few "
                "hundred means the DDIM sampler exploded — most likely clip_denoised "
                "fell back to 0 or the z-score/clamp fix was lost in eval/diffusion.py. "
                "Aborting before the experiment runs."
            )
        assert set(np.unique(y).tolist()) <= {0, 1}, (
            f"Bank for run {R} has labels outside {{0, 1}}: {np.unique(y)}"
        )
        banks[R] = SyntheticBank(held_out_run=R, X=X.astype(np.float32), y=y.astype(np.int64))
        print(f"  Bank run {R:>2}: {bank_size} epochs | std={std:.3f} | "
              f"class balance={np.bincount(y)}")
    return banks


def _load_synthetic_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    test_run: int,
    n_copies: int,
    bank: SyntheticBank,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for ``_create_offline_augmented_set`` that appends synthetic
    epochs instead of signal-augmented copies. For each class ``c`` it adds
    ``n_copies × count(y_real == c)`` synthetic epochs of that class, drawn without
    replacement from ``bank`` via a fold-seeded RNG. Real samples are prepended
    unchanged so Block 2 anchors on real signal — identical structure and per-fold
    sample count / class distribution to the augmented arm at the same ``n_copies``,
    which makes the two curves comparable sample-for-sample.

    Guard (must not be removed): asserts ``bank.held_out_run == test_run`` — a fold
    can never be augmented by a model that saw its test run. Removing/weakening this
    reintroduces the exact leakage the cross-fitting scheme exists to prevent.

    Returns ``(X_combined, y_combined)`` with ``len == len(y_real) * (1 + n_copies)``.
    """
    assert bank.held_out_run == test_run, (
        f"Leakage guard tripped: bank.held_out_run={bank.held_out_run} but "
        f"test_run={test_run}. A fold must draw only from the model that never "
        "saw its test run."
    )
    rng = np.random.default_rng(seed)
    syn_X_parts: list[np.ndarray] = []
    syn_y_parts: list[np.ndarray] = []
    for c in np.unique(y_real):
        n_needed = int(n_copies * int((y_real == c).sum()))
        bank_idx = np.where(bank.y == c)[0]
        if n_needed > len(bank_idx):
            raise ValueError(
                f"Bank too small: need {n_needed} class-{int(c)} synthetic samples "
                f"for test_run={test_run} but the bank holds only {len(bank_idx)}. "
                "Raise --bank-size."
            )
        chosen = rng.choice(bank_idx, size=n_needed, replace=False)
        syn_X_parts.append(bank.X[chosen])
        syn_y_parts.append(np.full(n_needed, int(c), dtype=np.int64))

    X_syn = np.concatenate(syn_X_parts).astype(np.float32)
    y_syn = np.concatenate(syn_y_parts).astype(np.int64)
    X_combined = np.concatenate([X_real.astype(np.float32), X_syn])
    y_combined = np.concatenate([y_real.astype(np.int64), y_syn])
    return X_combined, y_combined


def _save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, run_name: str, out: Path) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["class 0", "class 1"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {run_name} (all folds pooled)")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_channels: int,
    n_times: int,
    device: torch.device,
    augmentor,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    max_norm_fc: float,
    min_lr: float,
    checkpoint_path: Path,
) -> float:
    """Train EEGNet for fixed epochs on the full training fold. Returns best train loss."""
    model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    criterion = nn.CrossEntropyLoss()

    # Full-batch: all training samples in one batch → clean BatchNorm statistics,
    # deterministic gradient on small within-subject dataset.
    train_loader = DataLoader(
        EEGTorchDataset(X_train, y_train, augmentor=augmentor),
        batch_size=len(y_train), shuffle=True, num_workers=0,
    )

    best_train_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            model.apply_weight_constraints(max_norm_fc=max_norm_fc)
            epoch_loss += loss.item()

        scheduler.step()

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)

    return best_train_loss


def _eval_on_test(
    checkpoint_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_channels: int,
    n_times: int,
    device: torch.device,
    dropout: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load checkpoint and return (preds, labels) on test set."""
    model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=2, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    test_loader = DataLoader(
        EEGTorchDataset(X_test, y_test),
        batch_size=min(batch_size, len(y_test)), shuffle=False, num_workers=0,
    )
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            preds.extend(model(X_batch).argmax(dim=1).cpu().numpy())
            labels.extend(y_batch.numpy())
    return np.array(preds), np.array(labels)


def train(
    data_path: Path,
    run_name: str,
    checkpoint_dir: Path,
    mlflow_experiment: str = "phase0-baseline",
    seed: int = 42,
    max_epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
    max_norm_fc: float = 0.25,
    min_lr: float = 1e-5,
    n_copies: int = 2,
    bank_size: int = 2048,
) -> dict:
    """
    Within-subject LORO evaluation: train one EEGNet per fold, aggregate across subjects.
    Returns {"test_accuracy": float, "test_accuracy_std": float, "test_f1": float, "run_id": str}.
    """
    valid_runs = ("vanilla", "augmented", "augmented_offline", "tstr")
    if run_name not in valid_runs:
        raise ValueError(f"run_name must be one of {valid_runs}, got '{run_name}'")
    if not data_path.exists():
        sys.exit(
            f"Dataset not found: {data_path}\n"
            "Run `python data/eegbci_pipeline.py --subjects $(seq 1 109) "
            "--out data/processed.npz` first."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = EEGDataset.load(data_path)
    splits = within_subject_loro_split(ds.X, ds.y, ds.subject_ids, ds.run_ids)
    n_channels, n_times = ds.X.shape[1], ds.X.shape[2]
    device = get_device()
    online_augmentor = STANDARD_AUGMENTOR if run_name == "augmented" else None

    print(f"Device   : {device}")
    print(f"Subjects : {len(np.unique(ds.subject_ids))} | Folds: {len(splits)}")
    print(f"EEGNet   : {EEGNet(n_channels=n_channels, n_times=n_times)}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt = checkpoint_dir / f"_tmp_{run_name}.pt"
    best_ckpt = checkpoint_dir / f"best_{run_name}.pt"

    # TSTR: build one class-balanced synthetic bank per run-holdout model, once,
    # before the LORO loop. Each fold later draws from banks[test_run], guaranteeing
    # (via the assertion in _load_synthetic_set) it never sees its own test run.
    banks: dict[int, SyntheticBank] = {}
    if run_name == "tstr":
        print(f"Building synthetic banks ({bank_size} epochs/model) from {checkpoint_dir} ...")
        banks = _build_synthetic_banks(checkpoint_dir, device, bank_size=bank_size, seed=seed)

    subjects = sorted(set(s["subject"] for s in splits))
    fold_results: list[dict] = []
    all_preds_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    best_fold_acc = -1.0

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        base_params: dict = {
            "run_name": run_name,
            "augmented": run_name != "vanilla",
            "eval_protocol": "within_subject_loro",
            "fixed_epochs": True,
            "seed": seed,
            "max_epochs": max_epochs,
            "batch_size": "full",
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "max_norm_fc": max_norm_fc,
            "scheduler": "cosine",
            "min_lr": min_lr,
            "n_channels": n_channels,
            "n_times": n_times,
            "n_subjects": len(subjects),
            "n_folds_total": len(splits),
        }
        if run_name == "augmented_offline":
            base_params["augmentation_mode"] = "offline"
            base_params["n_copies"] = n_copies
            base_params["n_train_samples"] = 30 * (1 + n_copies)
        elif run_name == "augmented":
            base_params["augmentation_mode"] = "online"
        elif run_name == "tstr":
            base_params["augmentation_mode"] = "synthetic"
            base_params["n_copies"] = n_copies
            base_params["n_extra_per_fold"] = "n_copies * len(train)"
            base_params["bank_size"] = bank_size
            base_params["guidance_scale"] = 3.0
            base_params["ddim_steps"] = 50
            base_params["clip_denoised"] = 4.0
            base_params["diffusion_checkpoints"] = "diffusion_holdout_run{4,8,12}.pt"
        else:
            base_params["augmentation_mode"] = "none"
        mlflow.log_params(base_params)

        for subj in subjects:
            subj_splits = [s for s in splits if s["subject"] == subj]
            subj_accs: list[float] = []

            for i, split in enumerate(subj_splits):
                X_tr = split["X_train"]
                y_tr = split["y_train"]
                X_te = split["X_test"]
                y_te = split["y_test"]

                if run_name == "augmented_offline":
                    X_tr, y_tr = _create_offline_augmented_set(
                        X_tr, y_tr,
                        n_copies=n_copies,
                        augmentor=STANDARD_AUGMENTOR,
                        seed=seed + subj * 100 + i,
                    )
                    fold_augmentor = None
                elif run_name == "tstr":
                    X_tr, y_tr = _load_synthetic_set(
                        X_tr, y_tr,
                        test_run=split["test_run"],
                        n_copies=n_copies,
                        bank=banks[split["test_run"]],
                        seed=seed + subj * 100 + i,
                    )
                    fold_augmentor = None
                else:
                    fold_augmentor = online_augmentor

                print(
                    f"S{subj:03d} | fold {i+1}/{len(subj_splits)} "
                    f"| test_run={split['test_run']} "
                    f"| n_train={len(y_tr)} n_test={len(y_te)}",
                    end=" -> ",
                    flush=True,
                )

                _train_fold(
                    X_tr, y_tr,
                    n_channels, n_times, device, fold_augmentor,
                    max_epochs, lr, weight_decay,
                    dropout, max_norm_fc, min_lr, temp_ckpt,
                )

                preds, labels = _eval_on_test(
                    temp_ckpt, X_te, y_te, n_channels, n_times, device, dropout, len(y_te),
                )
                acc = float(accuracy_score(labels, preds))
                f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

                print(f"acc={acc:.1%} | f1={f1:.3f}")

                fold_results.append({
                    "subject": subj,
                    "test_run": split["test_run"],
                    "accuracy": acc,
                    "f1": f1,
                })
                subj_accs.append(acc)
                all_preds_list.append(preds)
                all_labels_list.append(labels)

                if acc > best_fold_acc:
                    best_fold_acc = acc
                    shutil.copy2(temp_ckpt, best_ckpt)

            subj_mean = float(np.mean(subj_accs))
            print(f"S{subj:03d} | MEAN = {subj_mean:.1%}")

        # -- Aggregation --
        per_subject: dict[int, list[float]] = {}
        for r in fold_results:
            per_subject.setdefault(r["subject"], []).append(r["accuracy"])

        subj_means = [float(np.mean(v)) for v in per_subject.values()]
        overall_accuracy = float(np.mean(subj_means))
        overall_std = float(np.std(subj_means))
        all_preds = np.concatenate(all_preds_list)
        all_labels = np.concatenate(all_labels_list)
        overall_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        mlflow.log_metrics({
            "test_accuracy_mean": overall_accuracy,
            "test_accuracy_std": overall_std,
            "test_f1_mean": overall_f1,
        })

        # Per-subject JSON artifact
        per_subj_data = [
            {
                "subject": s,
                "mean_accuracy": float(np.mean(v)),
                "fold_accuracies": [float(x) for x in v],
            }
            for s, v in per_subject.items()
        ]
        if run_name == "augmented_offline":
            json_stem = f"per_subject_results_augmented_{n_copies}x"
        elif run_name == "tstr":
            json_stem = f"per_subject_results_tstr_{n_copies}x"
        else:
            json_stem = f"per_subject_results_{run_name}"
        json_path = checkpoint_dir / f"{json_stem}.json"
        json_path.write_text(json.dumps(per_subj_data, indent=2))
        mlflow.log_artifact(str(json_path))

        if run_name == "augmented_offline":
            cm_stem = f"confusion_matrix_augmented_{n_copies}x"
        elif run_name == "tstr":
            cm_stem = f"confusion_matrix_tstr_{n_copies}x"
        else:
            cm_stem = f"confusion_matrix_{run_name}"
        cm_path = checkpoint_dir / f"{cm_stem}.png"
        _save_confusion_matrix(all_labels, all_preds, run_name, cm_path)
        mlflow.log_artifact(str(cm_path))

        print(f"\n{'='*60}")
        print(f"  Run        : {run_name}")
        print(f"  Subjects   : {len(subjects)}")
        print(f"  Folds      : {len(fold_results)}")
        print(f"  Accuracy   : {overall_accuracy:.4f} ± {overall_std:.4f}  ({overall_accuracy*100:.1f}%)")
        print(f"  F1 (macro) : {overall_f1:.4f}")
        print(f"{'='*60}\n")

        run_id = mlflow.active_run().info.run_id

    temp_ckpt.unlink(missing_ok=True)

    return {
        "test_accuracy": overall_accuracy,
        "test_accuracy_std": overall_std,
        "test_f1": overall_f1,
        "run_id": run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed.npz"))
    parser.add_argument(
        "--run",
        choices=["vanilla", "augmented", "augmented_offline", "tstr"],
        required=True,
        help=(
            "vanilla = no augmentation; "
            "augmented = online augmentation (regularisation, reference run); "
            "augmented_offline = additive offline copies (use with --n-copies); "
            "tstr = additive synthetic epochs from the run-holdout diffusion models "
            "(use with --n-copies; the RR-3 go/no-go gate)"
        ),
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=2,
        metavar="N",
        help="Number of extra copies added per real sample (augmented_offline / tstr). "
             "1→60 total, 2→90 total, 3→120 total. Default: 2.",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=2048,
        metavar="N",
        help="Synthetic epochs generated per run-holdout model for the tstr bank "
             "(class-balanced). Default: 2048 (1024/class).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment. If omitted: phase0-tstr for --run tstr, "
             "else phase0-baseline.",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-norm-fc", type=float, default=0.25)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        print(
            "WARNING: No GPU/MPS detected — training on CPU will be slow.\n"
            "Tip: add --max-epochs 50 for a quick sanity run first."
        )

    # Resolve the MLflow experiment default so tstr runs never pollute the baseline
    # experiment: --experiment omitted -> phase0-tstr for tstr, else phase0-baseline.
    experiment = args.experiment
    if experiment is None:
        experiment = "phase0-tstr" if args.run == "tstr" else "phase0-baseline"

    result = train(
        data_path=args.data,
        run_name=args.run,
        checkpoint_dir=args.checkpoint_dir,
        mlflow_experiment=experiment,
        seed=args.seed,
        max_epochs=args.max_epochs,
        lr=args.lr,
        dropout=args.dropout,
        max_norm_fc=args.max_norm_fc,
        min_lr=args.min_lr,
        n_copies=args.n_copies,
        bank_size=args.bank_size,
    )
    print(f"Run ID  : {result['run_id']}")
    print("Results logged. Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    main()
