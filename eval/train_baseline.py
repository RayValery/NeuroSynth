"""
Train EEGNet on EEGBCI using within-subject Leave-One-Run-Out evaluation.

Two runs establish the go/no-go comparison baselines:
  vanilla   — real data only, no augmentation
  augmented — real data + standard augmentation (noise, shift, dropout, crop)

For each subject, EEGNet is trained independently on 2 of 3 runs and tested
on the held-out run. Accuracy is averaged across 3 folds then across subjects.

Usage:
    python eval/train_baseline.py --data data/processed.npz --run vanilla
    python eval/train_baseline.py --data data/processed.npz --run augmented
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
) -> dict:
    """
    Within-subject LORO evaluation: train one EEGNet per fold, aggregate across subjects.
    Returns {"test_accuracy": float, "test_accuracy_std": float, "test_f1": float, "run_id": str}.
    """
    if run_name not in ("vanilla", "augmented"):
        raise ValueError(f"run_name must be 'vanilla' or 'augmented', got '{run_name}'")
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
    augmentor = STANDARD_AUGMENTOR if run_name == "augmented" else None

    print(f"Device   : {device}")
    print(f"Subjects : {len(np.unique(ds.subject_ids))} | Folds: {len(splits)}")
    print(f"EEGNet   : {EEGNet(n_channels=n_channels, n_times=n_times)}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_ckpt = checkpoint_dir / f"_tmp_{run_name}.pt"
    best_ckpt = checkpoint_dir / f"best_{run_name}.pt"

    subjects = sorted(set(s["subject"] for s in splits))
    fold_results: list[dict] = []
    all_preds_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    best_fold_acc = -1.0

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "run_name": run_name,
            "augmented": run_name == "augmented",
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
        })

        for subj in subjects:
            subj_splits = [s for s in splits if s["subject"] == subj]
            subj_accs: list[float] = []

            for i, split in enumerate(subj_splits):
                X_tr = split["X_train"]
                y_tr = split["y_train"]
                X_te = split["X_test"]
                y_te = split["y_test"]

                print(
                    f"S{subj:03d} | fold {i+1}/{len(subj_splits)} "
                    f"| test_run={split['test_run']} "
                    f"| n_train={len(y_tr)} n_test={len(y_te)}",
                    end=" -> ",
                    flush=True,
                )

                _train_fold(
                    X_tr, y_tr,
                    n_channels, n_times, device, augmentor,
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
        json_path = checkpoint_dir / f"per_subject_results_{run_name}.json"
        json_path.write_text(json.dumps(per_subj_data, indent=2))
        mlflow.log_artifact(str(json_path))

        cm_path = checkpoint_dir / f"confusion_matrix_{run_name}.png"
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
    parser.add_argument("--run", choices=["vanilla", "augmented"], required=True,
                        help="vanilla = no augmentation; augmented = standard signal augmentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--experiment", default="phase0-baseline")
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

    result = train(
        data_path=args.data,
        run_name=args.run,
        checkpoint_dir=args.checkpoint_dir,
        mlflow_experiment=args.experiment,
        seed=args.seed,
        max_epochs=args.max_epochs,
        lr=args.lr,
        dropout=args.dropout,
        max_norm_fc=args.max_norm_fc,
        min_lr=args.min_lr,
    )
    print(f"Run ID  : {result['run_id']}")
    print("Results logged. Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    main()
