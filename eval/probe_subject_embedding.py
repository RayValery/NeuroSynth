"""
Probe whether a subject-conditional generator (pivot_iter_02) actually USES its subject
embedding — the gate that decides whether a null TSTR result is real or just conditioning
collapse. Run this on a downloaded `diffusion_subj_holdout_run{R}.pt` BEFORE spending on
the TSTR passes.

Two checks:

  1. Same-seed A-vs-B  (embedding USED?)
     Generate for several subjects at the SAME initial noise + class + domain, so the ONLY
     thing that varies is the subject token. Identical output (cross-subject mean|Δ| ≈ 0)
     means the embedding is ignored/collapsed; a clearly-positive spread means it is used.
     (The training-time spread over *different* seeds is NOT diagnostic — two independent
     std-1 signals differ by ~2/sqrt(pi) ≈ 1.13 whether or not the embedding does anything.)

  2. PSD specificity  (embedding USEFUL / subject-specific?)
     For K subjects, compare each subject's generated mu/beta (8-30 Hz) PSD to EVERY
     subject's real imagery PSD. If conditioning is subject-specific, a subject's generated
     PSD tracks its OWN real PSD better than other subjects' (diagonal of the KxK distance
     matrix < off-diagonal; own-real is the nearest for most subjects).

Usage:
    python eval/probe_subject_embedding.py \
        --checkpoint models/checkpoints/diffusion_subj_holdout_run12.pt \
        --data data/processed_with_exec.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import welch

from data.eegbci_pipeline import EEGDataset
from eval.diffusion import DOMAIN_IMAGERY, EEGDiffusionModel, get_device

SAMPLING_RATE = 160  # Hz — EEGBCI
MU_BETA = (8, 30)    # Hz band the classifier depends on
SAME_SEED = 20_260   # fixed initial noise for the same-seed A-vs-B check


def _band_logpsd(X: np.ndarray) -> np.ndarray:
    """Mean log10 PSD over epochs+channels, restricted to the mu/beta band. X:(N,C,T)."""
    nperseg = min(256, X.shape[-1])
    f, pxx = welch(X, fs=SAMPLING_RATE, nperseg=nperseg, axis=-1)  # (N,C,F)
    band = (f >= MU_BETA[0]) & (f <= MU_BETA[1])
    return np.log10(pxx.mean(axis=(0, 1))[band] + 1e-12)


# Two independent std-1 signals differ by E|X-Y| = 2/sqrt(pi) ≈ 1.128 — the "fully
# different" reference the same-seed cross-subject spread is measured against.
NOISE_FLOOR = 2.0 / np.sqrt(np.pi)


def same_seed_check(model: EEGDiffusionModel, subjects: list[int], n: int, domain: int,
                    subject_guidance_scale: float = 0.0) -> float:
    """Cross-subject mean|Δ| at a fixed seed+class (only the subject token varies)."""
    g = subject_guidance_scale
    gens = {s: model.generate(n, subject_id=s, class_label=0, domain=domain, seed=SAME_SEED,
                              subject_guidance_scale=g) for s in subjects}
    # sanity: same subject + same seed must be identical (isolates the token, not RNG).
    repeat = model.generate(n, subject_id=subjects[0], class_label=0, domain=domain,
                            seed=SAME_SEED, subject_guidance_scale=g)
    self_delta = float(np.abs(gens[subjects[0]] - repeat).mean())

    diffs = [float(np.abs(gens[subjects[i]] - gens[subjects[j]]).mean())
             for i in range(len(subjects)) for j in range(i + 1, len(subjects))]
    spread = float(np.mean(diffs)) if diffs else float("nan")
    print(f"  same-subject/same-seed |Δ| (should be ~0)  : {self_delta:.6f}")
    print(f"  cross-subject/same-seed mean|Δ|            : {spread:.4f}  "
          f"({100*spread/NOISE_FLOOR:.1f}% of a full independent resample)")
    return spread


def psd_specificity(
    model: EEGDiffusionModel, ds: EEGDataset, subjects: list[int], n: int, domain: int,
    subject_guidance_scale: float = 0.0,
) -> tuple[float, float, float]:
    """KxK distance between gen-PSD(Si) and real-PSD(Sj); return (diag, offdiag, own-nearest frac)."""
    is_img = ds.domains == DOMAIN_IMAGERY if ds.domains is not None else np.ones(len(ds.y), bool)
    real_psd, gen_psd = {}, {}
    for s in subjects:
        real_s = ds.X[(ds.subject_ids == s) & is_img]
        real_psd[s] = _band_logpsd(real_s)
        gen_psd[s] = _band_logpsd(model.generate(n, subject_id=s, domain=domain,
                                                 seed=SAME_SEED + s,
                                                 subject_guidance_scale=subject_guidance_scale))

    K = len(subjects)
    D = np.zeros((K, K))
    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            D[i, j] = float(np.mean(np.abs(gen_psd[si] - real_psd[sj])))  # gen(i) vs real(j)

    diag = float(np.mean(np.diag(D)))
    off = float((D.sum() - np.trace(D)) / (K * K - K)) if K > 1 else float("nan")
    own_nearest = float(np.mean([D[i].argmin() == i for i in range(K)]))
    return diag, off, own_nearest


def _probe_one(model, ds, subjects, n, domain, g: float) -> dict:
    """Run both checks at one guidance scale; return the aim/fidelity metrics."""
    print(f"\n--- subject_guidance_scale = {g:g} ---")
    print("[1] Same-seed A-vs-B (is the embedding USED?)")
    spread = same_seed_check(model, subjects, n, domain, subject_guidance_scale=g)
    if ds is None:
        diag = off = own = float("nan")
    else:
        print("[2] PSD specificity (does gen track each subject's OWN real PSD?)")
        diag, off, own = psd_specificity(model, ds, subjects, n, domain, subject_guidance_scale=g)
        print(f"  gen-vs-own-real PSD dist (diag, fidelity↓) : {diag:.4f}")
        print(f"  gen-vs-other-real PSD dist (off-diag)      : {off:.4f}")
        print(f"  aim gap (off-diag, higher=more specific)   : {off - diag:+.4f}")
        print(f"  own real is the NEAREST for                : {own*100:.0f}% of subjects")
    return {"g": g, "spread": spread, "diag": diag, "off": off,
            "gap": (off - diag) if not np.isnan(diag) else float("nan"), "own": own}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="diffusion_subj_holdout_run{R}.pt to probe.")
    p.add_argument("--data", type=Path, default=Path("data/processed_with_exec.npz"))
    p.add_argument("--n-subjects", type=int, default=6, help="How many subjects to probe.")
    p.add_argument("--n-samples", type=int, default=64, help="Epochs generated per subject.")
    p.add_argument("--subject-guidance-scale", default="0",
                   help="Sample-time subject-CFG scale(s) to test on THIS checkpoint, for free "
                        "(no bank regen). Comma-separate to sweep, e.g. '0,1.5,3,5'. Each scale "
                        "trades aim (specificity ↑) against fidelity (distance to own real ↑).")
    args = p.parse_args()

    try:
        scales = [float(s) for s in str(args.subject_guidance_scale).split(",") if s.strip() != ""]
    except ValueError:
        sys.exit(f"--subject-guidance-scale must be number(s), got '{args.subject_guidance_scale}'")

    model = EEGDiffusionModel.load(args.checkpoint, device=get_device())
    if not model.config.use_subject_cond:
        sys.exit(f"{args.checkpoint.name} is NOT subject-conditional (use_subject_cond=False) — "
                 "nothing to probe.")
    domain = model.config.sample_domain
    cohort = list(model.config.subject_ids)
    subjects = cohort[:: max(1, len(cohort) // args.n_subjects)][: args.n_subjects]
    ds = EEGDataset.load(args.data) if args.data.exists() else None
    print(f"Checkpoint : {args.checkpoint.name}  (held_out_run={model.held_out_run}, "
          f"n_subjects={model.config.n_subjects})")
    print(f"Probing subjects: {subjects}  | scales: {scales}"
          + ("" if ds is not None else "  | (no --data → PSD/fidelity skipped)"))

    rows = [_probe_one(model, ds, subjects, args.n_samples, domain, g) for g in scales]

    # ---- comparison table (the aim-vs-fidelity trade-off across scales) ----
    print("\n" + "=" * 72)
    print(f"  {'scale':>6} {'same-seed':>10} {'fidelity(diag↓)':>16} {'aim gap↑':>10} {'own-nearest':>12}")
    for r in rows:
        diag_s = f"{r['diag']:.4f}" if not np.isnan(r["diag"]) else "n/a"
        gap_s = f"{r['gap']:+.4f}" if not np.isnan(r["gap"]) else "n/a"
        own_s = f"{r['own']*100:.0f}%" if not np.isnan(r["own"]) else "n/a"
        print(f"  {r['g']:>6g} {r['spread']:>10.4f} {diag_s:>16} {gap_s:>10} {own_s:>12}")
    print("  " + "-" * 68)

    if ds is None:
        print("  Pass --data to measure fidelity/aim and get a recommendation.")
    elif len(rows) == 1:
        r = rows[0]
        used = r["spread"] > 1e-3
        specific_ok = (not np.isnan(r["diag"])) and r["diag"] < r["off"] and r["own"] >= 0.5
        if not used:
            print("  VERDICT: ❌ embedding effectively UNUSED at this scale (same-seed barely moves).")
        elif specific_ok:
            print("  VERDICT: ✅ used AND subject-specific — OK to regenerate banks at this scale + TSTR.")
        else:
            print("  VERDICT: ⚠️ used but weakly subject-specific — try other scales (comma-sweep) or")
            print("           accept the data wall. Higher aim gap with fidelity(diag) near the g=0 row wins.")
    else:
        # Recommend the scale with the best aim (gap) whose fidelity hasn't blown out vs g=0.
        base = next((r for r in rows if r["g"] == 0), rows[0])
        cand = [r for r in rows if not np.isnan(r["gap"])]
        # keep fidelity within 1.6x of the g=0 baseline, then maximise the aim gap
        ok = [r for r in cand if base["diag"] <= 0 or r["diag"] <= 1.6 * base["diag"]] or cand
        best = max(ok, key=lambda r: (r["gap"], r["own"]))
        print(f"  Best aim-vs-fidelity trade-off: scale = {best['g']:g}  "
              f"(aim gap {best['gap']:+.4f}, fidelity {best['diag']:.4f} vs g0 {base['diag']:.4f})")
        print("  → regenerate banks ONCE at that scale, then run the TSTR passes:")
        print(f"     modal run --detach modal_train_diffusion_subj.py --subject-guidance-scale {best['g']:g}")
        print(f"     modal run --detach modal_train_tstr.py --variant subj --subject-guidance-scale {best['g']:g}")
    print("=" * 72)


if __name__ == "__main__":
    main()
