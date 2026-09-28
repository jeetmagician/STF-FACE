#!/usr/bin/env python3
"""Evaluate verification performance: ROC, EER, FAR/FRR, precision/recall.

Usage
-----
    python scripts/evaluate.py pairs.csv --root /data/faces --plot roc.png

    # Break results down by condition, to see where the system actually fails
    python scripts/evaluate.py pairs.csv --root /data/faces --by-condition

Condition breakdown
-------------------
Add an optional `condition` column to the CSV (e.g. `age_gap_10y`,
`facial_hair`, `post_surgical`, `lookalike`) and the report is computed per
condition as well as overall. This matters: an aggregate AUC of 0.98 can
conceal near-chance performance on the subgroup you actually care about, and
an aggregate number is the easiest way to ship a system that fails exactly
where it is most consequential.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.models.registry import get_embedder  # noqa: E402
from app.pipeline.similarity import cosine_similarity  # noqa: E402
from app.scoring.calibration import CalibrationProfile, load_profile  # noqa: E402


def roc_curve(
    genuine: np.ndarray, impostor: np.ndarray, steps: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, false_accept_rate, true_accept_rate)."""
    low = float(min(genuine.min(), impostor.min()))
    high = float(max(genuine.max(), impostor.max()))
    thresholds = np.linspace(low, high, steps)

    far = np.array([(impostor >= t).mean() for t in thresholds])
    tar = np.array([(genuine >= t).mean() for t in thresholds])
    return thresholds, far, tar


def compute_metrics(
    genuine: np.ndarray, impostor: np.ndarray, profile: CalibrationProfile | None = None
) -> dict:
    thresholds, far, tar = roc_curve(genuine, impostor)
    frr = 1.0 - tar

    # Equal error rate: where FAR and FRR cross.
    crossing = int(np.argmin(np.abs(far - frr)))
    eer = float((far[crossing] + frr[crossing]) / 2.0)
    eer_threshold = float(thresholds[crossing])

    # AUC by trapezoidal integration over FAR ascending.
    order = np.argsort(far)
    auc = float(np.trapezoid(tar[order], far[order]))

    metrics: dict = {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "genuine_mean": float(genuine.mean()),
        "genuine_std": float(genuine.std()),
        "impostor_mean": float(impostor.mean()),
        "impostor_std": float(impostor.std()),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
    }

    # Operating points at fixed false-accept rates - the way biometric systems
    # are normally specified in practice.
    for target in (0.001, 0.01, 0.05):
        feasible = np.where(far <= target)[0]
        if len(feasible):
            index = int(feasible[np.argmax(tar[feasible])])
            metrics[f"tar_at_far_{target}"] = float(tar[index])
            metrics[f"threshold_at_far_{target}"] = float(thresholds[index])
        else:
            metrics[f"tar_at_far_{target}"] = 0.0
            metrics[f"threshold_at_far_{target}"] = float(thresholds[-1])

    if profile is not None:
        settings = get_settings()
        threshold = profile.decision_threshold(settings.prior_same_person)
        metrics["default_threshold_raw"] = float(threshold)
        metrics["far_at_default"] = float((impostor >= threshold).mean())
        metrics["frr_at_default"] = float((genuine < threshold).mean())

        true_positive = float((genuine >= threshold).sum())
        false_positive = float((impostor >= threshold).sum())
        false_negative = float((genuine < threshold).sum())
        precision = true_positive / max(true_positive + false_positive, 1e-9)
        recall = true_positive / max(true_positive + false_negative, 1e-9)
        metrics["precision_at_default"] = precision
        metrics["recall_at_default"] = recall
        metrics["f1_at_default"] = (
            2 * precision * recall / max(precision + recall, 1e-9)
        )

        # Calibration quality: Brier score against the predicted posterior.
        predictions = np.concatenate(
            [
                [profile.posterior(s, settings.prior_same_person) for s in genuine],
                [profile.posterior(s, settings.prior_same_person) for s in impostor],
            ]
        )
        actuals = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
        metrics["brier_score"] = float(np.mean((predictions - actuals) ** 2))

    return metrics


def plot_roc(genuine: np.ndarray, impostor: np.ndarray, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ! matplotlib is not installed; skipping the plot.")
        return

    thresholds, far, tar = roc_curve(genuine, impostor)
    metrics = compute_metrics(genuine, impostor)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(far, tar, linewidth=2.2, color="#b8860b")
    axes[0].plot([0, 1], [0, 1], "--", linewidth=1, color="#999999")
    axes[0].set_xlabel("False acceptance rate")
    axes[0].set_ylabel("True acceptance rate")
    axes[0].set_title(f"ROC  (AUC = {metrics['auc']:.4f},  EER = {metrics['eer']:.4f})")
    axes[0].grid(alpha=0.25)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)

    bins = np.linspace(
        min(genuine.min(), impostor.min()), max(genuine.max(), impostor.max()), 60
    )
    axes[1].hist(impostor, bins=bins, alpha=0.62, label="Different people", color="#6b7fa3")
    axes[1].hist(genuine, bins=bins, alpha=0.62, label="Same person", color="#b8860b")
    axes[1].axvline(
        metrics["eer_threshold"],
        linestyle="--",
        color="#333333",
        label=f"EER threshold {metrics['eer_threshold']:.3f}",
    )
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Pair count")
    axes[1].set_title("Score distributions")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle(
        "Overlap between the two distributions is the irreducible error. "
        "It is never zero.",
        fontsize=10,
        y=0.005,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  · wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pairs_csv", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--by-condition", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    embedder = get_embedder(settings)
    profile = load_profile(
        settings.calibration_dir, settings.calibration_profile, embedder.info.backend
    )

    print(f"Backend:     {embedder.info.backend} ({embedder.info.recognizer})")
    print(f"Calibration: {profile.name} (fitted={profile.fitted})\n")

    cache: dict[Path, np.ndarray | None] = {}
    scores: list[tuple[float, int, str]] = []
    skipped = 0

    with args.pairs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows, start=1):
        if index % 50 == 0:
            print(f"  processed {index}/{len(rows)}...")

        paths = []
        for column in ("image_a", "image_b"):
            path = Path(row[column])
            if args.root is not None and not path.is_absolute():
                path = args.root / path
            paths.append(path)

        embeddings = []
        for path in paths:
            if path not in cache:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    cache[path] = None
                else:
                    faces = embedder.detect(image, max_faces=1)
                    cache[path] = embedder.embed(faces[0].aligned) if faces else None
            embeddings.append(cache[path])

        if any(e is None for e in embeddings):
            skipped += 1
            continue

        scores.append(
            (
                cosine_similarity(embeddings[0], embeddings[1]),
                int(row["label"]),
                row.get("condition", "all"),
            )
        )

    if skipped:
        print(f"\nSkipped {skipped} pairs where a face could not be detected.")

    genuine = np.array([s for s, label, _ in scores if label == 1])
    impostor = np.array([s for s, label, _ in scores if label == 0])

    if not len(genuine) or not len(impostor):
        raise SystemExit("Both genuine and impostor pairs are required.")

    print("\n" + "=" * 62)
    print("OVERALL")
    print("=" * 62)
    metrics = compute_metrics(genuine, impostor, profile)
    for key, value in metrics.items():
        print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else f"  {key:<28} {value}")

    if args.by_condition:
        grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for score, label, condition in scores:
            grouped[condition].append((score, label))

        for condition in sorted(grouped):
            subset = grouped[condition]
            sub_genuine = np.array([s for s, label in subset if label == 1])
            sub_impostor = np.array([s for s, label in subset if label == 0])
            print("\n" + "-" * 62)
            print(f"CONDITION: {condition}")
            print("-" * 62)
            if not len(sub_genuine) or not len(sub_impostor):
                print(
                    f"  Only one class present "
                    f"({len(sub_genuine)} genuine, {len(sub_impostor)} impostor); "
                    "discrimination metrics need both."
                )
                if len(sub_genuine):
                    print(f"  genuine mean  {sub_genuine.mean():.4f}")
                if len(sub_impostor):
                    print(f"  impostor mean {sub_impostor.mean():.4f}")
                continue
            sub_metrics = compute_metrics(sub_genuine, sub_impostor, profile)
            for key in ("n_genuine", "n_impostor", "eer", "auc", "frr_at_default", "far_at_default"):
                value = sub_metrics.get(key)
                if value is None:
                    continue
                print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else f"  {key:<28} {value}")

    if args.plot:
        print()
        plot_roc(genuine, impostor, args.plot)

    if not profile.fitted:
        print(
            "\nNOTE: the active calibration is unvalidated, so the "
            "posterior-dependent figures above (FAR/FRR at the default "
            "threshold, Brier score) describe the placeholder parameters "
            "rather than a fitted system. EER and AUC are calibration-"
            "independent and remain meaningful."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
