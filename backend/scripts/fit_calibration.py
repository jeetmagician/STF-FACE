#!/usr/bin/env python3
"""Fit calibration parameters on labelled image pairs.

WHY THIS IS NOT OPTIONAL
------------------------
A face model outputs a similarity score whose distribution depends on the
model, the population, the capture conditions, and the difficulty of the pairs.
There is no universal mapping from cosine similarity to "percent chance of the
same person", and any product that hard-codes one is presenting an assumption
as a measurement.

Concretely, a score of 0.45 might be a comfortable match for well-lit frontal
pairs of the same age, and an ambiguous result for pairs separated by fifteen
years. Same number, different meaning. Only data from *your* population can
tell you which you have.

This is especially true for the cases this tool is aimed at. Public benchmarks
are dominated by easy pairs; post-surgical and large-age-gap pairs sit in a
shifted part of the score distribution. Calibrating on benchmark-like data and
then applying it to hard cases will systematically overstate the evidence
against genuine pairs.

INPUT FORMAT
------------
A CSV with three columns:

    image_a,image_b,label

`label` is 1 for a genuine pair (same person) and 0 for an impostor pair.
Paths are resolved relative to --root, or absolute.

    image_a,image_b,label
    alice/2019.jpg,alice/2024.jpg,1
    alice/2019.jpg,bob/2021.jpg,0

HOW MANY PAIRS
--------------
Several hundred of each class is a workable minimum for the Gaussian method;
the logistic method wants more. Below roughly 200 per class the fitted
parameters are themselves noisy enough that the resulting percentages carry
false precision, and the script warns accordingly.

Compose the set to reflect real use: include the hard conditions you expect
(age gaps, lighting changes, facial hair, weight change, surgery where you can
obtain it ethically and lawfully) rather than only easy pairs.

USAGE
-----
    python scripts/fit_calibration.py pairs.csv --root /data/faces \\
        --method gaussian --profile production \\
        --description "1,842 pairs, internal KYC set, 2021-2026"
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.models.registry import get_embedder  # noqa: E402
from app.pipeline.similarity import cosine_similarity  # noqa: E402
from app.scoring.calibration import (  # noqa: E402
    CalibrationProfile,
    fit_gaussian,
    fit_logistic,
)

MIN_RECOMMENDED_PAIRS = 200


def load_pairs(csv_path: Path, root: Path | None) -> list[tuple[Path, Path, int]]:
    pairs: list[tuple[Path, Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_a", "image_b", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(
                f"CSV must have columns {sorted(required)}; found {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
            except (TypeError, ValueError):
                print(f"  ! row {row_number}: label is not an integer; skipping")
                continue
            if label not in (0, 1):
                print(f"  ! row {row_number}: label must be 0 or 1; skipping")
                continue

            path_a = Path(row["image_a"])
            path_b = Path(row["image_b"])
            if root is not None:
                path_a = path_a if path_a.is_absolute() else root / path_a
                path_b = path_b if path_b.is_absolute() else root / path_b
            pairs.append((path_a, path_b, label))
    return pairs


def embed_image(path: Path, embedder, settings) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    faces = embedder.detect(image, max_faces=1)
    if not faces:
        return None
    return embedder.embed(faces[0].aligned)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pairs_csv", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--method", choices=["gaussian", "logistic"], default="gaussian")
    parser.add_argument("--profile", default="production", help="Output profile name.")
    parser.add_argument(
        "--description", default="", help="What this dataset is. Recorded in the profile."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    embedder = get_embedder(settings)
    backend = embedder.info.backend
    print(f"Backend: {backend} ({embedder.info.recognizer})\n")

    pairs = load_pairs(args.pairs_csv, args.root)
    print(f"Loaded {len(pairs)} labelled pairs.\n")

    cache: dict[Path, np.ndarray | None] = {}
    genuine: list[float] = []
    impostor: list[float] = []
    skipped = 0

    for index, (path_a, path_b, label) in enumerate(pairs, start=1):
        if index % 50 == 0:
            print(f"  processed {index}/{len(pairs)}...")

        for path in (path_a, path_b):
            if path not in cache:
                cache[path] = embed_image(path, embedder, settings)

        embedding_a, embedding_b = cache[path_a], cache[path_b]
        if embedding_a is None or embedding_b is None:
            skipped += 1
            continue

        score = cosine_similarity(embedding_a, embedding_b)
        (genuine if label == 1 else impostor).append(score)

    print(f"\nUsable pairs: {len(genuine)} genuine, {len(impostor)} impostor")
    if skipped:
        print(f"Skipped {skipped} pairs where a face could not be detected.")

    if not genuine or not impostor:
        raise SystemExit("Both genuine and impostor pairs are required to calibrate.")

    genuine_array = np.array(genuine)
    impostor_array = np.array(impostor)

    warnings: list[str] = []
    if len(genuine) < MIN_RECOMMENDED_PAIRS or len(impostor) < MIN_RECOMMENDED_PAIRS:
        warnings.append(
            f"Fewer than {MIN_RECOMMENDED_PAIRS} pairs in at least one class. The "
            "fitted parameters are themselves uncertain, so the resulting "
            "percentages carry more precision than the data supports."
        )

    print(
        f"\nGenuine  mean {genuine_array.mean():.4f}  sd {genuine_array.std():.4f}"
        f"\nImpostor mean {impostor_array.mean():.4f}  sd {impostor_array.std():.4f}"
    )

    profile = CalibrationProfile(
        name=args.profile,
        backend=backend,
        method=args.method,
        fitted=True,
        n_genuine_pairs=len(genuine),
        n_impostor_pairs=len(impostor),
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_description=args.description or f"{args.pairs_csv.name}",
        notes=" ".join(warnings),
    )

    if args.method == "gaussian":
        (
            profile.genuine_mean,
            profile.genuine_std,
            profile.impostor_mean,
            profile.impostor_std,
        ) = fit_gaussian(genuine_array, impostor_array)
    else:
        profile.logistic_a, profile.logistic_b, profile.logistic_train_prior = fit_logistic(
            genuine_array, impostor_array
        )
        # Keep the Gaussian parameters populated too, for reference in reports.
        (
            profile.genuine_mean,
            profile.genuine_std,
            profile.impostor_mean,
            profile.impostor_std,
        ) = fit_gaussian(genuine_array, impostor_array)

    from evaluate import compute_metrics  # noqa: E402  (same scripts directory)

    metrics = compute_metrics(genuine_array, impostor_array, profile)
    profile.metrics = metrics

    print("\nPerformance on the fitting set (optimistic - use a held-out set for a")
    print("trustworthy estimate):")
    print(f"  EER                 {metrics['eer']:.4f} at raw score {metrics['eer_threshold']:.4f}")
    print(f"  AUC                 {metrics['auc']:.4f}")
    print(f"  FAR @ posterior 0.5 {metrics['far_at_default']:.4f}")
    print(f"  FRR @ posterior 0.5 {metrics['frr_at_default']:.4f}")

    output_dir = args.output_dir or settings.calibration_dir
    output_path = Path(output_dir) / f"{backend}.{args.profile}.json"
    profile.save(output_path)
    print(f"\nWrote {output_path}")
    print(f"Activate it with CALIBRATION_PROFILE={args.profile}")

    for warning in warnings:
        print(f"\nWARNING: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
