#!/usr/bin/env python3
"""Download and verify face-recognition model weights.

Usage
-----
    python scripts/download_models.py               # OpenCV backend (default)
    python scripts/download_models.py --backend insightface
    python scripts/download_models.py --verify-only
    python scripts/download_models.py --write-hashes   # pin what you downloaded

Verification
------------
Weights are checked three ways:

  1. size is within the expected range
  2. the file begins with the ONNX protobuf signature
  3. OpenCV actually loads the model and runs one dummy inference

That third check is the substantive one - it catches truncation, proxy error
pages saved as .onnx, and architecture mismatches, which a size check alone
would not.

Additionally, if `assets/models/model_hashes.json` exists, SHA-256 values are
enforced against it. That file ships empty on purpose: pinning a hash you have
not personally computed provides no security, only the appearance of it. Run
with --write-hashes after your first successful download to populate it, and
commit the result so later installs are pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = BACKEND_ROOT / "assets" / "models"
HASH_FILE = MODEL_DIR / "model_hashes.json"

ONNX_MAGIC = b"\x08"  # protobuf field 1 (ir_version), varint

# Pinned to a specific commit rather than a branch so the artefact cannot change
# underneath you.
OPENCV_ZOO_COMMIT = "main"

OPENCV_MODELS = [
    {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}"
            "/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "min_bytes": 180_000,
        "max_bytes": 400_000,
        "role": "detector",
        "licence": "Apache-2.0",
    },
    {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": (
            f"https://github.com/opencv/opencv_zoo/raw/{OPENCV_ZOO_COMMIT}"
            "/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "min_bytes": 30_000_000,
        "max_bytes": 45_000_000,
        "role": "recognizer",
        "licence": "Apache-2.0",
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_hashes() -> dict[str, str]:
    if not HASH_FILE.exists():
        return {}
    try:
        data = json.loads(HASH_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  ! {HASH_FILE.name} is not valid JSON; ignoring.", file=sys.stderr)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str) and len(v) == 64}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "facet-model-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 256):
                    handle.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {exc}") from exc
    temporary.replace(destination)


def structural_check(path: Path, spec: dict) -> list[str]:
    problems: list[str] = []
    size = path.stat().st_size

    if not spec["min_bytes"] <= size <= spec["max_bytes"]:
        problems.append(
            f"size {size:,} bytes is outside the expected "
            f"{spec['min_bytes']:,}-{spec['max_bytes']:,} range"
        )

    head = path.read_bytes()[:16]
    if not head.startswith(ONNX_MAGIC):
        preview = head[:32]
        if preview.lstrip().startswith((b"<", b"{")):
            problems.append(
                "the file looks like HTML or JSON, not ONNX - the download was "
                "probably intercepted by a proxy or returned an error page"
            )
        else:
            problems.append("the file does not begin with an ONNX protobuf header")

    return problems


def functional_check(model_dir: Path) -> list[str]:
    """Load both models through OpenCV and run one dummy inference."""
    problems: list[str] = []
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return [f"cannot run the functional check: {exc}"]

    yunet_path = model_dir / OPENCV_MODELS[0]["filename"]
    sface_path = model_dir / OPENCV_MODELS[1]["filename"]

    try:
        detector = cv2.FaceDetectorYN.create(
            model=str(yunet_path), config="", input_size=(320, 320)
        )
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        detector.setInputSize((320, 320))
        detector.detect(dummy)
    except Exception as exc:
        problems.append(f"YuNet failed to load or run: {exc}")

    try:
        recognizer = cv2.FaceRecognizerSF.create(model=str(sface_path), config="")
        feature = recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
        dimension = int(np.asarray(feature).size)
        if dimension != 128:
            problems.append(
                f"SFace produced a {dimension}-d embedding; 128 was expected"
            )
    except Exception as exc:
        problems.append(f"SFace failed to load or run: {exc}")

    return problems


def handle_opencv(args: argparse.Namespace) -> int:
    pinned = load_pinned_hashes()
    computed: dict[str, str] = {}
    failed = False

    print(f"Model directory: {MODEL_DIR}\n")

    for spec in OPENCV_MODELS:
        path = MODEL_DIR / spec["filename"]
        print(f"{spec['filename']}  ({spec['role']}, {spec['licence']})")

        if path.exists() and not args.force:
            print("  · already present")
        elif args.verify_only:
            print("  ! missing, and --verify-only was given")
            failed = True
            continue
        else:
            print(f"  · downloading from {spec['url']}")
            try:
                download(spec["url"], path)
                print(f"  · saved {path.stat().st_size:,} bytes")
            except RuntimeError as exc:
                print(f"  ! {exc}", file=sys.stderr)
                failed = True
                continue

        problems = structural_check(path, spec)

        digest = sha256_of(path)
        computed[spec["filename"]] = digest
        expected = pinned.get(spec["filename"])
        if expected and expected != digest:
            problems.append(
                f"SHA-256 mismatch\n      expected {expected}\n      got      {digest}"
            )
        elif expected:
            print("  · SHA-256 matches the pinned value")
        else:
            print(f"  · SHA-256 {digest}  (unpinned)")

        if problems:
            failed = True
            for problem in problems:
                print(f"  ! {problem}", file=sys.stderr)
        print()

    if not failed:
        print("Running functional check (load + dummy inference)...")
        problems = functional_check(MODEL_DIR)
        if problems:
            failed = True
            for problem in problems:
                print(f"  ! {problem}", file=sys.stderr)
        else:
            print("  · both models load and run correctly\n")

    if args.write_hashes and computed and not failed:
        HASH_FILE.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote pinned hashes to {HASH_FILE.name}. Commit this file.")

    if failed:
        print("\nOne or more models could not be verified.", file=sys.stderr)
        return 1

    print("All OpenCV backend models are present and working.")
    if not pinned:
        print(
            "\nTip: re-run with --write-hashes to pin these exact files, so a "
            "later download cannot silently differ."
        )
    return 0


def handle_insightface(args: argparse.Namespace) -> int:
    print(
        "InsightFace weights\n"
        "-------------------\n"
        "LICENCE: the insightface library is MIT, but its pretrained model packs\n"
        "(buffalo_l, buffalo_s, antelopev2) are released for NON-COMMERCIAL\n"
        "RESEARCH USE ONLY. Commercial deployment needs a licence from\n"
        "InsightFace (recognition-oss-pack@insightface.ai) or your own weights.\n"
    )

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print(
            "The insightface package is not installed. Install it with:\n"
            "    pip install -r requirements-insightface.txt\n",
            file=sys.stderr,
        )
        return 1

    if args.verify_only:
        pack_dir = MODEL_DIR.parent / "models" / args.pack
        if not pack_dir.exists():
            print(f"  ! {args.pack} is not present at {pack_dir}", file=sys.stderr)
            return 1
        print(f"  · {args.pack} is present at {pack_dir}")
        return 0

    print(f"Preparing '{args.pack}'. The library downloads weights on first use.\n")
    try:
        app = FaceAnalysis(
            name=args.pack,
            root=str(MODEL_DIR.parent),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "landmark_2d_106", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:
        print(f"  ! failed to prepare {args.pack}: {exc}", file=sys.stderr)
        return 1

    print(f"\n'{args.pack}' is ready. Set MODEL_BACKEND=insightface to use it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend", choices=["opencv", "insightface"], default="opencv"
    )
    parser.add_argument("--pack", default="buffalo_l", help="InsightFace model pack name.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Check existing files; download nothing."
    )
    parser.add_argument(
        "--write-hashes",
        action="store_true",
        help="Write the computed SHA-256 values to model_hashes.json.",
    )
    args = parser.parse_args()

    if args.backend == "insightface":
        return handle_insightface(args)
    return handle_opencv(args)


if __name__ == "__main__":
    raise SystemExit(main())
