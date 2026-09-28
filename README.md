# Facet

A facial **similarity** assessment tool. Upload photographs of a person taken
at different times — across ageing, weight change, facial hair, different
lighting, or cosmetic surgery — and get a calibrated similarity score together
with an account of how much that score is actually worth.

It reports similarity under a model. It does not determine identity.

---

## Read this before you run it

Three things are true of this system, and the interface says all three:

**1. It is not calibrated out of the box.** It ships with documented
placeholder parameters, not measurements. Every response carries
`calibration.is_validated: false` and a warning until you fit calibration on
your own labelled pairs. A percentage from an uncalibrated system describes an
assumption. See [docs/CALIBRATION.md](docs/CALIBRATION.md).

**2. A high score never establishes identity.** Siblings score highly.
Identical twins score very highly. No model resolves this and no threshold
exists above which it stops being true.

**3. The default model is chosen for licensing, not accuracy.** See below.

---

## Model backends

Two are implemented behind one interface. Swap with `MODEL_BACKEND`.

| | `opencv` (default) | `insightface` |
|---|---|---|
| Detector | YuNet | SCRFD |
| Recognizer | SFace, 128-d | ArcFace w600k_r50, 512-d |
| Landmarks | 5 keypoints | 106 points |
| Size | ~40 MB | ~330 MB |
| Licence | **Apache-2.0** | Library MIT; **weights non-commercial research only** |
| Accuracy | Lower on large age gaps and post-surgical pairs | Higher |

The InsightFace library is MIT, but the pretrained model packs it downloads
(`buffalo_l` and relatives) are released by the upstream maintainers for
non-commercial research use only — this applies to manual downloads and to the
library's automatic download alike. Commercial deployment requires a licence
from InsightFace (`recognition-oss-pack@insightface.ai`) or your own weights.

So `opencv` is the default: it works immediately, needs 40 MB, and you can ship
it. If you are doing research, or you hold a licence, `insightface` is the
better model for the hard cases this tool targets.

---

## Quick start

```bash
git clone <your-repo> facet && cd facet
cp .env.example .env

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py        # ~40 MB, Apache-2.0
python scripts/selfcheck.py              # verify the pipeline
uvicorn app.main:app --reload --port 8000

# Frontend, in a second terminal
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>. API docs at <http://localhost:8000/docs>.

Or with Docker:

```bash
docker compose up --build
```

`make help` lists the shortcuts.

---

## Model weights

```bash
cd backend

python scripts/download_models.py                 # OpenCV backend
python scripts/download_models.py --verify-only   # check existing files
python scripts/download_models.py --write-hashes  # pin what you downloaded
```

Weights are verified three ways: file size, ONNX protobuf header, and an actual
load-plus-inference through OpenCV. That third check is the substantive one —
it catches truncated downloads and proxy error pages saved as `.onnx`, which is
how this usually fails.

Hash pinning is opt-in rather than baked in. Shipping a SHA-256 that nobody
verified provides the appearance of supply-chain security without the
substance; `--write-hashes` records the hashes of files you actually downloaded,
and committing `model_hashes.json` pins every later install to those exact
bytes.

For InsightFace:

```bash
pip install -r requirements-insightface.txt   # needs Cython + a C++ toolchain
python scripts/download_models.py --backend insightface
export MODEL_BACKEND=insightface
```

Weights land in `~/.insightface/models/` or `backend/assets/models/`.

---

## Calibrating

The step that turns this from a demo into something whose numbers mean
anything. Build a CSV of labelled pairs:

```csv
image_a,image_b,label,condition
alice/2019.jpg,alice/2024.jpg,1,age_gap_5y
alice/2019.jpg,bob/2021.jpg,0,different_people
carol/pre.jpg,carol/post.jpg,1,post_surgical
```

```bash
cd backend
python scripts/fit_calibration.py pairs.csv --root /data/faces \
    --method gaussian --profile production \
    --description "1,842 pairs, internal set, 2021-2026"

export CALIBRATION_PROFILE=production
```

Then measure it on a **held-out** set:

```bash
python scripts/evaluate.py heldout.csv --root /data/faces \
    --plot roc.png --by-condition
```

`--by-condition` is the one to pay attention to. An aggregate AUC of 0.98 can
conceal near-chance performance on the subgroup you actually care about, and
the aggregate number is the easiest way to ship a system that fails precisely
where it matters most.

Full rationale: [docs/CALIBRATION.md](docs/CALIBRATION.md).

---

## How the score works

```
cosine similarity  →  likelihood ratio  →  + prior  →  percentage
```

A cosine is not a probability, and neither is `cosine × 100`. The system
estimates how the score distributes under "same person" and under "different
people", and reports the ratio of those densities — the evidence the
comparison carries, independent of any assumption about how likely a match was
to begin with. The percentage comes from combining that with an explicit,
configurable prior:

```
logit(posterior) = ln(LR) + logit(prior)
```

The prior is unavoidable, so it is printed with every result instead of hidden
inside a constant. A pair from a passport-renewal queue has a very different
base rate from two unrelated social media accounts, and the same score should
not produce the same percentage in both.

The likelihood ratio is capped at roughly 55:1. Estimating a ratio of a million
to one would mean characterising distribution tails that a few thousand
validation pairs say nothing about. The cap means the strongest available
result reads as ~98%, not 99.99% — a deliberate refusal to sound more certain
than the data allows.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Cosmetic surgery

The stated requirement was to weight stable facial regions more heavily than
the nose. Here is what is technically possible, and what this system does
instead.

A global face embedding **cannot be decomposed into per-region contributions**.
The 512-d vector has no region-indexed structure, so "weight the periocular
region more heavily" is not an operation that can be performed on it. Any
system claiming to do so is inventing an interpretation the model does not
support.

What the system does:

1. **Global embedding similarity** is the score. ArcFace and SFace are already
   substantially robust to nasal change, because they are trained across
   enormous variation and do not depend on any single feature.
2. **Region analysis** is reported separately and marked descriptive-only:
   gradient-orientation descriptors over canonical regions of the aligned face,
   plus anthropometric ratios normalised by inter-ocular distance. It is never
   folded into the score.

That produces exactly the output the brief asked for —

> "The jaw and chin, mouth regions show substantial appearance differences
> between the two photographs. The periocular and forehead regions — generally
> among the more persistent facial areas — remain comparatively consistent. The
> global embedding comparison places this pair in the 'Very High Similarity'
> band."

— without the region analysis pretending to be evidence of identity. The
narrative never asserts a cause: "the nasal region differs" is supportable,
"the subject has had rhinoplasty" is not.

Geometric ratios are also pose-sensitive, so they are suppressed from the
narrative when the two head poses differ by more than 15°, with the reason
stated.

---

## Detection fallback and quality gating

Two config flags (`backend/app/config.py`), both on by default:

- **`ENABLE_DETECTION_FALLBACK`** — if the primary backend finds *zero* faces
  in a pair comparison, `analyse_pair_with_fallback` retries the whole
  comparison with the OpenCV/YuNet backend before giving up. Both images in a
  pair are always embedded by the same backend, so a fallback restarts the
  full comparison rather than mixing models. It only triggers on a genuine
  detection failure — quality gates and multi-face disambiguation are
  untouched. Only wired into `POST /api/analyze` and `/api/analyze/resume`;
  the live per-photo check at `POST /api/face-detect` does not use it yet, so
  an upload can still be flagged "no face detected" there even for a pair
  that would have succeeded via fallback at analysis time.
- **`STRICT_QUALITY_GATING`** — when set to `false`, quality gates (blur,
  pose, lighting, etc.) that would otherwise block a comparison are
  downgraded to warnings instead, and the pipeline scores the pair anyway.
  Face detection itself is unaffected: with no face there is no embedding, so
  `NoFaceError` and multi-face responses still apply regardless of this flag.

Region analysis also now flags a likely **hairstyle or hair-colour change**:
`detect_hairstyle_influence` (`backend/app/pipeline/regions.py`) compares the
colour shift in the forehead/hairline band against the core-face band, and
flags it when the hairline shift is both above a noise floor and at least 3x
the core-face shift. This is descriptive only — it never changes the
similarity score — but the aligned 112x112 crop used for the global embedding
does include a thin hairline strip, so a drastic hair change can have a small,
real effect on the raw score. Surfaced as `region_analysis.hairstyle_likely`,
a narrative note, and an entry in `uncertainty_sources`.

---

## Testing

```bash
cd backend
python scripts/selfcheck.py     # 46 checks, no weights or pytest needed
pytest -v                       # full suite
```

`selfcheck.py` runs the entire pipeline against a deterministic mock backend,
so it verifies decoding, validation, alignment, quality gating, region
analysis, templating, calibration, banding, rendering and the ephemeral store
without any model weights. It does **not** measure recognition accuracy — the
mock reduces pixels rather than embedding faces. Accuracy needs real weights
and real labelled pairs; that is what `evaluate.py` is for.

Several tests pin ethical constraints as assertions: no band text asserts
identity, the serialised response never contains identity claims, the low band
never asserts that photographs show different people, and the region narrative
never claims a cause.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, active model, licence, whether calibration is validated |
| `POST /api/face-detect` | Detect faces in one image with per-face quality |
| `POST /api/analyze` | Compare two sets of photographs |
| `POST /api/analyze/resume` | Continue after a face selection |
| `DELETE /api/session/{token}` | Discard retained images immediately |

`POST /api/analyze` takes `old_photos[]` and `new_photos[]` (1–5 each), with
optional `old_face_indices` / `new_face_indices` JSON arrays.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "old_photos=@before.jpg" \
  -F "new_photos=@after.jpg"
```

Returns `similarity_score`, `confidence_level`, `face_detected_*`,
`image_quality_*`, `pose_difference`, `warnings`, `analysis`, plus `scoring`,
`calibration`, `model`, `templates`, `region_analysis`, `uncertainty_sources`
and base64 `visualisations`.

If several faces are found, the response is **409** with thumbnails and a
session token rather than a silent pick.

---

## Privacy and security

- Images are processed **entirely in memory**. Nothing is written to disk, so
  there is no temp-file cleanup that can fail.
- EXIF is stripped during decoding. Orientation is the only field read, and
  only to turn the image upright before discarding the rest.
- No uploaded image is ever given a URL. Visual output is embedded as base64 in
  the JSON response.
- No third-party service receives your photographs. Models run locally.
- Face-selection state lives in a TTL'd in-memory store (5 min default, capped,
  256-bit tokens, buffers overwritten on release). Disable with
  `RETAIN_FOR_FACE_SELECTION=false`.
- Upload validation by magic bytes, not declared MIME type. Size, dimension and
  decompression-bomb limits enforced.
- Optional `X-API-Key` auth, per-endpoint rate limiting, security headers,
  non-root containers, read-only root filesystem.

There is **no 1:N identification**. No database of faces, no search. That is
the capability that turns face comparison into surveillance, and it is
deliberately absent.

No sensitive characteristic is inferred — not race, ethnicity, religion,
health, sexuality, character or criminality.

---

## Project layout

```
facet/
├── backend/
│   ├── app/
│   │   ├── config.py              every threshold, env-overridable
│   │   ├── main.py                app factory, middleware, lifespan
│   │   ├── schemas.py             response contract
│   │   ├── api/                   routes and dependencies
│   │   ├── core/                  errors, security, ephemeral store
│   │   ├── models/                backend interface + implementations
│   │   ├── pipeline/              loader, quality, regions, similarity,
│   │   │                          visualise, orchestrator
│   │   └── scoring/               calibration, bands
│   ├── scripts/                   download_models, selfcheck,
│   │                              fit_calibration, evaluate
│   └── tests/
├── frontend/
│   ├── app/                       landing, compare, privacy, methodology
│   ├── components/                Dropzone, ScoreDial, ResultPanel,
│   │                              RegionTable, QualityPanel
│   └── lib/                       api client, types
└── docs/                          ARCHITECTURE, CALIBRATION, ETHICS,
                                   DEPLOYMENT
```

---

## Limits worth stating plainly

- **Uncalibrated by default.** The largest source of error in a fresh install.
- **2D geometry is pose-sensitive.** Ratios are suppressed when poses differ.
- **Occlusion detection is heuristic.** It misfires on shadow, dark-framed
  glasses and heavy facial hair.
- **Demographic performance varies.** Face recognition accuracy differs across
  demographic groups; a single global threshold does not treat everyone
  equally. Evaluate on your own population before deploying.
- **Twins and siblings.** Unresolvable by any model here.
- **The rate limiter is in-process.** Fine for one instance, wrong behind a
  load balancer — see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Licence

Application code: MIT. Model weights carry their own terms — Apache-2.0 for the
OpenCV Zoo models, non-commercial research only for InsightFace. See
[docs/ETHICS.md](docs/ETHICS.md) for use restrictions.
