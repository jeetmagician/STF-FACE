# Architecture

## Data flow

```
Browser
  │  multipart upload (1-5 photos per side)
  ▼
FastAPI
  │  magic-byte validation, size/dimension limits, bomb guard
  ▼
decode_image            in-memory BGR array; EXIF stripped, orientation applied
  ▼
detect                  YuNet or SCRFD; several faces -> 409, never a silent pick
  ▼
align                   Umeyama similarity transform onto the ArcFace template
  ▼
quality gate            blur, exposure, size, pose, occlusion, detection score
  │                     FAILS HERE -> refusal, not a score
  ▼
embed                   SFace 128-d or ArcFace 512-d, L2-normalised
  ▼
template                quality-weighted mean across photos on each side
  ▼
cosine similarity
  ▼
calibration             score -> log-likelihood ratio
  ▼
prior                   LLR + logit(prior) -> posterior -> 0-100
  ▼
band + narrative + region analysis + visualisations
  ▼
JSON (base64 images inline; no image is ever given a URL)
```

The quality gate sits **before** embedding on purpose. Once a number exists it
will be read as authoritative regardless of what produced it, so inputs that
cannot support a defensible comparison are refused rather than scored.

---

## Why these models

### The licensing problem comes first

InsightFace's ArcFace-R50 is the better model for this task. Its *code* is MIT.
Its *pretrained weights* are released for non-commercial research use only,
covering both manual downloads and the library's automatic download. Commercial
deployment requires a licence from InsightFace or your own weights.

That makes it the wrong default for a product prototype. The default is
therefore OpenCV Zoo's YuNet + SFace, both Apache-2.0: ~40 MB, no extra
dependencies (`FaceDetectorYN` and `FaceRecognizerSF` ship in OpenCV's main
`objdetect` module), and legally shippable. It is measurably weaker on large
age gaps and post-surgical pairs, which is a real cost, stated rather than
hidden.

Both sit behind `FaceEmbedder` in `app/models/base.py`. Replacing the model
means implementing two methods and registering the class; nothing outside
`app/models/` changes.

### Why embeddings rather than pixels

Two photographs of one person can differ in every pixel while depicting an
obviously identical face. Recognition networks are trained with margin-based
losses so that images of one identity cluster tightly and different identities
separate, which makes cosine similarity in embedding space the natural metric —
angular margin training optimises exactly that geometry.

### Why one alignment template for both backends

Both backends align to the same 5-point 112×112 ArcFace template. OpenCV's
`FaceRecognizerSF::alignCrop` uses that identical template internally, so
feeding it our own aligned crop is consistent with how SFace was trained, and
canonical region boxes mean the same thing across backends.

Keypoint ordering is normalised by **sorting on x** rather than trusting each
model's documented order. If a future release changes its convention, the code
stays correct instead of silently mirroring every face.

---

## Scoring

### The problem with a percentage

A cosine of 0.42 is not "42% likely to be the same person". It is a distance in
a learned space whose meaning depends on the model, the population, the capture
conditions, and pair difficulty. The same 0.42 may be a comfortable match for
well-lit frontal pairs a year apart, and thoroughly ambiguous across fifteen
years.

### Likelihood ratio, then prior

Two distributions are estimated — genuine and impostor — and the ratio of their
densities at the observed score is the evidence:

```
LR = p(score | same) / p(score | different)
```

This is prior-free. The percentage adds the prior explicitly:

```
logit(posterior) = ln(LR) + logit(prior)
```

`PRIOR_SAME_PERSON` defaults to 0.5, which is deliberately neutral and *not* an
empirical base rate. It is printed with every result rather than buried in a
constant, because it genuinely changes the answer and the caller is the only
one who knows how their pairs were selected.

### The monotonicity trap

With unequal variances the Gaussian LLR is **quadratic** in the score:

```
LLR(s) = a·s² + b·s + c,     a = ½(1/σᵢ² − 1/σ_g²)
```

Genuine scores are almost always more dispersed than impostor scores
(σ_g > σᵢ), which makes `a` positive. The parabola opens upward, so the LLR has
a minimum and **rises again** as the score falls past it. Left uncorrected, a
cosine of −1.0 — maximal dissimilarity — comes out as strong support for the
same-person hypothesis.

That is an artefact of extrapolating two Gaussians far outside the data, not an
inference. `_gaussian_vertex()` locates the turning point and the LLR is held
flat below it: every score that low is treated as equally strong evidence
against, because the model genuinely cannot distinguish among them.

`test_calibration.py` and `selfcheck.py` both assert monotonicity across the
full [−1, 1] range, including a deliberately pathological σ configuration.

### The evidence ceiling

`MAX_ABS_LLR = 4.0` (≈55:1), capping the displayed score near 98%. Estimating a
ratio of 10⁶ would require characterising distribution tails about which a few
thousand validation pairs say nothing — anything reported out there is
extrapolation dressed as measurement.

The verbal strength scale is expressed as *fractions* of that ceiling rather
than fixed log-units. With a fixed scale and a clamp at 4.0, the strongest
result the system can produce would read "moderate support" while the band
label said "Very High Similarity" — two parts of one response contradicting
each other.

### Shipped defaults

| Backend | μ_gen | σ_gen | μ_imp | σ_imp | Implied EER | Crossover |
|---|---|---|---|---|---|---|
| opencv | 0.61 | 0.19 | 0.17 | 0.12 | ~7.8% | 0.364 |
| insightface | 0.62 | 0.16 | 0.06 | 0.08 | ~1.0% | 0.262 |

The opencv crossover lands at 0.364, essentially SFace's published 0.363
reference threshold.

These are **starting points, not measurements**, and are marked
`fitted: false`. They were deliberately not tuned to look impressive: tightly
separated distributions would produce a near-step-function score reporting 5%
or 99% with nothing between, which is a presentation artefact rather than a
measurement. See [CALIBRATION.md](CALIBRATION.md).

---

## Multi-photo templates

Both sides accept 1–5 photographs, pooled by quality-weighted averaging of
L2-normalised embeddings, then renormalised — the standard set-based approach.
Averaging suppresses per-image noise; quality weighting stops a poor frame from
dragging the template around.

The full pairwise matrix is reported alongside the pooled score, because a wide
spread is itself diagnostic: it means the photographs disagree with one
another, which a single pooled number hides. Spread above 0.25 raises a
warning.

---

## Region analysis

### What cannot be done

The brief asked for stable regions to be weighted more heavily than the nose.
A global embedding **cannot be decomposed into per-region contributions** — the
vector has no region-indexed structure. There is no operation that weights the
periocular region inside a single 512-d embedding, and a system claiming to do
so is inventing an interpretation the model does not support.

Feeding region crops back through the network is also unsound: a periocular
crop is out-of-distribution for a model trained on full aligned faces, and its
similarity would need its own calibration to mean anything.

### What is done instead

Two signals, kept separate and labelled differently:

**Global embedding similarity** — the identity signal, and the only thing that
feeds the score. Recognition networks are already substantially nose-invariant,
being trained across enormous variation in pose, expression and appearance.

**Region analysis** — descriptive only, never fused into the score:

- *Appearance*: gradient-orientation descriptors (HOG-like, 9 bins over a 2×2
  cell grid) over canonical regions of the aligned crop, compared by cosine.
  Gradient orientation is far more robust to lighting than raw intensity, which
  matters because most before/after pairs differ in illumination. Histogram
  equalisation first, so a uniform exposure difference cannot masquerade as
  morphological change.
- *Geometry*: anthropometric ratios normalised by inter-ocular distance,
  measured in original image space and projected into an eye-line-aligned frame
  so roll does not contaminate vertical measurements.

Dense-landmark measures (face width, jaw width, facial index) select points by
**geometric extreme** rather than hardcoded index, so they stay correct if a
backend changes its landmark ordering — and they are simply absent on the
5-keypoint OpenCV backend rather than silently wrong.

### Stability weights

Heuristic persistence ranking, used to order the narrative, never to weight a
score:

| Region | Weight | |
|---|---|---|
| Periocular | 0.95 | most persistent |
| Eyes | 0.90 | |
| Forehead | 0.80 | |
| Mouth | 0.60 | |
| Cheeks | 0.55 | soft tissue, weight-sensitive |
| Jaw / chin | 0.50 | |
| Nose | 0.35 | rhinoplasty and ageing both alter it |

### Pose sensitivity

2D ratios are measured in the image plane and distort under pose. Above 15°
pose difference, geometry is dropped from the narrative and the response says
why — otherwise a projection artefact reads as morphological change.

### Language

The narrative never asserts a cause. "The nasal region differs" is supportable;
"the subject has had rhinoplasty" is not. A test asserts that the narrative
contains none of "has had", "underwent", "surgery was", "definitely", "proves".

---

## Quality gating

Measured on the aligned 112×112 crop wherever possible, so values are
comparable across image sizes.

| Metric | Method |
|---|---|
| Blur | Variance of Laplacian |
| Exposure | Mid-grey deviation × contrast × clipping, as a product so one severe problem is not averaged away |
| Face size | Bounding-box width in source pixels |
| Pose | solvePnP against a generic 3D model, folded to a sane range |
| Occlusion | Periocular darkness + flatness (glasses); lower-vs-upper texture ratio (mask) |
| Detection | Model confidence |

Composite is a weighted sum on 0–100. Hard gates refuse; advisory thresholds
warn. Every threshold is in `config.py` and env-overridable.

Occlusion detection is explicitly heuristic — it misfires on heavy shadow,
dark-framed glasses and facial hair — so it is reported as "suspected" and
never silently alters the score.

---

## Privacy architecture

No filesystem path exists for an uploaded image. Bytes arrive, are decoded to a
numpy array, processed, and released. This removes an entire class of bug:
there is no temp-file cleanup that can fail, no upload directory to
misconfigure, no stale file to leak.

The one exception is the face-selection store (`core/store.py`), which holds
decoded arrays in process memory so choosing a face does not require
re-uploading. Capped entries, 256-bit tokens, TTL sweep on a background task as
well as on access, buffers zeroed on release, cleared on shutdown, and
disableable entirely.

Visual output is base64-embedded in the JSON response. No uploaded image is
addressable by URL, so there is no address at which one could be fetched.

---

## Error handling

Typed errors in `core/errors.py`, each with a stable machine-readable code so
the frontend branches on type rather than string-matching prose.

| Condition | Status | Code |
|---|---|---|
| No face | 422 | `no_face_detected` |
| Several faces | 409 | `multiple_faces_detected` (+ thumbnails, token) |
| Quality failure | 422 | `insufficient_quality` (+ reasons) |
| Bad format / corrupt | 422 | `unsupported_format`, `corrupted_image` |
| Too large | 422 | `file_too_large`, `image_too_large` |
| Rate limited | 429 | `rate_limited` (+ `Retry-After`) |
| Missing key | 401 | `unauthorised` |
| Expired session | 404 | `session_expired` |
| Model unavailable | 503 | `model_unavailable` |

Unexpected exceptions are logged in full and returned as a generic 500 —
internal exception text leaks paths and library versions.

---

## Extension points

**New model**: implement `FaceEmbedder`, register in `models/registry.py`.
Calibration is per-backend and a profile fitted for one backend is refused for
another, so a swap cannot silently inherit the wrong mapping.

**New quality metric**: add to `QualityReport`, wire into `assess_quality`, add
thresholds to `config.py`.

**New region**: add to `CANONICAL_REGIONS` and `REGION_STABILITY`. Both
appearance comparison and the heatmap pick it up automatically.

**Persistent storage**: currently none, and adding it means revisiting every
claim on the privacy page.
