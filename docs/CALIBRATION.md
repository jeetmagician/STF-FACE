# Calibration

## Why this cannot be shipped in the box

A face model outputs a similarity score whose distribution depends on the
model, the population, the capture conditions, and the difficulty of the pairs.
There is no universal mapping from cosine similarity to "percent chance of the
same person", and any product hard-coding one is presenting an assumption as a
measurement.

Concretely: a score of 0.45 might be a comfortable match for well-lit frontal
pairs taken a year apart, and genuinely ambiguous for pairs separated by
fifteen years. Same number, different meaning. Only data from your population
can tell you which you have.

This bites hardest on exactly the cases this tool targets. Public benchmarks
are dominated by easy pairs. Photographs separated by many years, or by facial
surgery, sit in a shifted part of the score distribution. **A system calibrated
on benchmark-like data and pointed at hard cases will systematically overstate
the evidence against genuine pairs** — it will tell you two photographs show
different people when they show one person who has aged or had surgery.

That failure mode is the reason this document exists.

---

## What ships, and what it is

Documented starting parameters, marked `fitted: false`:

| Backend | μ_gen | σ_gen | μ_imp | σ_imp | Implied EER | Crossover |
|---|---|---|---|---|---|---|
| opencv | 0.61 | 0.19 | 0.17 | 0.12 | ~7.8% | 0.364 |
| insightface | 0.62 | 0.16 | 0.06 | 0.08 | ~1.0% | 0.262 |

The opencv crossover sits at 0.364 — essentially SFace's published 0.363
reference threshold — so an uncalibrated system at least makes its yes/no
decision at the model author's recommended operating point.

They were deliberately **not** tuned to look impressive. Tightly separated
distributions would produce a near-step-function: 5% below the threshold, 99%
above, nothing in between. That looks decisive and is a presentation artefact.
The shipped spreads produce a gradual curve that honestly reflects how much
these models actually overlap.

They remain assumptions. Every response says so:

```json
"calibration": { "is_validated": false, ... },
"warnings": ["This system is running on unvalidated default calibration..."],
"uncertainty_sources": [{ "factor": "Uncalibrated model", "severity": "high", ... }]
```

---

## Building a validation set

### Format

```csv
image_a,image_b,label,condition
alice/2019.jpg,alice/2024.jpg,1,age_gap_5y
alice/2019.jpg,bob/2021.jpg,0,different_people
carol/clean.jpg,carol/beard.jpg,1,facial_hair
dave/thin.jpg,dave/heavier.jpg,1,weight_change
erin/pre.jpg,erin/post.jpg,1,post_surgical
frank/a.jpg,grace/b.jpg,0,lookalike
```

`label`: 1 = genuine (same person), 0 = impostor. `condition` is optional and
enables per-subgroup reporting.

### How many

Several hundred per class is a workable minimum for the Gaussian method; the
logistic method wants more. Below ~200 per class the fitted parameters are
themselves noisy enough that the resulting percentages carry false precision,
and `fit_calibration.py` warns accordingly.

### Composition matters more than volume

Ten thousand easy pairs calibrate you for easy pairs. Compose the set to
reflect the conditions you expect:

- **Age gaps** at the ranges you will see. Five years and twenty years are
  different problems.
- **Lighting variation** — indoor/outdoor, flash/ambient, harsh/soft.
- **Facial hair** appearing and disappearing.
- **Weight change**, which alters cheek and jaw soft tissue.
- **Hairstyle and makeup**.
- **Post-surgical pairs**, if your use case involves them and you can obtain
  them lawfully and with consent. This is the hardest category to source
  ethically; if you cannot, say so in `--description` rather than pretending
  the calibration covers it.
- **Look-alikes** as hard negatives. Random impostor pairs are too easy and
  will flatter your false-accept rate. Include siblings where you have them.

### Demographics

Face recognition accuracy varies across demographic groups. A validation set
skewed toward one group yields a calibration that is wrong for the others —
and, because it is a single global mapping, wrong in a way that produces
systematically different error rates for different people.

If your deployment serves a broad population, your validation set must too, and
you should evaluate per-group as well as in aggregate.

---

## Fitting

```bash
cd backend

python scripts/fit_calibration.py pairs.csv \
    --root /data/faces \
    --method gaussian \
    --profile production \
    --description "1,842 pairs, internal KYC set 2021-2026, ages 18-75, mixed lighting"

export CALIBRATION_PROFILE=production
```

Writes `backend/assets/calibration/<backend>.<profile>.json`.

`--description` is not decoration. Six months from now it is the only record of
what the numbers actually mean, and whether they still apply.

### Gaussian or logistic

**Gaussian** (default) fits a normal to each class. Two parameters per class,
so it works with less data, and it models the full distribution — but it
assumes normality, and the tails are extrapolation. The monotonicity correction
described in [ARCHITECTURE.md](ARCHITECTURE.md#the-monotonicity-trap) exists
because of this.

**Logistic** (Platt scaling) fits the posterior directly. Monotone by
construction, makes no distributional assumption, and is usually better
calibrated in the middle where decisions actually get made — but it needs more
data and tells you nothing about the tails.

Start with Gaussian. Move to logistic once you have a few thousand pairs.

### Backend binding

A profile records which backend it was fitted for, and `load_profile` refuses
to apply a mismatched one, falling back to defaults with a loud log line.
Applying an ArcFace calibration to SFace scores would produce numbers that look
fine and are meaningless.

---

## Evaluating

Always on a **held-out** set. Metrics from the fitting set are optimistic.

```bash
python scripts/evaluate.py heldout.csv --root /data/faces \
    --plot roc.png --by-condition
```

### What you get

- **EER** — where false-accept equals false-reject. A single-number summary.
- **AUC** — ranking quality, independent of any threshold.
- **TAR @ FAR** at 0.1%, 1%, 5% — how biometric systems are actually
  specified in practice.
- **FAR / FRR at the operating point**.
- **Precision / recall / F1**.
- **Brier score** — calibration quality. Discrimination (AUC) and calibration
  are different things: a model can rank perfectly and still report badly
  wrong probabilities.

EER and AUC are calibration-independent, so they remain meaningful even on an
uncalibrated system.

### Use --by-condition

This is the important one.

An aggregate AUC of 0.98 can conceal near-chance performance on the subgroup
you actually care about. Aggregate numbers are the easiest way to ship a system
that fails precisely where it matters most — and the per-condition breakdown is
how you find out before your users do.

If `post_surgical` shows an EER of 0.31 while the aggregate shows 0.04, you do
not have a working system for post-surgical comparison. You have a working
system for easy pairs and a misleading number for hard ones.

---

## Choosing the prior

`PRIOR_SAME_PERSON` is the probability, before looking at the photographs, that
a submitted pair is genuine. It does not affect the likelihood ratio; it
affects the percentage derived from it.

| Setting | Reasonable prior |
|---|---|
| Document renewal, where most pairs really are the same person | 0.8–0.95 |
| General-purpose comparison tool | 0.5 (the neutral default) |
| Investigating whether two unrelated accounts share a person | 0.05–0.2 |

If you cannot justify a value, leave it at 0.5 and read the likelihood ratio
instead — that is the number that does not depend on this choice.

---

## Database search uses the same calibration

A database-search match score goes through the identical
`calibrated_score` / `band_for` path as a Compare-page result, using
whichever backend actually ran that search (see
[ARCHITECTURE.md](ARCHITECTURE.md#database-search) for when that is the
fallback rather than the primary backend). There is no separate calibration
for ranked search results: if your calibration is unvalidated, every score
on the search results list is exactly as unvalidated as a Compare result,
and the interface says so in both places.

## Recalibrating

Refit when:

- the model backend changes (mandatory — the refusal above enforces it)
- your population shifts materially
- capture conditions change (new camera, new lighting, new intake process)
- measured error rates drift from your validation estimates

Keep old profiles. `CALIBRATION_PROFILE` switches between them, which makes
comparing two calibrations on the same held-out set a one-line change.
