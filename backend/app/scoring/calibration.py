"""Score calibration.

WHY THIS MODULE EXISTS
----------------------
A cosine similarity is not a probability. Neither is `cosine * 100`. Two
separate things have to happen before a model score can be shown to a person as
a percentage:

1. CALIBRATION. Learn how the score distributes under the "same person"
   hypothesis and under the "different people" hypothesis. The ratio of those
   two densities at an observed score is the likelihood ratio - the amount of
   evidence the score carries, independent of any assumption about how likely a
   match was to begin with.

2. PRIOR COMBINATION. Convert evidence into a probability by combining it with
   a prior. This step is unavoidable and cannot be hidden: the probability that
   two photographs show the same person depends on how the pair was selected.
   A pair drawn from a passport renewal queue has a very different base rate
   from a pair drawn from two unrelated social media accounts.

   posterior_logit = LLR + logit(prior)

By keeping these separate, the system can report the evidence (LLR) honestly
while still producing the percentage the interface needs, with the assumption
made explicit rather than buried.

THE SHIPPED DEFAULTS ARE NOT VALIDATED. They are plausible starting parameters
derived from typical published score distributions for each model family. They
carry `fitted: false`, every API response says so, and they must be refit on
representative data before the numbers mean anything. See docs/CALIBRATION.md
and scripts/fit_calibration.py.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

CalibrationMethod = Literal["gaussian", "logistic"]

# Clamp the LLR so a single comparison can never be presented as overwhelming
# evidence. This is a real limit, not decoration: a likelihood ratio estimated
# from a few thousand validation pairs cannot support odds of 10^6 to 1, because
# the tails of the fitted distributions are extrapolation rather than
# measurement. 4.0 natural-log units is roughly 55:1, which caps the displayed
# similarity at about 98% under a neutral prior.
MAX_ABS_LLR = 4.0


@dataclass
class CalibrationProfile:
    """Parameters mapping a raw similarity score to an interpretable one."""

    name: str
    backend: str
    method: CalibrationMethod = "gaussian"

    # Gaussian parameters
    genuine_mean: float = 0.55
    genuine_std: float = 0.13
    impostor_mean: float = 0.05
    impostor_std: float = 0.08

    # Logistic (Platt) parameters: P(genuine|s) = sigmoid(a*s + b)
    logistic_a: float = 0.0
    logistic_b: float = 0.0
    logistic_train_prior: float = 0.5

    fitted: bool = False
    n_genuine_pairs: int = 0
    n_impostor_pairs: int = 0
    fitted_at: str | None = None
    dataset_description: str = ""
    notes: str = ""
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io ---
    @classmethod
    def from_dict(cls, data: dict) -> CalibrationProfile:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "backend": self.backend,
            "method": self.method,
            "genuine_mean": self.genuine_mean,
            "genuine_std": self.genuine_std,
            "impostor_mean": self.impostor_mean,
            "impostor_std": self.impostor_std,
            "logistic_a": self.logistic_a,
            "logistic_b": self.logistic_b,
            "logistic_train_prior": self.logistic_train_prior,
            "fitted": self.fitted,
            "n_genuine_pairs": self.n_genuine_pairs,
            "n_impostor_pairs": self.n_impostor_pairs,
            "fitted_at": self.fitted_at,
            "dataset_description": self.dataset_description,
            "notes": self.notes,
            "metrics": self.metrics,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # --------------------------------------------------------------- maths ---
    def _gaussian_vertex(self) -> float | None:
        """Score at which the Gaussian LLR turns around, if it turns at all.

        With unequal variances the Gaussian log-likelihood ratio is QUADRATIC in
        the score:

            LLR(s) = a*s^2 + b*s + c,   a = 0.5*(1/si^2 - 1/sg^2)

        Genuine scores are almost always more dispersed than impostor scores
        (sg > si), which makes `a` positive and the parabola open upward. The
        LLR therefore has a minimum and *rises again* as the score falls past
        it - so a cosine of -1.0, meaning maximal dissimilarity, would come out
        as strong support for the same-person hypothesis.

        That is an artefact of extrapolating two Gaussians far outside the data,
        not a real inference. We locate the turning point and hold the LLR flat
        below it: every score that low is treated as equally strong evidence
        against, which is honest, because the model cannot distinguish among
        them.
        """
        genuine_var = max(self.genuine_std, 1e-4) ** 2
        impostor_var = max(self.impostor_std, 1e-4) ** 2
        a = 0.5 * (1.0 / impostor_var - 1.0 / genuine_var)
        if a <= 1e-9:
            return None  # opens downward or is linear: already monotone here
        b = self.genuine_mean / genuine_var - self.impostor_mean / impostor_var
        return -b / (2.0 * a)

    def _raw_gaussian_llr(self, score: float) -> float:
        genuine_std = max(self.genuine_std, 1e-4)
        impostor_std = max(self.impostor_std, 1e-4)
        log_p_genuine = -0.5 * ((score - self.genuine_mean) / genuine_std) ** 2 - math.log(
            genuine_std
        )
        log_p_impostor = -0.5 * (
            (score - self.impostor_mean) / impostor_std
        ) ** 2 - math.log(impostor_std)
        return log_p_genuine - log_p_impostor

    def log_likelihood_ratio(self, score: float) -> float:
        """Natural-log likelihood ratio: ln[ p(score | same) / p(score | different) ].

        Guaranteed monotone non-decreasing in `score`, so a less similar pair can
        never receive a higher similarity result than a more similar one.
        """
        if self.method == "logistic":
            # sigmoid(a*s+b) estimates P(genuine|s) at the training prior, so
            # removing that prior's log-odds recovers the LLR. Linear in s, so
            # monotone by construction (given a >= 0).
            raw = self.logistic_a * score + self.logistic_b
            train_logit = math.log(
                self.logistic_train_prior / (1.0 - self.logistic_train_prior)
            )
            llr = raw - train_logit
        else:
            vertex = self._gaussian_vertex()
            effective_score = score
            if vertex is not None and score < vertex:
                effective_score = vertex
            llr = self._raw_gaussian_llr(effective_score)

        return float(np.clip(llr, -MAX_ABS_LLR, MAX_ABS_LLR))

    def posterior(self, score: float, prior: float = 0.5) -> float:
        """P(same person | score, prior)."""
        prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
        prior_logit = math.log(prior / (1.0 - prior))
        logit = self.log_likelihood_ratio(score) + prior_logit
        return float(1.0 / (1.0 + math.exp(-np.clip(logit, -30, 30))))

    def calibrated_score(self, score: float, prior: float = 0.5) -> float:
        """The 0-100 number shown in the interface."""
        return 100.0 * self.posterior(score, prior)

    def decision_threshold(self, prior: float = 0.5) -> float:
        """Raw score at which the posterior crosses 0.5, by bisection.

        Searches only the monotone region (above the Gaussian turning point),
        so the flat low-score tail cannot be mistaken for a crossing.
        """
        vertex = self._gaussian_vertex() if self.method == "gaussian" else None
        low = vertex if vertex is not None else -1.0
        high = 1.0

        if self.posterior(high, prior) < 0.5:
            return high
        if self.posterior(low, prior) > 0.5:
            return low

        for _ in range(100):
            mid = (low + high) / 2.0
            if self.posterior(mid, prior) < 0.5:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0


# --------------------------------------------------------------------------
# Unvalidated starting parameters
# --------------------------------------------------------------------------
# Approximated from typical published genuine/impostor cosine distributions for
# each model family on general-population benchmarks. They are a starting point
# for a system that has not yet been calibrated, NOT a validated result, and
# emphatically not appropriate for the hard cases this tool targets (large age
# gaps, post-surgical pairs), whose score distributions are shifted relative to
# benchmark data.

_UNVALIDATED_DEFAULTS: dict[str, dict] = {
    "opencv": {
        "genuine_mean": 0.61,
        "genuine_std": 0.19,
        "impostor_mean": 0.17,
        "impostor_std": 0.12,
        "notes": (
            "Unvalidated starting parameters for SFace. Chosen so the posterior "
            "crossover lands at 0.364, essentially OpenCV's published 0.363 "
            "reference threshold, with an implied equal error rate of about 7.8% "
            "- a realistic figure for this model on mixed-difficulty data. "
            "Deliberately NOT tuned to look impressive: a tight, well-separated "
            "pair of distributions would produce a near-step-function score that "
            "reports 5% or 99% with nothing in between, which would be a "
            "presentation artefact rather than a measurement. Refit before "
            "relying on any number."
        ),
    },
    "insightface": {
        "genuine_mean": 0.62,
        "genuine_std": 0.16,
        "impostor_mean": 0.06,
        "impostor_std": 0.08,
        "notes": (
            "Unvalidated starting parameters for ArcFace w600k_r50. Implied "
            "equal error rate about 1.0% with a posterior crossover at 0.262. "
            "That figure reflects general-population benchmark conditions; "
            "performance on large age gaps and post-surgical pairs is materially "
            "worse, so these parameters will overstate the evidence on exactly "
            "the cases this tool targets. Refit on representative data."
        ),
    },
}


def default_profile(backend: str) -> CalibrationProfile:
    params = _UNVALIDATED_DEFAULTS.get(backend, _UNVALIDATED_DEFAULTS["opencv"])
    return CalibrationProfile(
        name="default-unvalidated",
        backend=backend,
        method="gaussian",
        fitted=False,
        dataset_description="None. No validation data has been fitted.",
        **params,
    )


def load_profile(
    calibration_dir: Path, profile_name: str, backend: str
) -> CalibrationProfile:
    """Load `<dir>/<backend>.<profile>.json`, falling back to the unvalidated default."""
    path = Path(calibration_dir) / f"{backend}.{profile_name}.json"
    if not path.exists():
        logger.warning(
            "No calibration file at %s - using UNVALIDATED default parameters. "
            "Scores will be flagged as uncalibrated.",
            path,
        )
        return default_profile(backend)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = CalibrationProfile.from_dict(data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("Calibration file %s is invalid (%s); using defaults.", path, exc)
        return default_profile(backend)

    if profile.backend != backend:
        logger.error(
            "Calibration file %s was fitted for backend '%s' but the active "
            "backend is '%s'. Refusing to apply it; using defaults.",
            path,
            profile.backend,
            backend,
        )
        return default_profile(backend)

    logger.info(
        "Loaded calibration '%s' for backend '%s' (fitted=%s).",
        profile.name,
        backend,
        profile.fitted,
    )
    return profile


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_gaussian(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray
) -> tuple[float, float, float, float]:
    return (
        float(np.mean(genuine_scores)),
        float(max(np.std(genuine_scores), 1e-4)),
        float(np.mean(impostor_scores)),
        float(max(np.std(impostor_scores), 1e-4)),
    )


def fit_logistic(
    genuine_scores: np.ndarray, impostor_scores: np.ndarray
) -> tuple[float, float, float]:
    """Platt scaling by Newton-Raphson on the logistic log-likelihood.

    Implemented directly rather than pulling in scikit-learn for two parameters.
    """
    x = np.concatenate([genuine_scores, impostor_scores]).astype(np.float64)
    y = np.concatenate(
        [np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    ).astype(np.float64)

    a, b = 1.0, 0.0
    for _ in range(200):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gradient = np.array([np.sum((p - y) * x), np.sum(p - y)])
        w = p * (1.0 - p) + 1e-9
        hessian = np.array(
            [
                [np.sum(w * x * x), np.sum(w * x)],
                [np.sum(w * x), np.sum(w)],
            ]
        )
        try:
            step = np.linalg.solve(hessian + np.eye(2) * 1e-8, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
            break
        a -= float(step[0])
        b -= float(step[1])
        if np.max(np.abs(step)) < 1e-9:
            break

    train_prior = float(len(genuine_scores) / max(len(x), 1))
    train_prior = float(np.clip(train_prior, 1e-4, 1 - 1e-4))
    return a, b, train_prior
