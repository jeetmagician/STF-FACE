"""Score banding and the wording that goes with each band.

Every phrase here is constrained by one rule: the system reports similarity
under a model, never identity. There is no band whose text asserts that two
photographs show the same person, and there is no threshold above which the
language becomes conclusive.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass
class Band:
    key: str
    label: str
    statement: str
    guidance: str


# The verbal descriptions of evidence strength are modelled on the scales used
# in forensic comparison reporting, where a likelihood ratio is described in
# words rather than asserted as a conclusion.
#
# The cut points are expressed as fractions of MAX_ABS_LLR rather than as fixed
# log-units, so they stay coherent with the clamp. With a fixed scale and a
# clamp of 4.0, the strongest result the system can produce would read as only
# "moderate" while the band label said "Very High Similarity" - two parts of
# the same response contradicting each other.
def describe_evidence_strength(llr: float) -> str:
    from app.scoring.calibration import MAX_ABS_LLR

    magnitude = abs(llr)
    if magnitude < 0.7:
        return (
            "This comparison is essentially uninformative: it barely shifts the "
            "odds either way."
        )

    if magnitude < 0.40 * MAX_ABS_LLR:
        strength = "limited"
    elif magnitude < 0.75 * MAX_ABS_LLR:
        strength = "moderate"
    else:
        strength = "strong"

    direction = (
        "the same-person hypothesis" if llr > 0 else "the different-people hypothesis"
    )
    odds = round(pow(2.718281828, magnitude))
    ceiling_note = ""
    if magnitude >= MAX_ABS_LLR - 1e-6:
        ceiling_note = (
            " This is the ceiling the system will report from a single "
            "comparison; the underlying score may be more extreme, but "
            "finite validation data cannot support a stronger claim."
        )
    return (
        f"The similarity score provides {strength} support for {direction}: "
        f"roughly {odds}:1 in its favour, under this model and calibration."
        f"{ceiling_note}"
    )


def band_for(calibrated_score: float, settings: Settings) -> Band:
    if calibrated_score >= settings.band_very_high:
        return Band(
            key="very_high",
            label="Very High Similarity",
            statement=(
                "The photographs show a very high degree of facial similarity "
                "according to this model."
            ),
            guidance=(
                "Scores in this range are most often produced by images of the same "
                "person, but look-alikes, close relatives and identical twins can "
                "also score here. This is not proof of identity."
            ),
        )
    if calibrated_score >= settings.band_high:
        return Band(
            key="high",
            label="High Similarity",
            statement=(
                "The photographs show a high degree of facial similarity according "
                "to this model."
            ),
            guidance=(
                "Consistent with the same person, but also reachable by similar-"
                "looking individuals. Consider additional photographs before drawing "
                "any conclusion."
            ),
        )
    if calibrated_score >= settings.band_moderate:
        return Band(
            key="moderate",
            label="Moderate Similarity",
            statement=(
                "The photographs show a moderate degree of facial similarity "
                "according to this model."
            ),
            guidance=(
                "This range is genuinely ambiguous. It is reached both by the same "
                "person photographed under difficult conditions and by different "
                "people who resemble one another. Additional images would help more "
                "than re-reading this score."
            ),
        )
    return Band(
        key="low",
        label="Low Similarity",
        statement=(
            "The photographs show a low degree of facial similarity according to "
            "this model."
        ),
        guidance=(
            "A low score does not establish that the photographs show different "
            "people. Large age gaps, facial surgery, poor image quality and extreme "
            "pose differences all depress the score for genuine pairs."
        ),
    )


UNIVERSAL_DISCLAIMER = (
    "This result is probabilistic and should not be treated as proof of identity."
)

METHOD_STATEMENT = (
    "The score expresses similarity as measured by the selected face-recognition "
    "model and its current calibration. It is not a measure of identity, and it "
    "carries no legal or evidential weight."
)

UNCALIBRATED_WARNING = (
    "This system is running on unvalidated default calibration parameters. The "
    "percentage shown is not meaningful until calibration is fitted on data "
    "representative of your use case. See docs/CALIBRATION.md."
)
