# Ethics and use restrictions

## What this tool is

Primarily a one-to-one face **verification** tool. It compares photographs you
already have and reports how similar the faces are under a model.

It also has an optional, off-by-default **one-to-many search** of a single
folder the operator names explicitly (`DATABASE_SEARCH_DIR`) - e.g. searching
your own desktop photo library for the closest match to one photo. This is
scoped deliberately narrow and is not an exception to the line below so much
as its boundary made explicit:

- **Off unless turned on.** `ENABLE_DATABASE_SEARCH=false` by default.
- **One folder, named by the operator.** There is no way to search anything
  else - no crawling, no external data source, no combining folders at
  request time.
- **Never exposed by URL.** Matches return as a server-rendered thumbnail and
  a file path string, the same policy applied to every other image in this
  service.
- **Same evidentiary caveats as 1:1 comparison.** A match is a similarity
  score against a local library, not identification - look-alikes and
  relatives can score highly here exactly as they can in a pairwise
  comparison.

## What it deliberately is not

**No identification against public, shared or third-party data.** Database
search only ever reads the single local folder configured above. There is no
built-in way to point it at a shared drive, a scraped dataset, a public photo
feed, or any data source beyond what the operator names on their own machine.
That broader capability - matching a face against data you do not already
control - is what turns face comparison into surveillance, and its absence
here is a design decision, not an unfinished feature.

**No attribute inference.** No estimation of race, ethnicity, religion, health,
sexuality, character, emotion or criminality. These inferences are not
supported by face data. Systems that claim them are, at best, laundering
correlation from biased training sets; the code to attempt them does not exist
in this repository and should not be added to it.

**No identity determination.** No score, at any threshold, establishes that two
photographs depict the same person. This constraint is enforced in code — see
"Enforced in tests" below — not merely stated in documentation.

---

## Language rules

The system never says, and must never be modified to say:

- "This is the same person"
- "Identity confirmed" / "Verified as the same"
- "Definitely" / "Proves" / "Match confirmed"

It says instead:

> "The photographs show a high degree of facial similarity according to this
> model."

And on every result:

> "This result is probabilistic and should not be treated as proof of identity."

The low band is symmetric about this. A low score does **not** establish that
photographs show different people — large age gaps, surgery, poor quality and
extreme pose all depress the score for genuine pairs — and the guidance text
says so.

---

## Enforced in tests

These are assertions, not aspirations:

| Test | What it prevents |
|---|---|
| `test_no_band_asserts_identity` | Band text containing identity claims |
| `test_response_never_asserts_identity` | The whole serialised response, checked as text |
| `test_low_band_does_not_assert_different_people` | Treating a low score as exclusion |
| `test_high_bands_mention_lookalikes` | Dropping the twin/sibling caveat |
| `test_narrative_never_claims_a_cause` | "The subject has had rhinoplasty" |
| `test_region_disclaimer` | Region analysis being read as identity evidence |
| `test_uncalibrated_state_is_surfaced` | Silently shipping uncalibrated numbers |
| `test_lookalike_caveat_always_present` | The caveat being dropped at high scores |

If you fork this and remove these tests, you are changing what the product
claims about people. Do that deliberately, if at all.

---

## Mandatory disclosures in every result

- The probabilistic disclaimer
- Whether calibration is validated
- The prior used to derive the percentage
- The raw score and likelihood ratio, not only the percentage
- Image quality for both sides
- A ranked list of uncertainty sources
- The look-alike and relative caveat, at every score level

The look-alike caveat is unconditional. It appears at 98% as it appears at 12%,
because that is precisely when it is easiest to forget and most consequential.

---

## Legal considerations

Biometric processing of a face is regulated in many jurisdictions, and in
several it requires the subject's explicit consent regardless of how the
photograph was obtained.

Being able to see an image is not the same as being permitted to run biometric
analysis on it.

Frameworks that may apply include the EU GDPR (Article 9 — biometric data for
unique identification is a special category), the EU AI Act, Illinois BIPA,
Texas CUBI, and the biometric provisions of several US state privacy statutes.
BIPA in particular provides a private right of action, and its damages are not
nominal.

**This software gives you no legal basis you did not already have.**
Establishing one is your responsibility. Consult a lawyer before deploying in a
regulated context, and before processing anyone's face but your own. If you
turn on database search, this applies to every photograph in the indexed
folder, not only the query photo - biometric processing of a face someone
else appears in requires the same basis whether it is compared once or
indexed for repeated searching.

---

## Uses this tool should not be put to

- Accusing someone of being a person they deny being
- Deciding employment, housing, credit, insurance or immigration outcomes
- Covert investigation of a private individual
- Unmasking someone who has deliberately changed their appearance — including
  people who have transitioned, fled abuse, or entered witness protection
- Anything where a wrong answer harms someone and they have no way to contest
  it

That last category is the general case, and it is the one to reason from.

A specific note on the tool's own framing: comparing pre- and post-surgical
photographs is a legitimate technical problem, and it is also a capability that
can be used to strip someone of a change they made deliberately and at cost.
The same is true of ageing comparison. Holding both photographs does not mean
holding the right to link them.

---

## Known limitations that affect fairness

**Demographic performance varies.** Face recognition accuracy differs across
demographic groups — documented extensively in the NIST FRVT programme. A
single global threshold does not treat everyone equally: the same cutoff
produces different false-match rates for different populations. Evaluate on
your own population, per group, before deploying.

**Twins and close relatives.** Unresolvable by any model here. A very high
score is entirely consistent with a sibling.

**Calibration transfers badly.** A calibration fitted on one population,
camera, or difficulty level will misreport on another. See
[CALIBRATION.md](CALIBRATION.md).

**Quality correlates with circumstance.** People photographed in worse
conditions — older devices, poorer lighting, less control over the setting —
get lower-quality images, which compress genuine and impostor scores toward
each other. The quality gate partly protects against this by refusing rather
than guessing, but the correlation itself is a source of unequal treatment.

---

## If you deploy this

1. Calibrate on representative data. Not optional.
2. Evaluate per demographic group, not only in aggregate.
3. Keep every disclaimer. They are the product, as much as the score is.
4. Give people a route to contest a result, and staff it.
5. Log decisions, not photographs.
6. Set a retention policy and honour it. The default is zero retention; adding
   storage means revisiting every claim on the privacy page.
7. Do not let the number stand alone. A percentage stripped of its context
   becomes a verdict, which is exactly what this system is built not to issue.
