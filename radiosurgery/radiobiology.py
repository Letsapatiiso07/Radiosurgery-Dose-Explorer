"""Turning physical dose into biological effect.

The linear-quadratic (LQ) model says the share of cells surviving a dose d
is exp(-alpha d - beta d^2). From it come:

- BED, the biologically effective dose: BED = n d (1 + d / (alpha/beta)),
  for n fractions of d Gy.
- EQD2, the dose in 2 Gy fractions with the same effect:
  EQD2 = BED / (1 + 2 / (alpha/beta)).

alpha/beta is low (about 2 to 3 Gy) for slowly responding tissues such as
nerves and slow-growing tumours like schwannomas, so one large dose counts
for far more than the same dose split up. That is why 13 Gy in one session
matches roughly 50 to 60 Gy given the usual way.

At single doses above about 6 to 8 Gy the LQ model may overstate the effect,
because its curve keeps bending. The LQ-L model (Astrahan, 2008) continues in
a straight line beyond a transition dose d_T, with the slope LQ has at d_T.
Both are given, and the gap between them is part of the uncertainty.
"""

import numpy as np


def bed_lq(dose_per_fraction, fractions, ab):
    d = np.asarray(dose_per_fraction, float)
    return fractions * d * (1 + d / ab)


def bed_lql(dose_per_fraction, fractions, ab, transition):
    """LQ below d_T; above it, a straight line that meets LQ smoothly at d_T."""
    d = np.asarray(dose_per_fraction, float)
    lq = d * (1 + d / ab)
    at_t = transition * (1 + transition / ab)
    slope = 1 + 2 * transition / ab
    return fractions * np.where(d <= transition, lq, at_t + slope * (d - transition))


def eqd2(bed, ab):
    return np.asarray(bed, float) / (1 + 2 / ab)


def single_fraction_table(dose, abs_=(2.0, 3.0), transition=6.0):
    rows = []
    for ab in abs_:
        lq = float(bed_lq(dose, 1, ab))
        lql = float(bed_lql(dose, 1, ab, transition))
        rows.append({"alpha_beta": ab, "bed_lq": lq, "eqd2_lq": float(eqd2(lq, ab)),
                     "bed_lql": lql, "eqd2_lql": float(eqd2(lql, ab))})
    return rows
