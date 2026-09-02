"""
2_changepoint_detection.py

When did the AUD/JPY triangular basis change structure, and did it change
back?

The unit of analysis is the hour. For each open-market hour the dispersion
of the basis is summarised by its median absolute deviation, the diurnal
profile is removed, and the resulting series of log dispersions is modelled
as piecewise Gaussian.

Four models are compared:

    M0  no change                 one segment
    M1  permanent shift           one changepoint
    M2  excursion                 two changepoints, the level returns to
                                  the pre-shock segment's parameters
    M3  two free changes          two changepoints, three free segments

Including M0 is what makes the exercise a test: a one-changepoint model
always returns a changepoint, so fitting M1 alone answers a question the
data was never asked. Including M2 asks the substantive one — did the
market settle into a new regime, or pass through a bounded stress window
and return. Section 01 compared July against August; that split is a
calendar assumption, and if M2 wins it is the wrong one.

Estimation
    Each segment carries a Normal-Inverse-Gamma prior on (mu, sigma^2), so
    its marginal likelihood is available in closed form. There are at most
    two changepoints and they are discrete, so the posterior is computed by
    enumerating every admissible configuration rather than sampled. The
    result is exact: no chains, no convergence diagnostics, and identical
    numbers on every run. Model evidences come out of the same enumeration,
    which is why the model comparison costs nothing extra.

    No candidate date is supplied to any model. The BOJ decision appears in
    the figures as a reference line and enters no likelihood.

Assumptions, and why they are defensible here
    Gaussian log dispersion. The basis is leptokurtic, so hourly sd is
    driven by a handful of seconds; MAD is not. log MAD over ~3600
    observations per hour is close to symmetric, and the log makes the
    shift multiplicative, which is the natural scale for a volatility
    change.

    Diurnal profile as a nuisance. FX dispersion has a large session
    cycle. It is removed by subtracting the median log MAD by hour of day,
    estimated over the full sample so the adjustment is not conditioned on
    any candidate date. A common profile cannot manufacture a level shift;
    it can only shrink one, so this works against the finding.

    Segments are contiguous in retained hours, not in calendar time.
    Weekends are already absent. A segment may span one, which is
    immaterial for the level of dispersion but would not be for a model
    of the path.

Outputs
    output/data/02_regimes.parquet      regime label and probability per second
    output/tables/02_changepoint.csv    headline dates, intervals, ratios
    output/tables/02_model_evidence.csv
    output/tables/02_regime_stats.csv
    output/tables/02_sensitivity.csv
    output/figures/02_changepoint_posterior.png
    output/figures/02_regime_contrast.png
"""

from __future__ import annotations

from typing import NamedTuple

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import lru_cache

from scipy.special import gammaln, logsumexp

from utils import (
    make_dirs, set_style, in_rollover, robust_scale, acf1, roll_noise,
    noise_share, survival, save_table, save_fig, panel_title, annotate_event,
    annotate_interval, highlight_span,
    adjacent, check_grid,
    DAT_DIR, BOJ_SHOCK, TZ_LABEL, ROLLOVER_START, ROLLOVER_END,
    REGIME_COLOUR, regime_colours, MUTE, RULE,
)

# ------------------------------------------------------------- constants
#
# Every number below is one of three things, and saying which is the point.
# A reader who cannot tell a derived quantity from a tuned one has to
# assume the worst, and a filter deciding which buckets the model ever sees
# is exactly where a tuned threshold would hide.
#
#   derived      computed from something else in the project, so the two
#                cannot drift apart
#   convention   a stated choice with no data content, carried into
#                02_sensitivity so the reader sees it does not matter
#   measured     established from the data — none here; the measured
#                constants of this project live in utils.py
#
BUCKET = "1h"                    # convention; 30min and 2h in 02_sensitivity

# derived. A bucket is judged on the share of the seconds it *could* have
# held, not on a flat share of its length. The rollover window removes a
# known, fixed part of whichever bucket contains it, so charging that
# removal against coverage would mark one bucket a day defective for
# obeying the exclusion rule. The denominator is computed per bucket by
# `schedulable_seconds`, which leaves MIN_COVERAGE meaning one thing only:
# how much of the available time must actually be present.
MIN_COVERAGE = 0.90              # convention; varied in 02_sensitivity

# convention. Below roughly one move a minute the MAD is quantised by the
# tick grid rather than measuring dispersion. Stated as a count because
# estimator stability depends on the count; a share would silently change
# meaning whenever BUCKET changed.
MIN_MOVES_PER_BUCKET = 60        # varied in 02_sensitivity

# convention. A floor on the shortest regime the model may report, and so
# a statement about what counts as a regime rather than an episode.
# Episodes are 01's unit and run to minutes; six hours is well clear of
# them without reaching the multi-day scale the models are meant to find.
MIN_SEG_HOURS = 6.0              # varied at 3, 12 and 24 h in 02_sensitivity

CRED_MASS = 0.95                 # convention

# convention. Grid for the AR(1) error coefficient, marginalised under a
# uniform prior. Includes 0, so the independent-error model is nested and
# the data is free to return it. The upper end stops short of 1: at rho
# near unity a level shift and a random walk are indistinguishable, and
# admitting that region buys nothing but an unidentified mode.
RHO_GRID = np.linspace(0.0, 0.90, 8)

# convention. Buckets after a session reopen treated as a thin-liquidity
# ramp and given their own additive offset in `deseason`. Four hours covers
# the Sunday evening reopen before Tokyo arrives. Varied in 02_sensitivity,
# including 0, which switches the adjustment off entirely.
REOPEN_BUCKETS = 4

# convention. Null-calibration replicates. 200 is enough to place the
# observed evidence against the null to a couple of percent, which is all
# the precision the statement needs.
# Set to 0 while iterating: it is roughly three quarters of the runtime
# and nothing else depends on it.
# Set to 0 while iterating: the calibration is roughly 70% of the runtime
# and nothing else depends on it. The environment variable exists so the
# test harness can run every model path cheaply without editing the file.
NULL_DRAWS = int(os.environ.get("CP_NULL_DRAWS", 100))
NULL_SEED = 20240731

# Prior. k0 is the weight on the segment mean expressed in observations, so
# 0.01 is one hundredth of an hour's worth of information: weak, but proper,
# which matters because improper priors make the model comparison
# meaningless. a0 = 2 is the smallest shape giving a finite prior mean for
# sigma^2. b0 is set from the sample dispersion of the series being fitted,
# so the prior scales with the feature rather than assuming its units —
# empirical Bayes, and worth stating, because that dispersion is inflated
# by the very shift being tested and so shrinks the evidence for a change.
# Sensitivity to k0 is reported.
PRIOR_K0 = 0.01                  # convention; varied in 02_sensitivity
PRIOR_A0 = 2.0                   # convention

MODELS = ["M0 no change", "M1 permanent shift", "M2 excursion",
          "M3 two free changes"]
REGIME_NAMES = {
    "M0 no change": ["single regime"],
    "M1 permanent shift": ["before", "after"],
    "M2 excursion": ["calm", "stressed"],
    "M3 two free changes": ["segment 1", "segment 2", "segment 3"],
}


class Fit(NamedTuple):
    index: pd.DatetimeIndex   # bucket start times, one per retained bucket
    log_evidence: pd.Series   # by model
    p_model: pd.Series        # posterior model probability, equal priors
    best: str
    seg_prob: np.ndarray      # (n_buckets, n_regimes) pointwise posterior
    p_onset: np.ndarray       # marginal posterior over the first change
    p_return: np.ndarray      # marginal over the second, or None
    onset: pd.Timestamp
    onset_lo: pd.Timestamp
    onset_hi: pd.Timestamp
    ret: pd.Timestamp
    ret_lo: pd.Timestamp
    ret_hi: pd.Timestamp
    rho: float                # posterior mean AR(1) coefficient of the errors
    contiguous: np.ndarray    # bucket follows the previous one in clock time


# ------------------------------------------------------- marginal likelihood

def _segment_evidence(n, sww, swz, szz, m0, k0, a0, b0):
    """
    log p(segment) with (mu, sigma^2) integrated out under a
    Normal-Inverse-Gamma prior.

    The segment is written as a one-regressor regression z_t = mu w_t + e_t,
    e_t ~ N(0, sigma^2). With w_t = 1 this is the plain iid segment mean and
    the formula reduces to the familiar one; the general form is what lets
    the AR(1) transform below reuse it unchanged.

    Vectorised over segments. n, sum w^2, sum wz and sum z^2 are each
    obtainable in O(1) from cumulative sums, which is what makes exhaustive
    enumeration cheap enough to be worth preferring over a sampler.
    """
    n = np.asarray(n, dtype=float)
    kn = k0 + sww
    an = a0 + n / 2.0
    mn = (k0 * m0 + swz) / kn
    bn = b0 + 0.5 * (szz + k0 * m0 ** 2 - kn * mn ** 2)
    bn = np.maximum(bn, np.finfo(float).tiny)
    return (gammaln(an) - gammaln(a0)
            + a0 * np.log(b0) - an * np.log(bn)
            + 0.5 * (np.log(k0) - np.log(kn))
            - 0.5 * n * np.log(2.0 * np.pi))


def ar_transform(v, contiguous, rho):
    """
    Whiten an AR(1) series into z_t = mu w_t + e_t with e_t iid N(0, s^2).

    Two cases. Where the previous bucket is adjacent in clock time the
    one-step-ahead form applies, z = y - rho y_prev against w = 1 - rho.
    Where it is not — the first bucket of the sample, and the first bucket
    after every closure — there is no predecessor, so the stationary
    marginal is used instead, scaled by sqrt(1 - rho^2) to put it on the
    same sigma. Both are exact; mixing them is what makes the likelihood
    correct across a market that shuts every weekend.

    Returned as (conditional, stationary) pairs so that a segment can use
    the stationary form at its own first bucket, whose predecessor belongs
    to a different segment with a different mean.
    """
    v = np.asarray(v, dtype=float)
    root = np.sqrt(max(1.0 - rho ** 2, 1e-12))
    z_s = v * root
    w_s = np.full(v.size, root)

    z_c = np.empty_like(v)
    w_c = np.empty_like(v)
    z_c[0], w_c[0] = z_s[0], w_s[0]
    z_c[1:] = v[1:] - rho * v[:-1]
    w_c[1:] = 1.0 - rho
    gap = ~np.asarray(contiguous, dtype=bool)
    z_c[gap], w_c[gap] = z_s[gap], w_s[gap]
    return z_c, w_c, z_s, w_s


def credible_window(p, mass=CRED_MASS):
    """
    Shortest contiguous index window carrying at least `mass`.

    Reported rather than a highest-density set because the posterior over a
    changepoint is often multimodal, and a set of disjoint hours is not what
    a reader takes from a date range. The interval is conservative where the
    posterior has separated modes, and that is the correct direction to err.
    """
    p = np.asarray(p, dtype=float)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return 0, len(p) - 1
    q = p / total
    c = np.concatenate([[0.0], np.cumsum(q)])
    lo = np.arange(len(q))
    hi = np.searchsorted(c, c[:-1] + mass - 1e-12, side="left")
    width = np.where(hi <= len(q), hi - lo, np.iinfo(np.int32).max)
    i = int(np.argmin(width))
    return i, int(min(hi[i], len(q)) - 1)


# --------------------------------------------------------------- estimation

@lru_cache(maxsize=8)
def _pair_grid(n, min_seg):
    """
    Admissible ordered changepoint pairs. Cached: the enumeration depends
    only on the sample length and the minimum segment, so the rho grid and
    the null replicates all reuse one copy instead of rebuilding a pair of
    (n+1)^2 integer arrays on every call.
    """
    grid = np.arange(n + 1)
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    ok = (T1 >= min_seg) & (T2 - T1 >= min_seg) & (n - T2 >= min_seg)
    return T1[ok].copy(), T2[ok].copy()


def _fit_given_rho(v, contiguous, n, min_seg, m0, k0, a0, b0, rho):
    """Model evidences and changepoint weights at one value of rho."""
    z_c, w_c, z_s, w_s = ar_transform(v, contiguous, rho)

    # Cumulative sums of the conditional form. Position a of any segment is
    # substituted with the stationary form by _stats below.
    c_ww = np.concatenate([[0.0], np.cumsum(w_c ** 2)])
    c_wz = np.concatenate([[0.0], np.cumsum(w_c * z_c)])
    c_zz = np.concatenate([[0.0], np.cumsum(z_c ** 2)])

    def _stats(a, b):
        """(n, sum w^2, sum wz, sum z^2) over [a, b), stationary at a."""
        a = np.asarray(a); b = np.asarray(b)
        a1 = np.minimum(a + 1, b)
        return (b - a,
                w_s[a] ** 2 + (c_ww[b] - c_ww[a1]),
                w_s[a] * z_s[a] + (c_wz[b] - c_wz[a1]),
                z_s[a] ** 2 + (c_zz[b] - c_zz[a1]))

    def inside(a, b):
        return _segment_evidence(*_stats(a, b), m0, k0, a0, b0)

    log_z0 = float(inside(np.array([0]), np.array([n]))[0])

    t = np.arange(min_seg, n - min_seg + 1)
    ll1 = inside(np.zeros_like(t), t) + inside(t, np.full_like(t, n))
    log_z1 = float(logsumexp(ll1) - np.log(t.size))

    a1, a2 = _pair_grid(n, min_seg)
    zeros, full = np.zeros_like(a1), np.full_like(a2, n)

    # M2 and M3 share their segments: both need [0, a1), [a1, a2) and
    # [a2, n). M2 pools the outer two into one calm regime, M3 gives them
    # separate means. Computing the three sufficient-statistic blocks once
    # and combining them is the whole difference between five passes over
    # half a million pairs and three.
    sA = _stats(zeros, a1)                       # left block
    sB = _stats(a1, a2)                          # middle block
    sC = _stats(a2, full)                        # right block

    mid = _segment_evidence(*sB, m0, k0, a0, b0)
    ll3 = (_segment_evidence(*sA, m0, k0, a0, b0) + mid
           + _segment_evidence(*sC, m0, k0, a0, b0))
    ll2 = mid + _segment_evidence(*[x + z for x, z in zip(sA, sC)],
                                  m0, k0, a0, b0)
    log_z2 = float(logsumexp(ll2) - np.log(a1.size))
    log_z3 = float(logsumexp(ll3) - np.log(a1.size))

    return dict(log_z=np.array([log_z0, log_z1, log_z2, log_z3]),
                t=t, ll1=ll1, a1=a1, a2=a2, ll2=ll2, ll3=ll3)


def fit_changepoint(y, min_seg, k0=PRIOR_K0, a0=PRIOR_A0,
                    rho_grid=RHO_GRID, contiguous=None, force=None) -> Fit:
    """
    Exact posterior over M0-M3 and over the changepoints within each.

    `y` is a Series indexed by bucket start time. `min_seg` is the shortest
    admissible segment in buckets: it keeps segments estimable and stops the
    two-changepoint models buying evidence with a single outlying hour.

    Within a segment the errors are AR(1). This is not a refinement. With
    independent errors the model treats a run of above-average hours as a
    level shift, and hourly dispersion is strongly autocorrelated because
    volatility clusters, so the independent version selects an excursion on
    data containing no changepoint at all — at rho = 0.5 it does so in
    roughly four series out of five. See the null calibration written to
    02_null_calibration.csv.

    rho is marginalised over a grid rather than profiled, so the reported
    evidence accounts for not knowing it. Conditional on rho the segment
    likelihood is still a one-regressor conjugate regression, so the
    enumeration stays exact and the whole cost is len(rho_grid) times the
    independent model.
    """
    idx = pd.DatetimeIndex(y.index)
    v = np.asarray(y, dtype=float)
    n = v.size
    if n < 3 * min_seg + 3:
        raise ValueError(f"{n} buckets is too few for min_seg={min_seg}")

    if contiguous is None:
        step = pd.Timedelta(BUCKET) if len(idx) < 2 else idx.to_series().diff().median()
        contiguous = np.zeros(n, dtype=bool)
        contiguous[1:] = (np.diff(idx.to_numpy()) == step.to_timedelta64())
    contiguous = np.asarray(contiguous, dtype=bool)

    m0 = float(np.mean(v))
    b0 = float((a0 - 1.0) * np.var(v, ddof=1))

    rho_grid = np.asarray(rho_grid, dtype=float)
    fits = [_fit_given_rho(v, contiguous, n, min_seg, m0, k0, a0, b0, r)
            for r in rho_grid]

    # Marginalise rho under a uniform prior on the grid.
    stack = np.array([f["log_z"] for f in fits])              # (rho, model)
    log_prior = -np.log(len(rho_grid))
    log_z_arr = logsumexp(stack + log_prior, axis=0)
    log_z = pd.Series(log_z_arr, index=MODELS)
    p_model = pd.Series(np.exp(log_z - logsumexp(log_z.to_numpy())),
                        index=MODELS)
    best = str(p_model.idxmax())
    if force is not None:
        # Conditioning on a model rather than selecting one. Used by the
        # null calibration, which needs the largest excursion noise can
        # produce, not only those rare replicates where noise happens to
        # win the model comparison outright.
        best = force

    # Posterior over rho given the selected model, and the rho weights used
    # to average the changepoint marginals.
    bi = MODELS.index(best)
    log_w_rho = stack[:, bi] + log_prior
    w_rho = np.exp(log_w_rho - logsumexp(log_w_rho))
    rho_hat = float(np.sum(w_rho * rho_grid))

    t = fits[0]["t"]
    a1, a2 = fits[0]["a1"], fits[0]["a2"]
    if best == MODELS[1]:
        w1 = sum(wr * np.exp(f["ll1"] - logsumexp(f["ll1"]))
                 for wr, f in zip(w_rho, fits))
    key = "ll2" if best == MODELS[2] else "ll3"
    if best in (MODELS[2], MODELS[3]):
        w = sum(wr * np.exp(f[key] - logsumexp(f[key]))
                for wr, f in zip(w_rho, fits))

    # pointwise regime probabilities and changepoint marginals
    p_onset = np.zeros(n + 1)
    p_return = None

    if best == MODELS[0]:
        seg_prob = np.ones((n, 1))

    elif best == MODELS[1]:
        p_onset[t] = w1
        after = np.cumsum(p_onset)[:n]          # P(tau <= h)
        seg_prob = np.column_stack([1.0 - after, after])

    else:
        p_onset = np.bincount(a1, weights=w, minlength=n + 1)
        p_return = np.bincount(a2, weights=w, minlength=n + 1)
        if best == MODELS[2]:
            # P(bucket h is inside the excursion), by difference array so
            # the cost is O(pairs) rather than O(pairs x segment length).
            diff = np.zeros(n + 2)
            np.add.at(diff, a1, w)
            np.subtract.at(diff, a2, w)
            p_stress = np.cumsum(diff)[:n]
            seg_prob = np.column_stack([1.0 - p_stress, p_stress])
        else:
            p_first = 1.0 - np.cumsum(p_onset)[:n]
            p_third = np.cumsum(p_return)[:n]
            seg_prob = np.column_stack(
                [p_first, 1.0 - p_first - p_third, p_third])

    def summarise(p):
        if p is None or p.sum() <= 0:
            return (pd.NaT, pd.NaT, pd.NaT)
        k = int(np.argmax(p))
        lo, hi = credible_window(p[:n])
        # A changepoint at position k opens the segment starting at bucket k.
        return (idx[min(k, n - 1)], idx[min(lo, n - 1)], idx[min(hi, n - 1)])

    onset, onset_lo, onset_hi = summarise(p_onset if best != MODELS[0] else None)
    ret, ret_lo, ret_hi = summarise(p_return)

    return Fit(idx, log_z, p_model, best, seg_prob, p_onset, p_return,
               onset, onset_lo, onset_hi, ret, ret_lo, ret_hi,
               rho_hat, contiguous)


# ------------------------------------------------------------ feature build

def schedulable_seconds(bucket_starts, freq=BUCKET):
    """
    Seconds each bucket could hold: its length, less its overlap with the
    rollover window.

    Derived from ROLLOVER_START and ROLLOVER_END rather than restated, so
    changing the exclusion window in utils changes this denominator too.
    """
    span = int(pd.Timedelta(freq).total_seconds())
    if 86400 % span:
        raise ValueError(f"{freq} does not divide the day; buckets would "
                         f"straddle midnight and the overlap below is wrong")
    t = pd.DatetimeIndex(bucket_starts)
    sod = t.hour * 3600 + t.minute * 60 + t.second
    r0 = pd.Timestamp(ROLLOVER_START).hour * 3600 + pd.Timestamp(ROLLOVER_START).minute * 60
    r1 = pd.Timestamp(ROLLOVER_END).hour * 3600 + pd.Timestamp(ROLLOVER_END).minute * 60
    overlap = np.maximum(0, np.minimum(sod + span, r1) - np.maximum(sod, r0))
    return span - overlap


def bucket_features(d, rollover, freq=BUCKET, with_micro=False,
                    min_coverage=MIN_COVERAGE,
                    min_moves=MIN_MOVES_PER_BUCKET):
    """
    Per-bucket dispersion and microstructure summaries of the basis.

    Rollover seconds are dropped first: liquidity collapses there and the
    basis widens mechanically, which is a property of the clock rather than
    of the market. Increments are computed only across adjacent seconds, so
    a closure boundary never contributes a two-day 'increment'.
    """
    check_grid(d.index, name="01_clean.parquet index")

    b = d["basis"]
    adj = adjacent(d.index)

    dx = np.full(len(d), np.nan)
    dx[1:] = np.diff(b.to_numpy())
    dx[~adj] = np.nan

    keep = ~rollover.to_numpy()
    w = pd.DataFrame({"basis": b.to_numpy()[keep],
                      "dx": dx[keep],
                      "moved": (np.abs(dx) > 0)[keep]},
                     index=d.index[keep])
    key = w.index.floor(freq)
    g = w.groupby(key)

    out = pd.DataFrame({
        "n": g.size(),
        "mad": g["basis"].apply(robust_scale),
        "iqr": g["basis"].quantile(0.75) - g["basis"].quantile(0.25),
        "move_share": g["moved"].mean(),
    })
    if with_micro:
        out["rho1"] = g["dx"].apply(acf1)
        out["roll_c"] = g["dx"].apply(roll_noise)

    out["coverage"] = out["n"] / schedulable_seconds(out.index, freq)
    out["moves"] = out["move_share"] * out["n"]

    good = ((out["coverage"] >= min_coverage)
            & (out["mad"] > 0) & (out["iqr"] > 0)
            & (out["moves"] >= min_moves))
    dropped = int((~good).sum())

    if not good.any():
        # Nothing survived. Say which test did the killing, because the
        # alternative is a ValueError three frames later about an empty
        # array, which names a symptom and not a cause.
        raise ValueError(
            f"no {freq} bucket passed admission out of {len(out):,}. "
            f"coverage >= {min_coverage:.0%}: {int((out['coverage'] >= min_coverage).sum()):,} pass "
            f"(median {out['coverage'].median():.2f}); "
            f"moves >= {min_moves}: {int((out['moves'] >= min_moves).sum()):,} pass "
            f"(median {out['moves'].median():.0f}); "
            f"mad > 0: {int((out['mad'] > 0).sum()):,} pass. "
            f"A median move count of zero means every increment was "
            f"discarded — check the grid resolution first.")

    out = out[good]
    out.index.name = "bucket"
    return out, dropped


def session_age(index, freq=BUCKET):
    """Buckets elapsed since the session opened; 0 at each reopen."""
    idx = pd.DatetimeIndex(index)
    step = pd.Timedelta(freq).to_timedelta64()
    contig = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        contig[1:] = np.diff(idx.to_numpy()) == step
    age = np.zeros(len(idx), dtype=int)
    for i in range(1, len(idx)):
        age[i] = age[i - 1] + 1 if contig[i] else 0
    return age, contig


def deseason(series, freq=BUCKET, reopen_buckets=REOPEN_BUCKETS):
    """
    Remove the time-of-day profile and the reopen ramp.

    Both are properties of the clock rather than of the market, and both
    are removed for the same reason the rollover window is excluded
    outright: liquidity is thin, the basis widens mechanically, and a
    changepoint model will otherwise place a boundary on the artefact.

    The reopen term is the one that earns its place. A weekend closure is
    followed by a thin Sunday evening, so the first buckets of every week
    sit systematically high. Without this adjustment a regime that truly
    ended during a closure gets recorded a few hours after the reopen —
    the model cannot put a changepoint inside a gap where no bucket
    exists, so it puts it at the first bucket that looks normal. Estimated
    across every session in the sample, so no single weekend drives it.

    Both profiles are estimated over the full sample and so cannot have
    been fitted either side of a candidate date. Day of week is left in the
    residual variance: with nine weeks the cells are too thin to estimate,
    and the omission inflates within-segment variance, which is
    conservative.
    """
    s = pd.Series(np.asarray(series, dtype=float),
                  index=pd.DatetimeIndex(series.index))
    # Keyed on the within-day slot rather than the hour, so a 30-minute
    # bucket is compared against the same 30 minutes of other days instead
    # of being averaged with its neighbour.
    minutes = int(pd.Timedelta(freq).total_seconds() // 60)
    slot = pd.Index((s.index.hour * 60 + s.index.minute) // minutes)
    profile = s.groupby(slot.to_numpy()).median()
    r = s - profile.reindex(slot).to_numpy()

    if reopen_buckets <= 0:
        return r, profile, pd.Series(dtype=float)

    age, _ = session_age(s.index, freq)
    band = np.minimum(age, reopen_buckets)
    ramp = pd.Series(r.to_numpy()).groupby(band).median()
    # Settled hours define the baseline, so only the reopen band moves.
    ramp = ramp - ramp.get(reopen_buckets, 0.0)
    return r - ramp.reindex(band).to_numpy(), profile, ramp


# ---------------------------------------------------------------- reporting

def fmt(ts):
    return "—" if pd.isna(ts) else f"{ts:%d %b %H:%M}"


def num(x, spec=".2f"):
    """
    Format a number for a table, or an em dash if it is not defined.

    Quantities like a dispersion ratio have no value when the selected
    model has one segment, and NaN written into a LaTeX table that the
    report \\input s renders as a blank cell or the literal 'nan'. An em
    dash says 'not applicable' and is the only thing that should ever
    reach the report in place of a number.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else format(v, spec)


def modal_span_hours(fit: Fit, freq=BUCKET) -> float:
    """Open-market hours between the modal changepoints."""
    if pd.isna(fit.onset) or pd.isna(fit.ret):
        return float("nan")
    a = int(np.searchsorted(fit.index, fit.onset))
    b = int(np.searchsorted(fit.index, fit.ret))
    return (b - a) * pd.Timedelta(freq).total_seconds() / 3600.0


def change_bracket(fit: Fit, p, freq=BUCKET):
    """
    When the change happened, honestly.

    A changepoint at position k says the new segment opens with bucket k.
    All the data licenses is that the change happened after the previous
    retained bucket ended and by the time bucket k started. Where those two
    instants are one bucket apart that is the bucket; where a market
    closure sits between them it is the whole closure, and quoting an hour
    would be inventing resolution the sample does not contain.

    This matters here: a regime that ends over a weekend can only be
    recorded at the Sunday reopen, and the credible interval in bucket
    space is then spuriously narrow.
    """
    if p is None or not np.isfinite(p).any() or p.sum() <= 0:
        return "—", False
    idx, step = fit.index, pd.Timedelta(freq)
    n = len(idx)
    lo, hi = credible_window(p[:n])
    lo, hi = min(lo, n - 1), min(hi, n - 1)
    first = idx[lo - 1] + step if lo > 0 else idx[0]
    last = idx[hi]
    spans = bool((~fit.contiguous[lo:hi + 1]).any()) or (first < idx[lo])
    if last - first <= step:
        return fmt(last), spans
    return f"{fmt(first)} – {fmt(last)}", spans


def within_ratio(fit: Fit, y) -> float:
    """
    Size of the shift the model fitted, as a multiple: the widest segment
    against everything else.

    Cut at the reported changepoints rather than at the pointwise
    posterior. Two reasons. Under M3 the highest segment label is the calm
    tail, so keying on the label compares the quiet end of the sample
    against the rest and returns a number below one. And where the
    posterior is diffuse — which is exactly the case under the null — no
    bucket carries more than half its mass, every bucket takes label 0,
    and the ratio comes back undefined for the replicates that most need
    a value. Cutting at the modal changepoints always yields a number, and
    it is the number a reader would compute from the reported dates.
    """
    v = np.asarray(y, dtype=float)
    idx, n = fit.index, len(fit.index)
    if pd.isna(fit.onset):
        return float("nan")
    a = int(np.searchsorted(idx, fit.onset))
    b = n if pd.isna(fit.ret) else int(np.searchsorted(idx, fit.ret))
    lab = np.zeros(n, dtype=int)
    lab[a:b] = 1
    lab[b:] = 2 if fit.best == MODELS[3] else 0
    means = {k: v[lab == k].mean() for k in np.unique(lab)}
    if len(means) < 2:
        return float("nan")
    hot = lab == max(means, key=means.get)
    if not hot.any() or hot.all():
        return float("nan")
    return float(np.exp(v[hot].mean() - v[~hot].mean()))


def variant_row(name, fit: Fit, y=None):
    return {
        "variant": name,
        "selected": fit.best.split(" ", 1)[0],
        "P(selected)": float(fit.p_model.max()),
        "ratio": within_ratio(fit, y) if y is not None else float("nan"),
        "rho": float(fit.rho),
        "onset": fmt(fit.onset),
        "return": fmt(fit.ret),
    }


def regime_summary(d, rollover, labels, names, episodes):
    """
    Contrast between the estimated regimes.

    The labels come from dispersion alone. Persistence, microstructure noise
    and episode intensity were not used to place the changepoints, so a
    difference in those columns is corroboration from features the model
    never saw rather than a restatement of the fit.
    """
    open_rows = ~rollover.to_numpy()
    b = d["basis"].to_numpy()

    adj = adjacent(d.index)
    usable = np.zeros(len(d), dtype=bool)
    usable[1:] = (adj[1:]
                  & (labels[1:] == labels[:-1])
                  & open_rows[1:] & open_rows[:-1])
    i = np.flatnonzero(usable)
    prev, cur, lab_pair = b[i - 1], b[i], labels[i]

    dx = np.full(len(d), np.nan)
    dx[i] = b[i] - b[i - 1]

    ep_lab = None
    if episodes is not None and len(episodes):
        lookup = pd.Series(labels, index=d.index)
        ep_lab = lookup.reindex(pd.DatetimeIndex(episodes["start"])).to_numpy()

    cols = {}
    for k, name in enumerate(names):
        sel = (labels == k) & open_rows
        x = b[sel]
        hours = sel.sum() / 3600.0
        pair = lab_pair == k
        rho = (np.corrcoef(prev[pair], cur[pair])[0, 1] if pair.sum() > 2
               else np.nan)
        half = (np.log(0.5) / np.log(rho)
                if np.isfinite(rho) and 0 < rho < 1 else np.nan)
        col = {
            "open hours": hours,
            "seconds": int(sel.sum()),
            "basis MAD (pips)": robust_scale(x),
            "basis sd (pips)": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            # A segment can end up with no seconds when its posterior never
            # wins the pointwise argmax, and np.quantile raises on empty.
            "p99.9 |basis| (pips)": (float(np.quantile(np.abs(x), 0.999))
                                     if x.size else np.nan),
            "max |basis| (pips)": float(np.abs(x).max()) if x.size else np.nan,
            "rho_1 of increments": acf1(dx[sel]),
            "Roll noise c (pips)": roll_noise(dx[sel]),
            "noise share of variance": noise_share(dx[sel]),
            "AR(1) rho of level": rho,
            "half-life (s)": half,
        }
        if ep_lab is not None:
            mine = episodes[ep_lab == k]
            col["episodes"] = float(len(mine))
            col["episodes per 100 h"] = 100.0 * len(mine) / hours if hours else np.nan
            col["median |peak| (pips)"] = float(mine.peak.abs().median()) if len(mine) else np.nan
            col["widest |peak| (pips)"] = float(mine.peak.abs().max()) if len(mine) else np.nan
        cols[name] = col
    return pd.DataFrame(cols)


# --------------------------------------------------------------------- main

def main():
    make_dirs()
    set_style()

    print("load")
    d = pd.read_parquet(DAT_DIR / "01_clean.parquet").sort_index()
    if "basis" not in d.columns:
        raise KeyError("01_clean.parquet has no 'basis' column — rerun 01")
    info = check_grid(d.index, name="01_clean.parquet index")
    print(f" {len(d):,} open-market seconds, {d.index[0]} to {d.index[-1]}")
    print(f" resolution {info['resolution']}, "
          f"{info['adjacent_share']:.1%} of rows one second after the last")

    ep_path = DAT_DIR / "01_episodes.parquet"
    episodes = pd.read_parquet(ep_path) if ep_path.exists() else None
    if episodes is None:
        print(" 01_episodes.parquet absent; episode columns will be omitted")

    roll = in_rollover(d.index)

    print("buckets")
    feat, dropped = bucket_features(d, roll, BUCKET, with_micro=True)
    print(f" {len(feat):,} {BUCKET} buckets retained, {dropped} dropped "
          f"(coverage below {MIN_COVERAGE:.0%} of schedulable seconds, "
          f"or fewer than {MIN_MOVES_PER_BUCKET} moves)")

    y_raw = np.log(feat["mad"])
    y, profile, ramp = deseason(y_raw)
    amp = float(np.exp(profile.max() - profile.min()))
    print(f" diurnal profile removed; peak-to-trough {amp:.2f}x in dispersion")

    min_seg = max(2, int(round(MIN_SEG_HOURS
                               / (pd.Timedelta(BUCKET).total_seconds() / 3600))))

    print("fit")
    fit = fit_changepoint(y, min_seg)
    for m in MODELS:
        print(f" {m:<22} log Z {fit.log_evidence[m]:+11.2f}   "
              f"P {fit.p_model[m]:.3f}")
    runner_up = fit.p_model.drop(fit.best).idxmax()
    bf = float(fit.log_evidence[fit.best] - fit.log_evidence[runner_up])
    print(f" selected {fit.best}; log Bayes factor {bf:+.1f} over "
          f"{runner_up}")

    names = REGIME_NAMES[fit.best]
    print(f" onset  {fmt(fit.onset)}  [{fmt(fit.onset_lo)}, {fmt(fit.onset_hi)}]")
    if fit.p_return is not None:
        print(f" return {fmt(fit.ret)}  [{fmt(fit.ret_lo)}, {fmt(fit.ret_hi)}]")
        print(f" BOJ decision {BOJ_SHOCK:%d %b %H:%M} {TZ_LABEL} "
              f"(reference only, given to no model)")

    # ------------------------------------------------ labels, per second
    print("labels")
    prob = pd.DataFrame(fit.seg_prob, index=fit.index,
                        columns=[f"p_{n.replace(' ', '_')}" for n in names])
    grid = pd.date_range(d.index[0].floor(BUCKET), d.index[-1].floor(BUCKET),
                         freq=BUCKET)
    # Buckets dropped as short or frozen inherit the previous bucket's
    # posterior: they carry no information about where the change is, so
    # giving them a label of their own would invent one.
    prob = prob.reindex(grid).ffill().bfill()

    per_sec = prob.reindex(d.index.floor(BUCKET))
    per_sec.index = d.index
    labels = per_sec.to_numpy().argmax(axis=1).astype(np.int8)

    out = per_sec.astype(np.float32)
    out.insert(0, "regime", labels)
    out.index.name = "t"
    out.attrs["model"] = fit.best
    out.to_parquet(DAT_DIR / "02_regimes.parquet")
    shares = pd.Series(labels).value_counts(normalize=True).sort_index()
    print(" -> 02_regimes.parquet  "
          + ", ".join(f"{names[k]} {v:.1%}" for k, v in shares.items()))

    # ------------------------------------------------ tables
    print("tables")
    stats = regime_summary(d, roll, labels, names, episodes)
    save_table(
        stats, "02_regime_stats",
        caption=("The basis by estimated regime, open-market seconds outside "
                 "rollover. Regimes are estimated from hourly dispersion "
                 "alone; persistence, microstructure noise and episode counts "
                 "are contrasts on features the changepoint model never saw."),
        label="tab:regime_stats",
    )
    print(stats.to_string(float_format=lambda v: f"{v:.4g}"))

    evidence = pd.DataFrame({
        "changepoints": [0, 1, 2, 2],
        "log evidence": fit.log_evidence.to_numpy(),
        "log BF vs best": (fit.log_evidence - fit.log_evidence.max()).to_numpy(),
        "posterior probability": fit.p_model.to_numpy(),
    }, index=MODELS)
    save_table(
        evidence, "02_model_evidence",
        caption=("Marginal likelihood of each segmentation of hourly log "
                 "dispersion, computed exactly by enumerating every "
                 "admissible configuration. Model priors are equal; the "
                 "changepoint prior is uniform over configurations with all "
                 f"segments at least {MIN_SEG_HOURS:.0f} hours long."),
        label="tab:model_evidence",
    )

    def ratio(row):
        """
        Widest segment against narrowest, for one row of regime_stats.

        Defined this way rather than as segment 2 over segment 1 so that it
        means the same thing whatever model was selected. With three
        segments the calm tail, not the calm opening, is often the
        narrowest, and a fixed pair of indices would silently compare the
        wrong two.
        """
        if len(names) < 2 or row not in stats.index:
            return np.nan
        v = pd.to_numeric(stats.loc[row, names], errors="coerce").dropna()
        return np.nan if len(v) < 2 or v.min() == 0 else v.max() / v.min()
    # Reported as the span the data brackets, not as the modal bucket. A
    # change falling inside a closure can only be recorded at the reopen,
    # and quoting that hour would claim resolution the sample lacks.
    onset_win, onset_gap = change_bracket(fit, fit.p_onset)
    ret_win, ret_gap = change_bracket(fit, fit.p_return)

    def _median_cp(p):
        """Posterior median changepoint, in bucket time."""
        if p is None or not np.isfinite(p).any() or p.sum() <= 0:
            return pd.NaT
        m = len(fit.index)
        c = np.cumsum(p[:m]) / p[:m].sum()
        return fit.index[int(min(np.searchsorted(c, 0.5), m - 1))]

    # Mode and median both, because they can disagree and the disagreement
    # is itself the result. Where the onset posterior is bimodal its mode
    # sits on whichever spike is taller while half the mass lies elsewhere.
    # The per-second labels are the median segmentation — a bucket is
    # called elevated when more than half the posterior says it is — so
    # reporting only the mode leaves the headline describing a different
    # window from the one every downstream script consumes.
    onset_med, ret_med = _median_cp(fit.p_onset), _median_cp(fit.p_return)
    core_h = (int((fit.seg_prob.argmax(1) == 1).sum())
              * pd.Timedelta(BUCKET).total_seconds() / 3600.0)

    head = {
        "selected model": fit.best,
        "posterior probability": num(fit.p_model.max(), ".3f"),
        f"log Bayes factor vs {runner_up.split(' ', 1)[0]}": f"{bf:+.1f}",
        "AR(1) rho of the errors": num(fit.rho),
        "onset, posterior mode": fmt(fit.onset),
        "onset, posterior median": fmt(onset_med),
        "onset 95% window": onset_win + (" (spans a closure)" if onset_gap else ""),
    }
    if fit.p_return is not None:
        head.update({
            "return, posterior mode": fmt(fit.ret),
            "return, posterior median": fmt(ret_med),
            "return 95% window": ret_win + (" (spans a closure)" if ret_gap else ""),
            "elevated open-market hours, median segmentation (the labels)":
                num(core_h, ".0f"),
            "elevated open-market hours, modal segmentation":
                num(modal_span_hours(fit), ".0f"),
        })
    head.update({
        "BOJ decision (reference)": f"{BOJ_SHOCK:%d %b %H:%M}",
        "onset relative to BOJ, mode (hours)":
            ("—" if pd.isna(fit.onset) else
             f"{(fit.onset - BOJ_SHOCK).total_seconds() / 3600:+.1f}"),
        "onset relative to BOJ, median (hours)":
            ("—" if pd.isna(onset_med) else
             f"{(onset_med - BOJ_SHOCK).total_seconds() / 3600:+.1f}"),
        # Two different quantities, both previously called "dispersion".
        # The first is what the model fitted: how much wider the basis got
        # inside a given hour. The second pools every second in the regime
        # around one median, so it also absorbs the basis level wandering
        # between hours. The gap between them is that wandering, and it is
        # the same thing the half-life column reports.
        "within-hour dispersion ratio (fitted)": num(within_ratio(fit, y)),
        "pooled dispersion ratio, widest / narrowest segment (MAD)":
            num(ratio("basis MAD (pips)")),
        "microstructure noise ratio (Roll c)": num(ratio("Roll noise c (pips)")),
        "half-life ratio": num(ratio("half-life (s)")),
    })
    headline = pd.DataFrame({"value": pd.Series(head)})
    save_table(
        headline, "02_changepoint",
        caption=("Changepoint in the dispersion of the AUD/JPY triangular "
                 "basis. Dates are posterior marginal modes over hourly "
                 f"buckets; intervals are the shortest windows carrying "
                 f"{CRED_MASS:.0%} of the marginal posterior. The BOJ "
                 "decision is a reference and was supplied to no model."),
        label="tab:changepoint",
    )
    print(headline.to_string())

    # ------------------------------------------------ sensitivity
    print("sensitivity")
    rows = [variant_row("baseline (1h, deseasonalised log MAD)", fit, y)]

    rows.append(variant_row("no diurnal adjustment",
                            fit_changepoint(y_raw, min_seg), y_raw))
    y_iqr = deseason(np.log(feat["iqr"]))[0]
    rows.append(variant_row("log IQR instead of log MAD",
                            fit_changepoint(y_iqr, min_seg), y_iqr))

    # Switching the reopen adjustment off. If the return date moves to a
    # Sunday evening without it, the boundary was tracking thin post-
    # weekend liquidity rather than the market returning to normal.
    y_nr = deseason(y_raw, reopen_buckets=0)[0]
    rows.append(variant_row("no session-reopen adjustment",
                            fit_changepoint(y_nr, min_seg), y_nr))
    for freq, label in [("30min", "30-minute buckets"), ("2h", "2-hour buckets")]:
        f2, _ = bucket_features(d, roll, freq)
        y2 = deseason(np.log(f2["mad"]), freq)[0]
        m2 = max(2, int(round(MIN_SEG_HOURS
                              / (pd.Timedelta(freq).total_seconds() / 3600))))
        rows.append(variant_row(label, fit_changepoint(y2, m2), y2))
    for h in (3, 12, 24):
        rows.append(variant_row(f"minimum segment {h} h",
                                fit_changepoint(y, max(2, int(h))), y))
    for k0 in (0.001, 0.1):
        rows.append(variant_row(f"prior k0 = {k0:g}",
                                fit_changepoint(y, min_seg, k0=k0), y))

    # The admission filters. These decide which buckets the model ever
    # sees, so leaving them out of this table was the gap worth closing.
    for cov in (0.70, 0.98):
        f2, _ = bucket_features(d, roll, BUCKET, min_coverage=cov)
        yc = deseason(np.log(f2["mad"]))[0]
        rows.append(variant_row(f"bucket coverage >= {cov:.0%}",
                                fit_changepoint(yc, min_seg), yc))
    for mv in (10, 240):
        f2, _ = bucket_features(d, roll, BUCKET, min_moves=mv)
        ym = deseason(np.log(f2["mad"]))[0]
        rows.append(variant_row(f"minimum {mv} moves per bucket",
                                fit_changepoint(ym, min_seg), ym))

    # A different feature entirely. Persistence is scale-free, so if it
    # breaks at the same hour the change is in the process and not in the
    # units.
    rho = feat["rho1"].dropna().clip(-0.999, 0.999)
    y_rho = deseason(np.arctanh(rho))[0]
    rows.append(variant_row("feature: increment persistence (arctanh rho_1)",
                            fit_changepoint(y_rho, min_seg), y_rho))

    # A confound rather than a robustness check. Quote intensity rises
    # through the sample; if the dispersion changepoint merely tracked it,
    # the two would coincide.
    y_q = deseason(np.log(feat["move_share"]))[0]
    rows.append(variant_row("confound: quote-move rate",
                            fit_changepoint(y_q, min_seg), y_q))

    # The model this replaced. Kept in the table because the difference
    # between these two rows is the whole reason for the AR(1) term.
    rows.append(variant_row("independent errors (rho fixed at 0)",
                            fit_changepoint(y, min_seg,
                                            rho_grid=np.array([0.0])), y))

    # Placebo. The window must end before the decision, and the cut is
    # derived from BOJ_SHOCK rather than typed, for the same reason
    # BOJ_SHOCK is itself derived from UTC_OFFSET_HOURS: a literal date
    # drifts out of agreement with the offset it depends on. A bucket is
    # kept only if it *ends* at or before the announcement — slicing on
    # the bucket start admits the bucket containing it, which is the 23:00
    # bucket on 30 July, and with it both the decision and the
    # second-widest episode in the sample. A placebo containing the event
    # finds the event, and reports it as a false positive.
    pre = y.loc[(y.index + pd.Timedelta(BUCKET)) <= BOJ_SHOCK]
    rows.append(variant_row(
        f"placebo: truncated at {BOJ_SHOCK:%d %b %H:%M}, before the decision",
        fit_changepoint(pre, min_seg), pre))

    sens = pd.DataFrame(rows).set_index("variant")
    # A variant that selects M0 has no shift to size. Written as a dash for
    # the same reason as everywhere else: NaN must not reach the report.
    sens["ratio"] = [num(v, ".3f") for v in sens["ratio"]]
    sens["rho"] = [num(v, ".2f") for v in sens["rho"]]
    sens["P(selected)"] = [num(v, ".3f") for v in sens["P(selected)"]]
    save_table(
        sens, "02_sensitivity",
        caption=("Sensitivity of the selected segmentation. The first block "
                 "varies the summary statistic and the bucket, the second the "
                 "prior and the minimum segment length, the third the "
                 "admission filters, the fourth replaces the feature. The "
                 "placebo row fits the sample truncated before the policy "
                 "decision, where no change should be found. `ratio` is the "
                 "size of the fitted shift as a multiple: selection without "
                 "magnitude is not a finding, and Table~\\ref{tab:cp_null} "
                 "gives the range of ratios that noise alone produces."),
        label="tab:cp_sensitivity",
    )
    print(sens.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------ null calibration
    #
    # The Bayes factor answers "how much better is M2 than M0 on this
    # series", which is not the question a reader has. The question is
    # whether a series with no changepoint in it would have looked like
    # this. Simulate under the fitted null — same length, same session
    # structure, same AR(1) coefficient, no change anywhere — and read off
    # how often the machinery finds an excursion regardless, and how large
    # those spurious excursions are.
    print("null calibration")
    if NULL_DRAWS <= 0:
        print("  skipped (NULL_DRAWS = 0)")
    rng = np.random.default_rng(NULL_SEED)
    yv = np.asarray(y, dtype=float)
    resid = yv - np.array([yv[fit.seg_prob.argmax(1) == k].mean()
                           for k in fit.seg_prob.argmax(1)])
    sd = float(np.std(resid, ddof=1))
    nb = len(yv)
    # Length of the excursion actually estimated, in buckets: the null is
    # asked for the best excursion of at least this long.
    if pd.notna(fit.onset) and pd.notna(fit.ret):
        null_seg = max(min_seg, int(np.searchsorted(fit.index, fit.ret)
                                    - np.searchsorted(fit.index, fit.onset)))
    else:
        null_seg = min_seg

    picks, ratios, bfs = [], [], []
    t_null = time.perf_counter()
    for j in range(NULL_DRAWS):
        if j == 1:
            each = time.perf_counter() - t_null
            print(f"  {NULL_DRAWS} replicates at ~{each:.1f}s each, "
                  f"about {each * NULL_DRAWS / 60:.0f} min "
                  f"(set NULL_DRAWS = 0 to skip while iterating)")
        elif j and j % 25 == 0:
            print(f"  {j}/{NULL_DRAWS}")
        e = rng.standard_normal(nb)
        sim = np.empty(nb)
        sim[0] = e[0]
        for i in range(1, nb):
            sim[i] = (fit.rho * sim[i - 1] + np.sqrt(1 - fit.rho ** 2) * e[i]
                      if fit.contiguous[i] else e[i])
        sim = pd.Series(sim * sd, index=fit.index)
        g = fit_changepoint(sim, min_seg, contiguous=fit.contiguous)
        picks.append(g.best)
        bfs.append(float(g.log_evidence[MODELS[2]] - g.log_evidence[MODELS[0]]))
        # Forced to M2 and floored at the length actually estimated, so
        # every replicate contributes and the comparison is like for like.
        # Unforced, the null selects no change in 98 replicates out of 100
        # and the ratio is undefined for those; unmatched on duration, the
        # null is free to find a six-hour window whose ratio a sustained
        # fortnight could never reach, which understates the observation.
        h = fit_changepoint(sim, null_seg, contiguous=fit.contiguous,
                            force=MODELS[2])
        r = within_ratio(h, sim)
        if np.isfinite(r):
            ratios.append(r)

    if NULL_DRAWS <= 0:
        picks, bfs, ratios = [MODELS[0]], [float("nan")], [float("nan")]

    obs_ratio = within_ratio(fit, y)
    obs_bf = float(fit.log_evidence[MODELS[2]] - fit.log_evidence[MODELS[0]])
    ratios = np.array(ratios) if ratios else np.array([np.nan])
    null = pd.DataFrame({"value": pd.Series({
        "null replicates": f"{NULL_DRAWS}",
        "AR(1) rho used": f"{fit.rho:.2f}",
        "residual sd used": f"{sd:.3f}",
        "P(any change selected | no change)":
            f"{np.mean([p != MODELS[0] for p in picks]):.3f}",
        "P(excursion selected | no change)":
            f"{np.mean([p == MODELS[2] for p in picks]):.3f}",
        "null log BF, 95th percentile": f"{np.percentile(bfs, 95):+.1f}",
        "observed log BF (M2 vs M0)": f"{obs_bf:+.1f}",
        f"null excursion of >= {null_seg} buckets, median ratio":
            num(np.nanmedian(ratios) if np.isfinite(ratios).any() else np.nan, ".3f"),
        f"null excursion of >= {null_seg} buckets, 95th percentile":
            num(np.nanpercentile(ratios, 95) if np.isfinite(ratios).any() else np.nan, ".3f"),
        "observed within-hour ratio": num(obs_ratio, ".3f"),
        "observed ratio exceeds null replicates":
            num(np.mean(obs_ratio > ratios[np.isfinite(ratios)])
                if (np.isfinite(obs_ratio) and np.isfinite(ratios).any())
                else np.nan, ".3f"),
        "observed log BF exceeds null replicates":
            f"{np.mean(obs_bf > np.array(bfs)):.3f}",
    })})
    save_table(
        null, "02_null_calibration",
        caption=("Calibration under the null. Series of the same length and "
                 "session structure as the sample, with the fitted AR(1) "
                 "coefficient and residual scale but no changepoint "
                 "anywhere, refitted with the same machinery. The last row "
                 "is the fraction of null replicates the observed evidence "
                 "exceeds, and is the number to quote rather than the Bayes "
                 "factor alone."),
        label="tab:cp_null",
    )
    print(null.to_string())

    # ------------------------------------------------ figures
    print("figures")
    disp = pd.Series(np.exp(np.asarray(y, dtype=float)), index=fit.index)
    disp_pips = feat["mad"].reindex(fit.index).to_numpy()   # for the legend
    full = pd.date_range(disp.index.min(), disp.index.max(), freq=BUCKET)

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.36)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    if len(names) > 1:
        for k, colour in zip(range(len(names)), regime_colours(len(names))):
            level = float(np.exp(np.asarray(y)[fit.seg_prob.argmax(1) == k].mean()))
            # The plotted series is deseasonalised, so a value is a
            # multiple of the typical dispersion for that time of day, not
            # a pip count. The pip median is given alongside so the reader
            # can anchor it without the axis claiming units it does not have.
            pips = float(np.median(disp_pips[fit.seg_prob.argmax(1) == k]))
            ax0.axhline(level, color=colour, lw=1.2, zorder=2,
                        label=f"{names[k]}: {level:.2f}x  ({pips:.3f} pips)")
    if fit.p_return is not None:
        highlight_span(ax0, fit.onset, fit.ret)
    ax0.plot(full, disp.reindex(full).to_numpy(), lw=0.75, color=RULE, zorder=3)
    ax0.set_yscale("log")
    ax0.set_ylabel("dispersion, relative to\nthe same hour on a typical day")
    panel_title(ax0, "Hourly dispersion of the basis, diurnal profile removed",
                "1.0 = the same hour on a typical day; the 4× diurnal profile is divided out")
    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    annotate_event(ax0, BOJ_SHOCK, "BOJ")
    ax0.legend(loc="upper left", ncol=len(names))

    n = len(fit.index)
    if fit.p_onset is not None and fit.best != MODELS[0]:
        ax1.vlines(fit.index, 0, fit.p_onset[:n], color=REGIME_COLOUR[1],
                   lw=1.1, label=f"onset · {fmt(fit.onset)}")
        annotate_interval(ax1, fit.onset_lo, fit.onset_hi)
    if fit.p_return is not None:
        ax1.vlines(fit.index, 0, fit.p_return[:n], color=REGIME_COLOUR[0],
                   lw=1.1, label=f"return · {fmt(fit.ret)}")
        annotate_interval(ax1, fit.ret_lo, fit.ret_hi)
    ax1.set_ylabel("posterior mass")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_xlabel(TZ_LABEL)
    panel_title(ax1, "Posterior over the changepoints",
                "the model is given no candidate date; shaded bands are the "
                f"shortest {CRED_MASS:.0%} intervals")
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    annotate_event(ax1, BOJ_SHOCK, "BOJ", top=False)
    ax1.legend(loc="upper right")
    save_fig(fig, "02_changepoint_posterior")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3),
                           gridspec_kw={"wspace": 0.26})
    open_rows = ~roll.to_numpy()
    for k, name in enumerate(names):
        u, p = survival(d["basis"].to_numpy()[(labels == k) & open_rows])
        ax[0].loglog(u, p, lw=1.8, color=regime_colours(len(names))[k], label=name,
                     solid_capstyle="round")
    ax[0].set_xlabel("threshold $u$ (pips)")
    ax[0].set_ylabel(r"$P\,(\,|\mathrm{basis}| > u\,)$")
    panel_title(ax[0], "Tail by estimated regime", " the split is estimated, not calendar")
    ax[0].legend(loc="lower left")
    ax[0].grid(which="both", alpha=0.12)

    bucket_lab = prob.reindex(feat.index).to_numpy().argmax(axis=1)
    for k, name in enumerate(names):
        sel = bucket_lab == k
        ax[1].scatter(feat["roll_c"].to_numpy()[sel], feat["mad"].to_numpy()[sel],
                      s=16, c=regime_colours(len(names))[k], alpha=0.75, lw=0.3,
                      edgecolor="white", label=name)
    lim = np.array([np.nanmin(feat["roll_c"]) * 0.8, np.nanmax(feat["mad"]) * 1.2])
    ax[1].plot(lim, lim, color=MUTE, lw=0.8, ls=":", zorder=0)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("microstructure noise, Roll $c$ (pips)")
    ax[1].set_ylabel("dispersion, MAD (pips)")
    panel_title(ax[1], "Dispersion against noise, by hour",
                "noise alone would put every hour on one locus")
    ax[1].legend(loc="lower right")
    save_fig(fig, "02_regime_contrast")


if __name__ == "__main__":
    main()