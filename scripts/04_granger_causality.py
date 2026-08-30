"""
04_granger.py

Which leg breaks the triangle, and which leg repairs it?

Section 03 asked how fast the triangle closed. It answered with lambda, the
combination of the three adjustment coefficients, and it was careful to say
that the *split* of that closure across legs is the least precise thing in the
script. This one takes up the question the project set out with - "which pair
breaks the triangle" - and answers it three ways that share no machinery, in
increasing order of how much has to be assumed.

    1. An identity.  The basis is z = x_AUDJPY - x_AUDUSD - x_USDJPY, so the
       change in the basis over any window is exactly the sum of the three
       legs' contributions. For every episode section 01 detected, this script
       splits the opening and the closing of that episode across the three
       legs by arithmetic. No lag order, no regression, no regime label enters
       the calculation. This is the model-free answer and it is the one the
       report should lead with, for the same reason 03 leads with the episode
       ruler rather than with its own fit.

    2. A correlation.  The cross-correlation between each leg's increments and
       the basis's, at leads and lags out to a minute, computed only on pairs
       that sit inside one unbroken run. A leg that *leads* the basis shows
       correlation where its own past sits against the basis's future. Still
       no lag order and no fitted model, only a choice of horizon.

    3. A test.  Block Granger causality inside the error-correction model 03
       fitted: does the recent history of leg i help predict the next second
       of the basis, or of another leg, given everything else in the system.
       This is the formal version, and it is reported last because it has the
       most machinery between the data and the number.

Two different questions, kept apart
    "Who breaks it" and "who gives way" are not the same question, and the
    price-discovery literature conflates them often enough that it is worth
    separating them explicitly.

        Transitory causality (Gamma).  Does leg i's recent history predict the
        next move? This is Granger causality in the ordinary sense and it is
        about the order in which information arrives.

        Error-correction causality (alpha).  Does leg j respond to the level
        of the disequilibrium at all? A leg whose alpha is zero is weakly
        exogenous: it walks away and lets the others come to it, which is what
        carrying the permanent information looks like. 03 estimated these
        coefficients; this script tests them.

A degeneracy in the basis equation, and what is done about it
    The basis is a fixed combination of the three legs, so at every lag the
    three legs' increments together span the basis's own lagged increment.
    Dropping any one leg's block therefore removes part of the basis's own
    autoregression as well as that leg's information. When the basis is
    persistent - which under stress it is - all three blocks then test large
    for that reason alone, and the test cannot rank the legs even though each
    individual test is well posed.

    So the headline test conditions on the basis's own history explicitly. For
    each leg it fits a smaller model - the level of the basis, the basis's own
    lagged increments, and that one leg's lagged increments - and asks what
    the leg adds beyond the basis's own memory. That is the classical pairwise
    Granger setup and it isolates the leg. The full-system figure is reported
    beside it as a control, so the gap between the two can be read rather than
    argued about.

    Neither model is refitted from the data. Both are column subsets of one
    extended design whose cross-products are accumulated once, so every number
    in this script comes from the same rows and the same single pass.

Why there are no Hasbrouck or Gonzalo-Granger information shares here
    Those measures are defined for two prices of one asset: two series, one
    cointegrating vector, one common trend, and the trend's loadings then
    split into shares that sum to one. This system has three prices and one
    cointegrating restriction, so it has *two* common trends - an AUD trend
    and a JPY trend - and the orthogonal complement of alpha is 3x2 rather
    than a vector. There is no scalar share per leg without imposing a
    normalisation the data does not identify. Reporting one would be a number
    a reader could not trace, which is the one thing this project's
    conventions forbid. The weak-exogeneity tests below are the part of that
    apparatus this system does support, and they are reported instead.

Why a p-value is not the evidence here
    Every test in this script is computed on between one hundred thousand and
    a million seconds. At that sample size a coefficient of no economic
    consequence rejects at any conventional level, and a table of p-values
    reading "< 0.001" throughout says only that n is large. So every test is
    reported with an effect size next to it - the incremental predictable
    move, in pips per second, that the block buys - and every effect size is
    reported next to a *placebo* threshold obtained by giving the same test
    the same leg's increments from a different day. The placebo keeps each
    leg's own autocorrelation and its own volatility clustering and destroys
    only the cross-leg timing, so it is the distribution of the statistic when
    the null is true and everything else is as it really is. Without it there
    is no way to know what "found nothing" would look like.

Which seconds are used, and where that rule comes from
    A lag window must be contiguous in clock time, outside the daily rollover
    break, inside one regime and one trading day, and confidently labelled by
    the regime posterior. Those rules belong to the error-correction model
    this script tests, and they are imported from 03_var.py rather than
    reimplemented: a Granger test computed on a different set of seconds than
    the model it comments on is not a test of that model, and one definition
    of the guard that stops this pipeline differencing a price across a
    weekend is enough.

    Only that module is needed, not its outputs. This script reads the cleaned
    price file and the regime labels and requires nothing else. Where the
    error-correction model has already been fitted, its selected lag order is
    read and reused so that both describe one specification; where it has not,
    the same criterion on the same grid selects one here and the script says
    which happened.

Outputs

    Seven tables. Each is written twice, as output/tables/<name>.csv for
    reading and as report/tables/<name>.tex for the report to input, and each
    is also printed to the run log, so the log alone is a complete record of
    what this script found.

        04_attribution        which leg opened each dislocation and which
                              closed it, by arithmetic
        04_basis_causality    headline: what each leg's history says about
                              the next second of the basis
        04_leg_causality      the same test between the legs, both directions
        04_weak_exogeneity    which legs respond to the disequilibrium
        04_lead_lag           cross-correlations at matched leads and lags
        04_null_calibration   what the tests return when the null is true
        04_sensitivity        the headline under every convention varied

    Three figures, in output/figures/: 04_attribution.png, 04_lead_lag.png,
    04_granger.png.

    One dataset, output/data/04_granger_tests.parquet: every test in this
    script with its statistic, effect size and provenance, so the tables above
    can be rebuilt without refitting anything.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils import (
    make_dirs, set_style, in_rollover, robust_scale, save_table, save_fig,
    adjacent, check_grid,
    DAT_DIR, TAB_DIR, TEX_DIR,
    GRID_SECONDS, PAIR_LABEL, LEG_COLOUR, regime_colours,
    FOUND, MUTE, RULE,
)


# ------------------------------------------------------------------ 03 import
#
# A module name cannot begin with a digit, so the sibling script is loaded by
# path. It defines only functions and constants at import time - main() is
# guarded - so importing it runs nothing.

def _load_var_module():
    path = Path(__file__).resolve().with_name("03_var_analysis.py")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. 04 imports its row-admission rules from 03 so "
            f"that both scripts describe the same sample. It does not need 03 "
            f"to have been run, but it does need the file to be there.")
    spec = importlib.util.spec_from_file_location("var03", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["var03"] = module
    spec.loader.exec_module(module)
    return module


V = _load_var_module()

# Inherited, not restated. Every one of these is a decision 03 documented and
# exercised in its own sensitivity table; a Granger test computed under
# different rules would be commenting on a model nobody fitted.
LEGS = V.LEGS
LEG_SOURCE = V.LEG_SOURCE
W = V.W.astype(float)
SCALE = V.SCALE
LAG_GRID = V.LAG_GRID
LAG_CAP = V.LAG_CAP
P_MIN = V.P_MIN
NW_LAGS = V.NW_LAGS
CHUNK = V.CHUNK
MIN_CELL_SECONDS = V.MIN_CELL_SECONDS
SAMPLE_STEPS = V.SAMPLE_STEPS
EPISODE_TAIL_SECONDS = V.EPISODE_TAIL_SECONDS
CRIT = V.CRIT

num, ci, fmt = V.num, V.ci, V.fmt
usable_rows, drop_small_cells, centring = (V.usable_rows, V.drop_small_cells,
                                           V.centring)
adjacent_at, load_regimes, identify_stress = (V.adjacent_at, V.load_regimes,
                                              V.identify_stress)
regime_roles = V.regime_roles


# ---------------------------------------------------------------- constants
#
# Labelled by provenance, as everywhere else:
#
#   derived      computed from something else, so the two cannot drift apart
#   convention   a stated choice with no data content, exercised in
#                04_sensitivity rather than asserted
#   inherited    03's, imported above rather than restated here

# derived. The equations tested. The first three are the legs; the fourth is
# the basis, which is not a fourth equation but the W combination of the three
# - z_t = z_{t-1} + w'dx_t is an identity, so the basis equation's
# coefficients are exactly coef @ W and its residual is exactly U @ W. That is
# checked in `audit` rather than assumed.
EQUATIONS = (*LEGS, "basis")

# convention. Horizons reported in the lead-lag table, in seconds. Geometric
# rather than consecutive, for the same reason 03's lag grid is: the
# interesting comparison is one second against ten against a minute. The
# figure draws every lag in the range, so the table cannot hide the shape.
CCF_MAX = 60
CCF_REPORT = (1, 2, 5, 10, 30, 60)

# convention. How far either side of zero the lead-lag figure is drawn. The
# cross-correlation is a spike at zero and a shoulder either side of it; at
# sixty seconds the shoulder is four pixels wide and the figure says only that
# the two series are contemporaneously correlated, which nobody doubted.
CCF_ZOOM = 20

# convention. How far out the asymmetry is scanned for its peak. Ten seconds
# is an order of magnitude past the one-second bounce the grid manufactures
# and well inside the horizon over which 03 found the basis closing, so a lead
# this project could act on lives inside it.
CCF_PEAK = 10

# convention. Day offsets used to build the placebo. A leg's lagged increments
# are replaced by the same leg's increments from another day at the same time
# of day, which preserves its own autocorrelation, its own volatility
# clustering and the diurnal pattern, and destroys only the cross-leg timing.
# Whole days, and never one, because adjacent days share a volatility level.
PLACEBO_OFFSETS = (-7, -3, 3, 7)

# convention. Which placebo value a measured effect must clear. With four
# offsets the distribution is four draws, so the maximum is used rather than a
# percentile that would be interpolating between two of them.
PLACEBO_RULE = "max"

# convention. An episode is used for the attribution only where the basis
# actually moved over the window being decomposed, measured against that
# episode's own peak. The shares are a ratio whose denominator is that move;
# near zero they are noise amplified rather than a large attribution.
MIN_MOVE_SHARE = 0.25

# convention. The lag from which the "beyond bounce" test starts. Roll noise
# is a moving average of order one, so a leg's own quotation noise reaches the
# basis's next increment at the first lag and nowhere else; a test that starts
# at the second lag cannot be explained by it. Two rather than three because
# raising it further discards real information along with the mechanical part.
BEYOND_BOUNCE_LAG = 2

# convention. Minimum rows a regime needs before this script will test it.
# Well above the 4K rule 03's fit uses, because a Wald statistic on a barely
# identified design is still a number, and a number in a table gets read.
MIN_TEST_ROWS = 20_000


def eq_weight(name):
    """The 3-vector that turns the system's residuals into this equation's."""
    if name == "basis":
        return W.copy()
    e = np.zeros(3)
    e[LEGS.index(name)] = 1.0
    return e


def eq_label(name):
    return "basis" if name == "basis" else PAIR_LABEL[LEG_SOURCE[name]]


def conditioning_label(base, first_lag):
    return base if first_lag == 1 else f"{base}, lags {first_lag}+"


def leg_label(j):
    return PAIR_LABEL[LEG_SOURCE[LEGS[j]]]


# ------------------------------------------------------------------- design
#
# One extended design, from which every model in this script is a column
# subset:
#
#   0                 constant
#   1                 z(-1), centred on the cell
#   2 .. 2+3p-1       the three legs' increments at lags 1..p, in LEGS order
#   2+3p .. 2+4p-1    the basis's own increments at lags 1..p
#
# The first 2+3p columns are exactly 03's design, which `audit` asserts rather
# than assumes. The trailing block is redundant with them when every column
# comes from the same array - the basis increment is their W combination - so
# the extended matrix is singular as a whole and is never inverted as a whole.
# Only its subsets are, and each of those is of full rank.
#
# The trailing block is built from its own array. Under the placebo one leg's
# lagged increments are replaced by another day's, and the basis's own history
# has to stay real or the control would be contaminated by the very shift the
# placebo applies.

def n_full(p):
    return 2 + 3 * p


def n_ext(p):
    return 2 + 4 * p


def lag_columns(p):
    return np.arange(2, 2 + 3 * p, dtype=np.int64)


def block_columns(p, leg_index):
    """Design columns carrying one leg's increments at every lag."""
    return np.array([2 + 3 * (i - 1) + leg_index for i in range(1, p + 1)],
                    dtype=np.int64)


def ctrl_columns(p):
    """Design columns carrying the basis's own increments at every lag."""
    return np.arange(2 + 3 * p, 2 + 4 * p, dtype=np.int64)


def full_columns(p):
    return np.arange(0, 2 + 3 * p, dtype=np.int64)


def pair_columns(p, leg_index):
    """The classical pairwise model: level, basis memory, one leg."""
    return np.concatenate(([0, 1], ctrl_columns(p),
                           block_columns(p, leg_index)))


def design(rows, DX_lag, DX_y, p, zc, y_off, DX_ctrl=None, ext=True):
    """
    (X, Y) for a block of rows.

    Two departures from 03's `_design`, both of which exist so the placebo can
    be built without a second implementation of anything:

      DX_lag vs DX_y   the lag columns and the dependent variable may come
                       from different arrays, which is what lets a regressor
                       be shifted while the thing being predicted is not.
      DX_ctrl          the array the basis's own lagged increments are built
                       from, which under the placebo must stay real.

    Passing one array in all three roles with ext=False reproduces 03's
    `_design` exactly, and `audit` asserts that rather than trusting it - the
    an assumption of this kind is worth an assertion rather than a comment.
    """
    m = rows.size
    K = n_ext(p) if ext else n_full(p)
    X = np.empty((m, K), dtype=float)
    X[:, 0] = 1.0
    X[:, 1] = zc
    for i in range(1, p + 1):
        X[:, 2 + 3 * (i - 1):2 + 3 * i] = DX_lag[rows - i]
    if ext:
        ctrl = DX_lag if DX_ctrl is None else DX_ctrl
        for i in range(1, p + 1):
            X[:, 2 + 3 * p + (i - 1)] = ctrl[rows - i] @ W
    return X, DX_y[rows] - y_off


def _chunks(n, size=CHUNK):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def accumulate(rows, DX_lag, DX_y, p, zc, y_off, DX_ctrl=None):
    """
    X'X, X'Y and Y'Y on the extended design, in chunks.

    Every quantity this script reports is a function of these three matrices
    and of one long-run covariance, so the design is never formed whole. That
    is what makes the restricted fits free: dropping a block from the model is
    dropping rows and columns from X'X, not another pass over a million
    seconds.
    """
    K = n_ext(p)
    XtX = np.zeros((K, K))
    XtY = np.zeros((K, 3))
    YtY = np.zeros((3, 3))
    for a, b in _chunks(rows.size):
        X, Y = design(rows[a:b], DX_lag, DX_y, p, zc[a:b], y_off[a:b],
                      DX_ctrl)
        XtX += X.T @ X
        XtY += X.T @ Y
        YtY += Y.T @ Y
    return XtX, XtY, YtY


def solve_ols(A, b, name=""):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print(f"    note: {name}: normal equations are singular; "
              f"least-squares solution used")
        return np.linalg.lstsq(A, b, rcond=None)[0]


def embed(values, cols, K):
    """Place a subset solution back into the extended coordinate system."""
    out = np.zeros((K,) + values.shape[1:])
    out[cols] = values
    return out


# ------------------------------------------------------------ HAC covariance

def lrcov(rows, DX_lag, DX_y, p, zc, y_off, specs, run, n, K_eff,
          DX_ctrl=None, m=NW_LAGS + 1):
    """
    Newey-West covariance of Q'beta, for several models in one pass.

    Convention: `Q` already carries the relevant (X'X)^-1, so
    what comes back is the covariance of the linear functionals themselves and
    needs no further sandwiching, including the n/(n-K) finite-sample factor
    03 applies. The long-run covariance is taken from overlapping block sums
    of length m rather than from a loop over lags - summing over blocks of
    length m and dividing by m reproduces the Bartlett kernel with truncation
    m-1 exactly, up to the first and last m-1 rows of each run. Blocks that
    would straddle a break are refused, which is what `run` is for. `audit`
    checks the result against 03's own `alpha_hac_cov` on the one functional
    both compute, so this implementation is verified against a calibrated one
    rather than merely believed.

    The full HAC matrix of the system is never formed. Each spec tests at most
    1 + 3p coefficients, so its meat matrix is at most twenty-five by
    twenty-five at the lag order 03 selects.

    `specs` is a list of (weight, beta, Q). The residual for a spec is
    Y @ weight - X @ beta, which is what lets a restricted model be scored
    against the same rows as the unrestricted one without a second pass.
    """
    out = [np.zeros((s[2].shape[1], s[2].shape[1])) for s in specs]
    for a, b in _chunks(n):
        # Windows start in [a, b) but reach m-1 rows past their start, so the
        # slice is extended. The overlap is recomputed rather than carried,
        # which costs m-1 rows of design per chunk and removes the only piece
        # of cross-chunk state this function would otherwise need.
        hi = min(n, b + m - 1)
        X, Y = design(rows[a:hi], DX_lag, DX_y, p, zc[a:hi], y_off[a:hi],
                      DX_ctrl)
        rn = run[a:hi]
        n_local = hi - a
        starts = min(n_local - m + 1, b - a)
        if starts <= 0:
            continue
        same = rn[m - 1:m - 1 + starts] == rn[:starts]
        for si, (w, beta, Q) in enumerate(specs):
            u = Y @ w - X @ beta
            phi = (X @ Q) * u[:, None]
            C = np.zeros((n_local + 1, Q.shape[1]))
            np.cumsum(phi, axis=0, out=C[1:])
            B = (C[m:m + starts] - C[:starts])[same]
            if B.shape[0] >= 2:
                out[si] += B.T @ B
    return [(S / m) * (n / max(n - K_eff, 1)) for S in out]


# ------------------------------------------------------------------- testing

def block_test(src, dst, conditioning, XtX, xy, tss, cols, block, S, beta,
               n, K, first_lag=1):
    """
    One block test, with the two numbers that matter next to it.

    Returned fields, and why each is there:

      wald, df, p       the test. Reported because a reader expects it, and
                        with the caveat that at this n it cannot discriminate.
      delta_r2          the share of this equation's variation that only this
                        block explains. Sample-size invariant, so it is
                        comparable across regimes of very different length.
      gain_pips         the same thing as a magnitude: the standard deviation
                        of the part of the next second's move that only this
                        block predicts, in pips per second. This is the number
                        for the text.
      horizon_s         where the block's weight sits, as the coefficient-
                        weighted mean lag. A block that predicts through
                        one-second bounce and one that predicts over half a
                        minute are different findings.

    `cols` are the model's columns in the extended coordinate system and
    `block` the subset being tested; both restricted and unrestricted fits are
    submatrices of one accumulation, so nothing here touches the data.
    """
    keep = np.setdiff1d(cols, block, assume_unique=False)
    b_u = solve_ols(XtX[np.ix_(cols, cols)], xy[cols], f"{src}->{dst}")
    b_r = solve_ols(XtX[np.ix_(keep, keep)], xy[keep], f"{src}->{dst} (r)")
    rss_u = float(tss - b_u @ xy[cols])
    rss_r = float(tss - b_r @ xy[keep])
    gain = max(rss_r - rss_u, 0.0)

    rb = beta[block]
    try:
        wald = float(rb @ np.linalg.solve(S, rb))
    except np.linalg.LinAlgError:
        wald = float("nan")
    df = int(block.size)
    pval = float(stats.chi2.sf(wald, df)) if np.isfinite(wald) else float("nan")

    weight = np.abs(rb)
    horizon = (float(np.arange(first_lag, first_lag + df) @ weight
                     / weight.sum()) if weight.sum() > 0 else np.nan)

    return {"from": src, "to": dst, "conditioning": conditioning,
            "n": int(n), "df": df, "wald": wald, "p": pval,
            "delta_r2": gain / tss if tss > 0 else np.nan,
            "gain_pips": float(np.sqrt(gain / max(n - K, 1))),
            "resid_sd_pips": float(np.sqrt(max(rss_u, 0.0) / max(n - K, 1))),
            "horizon_s": horizon}


def holm(pvals):
    """
    Holm step-down adjustment.

    Here so the multiplicity of testing every ordered pair in every regime is
    handled rather than ignored. It changes nothing - at this sample size the
    raw values are already at the floor of double precision - and that is
    itself the point the caption makes.
    """
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    m = int(np.isfinite(p).sum())
    out = np.full(p.shape, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, (m - rank) * p[i])
        out[i] = min(running, 1.0)
    return out


# -------------------------------------------------------- one regime, fitted

class RegimeFit:
    """
    Everything the tests for one regime need, in two passes over the rows.

    Pass one accumulates the extended cross-products. Pass two computes the
    long-run covariance of every functional that will be tested. Nothing else
    reads the data.
    """

    def __init__(self, name, rows, run, DX, Z, p, cell, DX_lag=None,
                 legs=None):
        self.name, self.rows, self.run, self.p = name, rows, run, p
        self.n = rows.size
        self.K_full, self.K_ext = n_full(p), n_ext(p)
        self.K_pair = 2 + 2 * p
        # Kept so `audit` can rebuild the design this fit actually used rather
        # than one it assumes was used. The placebo passes a different array
        # for the lags than for the dependent variable, and an audit that
        # reconstructed from one of them would check the wrong thing.
        self.DX, self.DX_lag = DX, (DX if DX_lag is None else DX_lag)
        self.DX_ctrl = DX
        self.legs = tuple(range(3)) if legs is None else tuple(legs)

        if self.n <= 4 * self.K_ext:
            raise ValueError(f"{name}: {self.n:,} rows is too few for "
                             f"{self.K_ext} columns")

        code, z_cell, y_cell = centring(rows, Z, DX, cell)
        self.zc = Z[rows - 1] - z_cell[code]
        self.y_off = y_cell[code]
        self.z_mean = float(z_cell.mean())

        self.XtX, self.XtY, self.YtY = accumulate(
            rows, self.DX_lag, DX, p, self.zc, self.y_off, self.DX_ctrl)

        self.fcols = full_columns(p)
        self.coef = solve_ols(self.XtX[np.ix_(self.fcols, self.fcols)],
                              self.XtY[self.fcols], name)

        uu = self.YtY - self.coef.T @ self.XtY[self.fcols]
        self.sigma = 0.5 * (uu + uu.T) / max(self.n - self.K_full, 1)
        self.alpha = self.coef[1, :].copy()
        self.lam = float(W @ self.alpha)

        self._build_specs()

    # -- the functionals that will be tested, and their covariances ---------

    def _build_specs(self):
        p, K = self.p, self.K_ext
        self.test_full = np.concatenate(([1], lag_columns(p)))

        specs, self.spec_key = [], []

        # The four full-system equations. One Q serves all of them: the same
        # columns are tested in each, only the residual differs.
        E = np.zeros((self.fcols.size, self.test_full.size))
        E[np.searchsorted(self.fcols, self.test_full),
          np.arange(self.test_full.size)] = 1.0
        Qf = embed(solve_ols(self.XtX[np.ix_(self.fcols, self.fcols)], E,
                             self.name), self.fcols, K)
        for eq in EQUATIONS:
            w = eq_weight(eq)
            specs.append((w, embed(self.coef @ w, self.fcols, K), Qf))
            self.spec_key.append(("full", eq))

        # The pairwise models, one per leg, for the basis equation only. The
        # leg equations do not need them: a leg's own lagged increments are
        # already a separate block there, so dropping another leg's block does
        # not remove the dependent variable's own memory.
        self.pair = {}
        xy_z = self.XtY @ W
        for j in self.legs:
            cols = pair_columns(p, j)
            block = block_columns(p, j)
            G = self.XtX[np.ix_(cols, cols)]
            b = solve_ols(G, xy_z[cols], f"{self.name} pairwise {leg_label(j)}")
            self.pair[j] = (cols, block, b)
            Ep = np.zeros((cols.size, block.size))
            Ep[np.searchsorted(cols, block), np.arange(block.size)] = 1.0
            Qp = embed(solve_ols(G, Ep), cols, K)
            specs.append((W.copy(), embed(b, cols, K), Qp))
            self.spec_key.append(("pair", j))

        # The finite-sample factor uses the widest model actually fitted, so
        # every covariance in this regime carries the same correction and the
        # pairwise and full-system standard errors stay comparable.
        Ss = lrcov(self.rows, self.DX_lag, self.DX, self.p, self.zc,
                   self.y_off, specs, self.run, self.n, self.K_full,
                   self.DX_ctrl)
        self.S = dict(zip(self.spec_key, Ss))

    # -- tests --------------------------------------------------------------

    @property
    def z_index(self):
        """Where the error-correction functional sits inside Q and S."""
        return int(np.searchsorted(self.test_full, 1))

    def full_test(self, eq, j, first_lag=1):
        """Leg j's block in one equation, conditioning on the whole system."""
        w = eq_weight(eq)
        block = block_columns(self.p, j)[first_lag - 1:]
        S = self.S[("full", eq)]
        idx = np.searchsorted(self.test_full, block)
        return block_test(
            leg_label(j), eq_label(eq), conditioning_label("whole system",
                                                           first_lag),
            self.XtX, self.XtY @ w, float(w @ self.YtY @ w), self.fcols,
            block, S[np.ix_(idx, idx)],
            embed(self.coef @ w, self.fcols, self.K_ext), self.n,
            self.K_full, first_lag)

    def pair_test(self, j, first_lag=1):
        """
        Leg j's block in the basis equation, given the basis's own memory.

        `first_lag` exists to separate a market lead from bid-ask bounce. The
        basis contains each leg's own quotation noise, so that leg's increment
        one second ago mechanically predicts the basis's increment now,
        whether or not anything in the market leads anything. Under Roll's
        model that noise is a moving average of order one and lives entirely
        at the first lag. Setting first_lag to two therefore drops the
        mechanical part - the lag-1 coefficient stays in the model, only the
        test moves past it - and what is left is predictability over horizons
        at which a stale or bouncing quote cannot account for it.
        """
        cols, block_all, b = self.pair[j]
        block = block_all[first_lag - 1:]
        S = self.S[("pair", j)][first_lag - 1:, first_lag - 1:]
        return block_test(
            leg_label(j), "basis", conditioning_label("basis memory",
                                                      first_lag),
            self.XtX, self.XtY @ W, float(W @ self.YtY @ W), cols, block, S,
            embed(b, cols, self.K_ext), self.n, self.K_pair, first_lag)

    def alpha_se(self):
        """
        Standard error of each leg's adjustment coefficient, and of lambda.

        The z(-1) functional sits in the same Q as the lag blocks, so this
        costs nothing beyond the pass already taken. Lambda's standard error
        comes from the basis equation directly rather than from w'Cov(alpha)w,
        because the basis equation is the W combination of the other three and
        its HAC variance is that quadratic form already.
        """
        i = self.z_index
        var = np.array([self.S[("full", leg)][i, i] for leg in LEGS])
        return (np.sqrt(np.maximum(var, 0.0)),
                float(np.sqrt(max(self.S[("full", "basis")][i, i], 0.0))))


# --------------------------------------------------------------- model-free 1

def attribution(d, labels, episodes_path=None):
    """
    Split the opening and the closing of every episode across the three legs,
    by arithmetic.

    The basis is a linear combination of the three log prices, so over any
    window the change in the basis is exactly the sum of the three legs'
    contributions, w_j times leg j's change. Nothing is estimated. The only
    choices are where each window starts and stops.

        opening   the last second before the peak at which the basis had not
                  yet reached half of it, forward to the peak.
        closing   the peak forward to the first second at which it had fallen
                  back to half.

    The two windows are mirror images across the peak and span the same half
    of the same gap, so both denominators are peak/2 by construction. An
    earlier version ran the opening window from the episode's first second
    instead; see the comment in the code for what that produced and why it
    had to go. Both are censored at EPISODE_TAIL_SECONDS and truncated at the
    first break in the clock.

    Two guards. A window is truncated at any second whose predecessor is not
    one grid step earlier, because a leg's change measured across a closure is
    a weekend of drift and not a repair. And a window is dropped when the
    basis barely moved across it, because the shares are a ratio whose
    denominator is that move; near zero they are noise amplified rather than a
    large attribution. Both counts are reported rather than absorbed.

    Every window is oriented by the direction the basis moved before anything
    is pooled, because whether a gap opened upwards or downwards is not the
    finding and a pooled denominator that mixes the two nearly cancels.

    Shares are pooled - the sum of the contributions over the sum of the
    moves - so they sum to one exactly and can be audited. The spread across
    episodes is reported next to them, because a pooled share hides whether
    every episode looked like that or one of them did.
    """
    path = Path(episodes_path or (DAT_DIR / "01_episodes.parquet"))
    if not path.exists():
        return None
    ep = pd.read_parquet(path)
    if not len(ep):
        return None

    x = SCALE * d[list(LEGS)].to_numpy(dtype=float)
    z = x @ W
    adj = adjacent(d.index)
    index = pd.DatetimeIndex(d.index)
    starts = index.get_indexer(pd.DatetimeIndex(ep["start"]))
    ends = index.get_indexer(pd.DatetimeIndex(ep["end"]))
    n = len(d)

    def forward_to_break(a, b):
        """The last index at or before b reachable from a without a jump."""
        if b <= a:
            return a
        brk = np.flatnonzero(~adj[a + 1:b + 1])
        return a + int(brk[0]) if brk.size else b

    def back_to_break(a, b):
        """The first index at or after a reachable from b without a jump."""
        if b <= a:
            return b
        brk = np.flatnonzero(~adj[a + 1:b + 1])
        return a + 1 + int(brk[-1]) if brk.size else a

    rows, dropped, unresolved = [], 0, 0
    for a, e in zip(starts, ends):
        if a < 0 or e < a:
            continue
        e = forward_to_break(a, e)
        seg = np.abs(z[a:e + 1])
        if not seg.size:
            continue
        k = a + int(np.argmax(seg))
        peak = float(np.abs(z[k]))
        if not np.isfinite(peak) or peak <= 0:
            continue
        half = 0.5 * peak

        # Closing: the widest second forward to the first at which the gap has
        # halved. Opening: the mirror image, the last second before the peak
        # at which it had not yet reached half, forward to the peak.
        #
        # The mirror is the point. The first version of this ran the opening
        # window from the episode's first second, which on the real sample
        # meant a median of fifty-two seconds and a change in the basis of a
        # pip and a half. Over fifty seconds each leg moves several pips for
        # its own reasons, those moves very nearly cancel in the combination,
        # and the shares that come out are ratios of large offsetting numbers
        # divided by a small one: +4.83 and -3.76 for a decomposition whose
        # parts are supposed to be shares. Both windows now span the same half
        # of the same peak, so both denominators are peak/2 by construction
        # and neither can collapse.
        lo_limit = back_to_break(max(0, k - EPISODE_TAIL_SECONDS), k)
        before = np.abs(z[lo_limit:k + 1])
        under = np.flatnonzero(before <= half)
        open_lo = lo_limit + int(under[-1]) if under.size else lo_limit

        hi_limit = forward_to_break(k, min(n - 1, k + EPISODE_TAIL_SECONDS))
        after = np.abs(z[k:hi_limit + 1])
        over = np.flatnonzero(after <= half)
        close_hi = k + int(over[0]) if over.size else hi_limit

        for phase, lo, hi, censored in (("opening", open_lo, k,
                                         not under.size),
                                        ("closing", k, close_hi,
                                         not over.size)):
            dz = float(z[hi] - z[lo])
            if hi <= lo or abs(dz) < MIN_MOVE_SHARE * peak:
                dropped += 1
                continue
            unresolved += int(censored)
            # Oriented by the direction the basis actually moved. A gap can
            # open either way and the direction is not the finding, so a
            # window in which the basis moved down is flipped before anything
            # is pooled. The identity survives the flip: both sides are
            # multiplied by the same sign, so the shares still sum to one.
            sign = 1.0 if dz > 0 else -1.0
            contrib = sign * W * (x[hi] - x[lo])
            rows.append({
                "regime": int(labels[k]), "phase": phase,
                "seconds": int(hi - lo), "peak": peak,
                "delta_z": abs(dz), "positive_gap": bool(z[k] > 0),
                "censored": censored,
                # How much leg movement it took to produce one pip of gap. At
                # one, the legs moved only as much as the gap did. At five,
                # they moved five times as far and mostly cancelled, and the
                # shares are a ratio of large numbers over a small one and
                # should not be read as an attribution. Reported rather than
                # used as a filter, because the filter would be a judgement
                # and the number is a fact.
                "travel": float(np.abs(contrib).sum() / abs(dz)),
                **{f"c_{leg}": float(c) for leg, c in zip(LEGS, contrib)},
            })

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.attrs["dropped"] = dropped
    out.attrs["unresolved"] = unresolved
    return out


def attribution_table(att, names):
    """Pooled share per leg, with the spread across episodes beside it."""
    cols = {}
    for k, name in enumerate(names):
        for phase in ("opening", "closing"):
            sub = att[(att["regime"] == k) & (att["phase"] == phase)]
            col = {"episodes": f"{len(sub):,}"}
            total = float(sub["delta_z"].sum()) if len(sub) else 0.0
            if len(sub) and total != 0:
                for j, leg in enumerate(LEGS):
                    share = float(sub[f"c_{leg}"].sum()) / total
                    per = (sub[f"c_{leg}"] / sub["delta_z"]).replace(
                        [np.inf, -np.inf], np.nan).dropna()
                    col[f"share, {leg_label(j)}"] = num(share, ".3f")
                    col[f"spread, {leg_label(j)}"] = (
                        f"[{per.quantile(.25):.2f}, {per.quantile(.75):.2f}]"
                        if len(per) else "—")
                col["median seconds"] = num(sub["seconds"].median(), ".0f")
                col["median peak (pips)"] = num(sub["peak"].median(), ".2f")
                col["median leg travel per pip of gap"] = num(
                    float(sub["travel"].median()), ".1f")
                col["gap was positive, share"] = num(
                    float(sub["positive_gap"].mean()), ".2f")
                col["half not reached"] = f"{int(sub['censored'].sum()):,}"
            cols[f"{name}, {phase}"] = col
    return pd.DataFrame(cols).fillna("—")


# --------------------------------------------------------------- model-free 2

def cross_corr(a, b, run, lags):
    """
    corr(a_t, b_{t-h}) for each h in `lags`, using only pairs inside one run.

    Positive h means b leads a. Rows within a run are consecutive seconds by
    construction, so a lag of h positions is a lag of h seconds, and the run
    test is what stops a pair being formed across a weekend. Negative lags
    swap the roles rather than take a second code path.
    """
    out = []
    for h in lags:
        if h == 0:
            u, v = a, b
        elif h > 0:
            same = run[h:] == run[:-h]
            u, v = a[h:][same], b[:-h][same]
        else:
            k = -h
            same = run[k:] == run[:-k]
            u, v = a[:-k][same], b[k:][same]
        if u.size < 3 or np.ptp(u) == 0 or np.ptp(v) == 0:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(u, v)[0, 1]))
    return np.array(out)


def lead_lag(fits, names, DX, lags):
    """
    Cross-correlations of every leg against the basis and against every other
    leg, on the rows each regime was fitted on.

    Computed on the fitted rows rather than on the whole file so that this and
    the tests describe the same seconds. Nothing else is shared: no
    coefficient, no lag order and no centring enters a correlation.
    """
    out = {}
    for name in names:
        f = fits.get(name)
        if f is None:
            continue
        dx = DX[f.rows]
        dz = dx @ W
        table = {}
        for j in range(3):
            table[(leg_label(j), "basis")] = cross_corr(dz, dx[:, j], f.run,
                                                        lags)
        for j in range(3):
            for i in range(3):
                if i != j:
                    table[(leg_label(i), leg_label(j))] = cross_corr(
                        dx[:, j], dx[:, i], f.run, lags)
        out[name] = table
    return out


def asymmetry(cc, fits, lags, names, horizon=CCF_PEAK):
    """
    How one-sided each leg's cross-correlation with the basis is.

    corr(basis_t, leg_{t-h}) minus corr(basis_t, leg_{t+h}). Anything a leg
    contributes to the basis contemporaneously - its own quotation noise, a
    stale quote, a shared shock - lands on both sides equally and cancels
    here. Only precedence survives. That is what makes this, and not the block
    effect sizes, the quantity that distinguishes a lead from the mechanics of
    a forward-filled grid.

    Reported at one second and at whichever horizon out to `horizon` the
    asymmetry is largest, because a lead can sit anywhere in that range and
    which second it sits at is itself informative. The reference is the
    standard 2/sqrt(n) for a correlation, widened for the peak: taking a
    maximum over `horizon` roughly independent horizons inflates it, and the
    widened band is the level a standard normal maximum of that many draws
    clears about five percent of the time. No placebo has to be constructed
    for either, which is the point of using a correlation here.
    """
    band_factor = float(np.sqrt(2.0 * np.log(max(horizon, 2))) + 0.7)
    out = {}
    for name in names:
        f = fits.get(name)
        table = cc.get(name)
        if f is None or table is None:
            continue
        base = 2.0 / np.sqrt(f.n)
        for j in range(3):
            v = table[(leg_label(j), "basis")]

            def at(h):
                return float(v[int(np.flatnonzero(lags == h)[0])])

            hs = np.arange(1, horizon + 1)
            vals = np.array([at(int(h)) - at(int(-h)) for h in hs])
            k = int(np.nanargmax(np.abs(vals))) if np.isfinite(vals).any() else 0
            out[(name, j)] = {"at_1": vals[0], "peak": vals[k],
                              "peak_h": int(hs[k]), "band": base,
                              "band_peak": band_factor / np.sqrt(f.n)}
    return out


def lead_lag_table(cc, names, lags):
    """
    The asymmetry, not the whole function.

    A correlation at +h and the same correlation at -h say opposite things
    about who leads, so the informative number is their difference. The table
    reports both sides at a few horizons and the difference next to them; the
    figure shows the whole function so the table cannot hide its shape.
    """
    rows = []
    for name in names:
        for (src, dst), values in cc.get(name, {}).items():
            for h in CCF_REPORT:
                pos = float(values[np.flatnonzero(lags == h)[0]])
                neg = float(values[np.flatnonzero(lags == -h)[0]])
                rows.append({"regime": name, "leads": src, "follows": dst,
                             "horizon (s)": h,
                             "corr, leads": num(pos, ".4f"),
                             "corr, follows": num(neg, ".4f"),
                             "asymmetry": num(pos - neg, ".4f")})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- placebo

def shifted_lags(d, DX, days):
    """
    Each leg's increments taken from another day at the same time of day.

    The placebo has to destroy the cross-leg timing and leave everything else
    alone. Shifting by a whole number of days at the same clock time preserves
    the leg's own autocorrelation, its own volatility level and the diurnal
    pattern of quoting; only the alignment with the other two legs is broken.
    Seconds whose counterpart does not exist - the shift lands in a weekend or
    outside the sample - are marked invalid and excluded by the same row
    admission the baseline uses, so the placebo is never fitted on a row the
    baseline would have refused for some other reason.
    """
    index = pd.DatetimeIndex(d.index)
    src = index.get_indexer(index + pd.Timedelta(days=days))
    valid = src >= 0
    out = np.full_like(DX, np.nan)
    out[valid] = DX[src[valid]]
    return out, valid


def placebo_threshold(placebo, regime, src, dst, conditioning):
    if placebo is None or not len(placebo):
        return np.nan
    sub = placebo[(placebo["regime"] == regime) & (placebo["from"] == src)
                  & (placebo["to"] == dst)
                  & (placebo["conditioning"] == conditioning)]
    v = pd.to_numeric(sub["gain_pips"], errors="coerce").dropna()
    if not len(v):
        return np.nan
    return float(v.max() if PLACEBO_RULE == "max" else v.quantile(0.95))


# --------------------------------------------------------------------- tables

def emit(frame, name, caption, label, index=True, view=None):
    """
    Write a table and show it.

    Everything this script produces is meant to be read from the run log as
    well as from the files, so a table that is saved without being printed is
    a table somebody has to go and open. `view` is for the two frames that are
    long in the file and only interesting in summary on screen.
    """
    save_table(frame, name, caption=caption, label=label, index=index)
    shown = frame if view is None else view
    print(f"\n{name}")
    print(shown.to_string(index=index if view is None else True))
    return frame


def causality_frame(fits, names):
    """Every test in every regime, in one long frame."""
    rows = []
    for name in names:
        f = fits.get(name)
        if f is None:
            continue
        for j in range(3):
            for first in (1, BEYOND_BOUNCE_LAG):
                if first > f.p:
                    continue
                rec = f.pair_test(j, first)
                rec["regime"] = name
                rows.append(rec)
        for eq in EQUATIONS:
            for j in range(3):
                if eq == LEGS[j]:
                    continue
                rec = f.full_test(eq, j)
                rec["regime"] = name
                rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def verdict(value, threshold):
    """
    A ratio, and a word that says only what the ratio supports.

    "Clears" rather than "yes": the threshold this is measured against bounds
    sampling variability, not the predictability the forward-filled grid
    creates on its own, so clearing it is a necessary condition for a finding
    and nowhere near a sufficient one. The multiple is what carries the
    information and it is printed for every cell, cleared or not.
    """
    if not (np.isfinite(value) and np.isfinite(threshold)) or threshold <= 0:
        return "—"
    ratio = value / threshold
    return (f"below ({ratio:.2f}x)" if ratio < 1.0
            else f"clears ({ratio:.1f}x)")


def pfmt(p):
    """
    A p-value that is zero to machine precision should say so.

    Writing 0.0000 into a table invites a reader to believe the number; this
    at least says the computation reached its floor. The caption says why none
    of these values carries information at this sample size.
    """
    if not np.isfinite(p):
        return "—"
    return "< 1e-12" if p < 1e-12 else f"{p:.2e}"


def basis_table(frame, placebo, names, asym):
    """
    The headline, one column per regime.

    Built as strings. Half these cells are verdicts or intervals with no value
    when a regime failed to fit, and the report inputs the file directly, so a
    NaN reaching it would print as a blank or as the word nan.
    """
    cols = {}
    for name in names:
        col = {}
        for j in range(3):
            tag = f"{leg_label(j)} -> basis"
            pair = frame[(frame["regime"] == name)
                         & (frame["from"] == leg_label(j))
                         & (frame["to"] == "basis")
                         & (frame["conditioning"] == "basis memory")]
            full = frame[(frame["regime"] == name)
                         & (frame["from"] == leg_label(j))
                         & (frame["to"] == "basis")
                         & (frame["conditioning"] == "whole system")]
            if not len(pair):
                continue
            r = pair.iloc[0]

            # The model-free quantity first, because it is the one that
            # separates precedence from the mechanics of a forward-filled
            # grid. Everything below it is corroboration.
            a = asym.get((name, j))
            if a is not None:
                col[f"{tag}, lead-lag asymmetry at 1 s"] = num(a["at_1"],
                                                               ".4f")
                col[f"{tag}, asymmetry, 2/sqrt(n)"] = num(a["band"], ".4f")
                col[f"{tag}, largest asymmetry to {CCF_PEAK} s"] = num(
                    a["peak"], ".4f")
                col[f"{tag}, at lag (s)"] = f"{a['peak_h']}"
                col[f"{tag}, asymmetry above the reference"] = verdict(
                    abs(a["peak"]), a["band_peak"])

            thr = placebo_threshold(placebo, name, leg_label(j), "basis",
                                    "basis memory")
            col[f"{tag}, incremental sd (pips/s)"] = num(r["gain_pips"], ".4f")
            col[f"{tag}, placebo threshold"] = num(thr, ".4f")
            col[f"{tag}, above sampling noise"] = verdict(r["gain_pips"], thr)
            col[f"{tag}, share of variance"] = num(r["delta_r2"], ".4f")
            col[f"{tag}, mean lag (s)"] = num(r["horizon_s"], ".1f")
            col[f"{tag}, Wald ({int(r['df'])} df)"] = num(r["wald"], ".1f")
            col[f"{tag}, p (Holm)"] = pfmt(r.get("p_holm", np.nan))
            beyond = frame[(frame["regime"] == name)
                           & (frame["from"] == leg_label(j))
                           & (frame["to"] == "basis")
                           & (frame["conditioning"]
                              == conditioning_label("basis memory",
                                                    BEYOND_BOUNCE_LAG))]
            if len(beyond):
                rb = beyond.iloc[0]
                tb = placebo_threshold(
                    placebo, name, leg_label(j), "basis",
                    conditioning_label("basis memory", BEYOND_BOUNCE_LAG))
                col[f"{tag}, beyond one second (pips/s)"] = num(rb["gain_pips"],
                                                                ".4f")
                col[f"{tag}, beyond one second, above sampling noise"] = (
                    verdict(rb["gain_pips"], tb))
            col[f"{tag}, conditioning on the whole system instead"] = (
                num(full["gain_pips"].iloc[0], ".4f") if len(full) else "—")
        cols[name] = col
    return pd.DataFrame(cols).fillna("—")


def leg_table(frame, placebo, names):
    cols = {}
    for name in names:
        col = {}
        for eq in LEGS:
            for j in range(3):
                if eq == LEGS[j]:
                    continue
                sub = frame[(frame["regime"] == name)
                            & (frame["from"] == leg_label(j))
                            & (frame["to"] == eq_label(eq))
                            & (frame["conditioning"] == "whole system")]
                if not len(sub):
                    continue
                r = sub.iloc[0]
                tag = f"{leg_label(j)} -> {eq_label(eq)}"
                thr = placebo_threshold(placebo, name, leg_label(j),
                                        eq_label(eq), "whole system")
                col[f"{tag}, incremental sd (pips/s)"] = num(r["gain_pips"],
                                                             ".4f")
                col[f"{tag}, placebo threshold"] = num(thr, ".4f")
                col[f"{tag}, above sampling noise"] = verdict(r["gain_pips"],
                                                             thr)
                col[f"{tag}, mean lag (s)"] = num(r["horizon_s"], ".1f")
        cols[name] = col
    return pd.DataFrame(cols).fillna("—")


def exogeneity_table(fits, names):
    """
    Does each leg respond to the level of the disequilibrium at all.

    A leg whose alpha is indistinguishable from zero is weakly exogenous: it
    does not return to the triangle, the others come to it. That is the
    error-correction half of the causality question, and it is the half that
    identifies which leg carries the permanent information. Signs are
    predicted before estimation - closure needs AUD/JPY negative and the other
    two positive - so the sign row is a test and not a description.
    """
    cols = {}
    for name in names:
        f = fits.get(name)
        if f is None:
            cols[name] = {}
            continue
        se, lam_se = f.alpha_se()
        col = {}
        for j, leg in enumerate(LEGS):
            lab = leg_label(j)
            a = float(f.alpha[j])
            expect = "< 0" if leg == "x_audjpy" else "> 0"
            got = (a < 0) if leg == "x_audjpy" else (a > 0)
            col[f"alpha, {lab}"] = num(a, ".4f")
            col[f"alpha, {lab}, HAC 95%"] = ci(a - CRIT * se[j],
                                               a + CRIT * se[j])
            col[f"alpha, {lab}, t"] = num(a / se[j] if se[j] > 0 else np.nan,
                                          ".1f")
            col[f"alpha, {lab}, predicted {expect}"] = ("holds" if got
                                                        else "FAILS")
        col["lambda"] = num(f.lam, ".4f")
        col["lambda, HAC 95%"] = ci(f.lam - CRIT * lam_se,
                                    f.lam + CRIT * lam_se)
        cols[name] = col
    return pd.DataFrame(cols).fillna("—")


# --------------------------------------------------------------------- figures

def panel_title(ax, title, note):
    """
    Title with a subtitle underneath it.

    The subtitle sits just above the axes, so the title has to be pushed clear
    of it. rcParams sets a pad of 10, which is the height of the subtitle
    itself and therefore exactly enough for the two to overlap; this passes a
    local pad instead of editing utils, because changing the default there
    would move every other figure in the project with it.
    """
    ax.set_title(title, pad=19)
    ax.annotate(note, xy=(0, 1.012), xycoords="axes fraction", fontsize=7.5,
                color="#666666", va="bottom", ha="left")


def figure_attribution(att, names, colours):
    # One scale across both panels. They are the same quantity measured over
    # the two halves of the same episode, and on separate scales a share of
    # 0.7 in one panel sits at the same height as a share of 0.9 in the other.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True,
                           gridspec_kw={"wspace": 0.12})
    width = 0.8 / max(len(names), 1)
    base = np.arange(3, dtype=float)

    for panel, phase in enumerate(("opening", "closing")):
        for k, name in enumerate(names):
            sub = att[(att["regime"] == k) & (att["phase"] == phase)]
            total = float(sub["delta_z"].sum()) if len(sub) else 0.0
            if not len(sub) or total == 0:
                continue
            vals = [float(sub[f"c_{leg}"].sum()) / total for leg in LEGS]
            ax[panel].bar(base + (k - (len(names) - 1) / 2) * width, vals,
                          width=width * 0.9, color=colours[k], lw=0,
                          label=name)
        ax[panel].axhline(0, color=RULE, lw=0.6)
        ax[panel].set_xticks(base)
        ax[panel].set_xticklabels([leg_label(j) for j in range(3)])
        if panel == 0:
            ax[panel].set_ylabel("share of the move")
        ax[panel].margins(y=0.18)
        ax[panel].legend(loc="best", ncol=len(names))

    panel_title(ax[0], "Which leg opened the gap",
                "half the peak up to the widest second; no model, shares sum "
                "to one")
    panel_title(ax[1], "Which leg closed it",
                "the widest second back down to half — the mirror of the "
                "left")
    save_fig(fig, "04_attribution")


def figure_lead_lag(cc, names, lags, fits, asym):
    """
    Two rows, because the interesting quantity is not the one the eye goes to.

    The top row is the cross-correlation itself, zoomed to the seconds either
    side of zero and with the contemporaneous value removed. That value is
    large, it is the same in every regime, and it is not a finding: the basis
    is built out of these three legs, so of course it moves with them within
    the second. Left in, it is a spike four pixels wide that flattens
    everything the figure exists to show.

    The bottom row is the asymmetry - the correlation at +h minus the
    correlation at -h - against the ±2/sqrt(n) reference. Anything a leg
    contributes to the basis contemporaneously lands on both sides equally and
    cancels here, so what is left is precedence and nothing else. A leg that
    leads shows a positive excursion; a leg that merely moves with the basis
    sits inside the band.
    """
    n = max(len(names), 1)
    # Shared within a row, and this matters for the bottom one: on its own
    # scale the calm panel's noise fills the frame and looks like structure.
    # On a scale the stressed panel sets, it is the flat line it is.
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6.6), squeeze=False,
                             sharey="row",
                             gridspec_kw={"hspace": 0.45, "wspace": 0.14})
    keep = np.abs(lags) <= CCF_ZOOM
    hs = np.arange(1, CCF_ZOOM + 1)

    for k, name in enumerate(names):
        table = cc.get(name, {})
        f = fits.get(name)
        top, bottom = axes[0][k], axes[1][k]

        for j in range(3):
            key = (leg_label(j), "basis")
            if key not in table:
                continue
            colour = LEG_COLOUR[LEG_SOURCE[LEGS[j]]]
            v = table[key].astype(float).copy()
            v[lags == 0] = np.nan          # the identity, not a result
            top.plot(lags[keep], v[keep], lw=1.3, color=colour,
                     label=leg_label(j))

            a = np.array([table[key][int(np.flatnonzero(lags == h)[0])]
                          - table[key][int(np.flatnonzero(lags == -h)[0])]
                          for h in hs])
            bottom.plot(hs, a, lw=1.4, color=colour, label=leg_label(j))

        if f is not None and f.n > 0:
            band = 2.0 / np.sqrt(f.n)
            bottom.axhspan(-band, band, color=MUTE, alpha=0.22, lw=0, zorder=0)
        for ax in (top, bottom):
            ax.axhline(0, color=RULE, lw=0.6, zorder=1)
        top.axvline(0, color=MUTE, lw=0.7, ls=":", zorder=0)

        top.set_xlabel("lag h (s); h > 0 means the leg leads")
        bottom.set_xlabel("lag h (s)")
        if k == 0:
            top.set_ylabel(r"corr(basis$_t$, leg$_{t-h}$)")
            bottom.set_ylabel("asymmetry: corr at $+h$ minus corr at $-h$")
        panel_title(top, name, "h = 0 removed: the identity, not a result")
        panel_title(bottom, f"{name} — who goes first", "band is ±2/√n")
        if k == 0:
            top.legend(loc="best")
            bottom.legend(loc="best")
    save_fig(fig, "04_lead_lag")


def figure_granger(frame, placebo, names, colours):
    sub_all = frame[(frame["to"] == "basis")
                    & (frame["conditioning"] == "basis memory")]
    beyond = frame[(frame["to"] == "basis")
                   & (frame["conditioning"]
                      == conditioning_label("basis memory",
                                            BEYOND_BOUNCE_LAG))]
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    width = 0.8 / max(len(names), 1)
    base = np.arange(3, dtype=float)

    for k, name in enumerate(names):
        vals, thr = [], []
        for j in range(3):
            row = sub_all[(sub_all["regime"] == name)
                          & (sub_all["from"] == leg_label(j))]
            vals.append(float(row["gain_pips"].iloc[0]) if len(row) else np.nan)
            thr.append(placebo_threshold(placebo, name, leg_label(j), "basis",
                                         "basis memory"))
        x = base + (k - (len(names) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.9, color=colours[k], lw=0, label=name)
        # The same test from the second lag onward, marked on the bar rather
        # than drawn as a second bar: a paler bar of the same hue is exactly
        # what a lighter regime shade looks like, and a reader would have to
        # check the legend to know which was which. A dashed rule cannot be
        # mistaken for a regime. What sits below it is the leg's own quotation
        # noise reaching the basis; what sits above it is not.
        inner = []
        for j in range(3):
            row = beyond[(beyond["regime"] == name)
                         & (beyond["from"] == leg_label(j))]
            inner.append(float(row["gain_pips"].iloc[0]) if len(row)
                         else np.nan)
        ax.plot(np.concatenate([[xi - width * 0.45, xi + width * 0.45, np.nan]
                                for xi in x]),
                np.concatenate([[v, v, np.nan] for v in inner]),
                color=RULE, lw=1.0, ls=(0, (3, 2)), zorder=6)
        # The placebo, drawn as a cap over each bar rather than as one line:
        # it is a different number in every cell, and a single line would
        # invite a reader to compare a bar against the wrong threshold.
        caps_x = np.concatenate([[xi - width * 0.45, xi + width * 0.45, np.nan]
                                 for xi in x])
        caps_y = np.concatenate([[t, t, np.nan] for t in thr])
        ax.plot(caps_x, caps_y, color=FOUND, lw=1.2, zorder=5)
    ax.plot([], [], color=RULE, lw=1.0, ls=(0, (3, 2)),
            label="of which beyond one second")
    ax.plot([], [], color=FOUND, lw=1.2,
            label="placebo (same leg, another day)")
    ax.set_xticks(base)
    ax.set_xticklabels([f"{leg_label(j)} → basis" for j in range(3)])
    ax.set_ylabel("incremental predictable move (pips/s)")
    ax.margins(y=0.24)
    panel_title(ax, "What one leg's recent history says about the next second "
                    "of the basis",
                "given the basis's own memory; the orange cap is sampling "
                "noise, which every bar clears on a one-second grid")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(names) + 2, borderaxespad=0.0)
    save_fig(fig, "04_granger")


# ------------------------------------------------------------------ self-check

def check_against_03(fits, names):
    """
    Every coefficient this script fits, against the ones 03 wrote out.

    The gap this closes: the audit already compared this script's Newey-West
    machinery against 03's, but it did so using *this* script's coefficients
    and residuals on both sides. A disagreement in the fit itself - a column
    in the wrong order, a row admitted here and not there, a centring applied
    twice - would have passed every check while making all of section 4 a
    commentary on a model nobody estimated. 04 exists to test what 03 fitted,
    so it has to prove it fitted the same thing.

    Skipped when 03 has not run, because the lag order then came from this
    script's own selection and there is nothing to compare against.
    """
    path = DAT_DIR / "03_var_model.parquet"
    if not path.exists():
        return []
    try:
        ref = pd.read_parquet(path)
    except Exception as exc:
        return [f"03_var_model.parquet could not be read: {exc}"]
    if not {"regime", "equation", "term", "coef"} <= set(ref.columns):
        return ["03_var_model.parquet has an unexpected shape; "
                "the cross-check against 03 was skipped"]

    problems = []
    for name in names:
        f = fits.get(name)
        sub = ref[ref["regime"] == name]
        if f is None or not len(sub):
            continue

        # 03's own lag order, read from its term names rather than assumed.
        # If it does not match this fit's, the two scripts are describing
        # different models and nothing below is worth comparing.
        depth = sub["term"].astype(str).str.extract(r"\(-(\d+)\)")[0].dropna()
        if len(depth) and int(depth.astype(int).max()) != f.p:
            problems.append(
                f"{name}: 03's model has lag order "
                f"{int(depth.astype(int).max())} and this fit has {f.p}; "
                f"rerun 03, or 04, so the two describe one model")
            continue

        terms = ["const", "z(-1)"]
        for i in range(1, f.p + 1):
            terms += [f"d {leg_label(j)} (-{i})" for j in range(3)]

        # A file that does not carry all three equations at the full depth did
        # not come out of a completed 03 run, so it is not evidence about this
        # script. Said out loud rather than treated as a disagreement, and
        # said out loud rather than skipped silently.
        table = {}
        for j in range(3):
            got = sub[sub["equation"] == leg_label(j)].set_index("term")["coef"]
            if any(t not in got.index for t in terms):
                print(f"    note: 03_var_model.parquet is incomplete for "
                      f"{name}/{leg_label(j)}; the cross-check against 03 was "
                      f"skipped for it")
                table = None
                break
            table[j] = got
        if table is None:
            continue

        for j, got in table.items():
            mine = f.coef[:, j]
            theirs = got.reindex(terms).to_numpy(dtype=float)
            scale = max(float(np.abs(theirs).max()), 1e-6)
            worst = float(np.abs(mine - theirs).max()) / scale
            if worst > 1e-6:
                problems.append(
                    f"{name}/{leg_label(j)}: coefficients differ from 03's by "
                    f"{worst:.2e} relative; 04 is not testing the model 03 "
                    f"fitted")
    return problems


def audit(fits, names, att, prefix="04_"):
    """
    Internal consistency.

    Every check corresponds to something that has gone wrong in this project,
    or that would be invisible if it went wrong here: a HAC implementation
    that disagrees with the calibrated one it was modelled on, a basis
    equation that is not the combination it claims to be, a design matrix that
    does not reproduce 03's, shares that do not sum to one, and the literal
    string 'nan' reaching a generated LaTeX table that the report inputs.
    """
    problems = []

    for name in names:
        f = fits.get(name)
        if f is None:
            continue

        # The basis equation is the W combination of the three, as an
        # identity. If this fails, every number in the headline is about
        # something other than the basis.
        if not np.allclose(f.coef @ W, f.coef @ eq_weight("basis"),
                           atol=1e-12):
            problems.append(f"{name}: basis equation is not w'coef")

        for key, S in f.S.items():
            scale = max(1.0, float(np.abs(S).max()))
            if not np.allclose(S, S.T, atol=1e-8 * scale):
                problems.append(f"{name}/{key}: HAC matrix is not symmetric")
            if np.linalg.eigvalsh(0.5 * (S + S.T)).min() < -1e-8 * scale:
                problems.append(f"{name}/{key}: HAC matrix is not PSD")

        # The extended design, restricted to its first 2+3p columns, must be
        # 03's design exactly. If it is not, this script is testing a model 03
        # did not fit.
        probe = slice(0, min(256, f.n))
        rows = f.rows[probe]
        Xa, Ya = design(rows, f.DX_lag, f.DX, f.p, f.zc[probe],
                        f.y_off[probe], f.DX_ctrl, ext=False)
        Xb, Yb = V._design(rows, f.DX_lag, f.p, f.zc[probe], f.y_off[probe])
        if not np.array_equal(Xa, Xb):
            problems.append(f"{name}: design disagrees with 03's _design")
        if not np.array_equal(Ya, Yb):
            problems.append(f"{name}: dependent variable disagrees with 03's")

        # The trailing control block must be the W combination of the leg
        # columns whenever both come from the same array, which is the whole
        # justification for calling it the basis's own memory. Under the
        # placebo they do not, and the check does not apply.
        if f.DX_lag is f.DX_ctrl:
            Xe, _ = design(rows, f.DX_lag, f.DX, f.p, f.zc[probe],
                           f.y_off[probe], f.DX_ctrl)
            for i in range(1, f.p + 1):
                lhs = Xe[:, 2 + 3 * (i - 1):2 + 3 * i] @ W
                rhs = Xe[:, 2 + 3 * f.p + (i - 1)]
                if not np.allclose(lhs, rhs, atol=1e-9, equal_nan=True):
                    problems.append(f"{name}: control column at lag {i} is "
                                    f"not the basis increment")
                    break

        # This script's HAC against 03's, on the one functional both compute.
        # 03's is the calibrated implementation; agreement here is what makes
        # the block tests believable.
        e = np.zeros((f.fcols.size, 1))
        e[int(np.searchsorted(f.fcols, 1)), 0] = 1.0
        q = solve_ols(f.XtX[np.ix_(f.fcols, f.fcols)], e, name)[:, 0]
        U = np.empty((f.n, 3))
        hq = np.empty(f.n)
        for a, b in _chunks(f.n):
            X, Y = design(f.rows[a:b], f.DX_lag, f.DX, f.p, f.zc[a:b],
                          f.y_off[a:b], f.DX_ctrl, ext=False)
            U[a:b] = Y - X @ f.coef
            hq[a:b] = X @ q
        ref = V.alpha_hac_cov(hq, U, f.run, NW_LAGS + 1, f.n, f.K_full)
        i = f.z_index
        mine = np.array([f.S[("full", leg)][i, i] for leg in LEGS])
        rel = np.abs(mine - np.diag(ref)) / np.maximum(np.abs(np.diag(ref)),
                                                       1e-300)
        if np.nanmax(rel) > 1e-8:
            problems.append(
                f"{name}: HAC variance of alpha disagrees with 03's "
                f"alpha_hac_cov by {np.nanmax(rel):.2e} relative")

    if att is not None and len(att):
        for (k, phase), sub in att.groupby(["regime", "phase"]):
            total = float(sub["delta_z"].sum())
            if total == 0:
                continue
            s = sum(float(sub[f"c_{leg}"].sum()) for leg in LEGS) / total
            if abs(s - 1.0) > 1e-6:
                problems.append(
                    f"regime {k}/{phase}: attribution shares sum to {s:.6f}, "
                    f"not one; the identity has been broken")

    problems.extend(check_against_03(fits, names))

    bad = re.compile(r"(?<![A-Za-z])(nan|inf|-inf|None)(?![A-Za-z])",
                     re.IGNORECASE)
    for folder, suffix in ((TAB_DIR, ".csv"), (TEX_DIR, ".tex")):
        for path in sorted(folder.glob(f"{prefix}*{suffix}")):
            hit = bad.search(path.read_text(encoding="utf-8"))
            if hit:
                problems.append(f"{path.name} contains '{hit.group(0)}'")

    if problems:
        for line in problems:
            print(f"  FAIL {line}")
        raise AssertionError(f"{len(problems)} internal check(s) failed")
    print("  all internal checks passed")


# ------------------------------------------------------------------ lag order

def lag_order(by_regime, DX, Z, cell):
    """
    03's lag order if 03 has run, otherwise the same criterion on the same
    grid.

    Reading it is preferred over recomputing it: 04 is a set of tests about
    the model 03 fitted, and a test at a different lag order is a test of a
    different model. Recomputing is the fallback rather than the default so
    that this script still runs from the regime labels alone, which is the
    only upstream input it truly requires.
    """
    path = DAT_DIR / "03_var_model.parquet"
    if path.exists():
        try:
            terms = pd.read_parquet(path, columns=["term"])["term"]
            found = terms.astype(str).str.extract(r"\(-(\d+)\)")[0].dropna()
            if len(found):
                p = int(found.astype(int).max())
                print(f" lag order p = {p}, read from 03_var_model.parquet; "
                      f"04 tests the specification 03 fitted")
                return p
        except Exception:
            print("    note: 03_var_model.parquet present but its lag order "
                  "could not be read")

    print(" 03 has not run; selecting a lag order here by the same criterion")
    best, best_bic = None, np.inf
    for p in LAG_GRID:
        total_n, total_ll, ok = 0, 0.0, True
        for name, (rows, run) in by_regime.items():
            try:
                f = V.fit_vecm(name, rows, run, DX, Z, p, cell, light=True)
            except (ValueError, np.linalg.LinAlgError):
                ok = False
                break
            sign, logdet = np.linalg.slogdet(f.sigma)
            if sign <= 0:
                ok = False
                break
            total_n += f.n
            total_ll += f.n * logdet
        if not ok:
            continue
        params = len(by_regime) * 3 * (2 + 3 * p)
        bic = total_ll + params * np.log(total_n)
        if bic < best_bic:
            best, best_bic = p, bic
    if best is None:
        raise RuntimeError("no lag order in LAG_GRID could be fitted")
    print(f" BIC selects p = {best}")
    return best


# ----------------------------------------------------------------- sensitivity

def sensitivity(d, DX, Z, X, labels, names, fits, ok, roll, pmax, cell,
                p_star, stress_k):
    """
    One row per variant, reporting the headline effect size for every leg into
    the basis.

    Exactly one thing changes per row, and `ok_v` replaces the baseline mask
    outright rather than intersecting with it, so a variant that puts rollover
    back in is not silently still excluding it.
    """
    epoch_s = (pd.DatetimeIndex(d.index).to_numpy()
               .astype("datetime64[s]").astype(np.int64))

    def summarise(tag, fitted, note=""):
        row = {"variant": tag}
        for name in names:
            g = fitted.get(name)
            for j in range(3):
                row[f"{leg_label(j)} -> basis, {name}"] = (
                    num(g.pair_test(j)["gain_pips"], ".4f") if g is not None
                    else "—")
        row["note"] = note
        return row

    def fit_all(tag, rows_all, run_all, lab, DXv, Zv, cel, p, floor):
        out = {}
        for k, name in enumerate(names):
            sel = lab[rows_all] == k
            if sel.sum() < floor:
                out[name] = None
                continue
            try:
                out[name] = RegimeFit(name, rows_all[sel], run_all[sel], DXv,
                                      Zv, p, cel)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"  {tag} / {name}: {exc}")
                out[name] = None
        return out

    def run_variant(tag, p=None, min_prob=None, ok_v=None, centre=None,
                    note=""):
        p = p_star if p is None else p
        if min_prob is not None:
            base_ok = (~roll) & (pmax >= min_prob)
        elif ok_v is not None:
            base_ok = ok_v
        else:
            base_ok = ok
        cel = cell if centre is None else centre
        r_all, run_v = usable_rows(d.index, LAG_CAP, base_ok, cel)
        r_all, run_v, _ = drop_small_cells(r_all, run_v, cel, MIN_CELL_SECONDS)
        return summarise(tag, fit_all(tag, r_all, run_v, labels, DX, Z, cel, p,
                                      MIN_TEST_ROWS), note)

    def frequency_variant(step_s):
        """
        The same tests on a coarser sample of the same seconds.

        The check that matters most for a lead-lag result on this data. The
        one-second grid is forward-filled, so a leg that appears to lead may
        only be quoting more often than the others. A lead that is real in the
        market survives being sampled less often; one manufactured by the fill
        does not. The lag order is set to span the same wall-clock memory at
        every frequency, so the only thing changing is the sample.
        """
        sel = np.flatnonzero(epoch_s % step_s == 0)
        sub = d.index[sel]
        Xs = X[sel]
        Zs = Xs @ W
        DXs = np.full_like(Xs, np.nan)
        DXs[1:] = Xs[1:] - Xs[:-1]
        DXs[~adjacent_at(sub, step_s)] = np.nan
        cell_s, lab_s = cell[sel], labels[sel]
        p_s = max(1, int(round(p_star * GRID_SECONDS / step_s)))
        r_all, run_v = usable_rows(sub, p_s, ok[sel], cell_s,
                                   step_seconds=step_s)
        r_all, run_v, _ = drop_small_cells(
            r_all, run_v, cell_s, max(2, MIN_CELL_SECONDS // step_s))
        # The row floor is expressed per second; at a coarser sample the same
        # wall-clock window holds proportionally fewer rows, so the floor has
        # to scale with it or every frequency row would come back empty.
        floor = max(4 * n_ext(p_s), MIN_TEST_ROWS // step_s)
        return summarise(f"sampled every {step_s} s",
                         fit_all(f"{step_s}s", r_all, run_v, lab_s, DXs, Zs,
                                 cell_s, p_s, floor),
                         f"p = {p_s}, the same {p_s * step_s} s of memory")

    variants = [summarise(f"baseline (p = {p_star}, posterior at least "
                          f"{P_MIN:.0%})", fits,
                          "the fit everything else is compared against")]
    for p in sorted({LAG_GRID[0], 5, LAG_CAP} - {p_star}):
        variants.append(run_variant(f"lag order p = {p}", p=p,
                                    note="transitory memory"))
    variants.append(run_variant("posterior at least 50% (hard boundary)",
                                min_prob=0.50,
                                note="the buffer around the changepoints "
                                     "switched off"))
    variants.append(run_variant("posterior at least 99%", min_prob=0.99,
                                note="a wider buffer"))
    variants.append(run_variant("rollover seconds included",
                                ok_v=(pmax >= P_MIN),
                                note="17:00-17:30 back in"))
    variants.append(run_variant("centred on the regime, not the day",
                                centre=labels.astype(np.int64),
                                note="one level for the whole regime rather "
                                     "than one for each day"))
    for step_s in SAMPLE_STEPS:
        variants.append(frequency_variant(step_s))

    out = pd.DataFrame(variants).set_index("variant")

    # Say out loud which rows move the answer, so a choice that changes the
    # headline has to be argued for in the text instead of being buried.
    key = [c for c in out.columns if c.endswith(f", {names[stress_k]}")]
    if key:
        base = pd.to_numeric(out[key[0]].iloc[0], errors="coerce")
        moved = [(tag, v) for tag, v in
                 zip(out.index, pd.to_numeric(out[key[0]], errors="coerce"))
                 if np.isfinite(base) and np.isfinite(v) and base > 0
                 and max(v / base, base / max(v, 1e-12)) > 1.5]
        if moved:
            print(f" variants that move '{key[0]}' from {base:.4f}:")
            for tag, v in moved:
                print(f"   {v:8.4f}  {tag}")
        else:
            print(" no variant moves the headline effect size by more than "
                  "half again")
    return out


# ----------------------------------------------------------------------- main

def main():
    make_dirs()
    set_style()

    # ---------------------------------------------------------------- load
    print("load")
    path = DAT_DIR / "01_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 01 first")
    d = pd.read_parquet(path).sort_index()
    missing = [c for c in (*LEGS, "basis") if c not in d.columns]
    if missing:
        raise KeyError(f"01_clean.parquet has no {missing} — rerun 01")
    info = check_grid(d.index, name="01_clean.parquet index")
    print(f" {len(d):,} open-market seconds, {d.index[0]} to {d.index[-1]}")
    print(f" resolution {info['resolution']}, "
          f"{info['adjacent_share']:.1%} of rows one second after the last")

    X = SCALE * d[list(LEGS)].to_numpy(dtype=float)
    Z = X @ W
    DX = np.full_like(X, np.nan)
    DX[1:] = X[1:] - X[:-1]
    DX[~adjacent(d.index)] = np.nan

    ratio = robust_scale(Z) / robust_scale(d["basis"].to_numpy())
    print(f" scaled log basis is {ratio:.4f} x the pip basis of 01")

    # ------------------------------------------------------------- regimes
    print("regimes")
    labels, prob, seg_names = load_regimes(d)
    n_regimes = prob.shape[1]
    pmax = prob.max(axis=1)
    stress_k, boj_k, mad = identify_stress(d, labels, n_regimes)
    names = regime_roles(n_regimes, stress_k)
    colours = list(regime_colours(n_regimes))

    for k, name in enumerate(names):
        sel = np.flatnonzero(labels == k)
        lo, hi = ((d.index[sel[0]], d.index[sel[-1]]) if sel.size
                  else (pd.NaT, pd.NaT))
        print(f"  {seg_names[k]:<12} -> {name:<8} {lo} .. {hi}  "
              f"MAD {num(mad[k], '.4f')} pips")
    if boj_k != stress_k:
        print(f"  WARNING: dispersion says segment {stress_k + 1} is the "
              f"stressed one, but the decision falls in segment {boj_k + 1}. "
              f"The label is taken from dispersion; check the segmentation.")

    roll = in_rollover(d.index).to_numpy()
    day_code = pd.factorize(pd.DatetimeIndex(d.index).floor("D"))[0]
    cell = labels.astype(np.int64) * (day_code.max() + 1) + day_code
    ok = (~roll) & (pmax >= P_MIN)

    # ---------------------------------------------------------------- rows
    print("rows")
    rows_all, run_all = usable_rows(d.index, LAG_CAP, ok, cell)
    rows_all, run_all, tiny = drop_small_cells(rows_all, run_all, cell,
                                               MIN_CELL_SECONDS)
    print(f" {rows_all.size:,} of {len(d):,} seconds admitted at p = "
          f"{LAG_CAP} (03's rules, imported not restated); {tiny:,} dropped "
          f"in day cells shorter than {MIN_CELL_SECONDS:,} seconds")

    by_regime = {}
    for k, name in enumerate(names):
        sel = labels[rows_all] == k
        by_regime[name] = (rows_all[sel], run_all[sel])
        print(f"  {name:<8} {sel.sum():,} rows ({sel.sum() / 3600:.1f} h)")

    p_star = lag_order(by_regime, DX, Z, cell)

    # ------------------------------------------------------------ baseline
    print("fit")
    fits = {}
    for name in names:
        rows, run = by_regime[name]
        if rows.size < MIN_TEST_ROWS:
            print(f" {name:<8} {rows.size:,} rows is below the "
                  f"{MIN_TEST_ROWS:,} this script will test on; skipped")
            continue
        try:
            f = RegimeFit(name, rows, run, DX, Z, p_star, cell)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f" {name:<8} not fitted: {exc}")
            continue
        fits[name] = f
        print(f" {name:<8} n {f.n:>10,}  lambda {f.lam:+.4f}  "
              f"columns {f.K_full} full / {f.K_pair} pairwise  "
              f"basis innovation sd {np.sqrt(W @ f.sigma @ W):.4f} pips")

    if not fits:
        raise RuntimeError("no regime had enough admissible rows to test")
    fitted = list(fits)

    # -------------------------------------------------------- model-free 1
    print("attribution")
    att = attribution(d, labels)
    if att is None:
        print(" 01_episodes.parquet absent or empty; the model-free "
              "attribution is omitted")
    else:
        print(f" {len(att):,} episode windows decomposed, "
              f"{att.attrs['dropped']:,} dropped as too small a move, "
              f"{att.attrs['unresolved']:,} that never reached half the peak")
        for k, name in enumerate(names):
            for phase in ("opening", "closing"):
                sub = att[(att["regime"] == k) & (att["phase"] == phase)]
                total = float(sub["delta_z"].sum()) if len(sub) else 0.0
                if not len(sub) or total == 0:
                    continue
                shares = "  ".join(
                    f"{leg_label(j)} "
                    f"{float(sub[f'c_{LEGS[j]}'].sum()) / total:+.2f}"
                    for j in range(3))
                print(f"  {name:<8} {phase:<8} n {len(sub):>5,}   {shares}"
                      f"   travel {sub['travel'].median():.1f}x")

    # -------------------------------------------------------- model-free 2
    print("lead-lag")
    lags = np.arange(-CCF_MAX, CCF_MAX + 1)
    cc = lead_lag(fits, names, DX, lags)
    at1 = int(np.flatnonzero(lags == 1)[0])
    at_1 = int(np.flatnonzero(lags == -1)[0])
    for name in fitted:
        for j in range(3):
            v = cc[name][(leg_label(j), "basis")]
            print(f"  {name:<8} {leg_label(j):<8} leads {v[at1]:+.4f}  "
                  f"follows {v[at_1]:+.4f}  "
                  f"asymmetry {v[at1] - v[at_1]:+.4f}")

    # --------------------------------------------------------------- tests
    print("tests")
    tests = causality_frame(fits, fitted)
    for _, r in tests[(tests["to"] == "basis")
                      & (tests["conditioning"] == "basis memory")].iterrows():
        print(f"  {r['regime']:<8} {r['from']:<8} -> basis   "
              f"gain {r['gain_pips']:.4f} pips/s   "
              f"share of variance {r['delta_r2']:.4f}   "
              f"mean lag {r['horizon_s']:.1f} s")

    # ------------------------------------------------------------- placebo
    #
    # The same tests, with one leg's lagged increments taken from another day.
    # Everything else - the dependent variable, the error-correction term, the
    # basis's own memory, the other two legs, the row admission - is the
    # baseline's.
    print("placebo")
    placebo_rows = []
    for days in PLACEBO_OFFSETS:
        DXs, valid = shifted_lags(d, DX, days)
        r_all, run_p = usable_rows(d.index, LAG_CAP, ok & valid, cell)
        r_all, run_p, _ = drop_small_cells(r_all, run_p, cell,
                                           MIN_CELL_SECONDS)
        for k, name in enumerate(names):
            if name not in fits:
                continue
            sel = labels[r_all] == k
            if sel.sum() < MIN_TEST_ROWS:
                continue
            # One leg at a time. Shifting all three together would preserve
            # their mutual timing, which is the one thing the placebo has to
            # destroy; shifting each by a different offset would leave no
            # unshifted control to measure the block against.
            for j in range(3):
                mixed = DX.copy()
                mixed[:, j] = DXs[:, j]
                try:
                    g = RegimeFit(name, r_all[sel], run_p[sel], DX, Z, p_star,
                                  cell, DX_lag=mixed, legs=(j,))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    print(f"  {days:+d}d / {name} / {leg_label(j)}: {exc}")
                    continue
                for first in (1, BEYOND_BOUNCE_LAG):
                    if first > g.p:
                        continue
                    rec = g.pair_test(j, first)
                    rec.update({"regime": name, "offset_days": days})
                    placebo_rows.append(rec)
                for eq in EQUATIONS:
                    if eq == LEGS[j]:
                        continue
                    rec = g.full_test(eq, j)
                    rec.update({"regime": name, "offset_days": days})
                    placebo_rows.append(rec)
        print(f"  {days:+d} days: {r_all.size:,} rows admitted")

    placebo = pd.DataFrame(placebo_rows)
    if len(placebo):
        # A threshold is a maximum over the offsets that survived, and an
        # offset does not survive if shifting by it lands on a weekend or off
        # the end of the sample often enough to take the regime below the row
        # floor. A maximum over one draw is not a bound, so it is named here
        # rather than quietly used.
        counts = placebo.groupby(["regime", "conditioning", "from",
                                  "to"]).size()
        thin = counts[counts < 2]
        if len(thin):
            print(f"  WARNING: {len(thin)} placebo threshold(s) rest on a "
                  f"single offset, so they bound nothing; widen "
                  f"PLACEBO_OFFSETS or read those cells as unbounded")
            for key in list(thin.index)[:6]:
                print(f"    {' / '.join(str(k) for k in key)}")
        summary = (placebo[(placebo["to"] == "basis")
                           & (placebo["conditioning"] == "basis memory")]
                   .groupby(["regime", "from"])["gain_pips"]
                   .agg(["median", "max"]).reset_index())
        print(" placebo effect sizes into the basis, pips/s:")
        for _, r in summary.iterrows():
            print(f"  {r['regime']:<8} {r['from']:<8} median "
                  f"{r['median']:.4f}   max {r['max']:.4f}")

    # -------------------------------------------------------------- tables
    print("tables")
    asym = asymmetry(cc, fits, lags, fitted)
    head = basis_table(tests, placebo, fitted, asym)
    emit(
        head, "04_basis_causality",
        caption=("Who leads the basis, by estimated regime. The first "
                 "rows of each block are model-free: the asymmetry of the "
                 "cross-correlation between that leg's increments and the "
                 "basis's, which is what separates precedence from the "
                 "mechanics of a forward-filled grid, because anything a leg "
                 "contributes to the basis contemporaneously lands on both "
                 "sides of the correlation equally and cancels. The rest are "
                 "block Granger tests. Each block is one leg's increments at "
                 "every lag from one second to the selected order. The headline rows "
                 "condition on the basis's own lagged increments and on the "
                 "level of the disequilibrium, so what they measure is what "
                 "the leg adds beyond the basis's own memory; the last row "
                 "of each block gives the same quantity conditioning on the "
                 "whole system instead, where the three legs together span "
                 "that memory and the three tests are therefore not "
                 "comparable across legs. The incremental standard deviation "
                 "is the part of the next second's move in the basis that "
                 "only this block predicts. The placebo threshold is the "
                 "largest value the same test produced when that leg's "
                 "history was replaced by its own increments from another "
                 "day at the same time, which leaves its autocorrelation and "
                 "its volatility clustering intact and destroys only the "
                 "alignment with the other legs. It bounds sampling "
                 "variability and nothing else: the basis contains each "
                 "leg's own quotation noise, so on a one-second grid every "
                 "block clears its placebo whether or not anything leads "
                 "anything, and a run of this script on a synthetic series "
                 "with no lead in it confirms that every cell still reads "
                 "two to seven times its placebo. Clearing the placebo is "
                 "therefore necessary and far from sufficient, and the "
                 "wording of those rows says so. The beyond-one-second rows "
                 "repeat the test from the second lag onward: the basis "
                 "contains each leg's own quotation noise, so a leg's last "
                 "second predicts the basis mechanically whether or not "
                 "anything leads anything, and under Roll's model that noise "
                 "lives entirely at the first lag. A block that still clears "
                 "its placebo from the second lag onward is not bid-ask "
                 "bounce. The Wald statistics are Newey-West and the "
                 "p-values are Holm-adjusted across every test in this "
                 "section. They were included on the expectation that at "
                 "these sample sizes they would reject everywhere and "
                 "therefore say nothing; in the event several blocks do not "
                 "reject at all, which is worth more than the rejections. "
                 "Where a block rejects, the effect size and the placebo "
                 "columns are still what decide whether it matters."),
        label="tab:basis_causality",
    )

    legs_out = leg_table(tests, placebo, fitted)
    emit(
        legs_out, "04_leg_causality",
        caption=("The same test between the legs themselves, in both "
                 "directions, conditioning on the whole system. A pair in "
                 "which one direction clears its placebo and the reverse "
                 "does not is an ordering of information; a pair in which "
                 "both clear it is a common factor reaching the two at "
                 "slightly different speeds and should not be read as one "
                 "leg driving the other."),
        label="tab:leg_causality",
    )

    exo = exogeneity_table(fits, fitted)
    emit(
        exo, "04_weak_exogeneity",
        caption=("Does each leg respond to the level of the disequilibrium. "
                 "A leg whose adjustment coefficient cannot be distinguished "
                 "from zero is weakly exogenous: it does not return to the "
                 "triangle, the others come to it. Signs were predicted "
                 "before estimation, so the last row of each block is a test "
                 "and not a description. Intervals are Newey-West at "
                 f"{NW_LAGS} seconds and reproduce those of section 3, which "
                 "is checked rather than asserted. Information shares of the "
                 "Hasbrouck or Gonzalo-Granger kind are not reported: three "
                 "prices under one cointegrating restriction leave two "
                 "common trends, so no scalar share per leg is identified."),
        label="tab:weak_exogeneity",
    )

    if att is not None:
        att_out = attribution_table(att, names)
        emit(
            att_out, "04_attribution",
            caption=("Which leg opened each dislocation and which leg closed "
                     "it, by arithmetic. The basis is a fixed linear "
                     "combination of the three log prices, so its change "
                     "over any window is exactly the sum of the three legs' "
                     "contributions and the shares sum to one; no model, no "
                     "lag order and no regime label enters the calculation. "
                     "The two windows are mirror images across the peak: "
                     "from the last second at which the gap had not yet "
                     "reached half its eventual width, up to the peak; and "
                     "from the peak back down to half. Both therefore span "
                     "the same half of the same gap and neither denominator "
                     "can collapse. Windows are truncated at any break in "
                     "the clock. The travel row is the diagnostic to read "
                     "before the shares: it is how far the three legs moved "
                     "in total per pip of gap produced, so at one the legs "
                     "moved only as much as the gap did and at five they "
                     "moved five times as far and mostly cancelled, and "
                     "shares outside the unit interval are to be expected. "
                     "The spread is the interquartile range across episodes, "
                     "which is what says whether the pooled share describes "
                     "every episode or one of them."),
            label="tab:attribution",
        )

    ll_out = lead_lag_table(cc, fitted, lags)
    # Long in the file - every pair at every horizon - and only the legs
    # against the basis are worth a screen, so the console gets the pivot.
    ll_view = (ll_out[ll_out["follows"] == "basis"]
               .pivot_table(index=["leads", "horizon (s)"], columns="regime",
                            values="asymmetry", aggfunc="first"))
    emit(
        ll_out, "04_lead_lag", index=False, view=ll_view,
        caption=("Cross-correlation between one series' increments and "
                 "another's at matched leads and lags, computed only on "
                 "pairs inside one unbroken run so nothing is correlated "
                 "across a closure. The asymmetry column is the informative "
                 "one: a symmetric cross-correlation is common information "
                 "reaching both, and only the difference between the two "
                 "sides is an ordering."),
        label="tab:lead_lag",
    )

    if len(placebo):
        null_out = (placebo.groupby(["regime", "conditioning", "from", "to"])
                    ["gain_pips"].agg(offsets="size", smallest="min",
                                      median="median", largest="max")
                    .reset_index())
        for c in ("smallest", "median", "largest"):
            null_out[c] = [num(v, ".4f") for v in null_out[c]]
        emit(
            null_out, "04_null_calibration", index=False,
            caption=("What every test in this section returns when the null "
                     "is true. One leg's lagged increments are replaced by "
                     "its own increments from another day at the same time "
                     "of day, which preserves that leg's autocorrelation, "
                     "its volatility clustering and the diurnal pattern of "
                     "quoting, and destroys only its alignment with the "
                     "other two. Everything else is the baseline's. The "
                     "largest value over the offsets is the threshold used "
                     "in the tables above."),
            label="tab:null_calibration",
        )

    # --------------------------------------------------------- sensitivity
    print("sensitivity")
    sens = sensitivity(d, DX, Z, X, labels, names, fits, ok, roll, pmax, cell,
                       p_star, stress_k)
    emit(
        sens, "04_sensitivity",
        caption=("Sensitivity of the causality result, reported as the "
                 "incremental predictable move each leg buys in the basis "
                 "equation. The first block varies the transitory lag order, "
                 "the second the posterior threshold that buffers the "
                 "regime boundaries, the third the excluded rollover and "
                 "what the error-correction term is measured against, and "
                 "the fourth the sampling interval. The frequency rows are "
                 "the ones to read first: a lead that is real in the market "
                 "survives being sampled less often, while one manufactured "
                 "by forward-filling a one-second grid does not."),
        label="tab:granger_sensitivity",
    )

    # -------------------------------------------------------------- figures
    print("figures")
    if att is not None:
        figure_attribution(att, names, colours)
    figure_lead_lag(cc, fitted, lags, fits, asym)
    figure_granger(tests, placebo, fitted, colours)

    # ---------------------------------------------------------------- write
    print("data")
    frames = [tests.assign(family="measured")]
    if len(placebo):
        frames.append(placebo.assign(family="placebo"))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(DAT_DIR / "04_granger_tests.parquet", index=False)
    print(f" -> 04_granger_tests.parquet ({len(out):,} tests, p = {p_star})")

    print("checks")
    audit(fits, fitted, att)


if __name__ == "__main__":
    main()