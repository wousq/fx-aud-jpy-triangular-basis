"""
03_var.py

Does the triangle still close under stress, how fast, and which leg does the
closing?

Script 02 established *when* the basis changed structure and wrote a
per-second regime label. This script conditions on that label and asks what
the price dynamics look like inside each regime.

The model
    Triangular no-arbitrage is a cointegrating restriction with a *known*
    vector. Writing x for scaled log prices,

        z_t = x^AUDJPY_t - x^AUDUSD_t - x^USDJPY_t

    is stationary if the triangle holds, and it is (to first order) the same
    quantity 01 and 02 measured in pips. Nothing has to be estimated to know
    the cointegrating vector, which removes the entire Johansen apparatus and
    with it the usual argument about rank. What is left is a VECM with the
    restriction imposed:

        dx_t = c + alpha z_{t-1} + sum_{i=1..p} Gamma_i dx_{t-i} + e_t

    fitted separately within each estimated regime. alpha is the object of
    interest: the share of the current gap each leg erases per second. Its
    combination

        lambda = w'alpha,     w = (-1, -1, +1)

    is the closure rate of the basis itself, because z_t = z_{t-1} + w'dx_t
    holds as an identity, not as an approximation.

    Signs are predicted before estimation, which makes them a test rather
    than a description. z > 0 means the direct rate is rich against the
    synthetic, so closure requires alpha_AUDJPY < 0, alpha_AUDUSD > 0,
    alpha_USDJPY > 0, and therefore lambda < 0.

Why this and not a plain VAR in returns
    A VAR in returns throws away the level of the basis, which is the only
    variable in the system that is not a martingale and the only one the
    no-arbitrage condition says anything about. The error-correction term is
    the arbitrage.

Units
    Log prices are scaled by 1e4. alpha and lambda are ratios of two
    quantities on that scale, so they are dimensionless per-second rates and
    the scale cancels; it survives only in the residual standard deviations,
    where a unit is an AUD/JPY pip to within about two percent at the sample
    average of the cross. The measured ratio is printed at load so the claim
    is checked rather than asserted.

Regimes, and the choice this script makes
    The regime labels are read from 02_regimes.parquet, and 02 wrote the
    *median* segmentation: a second is elevated when more than half the
    posterior mass says it is. That segmentation opens the stressed regime on
    30 July, 8.5 h before the policy decision. The competing 24 July modal
    onset is a documented carry-unwind episode and a real feature of the
    posterior, but it is not what this project is about, and 02's sensitivity
    table showed that including 24-30 July dilutes the estimated shift rather
    than strengthening it. So: pre-shock is everything before the 30 July
    boundary, and the 24 July mode enters only as a sensitivity row, where
    its cost can be read off instead of argued about.

    The boundary is never typed as a date. It is read from the labels, and
    the script checks that the segment it calls "stress" is both the widest
    by dispersion and the one containing the decision, and says so if not.

    Following 02's own recommendation, membership uses the posterior
    probability rather than the hard boundary: a second is admitted only
    where the winning segment carries at least P_MIN of the mass. Both
    changepoints span market closures, so this puts a buffer around them that
    is derived from the posterior instead of chosen. Varied in the
    sensitivity table, including at 0.5, which switches the buffer off.

Estimation notes that matter
    Rows are admitted only where the whole lag window - p+2 consecutive
    seconds - is contiguous in clock time, outside rollover, inside one
    regime, and confidently labelled. Nothing is ever differenced across a
    weekend, a closure, a rollover window or a regime boundary. This is the
    same guard that 01's episode detection needed and did not originally
    have.

    The row set is fixed once, at LAG_CAP, and reused for every lag order and
    every regime, so no two numbers in the output are computed on different
    samples.

    The three alphas are estimated far less precisely than their
    combination, and the intervals in the headline table show it plainly.
    This is not a defect of the fit. Each leg's own volatility is orders of
    magnitude larger than the basis, so a regression of that leg's return on
    the basis has an enormous error variance, while lambda is a combination
    in which the common moves cancel by construction. The consequence for
    the report is that "the triangle closed ten times slower" is a much
    sturdier claim than "AUD/JPY did most of the closing", and the two
    should not be given equal weight.

    Standard errors on alpha are Newey-West. Serial correlation in the
    residuals is not a nuisance to be assumed away here: the increments carry
    a large negative first-order autocorrelation from bid-ask bounce
    (rho_1 ~ -0.35 in 01), and OLS standard errors would be wrong by a factor
    that is not small. The full 3K x 3K HAC matrix is never formed. Only the
    3 x 3 covariance of the three alphas is needed, and it can be had exactly
    from one scalar series per equation - see `alpha_hac_cov`.

Outputs
    output/data/03_resid.parquet          per-second VECM residuals, for 06
    output/data/03_var_model.parquet      coefficients, for 06
    output/tables/03_error_correction.csv headline: alpha, lambda, half-lives
    output/tables/03_var_fit.csv          fit and residual diagnostics
    output/tables/03_lag_selection.csv    BIC over the lag grid
    output/tables/03_coefficients.csv     every fitted coefficient
    output/tables/03_closure_speed.csv    daily closure speed
    output/tables/03_sensitivity.csv
    output/figures/03_error_correction.png
    output/figures/03_closure_speed.png
    output/figures/03_var_diagnostics.png
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from utils import (
    make_dirs, set_style, in_rollover, robust_scale, save_table, save_fig,
    adjacent, check_grid, annotate_event, highlight_span,
    DAT_DIR, TAB_DIR, TEX_DIR, BOJ_SHOCK, TZ_LABEL,
    GRID_SECONDS, PAIR_LABEL, regime_colours,
    MUTE, RULE, BAND_ALPHA,
)

# ------------------------------------------------------------- constants
#
# Labelled by provenance, as everywhere else in this project:
#
#   derived      computed from something else, so the two cannot drift apart
#   convention   a stated choice with no data content, exercised in
#                03_sensitivity rather than asserted
#   measured     established from the data; none here
#
# The system, in the order the columns appear everywhere below. Written as
# log-price column names from 01 so a reordering upstream cannot silently
# permute alpha.
LEGS = ("x_audusd", "x_usdjpy", "x_audjpy")

# derived. The mid-price column each log price came from, used only to fetch
# the right colour and label from utils, so a leg cannot change colour
# between section 01's figures and this script's.
LEG_SOURCE = {"x_audusd": "audusd_mid", "x_usdjpy": "usdjpy_mid",
              "x_audjpy": "audjpy_direct"}

# derived. z = x_audjpy - x_audusd - x_usdjpy, in the LEGS order. This is the
# triangular restriction and it is the reason no cointegrating vector is
# estimated.
W = np.array([-1.0, -1.0, 1.0])

# convention. Log prices times 1e4. Chosen so a unit is an AUD/JPY pip to
# within the cross's deviation from 100; the actual ratio is measured at load
# and printed, and nothing in alpha or lambda depends on it.
SCALE = 1e4

# convention. Lag grid for the transitory terms, in seconds. Geometric rather
# than consecutive: the interesting comparison is 1 vs 5 vs 20 seconds of
# microstructure memory, and eight points spaced that way say more than
# twenty consecutive ones at ten times the cost.
LAG_GRID = (1, 2, 3, 5, 8, 13, 21)

# convention. Hard cap on p, and the lag depth at which the admissible row
# set is built. Two jobs in one number: it bounds the cost of the enumeration
# above, and it fixes one sample for every fit in the script. If BIC selects
# the cap the lag-selection table shows it sitting on the boundary, which is
# the honest way to report a binding constraint.
LAG_CAP = max(LAG_GRID)

# convention. Minimum posterior mass the winning segment must carry for a
# second to be admitted. Not a tuning knob for the result: it is the buffer
# around the two changepoints, both of which span a market closure, and 02
# asked for exactly this instead of a hard boundary. Varied in
# 03_sensitivity down to 0.5, which is the hard boundary.
P_MIN = 0.90

# convention. Newey-West truncation for the alpha standard errors, in
# seconds. Well beyond where the residual autocorrelation function has
# died - which is plotted in 03_var_diagnostics, so the choice can be
# checked rather than taken on trust.
NW_LAGS = 30

# convention. Horizon over which a unit dislocation is followed, in seconds.
# Half an hour is an order of magnitude past the longest half-life 02
# reported, so a regime that fails to halve inside it is reported as failing
# rather than extrapolated.
DECAY_HORIZON = 1800

# convention. Rows per chunk when accumulating cross-products. Purely an
# engineering number: with millions of seconds and up to 65 regressors the
# design matrix is several hundred megabytes if it is ever formed whole, and
# it never needs to be.
CHUNK = 200_000

# convention. Shortest window that gets its own fit in the daily closure
# speed series. A day with less than this after closures, rollover and the
# lag-window guard is reported as missing rather than estimated.
MIN_WINDOW_ROWS = 5_000

CRIT = 1.959963984540054   # 95% normal quantile, for HAC intervals


class Fit(NamedTuple):
    """One VECM fitted on one row set."""
    name: str
    n: int
    p: int
    terms: tuple
    coef: np.ndarray             # (K, 3), columns in LEGS order
    sigma: np.ndarray            # (3, 3) residual covariance
    r2: np.ndarray               # (3,)
    zbar: float
    cond: float
    alpha: np.ndarray            # (3,) adjustment coefficients
    alpha_cov: Optional[np.ndarray]   # (3, 3) HAC, None in light mode
    lam: float                   # w'alpha, the basis closure rate
    lam_se: float
    half_life: float             # from the full system
    half_life_step: float        # from lambda alone
    decay: np.ndarray            # (DECAY_HORIZON + 1,)
    max_eig: float
    resid: Optional[np.ndarray]  # (n, 3), None in light mode
    rows: Optional[np.ndarray]
    run: Optional[np.ndarray]


# ---------------------------------------------------------------- reporting

def num(x, spec=".3f"):
    """
    Format a number for a table, or an em dash if it is not defined.

    The same helper 02 carries, and for the same reason: a NaN written into
    a LaTeX table that the report \\input s renders as a blank cell or the
    literal 'nan'. An em dash says 'not applicable' and is the only thing
    that should ever reach the report in place of a number.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else format(v, spec)


def ci(lo, hi, spec=".4f"):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"[{format(lo, spec)}, {format(hi, spec)}]"


def fmt(ts):
    return "—" if pd.isna(ts) else f"{ts:%d %b %H:%M}"


# ------------------------------------------------------------------ regimes

def regime_roles(n_regimes, stress_k):
    """
    Report names for the estimated segments.

    Keyed on which segment is stressed rather than on how many there are.
    02 selected M3 by a 0.616-to-0.384 margin over M2; had the coin landed
    the other way the parquet would carry two probability columns instead of
    three, and a script that indexed a fixed tuple of names would break on
    the day that happens. The suffix only appears when a model returns more
    than one segment on a side, which no model in 02 can currently do.
    """
    if n_regimes <= 1:
        return ["single regime"]
    if n_regimes == 2:
        # M2 pools the calm before and after into one segment, so there is no
        # pre and post to name. Calling the first half "pre" would assert a
        # distinction that model explicitly does not make.
        return ["stress" if k == stress_k else "calm" for k in range(2)]
    raw = ["stress" if k == stress_k else ("pre" if k < stress_k else "post")
           for k in range(n_regimes)]
    counts = {name: raw.count(name) for name in set(raw)}
    seen: dict[str, int] = {}
    out = []
    for name in raw:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name} {seen[name]}")
    return out


def load_regimes(d):
    """
    Per-second labels and posterior probabilities from 02.

    Everything about the segmentation is read, nothing is assumed. The number
    of segments comes from the columns present, the labels are recomputed
    from the probabilities and checked against the stored column, and the
    stressed segment is identified from dispersion rather than from its
    position. DataFrame.attrs does not reliably survive a parquet round trip,
    so the selected model is inferred from the column names and treated as a
    label for printing only.
    """
    path = DAT_DIR / "02_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 02 before 03: the regime labels are this "
            f"script's only source of a pre/stress/post split, and inventing "
            f"one from the calendar is exactly what 02 exists to replace.")

    r = pd.read_parquet(path).sort_index()
    prob_cols = [c for c in r.columns if c.startswith("p_")]
    if not prob_cols:
        raise KeyError(
            f"{path} carries no p_* probability columns, only "
            f"{list(r.columns)}. 03 conditions on the posterior, not on the "
            f"hard label; rerun 02.")

    r = r.reindex(d.index)
    missing = int(r[prob_cols].isna().any(axis=1).sum())
    if missing:
        raise ValueError(
            f"02_regimes.parquet is missing {missing:,} of the "
            f"{len(d):,} seconds in 01_clean.parquet. The two were written "
            f"from different runs; rerun 02.")

    prob = r[prob_cols].to_numpy(dtype=float)
    total = prob.sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-3):
        raise ValueError(
            f"segment probabilities do not sum to one "
            f"(min {total.min():.4f}, max {total.max():.4f})")

    labels = prob.argmax(axis=1).astype(np.int8)
    if "regime" in r.columns:
        stored = r["regime"].to_numpy()
        disagree = int((stored != labels).sum())
        if disagree:
            print(f"    note: {disagree:,} seconds where the stored label "
                  f"disagrees with the argmax of the probabilities; the "
                  f"argmax is used")

    names = [c[2:].replace("_", " ") for c in prob_cols]
    return labels, prob, names


def identify_stress(d, labels, n_regimes):
    """
    Which segment is the stressed one, established two ways.

    Dispersion decides, because that is the feature 02 fitted and it orders
    the segments without reference to the calendar. The BOJ decision is then
    used only to check the answer: it is a reference in this project and
    never an input, so it may confirm a label but must not assign one.
    """
    mad = np.array([robust_scale(d["basis"].to_numpy()[labels == k])
                    if (labels == k).any() else np.nan
                    for k in range(n_regimes)])
    stress_k = int(np.nanargmax(mad))
    pos = int(pd.DatetimeIndex(d.index).get_indexer([BOJ_SHOCK],
                                                    method="nearest")[0])
    boj_k = int(labels[pos])
    return stress_k, boj_k, mad


# ------------------------------------------------------------- row admission

def usable_rows(index, p, ok, group):
    """
    Positional indices whose whole lag window is safe to difference.

    Row t of the design needs prices at t-p-1 ... t: dx_t uses x_t and
    x_{t-1}, the lag block uses dx_{t-i} for i = 1..p, and z_{t-1} uses
    x_{t-1}. So p+2 consecutive rows must all be one grid second apart, all
    admissible under `ok`, and all carry the same value of `group`.

    Both guards are load-bearing rather than defensive. Without the
    adjacency test a 48-hour weekend becomes a single 'increment' of two
    days, which is the same bug that once merged two dislocations either
    side of a closure into one 48-hour episode. Without the group test a
    regression labelled 'pre' would be fitted partly on stressed seconds
    wherever a lag window straddles the changepoint.

    Timestamps are compared through utils.adjacent, never as integer
    nanoseconds: an index arriving from parquet is datetime64[us], and an
    int64 comparison against a nanosecond constant is False everywhere,
    which turns every guard in this function off without raising anything.
    """
    adj = adjacent(index)
    ok = np.asarray(ok, dtype=bool)
    group = np.asarray(group)

    # chain[s]: row s continues an unbroken, admissible, single-group run.
    chain = np.zeros(len(ok), dtype=bool)
    chain[1:] = (adj[1:] & ok[1:] & ok[:-1] & (group[1:] == group[:-1]))

    # Length of the run of consecutive True chain flags ending at each row,
    # by the standard reset-and-accumulate trick rather than a Python loop
    # over several million elements.
    pos = np.arange(len(chain))
    reset = np.maximum.accumulate(np.where(chain, 0, pos))
    run_len = pos - reset

    rows = np.flatnonzero(run_len >= p + 1).astype(np.int64)
    return rows, reset[rows].astype(np.int64)


# -------------------------------------------------------------- estimation

def term_names(p):
    short = {"x_audusd": "AUD/USD", "x_usdjpy": "USD/JPY",
             "x_audjpy": "AUD/JPY"}
    terms = ["const", "z(-1)"]
    for i in range(1, p + 1):
        terms += [f"d {short[c]} (-{i})" for c in LEGS]
    return tuple(terms)


def _design(rows, DX, Z, p, zbar):
    """
    (X, Y) for a block of rows. Never called on the whole sample at once.

    `rows` is guaranteed by `usable_rows` to satisfy rows >= p + 1, so every
    backward index below is in range without a bounds check that would only
    ever hide a violated precondition.
    """
    m = rows.size
    X = np.empty((m, 2 + 3 * p), dtype=float)
    X[:, 0] = 1.0
    X[:, 1] = Z[rows - 1] - zbar
    for i in range(1, p + 1):
        X[:, 2 + 3 * (i - 1):2 + 3 * i] = DX[rows - i]
    return X, DX[rows]


def _chunks(n, size=CHUNK):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def companion(alpha, gammas):
    """
    Transition matrix of the state (z_t, dx_t, ..., dx_{t-p+1}).

    Built from the fitted VECM plus the identity z_t = z_{t-1} + w'dx_t, so
    the first row is not an extra assumption. Its spectral radius is the
    stationarity check the error-correction interpretation needs: with the
    cointegrating vector imposed the system has no unit root left in this
    state, so an eigenvalue at or above one means the regime's basis is not
    mean-reverting at all.
    """
    p = len(gammas)
    dim = 1 + 3 * p
    M = np.zeros((dim, dim))
    M[0, 0] = 1.0 + float(W @ alpha)
    M[1:4, 0] = alpha
    for i, G in enumerate(gammas):
        cols = slice(1 + 3 * i, 1 + 3 * (i + 1))
        M[0, cols] = W @ G
        M[1:4, cols] = G
    for i in range(1, p):
        M[1 + 3 * i:1 + 3 * (i + 1), 1 + 3 * (i - 1):1 + 3 * i] = np.eye(3)
    return M


def decay_path(M, horizon=DECAY_HORIZON):
    """
    Path of a unit dislocation with no further shocks.

    This is what 'how long does the triangle take to close' means once the
    transitory terms are in the model: lambda alone describes the first
    second only, and the Gamma feedback can either speed the rest up or drag
    it out. Reported alongside the one-step number so the difference between
    them is visible rather than buried.
    """
    s = np.zeros(M.shape[0])
    s[0] = 1.0
    out = np.empty(horizon + 1)
    out[0] = 1.0
    for h in range(1, horizon + 1):
        s = M @ s
        out[h] = s[0]
    return out


def half_life(path):
    """
    First crossing of one half, interpolated between the two seconds that
    bracket it. NaN when the path never gets there inside the horizon, which
    is reported as an em dash rather than extrapolated.
    """
    below = np.flatnonzero(path <= 0.5)
    if below.size == 0:
        return float("nan")
    h = int(below[0])
    if h == 0:
        return 0.0
    a, b = path[h - 1], path[h]
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return float(h)
    return float((h - 1) + (a - 0.5) / (a - b)) * GRID_SECONDS


def alpha_hac_cov(h, U, run, m, n, K):
    """
    Newey-West covariance of the three adjustment coefficients.

    The full HAC matrix of a three-equation system with K regressors is
    3K x 3K and needs the lagged cross-products of a (n, 3K) array; at a
    million seconds and sixty regressors that is not worth forming. It is
    also not needed. Only the z(-1) coefficient is of interest, and for a
    single linear functional q'beta the sandwich collapses: with
    q = (X'X)^-1 e and the scalar h_t = q'x_t, the meat term is the long-run
    covariance of h_t u_t, which is one series per equation.

    That long-run covariance is then taken from overlapping block sums
    rather than from a loop over lags. Summing over blocks of length m and
    dividing by m reproduces the Bartlett kernel with truncation m-1
    exactly, up to the first and last m-1 rows of each run; with runs
    hours long and m of the order of a minute that edge term is around
    1e-5 of the total. Blocks are refused wherever they would straddle a
    break, which is what `run` is for - the same discipline as everywhere
    else in this script.
    """
    phi = h[:, None] * U                       # (n, 3), sums to zero by OLS
    C = np.zeros((phi.shape[0] + 1, 3))
    np.cumsum(phi, axis=0, out=C[1:])
    B = C[m:] - C[:-m]
    valid = run[m - 1:] == run[:len(run) - m + 1]
    B = B[valid]
    if B.shape[0] < 2:
        return np.full((3, 3), np.nan)
    S = (B.T @ B) / m
    return S * (n / max(n - K, 1))


def fit_vecm(name, rows, run, DX, Z, p, light=False, nw=NW_LAGS):
    """
    One regime, one lag order.

    Cross-products are accumulated in chunks, which keeps peak memory at the
    chunk rather than at the sample and costs nothing: X'X, X'Y and Y'Y are
    all the normal equations need, and the residual covariance follows from
    them as Y'Y - B'X'Y without a second pass. `light` stops there and is
    what the sensitivity table and the daily series use; the full path also
    materialises the residuals, which are needed for the HAC term, the
    residual autocorrelations and 06.
    """
    n, K = rows.size, 2 + 3 * p
    if n <= 4 * K:
        raise ValueError(f"{name}: {n:,} rows is too few for {K} regressors")

    zbar = float(Z[rows - 1].mean())

    XtX = np.zeros((K, K))
    XtY = np.zeros((K, 3))
    YtY = np.zeros((3, 3))
    sumY = np.zeros(3)
    for a, b in _chunks(n):
        X, Y = _design(rows[a:b], DX, Z, p, zbar)
        XtX += X.T @ X
        XtY += X.T @ Y
        YtY += Y.T @ Y
        sumY += Y.sum(axis=0)

    cond = float(np.linalg.cond(XtX))
    try:
        coef = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        print(f"    note: {name}: X'X is singular at p={p} "
              f"(condition {cond:.3e}); falling back to a least-squares "
              f"solution")
        coef = np.linalg.lstsq(XtX, XtY, rcond=None)[0]

    # U'U = Y'Y - B'X'Y, exact, because B solves the normal equations.
    uu = YtY - coef.T @ XtY
    uu = 0.5 * (uu + uu.T)                     # kill asymmetry from rounding
    sigma = uu / max(n - K, 1)
    sst = np.diag(YtY) - sumY ** 2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1.0 - np.diag(uu) / sst, np.nan)

    alpha = coef[1, :].copy()
    lam = float(W @ alpha)

    gammas = [coef[2 + 3 * (i - 1):2 + 3 * i, :].T for i in range(1, p + 1)]
    M = companion(alpha, gammas)
    path = decay_path(M)
    hl = half_life(path)
    hl_step = (float(np.log(0.5) / np.log1p(lam)) * GRID_SECONDS
               if -2.0 < lam < 0.0 else float("nan"))
    max_eig = float(np.abs(np.linalg.eigvals(M)).max())

    if light:
        return Fit(name, n, p, term_names(p), coef, sigma, r2, zbar, cond,
                   alpha, None, lam, float("nan"), hl, hl_step, path,
                   max_eig, None, None, None)

    # Residuals, and the single scalar series each equation's HAC term needs.
    q = np.linalg.solve(XtX, np.eye(K)[1])
    U = np.empty((n, 3))
    hq = np.empty(n)
    for a, b in _chunks(n):
        X, Y = _design(rows[a:b], DX, Z, p, zbar)
        U[a:b] = Y - X @ coef
        hq[a:b] = X @ q

    cov = alpha_hac_cov(hq, U, run, nw + 1, n, K)
    lam_se = float(np.sqrt(max(W @ cov @ W, 0.0)))

    return Fit(name, n, p, term_names(p), coef, sigma, r2, zbar, cond,
               alpha, cov, lam, lam_se, hl, hl_step, path, max_eig,
               U, rows, run)


def resid_acf(u, run, lags):
    """
    Autocorrelation of a residual series at each lag, using only pairs that
    sit inside one unbroken run. Rows are consecutive seconds within a run,
    so a lag-l pair is a row and the row l positions earlier with the same
    run id, and nothing is correlated across a weekend.
    """
    out = []
    for l in lags:
        same = run[l:] == run[:-l]
        a, b = u[l:][same], u[:-l][same]
        if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


# ----------------------------------------------------------------- variants

def relabel_drop(labels, ok, index, lo, hi, which):
    """
    Drop the seconds of segment `which` that fall in [lo, hi).

    Used by the sensitivity rows that redefine where the pre-shock window
    ends. Dropping is the honest operation here: those seconds are being
    declared unclassifiable under the variant's definition, not moved into a
    regime the variant has no evidence for.
    """
    t = pd.DatetimeIndex(index)
    hit = (t >= lo) & (t < hi) & (labels == which)
    return ok & ~hit, int(hit.sum())


def relabel_move(labels, index, lo, hi, frm, to):
    """Reassign the seconds of segment `frm` in [lo, hi) to segment `to`."""
    t = pd.DatetimeIndex(index)
    hit = (t >= lo) & (t < hi) & (labels == frm)
    out = labels.copy()
    out[hit] = to
    return out, int(hit.sum())


def modal_onset(index):
    """
    The 24 July modal onset, parsed from 02's own headline table.

    Read rather than typed, and the year comes from the sample rather than
    from the string, so this cannot drift out of agreement with 02 or with
    the data. Returns None when the table is absent or the field is not a
    date, in which case the sensitivity row is skipped and said to be
    skipped.
    """
    path = TAB_DIR / "02_changepoint.csv"
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path, index_col=0)
        raw = str(table.loc["onset, posterior mode", "value"]).strip()
        year = pd.DatetimeIndex(index)[0].year
        return pd.to_datetime(f"{raw} {year}", format="%d %b %H:%M %Y")
    except Exception:
        return None


# ------------------------------------------------------------------- tables

def headline_table(fits, names, cross):
    """
    The result, one column per regime.

    Built as strings rather than floats. Half of these cells are intervals
    or ratios that have no value when a regime failed to fit, and the report
    \\input s this file directly, so a NaN reaching it would print as a blank
    or as the word nan.
    """
    cols = {}
    for name in names:
        f = fits.get(name)
        if f is None:
            cols[name] = {}
            continue
        col = {"open-market seconds": f"{f.n:,}",
               "hours": num(f.n / 3600.0, ".1f")}
        for j, leg in enumerate(LEGS):
            lab = PAIR_LABEL[LEG_SOURCE[leg]]
            se = (np.sqrt(f.alpha_cov[j, j])
                  if f.alpha_cov is not None else np.nan)
            col[f"alpha {lab} (per second)"] = num(f.alpha[j], ".4f")
            col[f"alpha {lab}, HAC 95%"] = ci(f.alpha[j] - CRIT * se,
                                              f.alpha[j] + CRIT * se)
        col["lambda, basis closure rate (per second)"] = num(f.lam, ".4f")
        col["lambda, HAC 95%"] = ci(f.lam - CRIT * f.lam_se,
                                    f.lam + CRIT * f.lam_se)
        for j, leg in enumerate(LEGS):
            lab = PAIR_LABEL[LEG_SOURCE[leg]]
            share = (W[j] * f.alpha[j] / f.lam) if f.lam != 0 else np.nan
            col[f"share of closure, {lab}"] = num(share, ".2f")
        col["half-life, one-step (s)"] = num(f.half_life_step, ".2f")
        col["half-life, full system (s)"] = num(f.half_life, ".2f")
        col["02 AR(1) half-life (s)"] = num(cross.get(f.name, np.nan), ".2f")
        col["largest eigenvalue"] = num(f.max_eig, ".4f")
        cols[name] = col
    return pd.DataFrame(cols).fillna("—")


def fit_table(fits, names, acf, boundaries):
    cols = {}
    for name in names:
        f = fits.get(name)
        if f is None:
            cols[name] = {}
            continue
        lo, hi = boundaries[name]
        sd = np.sqrt(np.diag(f.sigma))
        corr = f.sigma / np.outer(sd, sd)
        col = {"first second": fmt(lo), "last second": fmt(hi),
               "seconds fitted": f"{f.n:,}",
               "lag order p": f"{f.p}", "regressors": f"{2 + 3 * f.p}",
               "condition number of X'X": num(f.cond, ".3e"),
               "mean of z on the fitted rows (pips)": num(f.zbar, ".4f")}
        for j, leg in enumerate(LEGS):
            lab = PAIR_LABEL[LEG_SOURCE[leg]]
            col[f"R2, {lab} equation"] = num(f.r2[j], ".4f")
            col[f"residual sd, {lab} (pips)"] = num(sd[j], ".4f")
        pairs = [(0, 1), (0, 2), (1, 2)]
        for a, b in pairs:
            la = PAIR_LABEL[LEG_SOURCE[LEGS[a]]]
            lb = PAIR_LABEL[LEG_SOURCE[LEGS[b]]]
            col[f"residual corr, {la} vs {lb}"] = num(corr[a, b], ".3f")
        col["residual rho_1, basis innovation"] = num(acf[name][0], ".3f")
        col["residual rho_60, basis innovation"] = num(acf[name][-1], ".3f")
        cols[name] = col
    return pd.DataFrame(cols).fillna("—")


def coefficient_frame(fits, names):
    rows = []
    for k, name in enumerate(names):
        f = fits.get(name)
        if f is None:
            continue
        for i, term in enumerate(f.terms):
            for j, leg in enumerate(LEGS):
                rows.append({"regime_id": k, "regime": name,
                             "equation": PAIR_LABEL[LEG_SOURCE[leg]],
                             "term": term, "coef": float(f.coef[i, j])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def panel_title(ax, title, note):
    """
    Title with a subtitle underneath it.

    The subtitle sits just above the axes, so the title has to be pushed
    clear of it. rcParams sets a pad of 10, which is the height of the
    subtitle itself and therefore exactly enough for the two to overlap;
    this passes a local pad instead of editing utils, because changing the
    default there would move the titles in 01 and 02 as well.
    """
    ax.set_title(title, pad=19)
    ax.annotate(note, xy=(0, 1.012), xycoords="axes fraction", fontsize=7.5,
                color="#666666", va="bottom", ha="left")


def figure_error_correction(fits, names, colours):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4),
                           gridspec_kw={"wspace": 0.26})

    width = 0.8 / max(len(names), 1)
    base = np.arange(len(LEGS) + 1, dtype=float)
    for k, name in enumerate(names):
        f = fits.get(name)
        if f is None:
            continue
        est = np.append(f.alpha, f.lam)
        if f.alpha_cov is not None:
            se = np.append(np.sqrt(np.diag(f.alpha_cov)), f.lam_se)
        else:
            se = np.full(len(LEGS) + 1, np.nan)
        x = base + (k - (len(names) - 1) / 2) * width
        ax[0].errorbar(x, est, yerr=CRIT * se, fmt="o", ms=4.5,
                       color=colours[k], lw=0, elinewidth=1.4, capsize=2.5,
                       ecolor=colours[k], label=name)
    ax[0].axhline(0, color=RULE, lw=0.6, zorder=1)
    ax[0].axvline(len(LEGS) - 0.5, color=MUTE, lw=0.7, ls=":", zorder=0)
    ax[0].set_xticks(base)
    ax[0].set_xticklabels([PAIR_LABEL[LEG_SOURCE[c]] for c in LEGS]
                          + [r"$\lambda$"])
    ax[0].set_ylabel("adjustment per second")
    ax[0].margins(y=0.16)
    panel_title(ax[0], "How each leg responds to an open triangle",
                "closure needs AUD/JPY negative, the other two positive, "
                r"and $\lambda$ below zero")
    # Below the axes, not inside them. This panel's data are points with
    # error bars, and an unframed legend inside the frame puts three more
    # coloured dots among them; a reader should never have to work out which
    # dots are estimates.
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                 ncol=len(names), borderaxespad=0.0)

    horizon = np.arange(DECAY_HORIZON + 1) * GRID_SECONDS
    for k, name in enumerate(names):
        f = fits.get(name)
        if f is None:
            continue
        label = (f"{name} · {num(f.half_life, '.1f')} s"
                 if np.isfinite(f.half_life) else f"{name} · no crossing")
        ax[1].plot(horizon[1:], f.decay[1:], lw=1.6, color=colours[k],
                   label=label)
        if np.isfinite(f.half_life):
            ax[1].plot([f.half_life], [0.5], marker="o", ms=4.5,
                       color=colours[k], zorder=4)
    ax[1].axhline(0.5, color=RULE, lw=0.7, ls=":", zorder=1)
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, 1.02)
    # Trimmed to where the slowest regime has finished decaying. The horizon
    # is long enough to prove a regime does eventually close, which is not
    # the same as being worth plotting.
    slowest = max((f.half_life for f in fits.values()
                   if f is not None and np.isfinite(f.half_life)),
                  default=float("nan"))
    right = (min(DECAY_HORIZON, max(60.0, 12.0 * slowest))
             if np.isfinite(slowest) else DECAY_HORIZON)
    ax[1].set_xlim(1, right)
    ax[1].set_xlabel("seconds since the dislocation opened")
    ax[1].set_ylabel("share of the dislocation still open")
    panel_title(ax[1], "How long a dislocation survives",
                "no further shocks; the fitted system left to run down")
    ax[1].legend(loc="upper right")
    save_fig(fig, "03_error_correction")


def figure_diagnostics(fits, names, colours, acf, lags):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3),
                           gridspec_kw={"wspace": 0.26})

    for k, name in enumerate(names):
        if name not in acf:
            continue
        ax[0].plot(lags, acf[name], lw=1.4, color=colours[k], label=name)
    ax[0].axhline(0, color=RULE, lw=0.6)
    # What zero looks like at this sample size. Without it the eye reads any
    # wiggle as structure; the smallest regime sets the widest band, so the
    # rule drawn is the conservative one.
    smallest = min((f.n for f in fits.values() if f is not None), default=0)
    if smallest > 0:
        band = 2.0 / np.sqrt(smallest)
        for sign in (-1, 1):
            ax[0].axhline(sign * band, color=MUTE, lw=0.7, ls=":", zorder=0)
        ax[0].annotate(f"±2/√n at n = {smallest:,}", xy=(0.99, band),
                       xycoords=("axes fraction", "data"), xytext=(0, 3),
                       textcoords="offset points", ha="right", va="bottom",
                       fontsize=7, color=MUTE)
    ax[0].set_xlabel("lag (seconds)")
    ax[0].set_ylabel("autocorrelation")
    panel_title(ax[0], "What the model left behind",
                "residual autocorrelation of the basis innovation; flat "
                "means p was deep enough")
    ax[0].legend(loc="best")

    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [f"{PAIR_LABEL[LEG_SOURCE[LEGS[a]]]}\n{PAIR_LABEL[LEG_SOURCE[LEGS[b]]]}"
              for a, b in pairs]
    width = 0.8 / max(len(names), 1)
    base = np.arange(len(pairs), dtype=float)
    for k, name in enumerate(names):
        f = fits.get(name)
        if f is None:
            continue
        sd = np.sqrt(np.diag(f.sigma))
        corr = f.sigma / np.outer(sd, sd)
        vals = [corr[a, b] for a, b in pairs]
        ax[1].bar(base + (k - (len(names) - 1) / 2) * width, vals,
                  width=width * 0.9, color=colours[k], label=name, lw=0)
    ax[1].axhline(0, color=RULE, lw=0.6)
    ax[1].set_xticks(base)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("residual correlation")
    ax[1].margins(y=0.18)
    panel_title(ax[1], "Simultaneous co-movement of the innovations",
                "what the lags and the error-correction term could not "
                "account for")
    ax[1].legend(loc="best", ncol=len(names))
    save_fig(fig, "03_var_diagnostics")


def figure_closure_speed(daily, fits, names, colours, boundaries):
    fig, ax = plt.subplots(figsize=(10, 4.2))

    for k, name in enumerate(names):
        lo, hi = boundaries[name]
        if pd.isna(lo) or pd.isna(hi):
            continue
        highlight_span(ax, lo, hi, colour=colours[k], alpha=BAND_ALPHA)
        f = fits.get(name)
        if f is not None and np.isfinite(f.half_life):
            ax.plot([lo, hi], [f.half_life] * 2, color=colours[k], lw=2.2,
                    solid_capstyle="butt", zorder=4,
                    label=f"{name} · {f.half_life:.1f} s")

    ok = daily["half_life_s"].notna()
    ax.plot(daily.index[ok], daily.loc[ok, "half_life_s"], lw=1.1,
            color=RULE, marker="o", ms=2.6, zorder=3,
            label="fitted day by day")
    ax.set_yscale("log")
    # A log axis whose range is a single decade shows one label by default,
    # which is no axis at all. Ticks are placed on the round seconds a
    # reader would actually name.
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("half-life of a dislocation (s)")
    ax.set_xlabel(TZ_LABEL)
    ax.margins(y=0.22)
    panel_title(ax, "How fast the triangle closed, day by day",
                "each day fitted on its own, with no regime label supplied; "
                "bands are the regimes 02 estimated")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    annotate_event(ax, BOJ_SHOCK, "BOJ")
    ax.legend(loc="upper left", ncol=2)
    save_fig(fig, "03_closure_speed")


# ---------------------------------------------------------------- self-check

def audit(fits, names, prefix="03_"):
    """
    Internal consistency, in the spirit of 02's harness.

    Every check here corresponds to something that has actually gone wrong
    in this project or in a script like it: a residual that is not orthogonal
    to the design, a covariance that is not a covariance, and the literal
    string 'nan' reaching a generated LaTeX table that the report inputs.
    """
    problems = []
    for name in names:
        f = fits.get(name)
        if f is None or f.resid is None:
            continue
        if not np.isfinite(f.resid).all():
            problems.append(f"{name}: residuals contain non-finite values")
        if abs(float(f.resid.mean())) > 1e-6 * float(np.abs(f.resid).mean() + 1):
            problems.append(f"{name}: residuals are not mean zero")
        if not np.allclose(f.sigma, f.sigma.T, atol=1e-12):
            problems.append(f"{name}: residual covariance is not symmetric")
        if np.linalg.eigvalsh(f.sigma).min() <= 0:
            problems.append(f"{name}: residual covariance is not positive "
                            f"definite")

    # Whole tokens, not substrings: 'nan' sits inside ordinary English words
    # and a scan that flagged those would be turned off within a week.
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


# --------------------------------------------------------------------- main

def main():
    make_dirs()
    set_style()

    # ------------------------------------------------------------- load
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

    # The scaled log basis and the pip basis of 01 and 02 are the same
    # quantity to first order. Measured rather than asserted, because if the
    # two ever disagreed it would mean the synthetic leg is not the product
    # of the other two and every number below would be about something else.
    ratio = robust_scale(Z) / robust_scale(d["basis"].to_numpy())
    print(f" scaled log basis is {ratio:.4f} x the pip basis of 01; "
          f"alpha is a ratio of two quantities on this scale, so the scale "
          f"cancels")

    # ---------------------------------------------------------- regimes
    print("regimes")
    labels, prob, seg_names = load_regimes(d)
    n_regimes = prob.shape[1]
    pmax = prob.max(axis=1)
    stress_k, boj_k, mad = identify_stress(d, labels, n_regimes)
    names = regime_roles(n_regimes, stress_k)
    colours = list(regime_colours(n_regimes))

    boundaries = {}
    for k, name in enumerate(names):
        sel = np.flatnonzero(labels == k)
        boundaries[name] = ((d.index[sel[0]], d.index[sel[-1]]) if sel.size
                            else (pd.NaT, pd.NaT))

    print(f" {n_regimes} segments in 02_regimes.parquet")
    for k, name in enumerate(names):
        lo, hi = boundaries[name]
        print(f"  {seg_names[k]:<12} -> {name:<8} {lo} .. {hi}  "
              f"MAD {num(mad[k], '.4f')} pips")
        # The boundaries printed above are the first and last second with
        # this label, which is only a description of the segment if the
        # label holds throughout. 02's pointwise argmax is not guaranteed to
        # be monotone, so it is checked rather than assumed.
        flag = (labels == k)
        blocks = int(np.count_nonzero(flag[1:] & ~flag[:-1])) + int(flag[0])
        if blocks > 1:
            print(f"    WARNING: {name} is {blocks} separate blocks, not "
                  f"one; the dates above are its outer edges only")
    if boj_k != stress_k:
        print(f"  WARNING: dispersion says segment {stress_k + 1} is the "
              f"stressed one, but the BOJ decision falls in segment "
              f"{boj_k + 1}. The label is taken from dispersion; check 02.")
    else:
        print(f"  the stressed segment is the widest by dispersion and "
              f"contains the {BOJ_SHOCK:%d %b %H:%M} decision")

    if stress_k > 0:
        onset = boundaries[names[stress_k]][0]
        gap = (onset - BOJ_SHOCK).total_seconds() / 3600.0
        print(f"  pre-shock ends {boundaries[names[stress_k - 1]][1]}; "
              f"stress opens {onset} ({gap:+.1f} h relative to the decision)")
        print("  this is 02's median segmentation. The 24 Jul modal onset is "
              "not used for the split; it appears in 03_sensitivity only")
        if abs(gap) > 48:
            print(f"  WARNING: the stress window opens {gap:+.1f} h from the "
                  f"decision, further than this project's framing assumes")

    roll = in_rollover(d.index).to_numpy()
    print(f" excluding {roll.mean():.2%} of seconds as rollover")

    # ------------------------------------------------------------ rows
    #
    # One row set, built once at the deepest lag order the script will ever
    # fit, and reused by every fit below. Fitting p = 1 on more rows than
    # p = 21 would make the lag-selection table compare models on different
    # samples, which is not a comparison.
    print("rows")
    ok = (~roll) & (pmax >= P_MIN)
    rows_all, run_all = usable_rows(d.index, LAG_CAP, ok, labels)
    print(f" {rows_all.size:,} of {len(d):,} seconds admitted at p = "
          f"{LAG_CAP} (contiguous lag window, outside rollover, one regime, "
          f"posterior at least {P_MIN:.0%})")
    dropped_prob = int(((~roll) & (pmax < P_MIN)).sum())
    print(f" {dropped_prob:,} seconds set aside as too close to a "
          f"changepoint to label confidently")

    by_regime = {}
    for k, name in enumerate(names):
        sel = labels[rows_all] == k
        by_regime[name] = (rows_all[sel], run_all[sel])
        print(f"  {name:<8} {sel.sum():,} rows "
              f"({sel.sum() / 3600:.1f} h)")

    # -------------------------------------------------- lag selection
    #
    # Selected on the sum of the regimes' evidence rather than on a pooled
    # fit, because every other quantity in this script is regime-specific and
    # a pooled fit would choose the lag order for a model nobody reports.
    print("lag selection")
    sel_rows = []
    for p in LAG_GRID:
        total_n, total_ll, ok_grid = 0, 0.0, True
        per = {}
        for name in names:
            r, run = by_regime[name]
            try:
                f = fit_vecm(name, r, run, DX, Z, p, light=True)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"  p={p} {name}: {exc}")
                ok_grid = False
                break
            sign, logdet = np.linalg.slogdet(f.sigma)
            if sign <= 0:
                ok_grid = False
                break
            per[name] = logdet
            total_n += f.n
            total_ll += f.n * logdet
        if not ok_grid:
            continue
        params = len(names) * 3 * (2 + 3 * p)
        row = {"p (seconds)": p, "parameters": params,
               "sum n log|Sigma|": total_ll,
               "AIC": total_ll + 2 * params,
               "BIC": total_ll + params * np.log(total_n)}
        row.update({f"log|Sigma| {name}": per[name] for name in names})
        sel_rows.append(row)

    if not sel_rows:
        raise RuntimeError("no lag order in LAG_GRID could be fitted")
    lag_table = pd.DataFrame(sel_rows).set_index("p (seconds)")
    p_star = int(lag_table["BIC"].idxmin())
    print(lag_table[["parameters", "AIC", "BIC"]].to_string(
        float_format=lambda v: f"{v:,.1f}"))
    print(f" BIC selects p = {p_star}"
          + (" — the cap; the transitory terms want more memory than the "
             "grid allows, and the sensitivity table refits at the ends of "
             "the grid" if p_star == LAG_CAP else ""))
    save_table(
        lag_table, "03_lag_selection",
        caption=("Lag order for the transitory terms, chosen on the sum of "
                 "the regimes' evidence. The row set is the same at every "
                 "lag order, so the criteria are comparable. The "
                 "error-correction coefficient is the object of interest and "
                 "the transitory terms are a nuisance; the sensitivity table "
                 "refits at both ends of this grid."),
        label="tab:var_lags",
    )

    # -------------------------------------------------------- baseline
    print("fit")
    fits, acf = {}, {}
    lags = np.arange(1, NW_LAGS * 2 + 1)
    for name in names:
        r, run = by_regime[name]
        f = fit_vecm(name, r, run, DX, Z, p_star)
        fits[name] = f
        acf[name] = resid_acf(f.resid @ W, run, lags)
        print(f" {name:<8} n {f.n:>10,}  lambda {f.lam:+.4f} "
              f"({f.lam - CRIT * f.lam_se:+.4f}, "
              f"{f.lam + CRIT * f.lam_se:+.4f})  "
              f"half-life {num(f.half_life, '.2f')} s  "
              f"max|eig| {f.max_eig:.4f}")

    # 02 estimated the same quantity from a plain AR(1) on the level of the
    # basis, with no lags and no other legs in the system. Quoting the two
    # side by side is a cross-check on both, and the numbers are not
    # constructed to agree.
    cross = {}
    cp = TAB_DIR / "02_regime_stats.csv"
    if cp.exists():
        try:
            prev = pd.read_csv(cp, index_col=0)
            row = prev.loc["half-life (s)"]
            for k, name in enumerate(names):
                if k < len(row):
                    cross[name] = float(row.iloc[k])
        except Exception:
            print("  note: 02_regime_stats.csv present but its half-life row "
                  "could not be read; the cross-check column is left empty")

    # ---------------------------------------------------------- tables
    print("tables")
    head = headline_table(fits, names, cross)
    save_table(
        head, "03_error_correction",
        caption=("Error-correction dynamics of the triangle by estimated "
                 "regime. The cointegrating vector is imposed, not "
                 "estimated: it is the triangular identity. alpha is the "
                 "share of the current gap each leg erases per second, "
                 "lambda their combination and so the closure rate of the "
                 "basis itself; intervals are Newey-West at "
                 f"{NW_LAGS} seconds. The per-leg intervals are much wider "
                 "than the interval on lambda, because each leg's own "
                 "volatility dwarfs the basis while the common moves cancel "
                 "in the combination; the closure rate is a far sturdier "
                 "quantity than the split of it across legs. The last two "
                 "rows are checks, not results: the 02 column comes from a "
                 "plain AR(1) on the level of the basis and shares no "
                 "machinery with this fit."),
        label="tab:error_correction",
    )
    print(head.to_string())

    fitinfo = fit_table(fits, names, acf, boundaries)
    save_table(
        fitinfo, "03_var_fit",
        caption=("Fit and residual diagnostics. Residual standard deviations "
                 "are in the scaled log units of the text, which are AUD/JPY "
                 "pips to within the cross's deviation from 100. The two "
                 "autocorrelation rows say whether the lag order was deep "
                 "enough for the Newey-West truncation to cover what is "
                 "left."),
        label="tab:var_fit",
    )

    coefs = coefficient_frame(fits, names)
    save_table(
        coefs.drop(columns=["regime_id"]), "03_coefficients", index=False,
        caption=("Every fitted coefficient, so the impulse responses in 06 "
                 "can be reconstructed from a file rather than from a rerun. "
                 "Standard errors are reported only for the "
                 "error-correction terms, in Table~\\ref{tab:error_correction}: "
                 "the transitory terms are a nuisance and are not "
                 "interpreted."),
        label="tab:var_coefficients",
    )

    # ----------------------------------------------------- sensitivity
    print("sensitivity")

    def summarise(tag, fitted, note=""):
        row = {"variant": tag}
        for name in names:
            f = fitted.get(name)
            row[f"lambda {name}"] = num(f.lam, ".4f") if f else "—"
            row[f"half-life {name} (s)"] = (num(f.half_life, ".2f") if f
                                            else "—")
        stress = fitted.get(names[stress_k])
        others = [fitted[n].half_life for n in names
                  if n != names[stress_k] and fitted.get(n) is not None
                  and np.isfinite(fitted[n].half_life)]
        ratio_v = (stress.half_life / min(others)
                   if stress is not None and np.isfinite(stress.half_life)
                   and others and min(others) > 0 else np.nan)
        row["stress / calmest"] = num(ratio_v, ".2f")
        row["note"] = note
        return row

    def run_variant(tag, p=None, min_prob=None, labels_v=None, ok_v=None,
                    note=""):
        """
        One row of 03_sensitivity. Exactly one thing changes per call: the
        lag order, the posterior threshold, the admissible seconds, or the
        labels. `ok_v` replaces the baseline mask outright rather than
        intersecting with it, so a variant that puts rollover back in is not
        silently still excluding it.
        """
        p = p_star if p is None else p
        lab = labels if labels_v is None else labels_v
        if min_prob is not None:
            base_ok = (~roll) & (pmax >= min_prob)
            if ok_v is not None:
                base_ok = base_ok & ok_v
        elif ok_v is not None:
            base_ok = ok_v
        else:
            base_ok = ok
        r_all, run_v = usable_rows(d.index, LAG_CAP, base_ok, lab)
        out = {}
        for k, name in enumerate(names):
            sel = lab[r_all] == k
            if sel.sum() <= 4 * (2 + 3 * p):
                out[name] = None
                continue
            try:
                out[name] = fit_vecm(name, r_all[sel], run_v[sel], DX, Z, p,
                                     light=True)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"  {tag} / {name}: {exc}")
                out[name] = None
        return summarise(tag, out, note)

    variants = [summarise(f"baseline (p = {p_star}, "
                          f"posterior at least {P_MIN:.0%})", fits,
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

    ok_roll = pmax >= P_MIN
    variants.append(run_variant("rollover seconds included", ok_v=ok_roll,
                                note="17:00-17:30 back in"))

    if stress_k > 0:
        onset = boundaries[names[stress_k]][0]
        pre_name = names[stress_k - 1]

        cut = onset.normalize()
        ok_v, n_cut = relabel_drop(labels, ok, d.index, cut, onset,
                                   stress_k - 1)
        variants.append(run_variant(
            f"pre-shock ends {cut:%d %b} 00:00", ok_v=ok_v,
            note=f"{n_cut:,} seconds of {pre_name} dropped so it ends before "
                 f"the 30th"))

        ok_v, n_cut = relabel_drop(labels, ok, d.index, onset, BOJ_SHOCK,
                                   stress_k)
        variants.append(run_variant(
            "stress starts at the decision", ok_v=ok_v,
            note=f"{n_cut:,} pre-announcement stressed seconds dropped"))

        mode = modal_onset(d.index)
        if mode is not None and mode < onset:
            lab_v, n_moved = relabel_move(labels, d.index, mode, onset,
                                          stress_k - 1, stress_k)
            variants.append(run_variant(
                f"02's modal onset, {mode:%d %b %H:%M}", labels_v=lab_v,
                min_prob=0.0,
                note=f"{n_moved:,} seconds moved from {pre_name} into "
                     f"stress; the reading this project does not take"))
        else:
            print("  note: 02's modal onset could not be read from "
                  "02_changepoint.csv; that sensitivity row is omitted")

    sens = pd.DataFrame(variants).set_index("variant")
    save_table(
        sens, "03_sensitivity",
        caption=("Sensitivity of the error-correction result. The first "
                 "block varies the transitory lag order, the second the "
                 "posterior threshold that buffers the changepoints, the "
                 "third the excluded rollover window, and the fourth where "
                 "the pre-shock window is taken to end - including the "
                 "24 July modal onset this project does not adopt, so its "
                 "cost can be read rather than argued."),
        label="tab:var_sensitivity",
    )
    print(sens.to_string())

    # ------------------------------------------------ daily closure speed
    #
    # Fitted with no regime label supplied, so the figure it feeds is not a
    # restatement of the partition it is drawn over.
    print("daily closure speed")
    day = pd.DatetimeIndex(d.index).floor("D")
    day_key = pd.factorize(day)[0]
    rows_day, _ = usable_rows(d.index, LAG_CAP, ~roll, day_key)
    daily_rows = []
    for code in np.unique(day_key[rows_day]):
        sel = rows_day[day_key[rows_day] == code]
        stamp = day[np.flatnonzero(day_key == code)[0]]
        record = {"day": stamp, "seconds": int(sel.size),
                  "lambda": np.nan, "half_life_s": np.nan}
        if sel.size >= MIN_WINDOW_ROWS:
            try:
                f = fit_vecm(str(stamp.date()), sel,
                             np.zeros(sel.size, dtype=np.int64), DX, Z,
                             p_star, light=True)
                record["lambda"] = f.lam
                record["half_life_s"] = f.half_life
            except (ValueError, np.linalg.LinAlgError):
                pass
        daily_rows.append(record)

    daily = pd.DataFrame(daily_rows).set_index("day")
    fitted_days = int(daily["half_life_s"].notna().sum())
    print(f" {fitted_days} of {len(daily)} days fitted "
          f"(a day needs {MIN_WINDOW_ROWS:,} admissible seconds)")
    save_table(
        daily.assign(**{
            "lambda": [num(v, ".4f") for v in daily["lambda"]],
            "half_life_s": [num(v, ".2f") for v in daily["half_life_s"]],
        }),
        "03_closure_speed",
        caption=("The same model fitted one day at a time, with no regime "
                 "label supplied. Days with fewer than "
                 f"{MIN_WINDOW_ROWS:,} admissible seconds are left "
                 "unestimated rather than fitted on a stub."),
        label="tab:closure_speed",
    )

    # --------------------------------------------------------- figures
    print("figures")
    figure_error_correction(fits, names, colours)
    figure_closure_speed(daily, fits, names, colours, boundaries)
    figure_diagnostics(fits, names, colours, acf, lags)

    # ----------------------------------------------------------- write
    print("data")
    frames = []
    for k, name in enumerate(names):
        f = fits.get(name)
        if f is None:
            continue
        frame = pd.DataFrame(
            f.resid.astype(np.float32),
            index=d.index[f.rows],
            columns=[f"e_{c.removeprefix('x_')}" for c in LEGS])
        frame.insert(0, "regime", np.int8(k))
        frame["z_lag"] = (Z[f.rows - 1] - f.zbar).astype(np.float32)
        # A run is a maximal stretch of consecutive seconds inside one
        # regime. 06 needs it: without it a lagged difference taken on this
        # file would silently span a weekend, which is the one mistake this
        # pipeline has already made once.
        frame["run"] = f.run
        frames.append(frame)

    resid = pd.concat(frames).sort_index()
    resid.index.name = "t"
    # Run ids arrive as positions in the full grid, unique across regimes by
    # construction. Renumbered to consecutive integers only to keep the file
    # small; uniqueness is preserved, so the id never has to be read together
    # with the regime column.
    resid["run"] = pd.factorize(resid["run"])[0].astype(np.int32)
    resid.to_parquet(DAT_DIR / "03_resid.parquet")
    print(f" -> 03_resid.parquet ({len(resid):,} rows, "
          f"{resid['run'].nunique():,} unbroken runs)")

    coefs.to_parquet(DAT_DIR / "03_var_model.parquet", index=False)
    print(f" -> 03_var_model.parquet ({len(coefs):,} coefficients, "
          f"p = {p_star})")

    print("checks")
    audit(fits, names)


if __name__ == "__main__":
    main()