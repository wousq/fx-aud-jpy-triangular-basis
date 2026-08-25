"""
utils.py - shared helpers. Definitions only; no execution at import.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synchronized_rates.parquet"
FIG_DIR = ROOT / "output" / "figures"
TAB_DIR = ROOT / "output" / "tables"
TEX_DIR = ROOT / "report" / "tables"
DAT_DIR = ROOT / "output" / "data"

PIP = {"AUDUSD": 1e-4, "USDJPY": 1e-2, "AUDJPY": 1e-2}
PAIRS = ["audusd_mid", "usdjpy_mid", "audjpy_direct"]
PAIR_LABEL = {"audusd_mid": "AUD/USD", "usdjpy_mid": "USD/JPY",
              "audjpy_direct": "AUD/JPY"}

# Grid resolution. Used to detect clock jumps, so it is named rather than
# written as a literal at the point of use.
GRID_SECONDS = 1
GRID_NS = GRID_SECONDS * 1_000_000_000

# Timestamps are New York local time. The vendor documents EST, but the
# sample spans 2024-07 to 2024-08, entirely inside US daylight saving
# (2024-03-10 to 2024-11-03), so the operative offset is EDT = UTC-4.
#
# The 17:00 Friday to 17:00 Sunday week boundary does NOT establish this.
# FX closes at 17:00 New York local year-round, so that boundary appears
# under either offset and is uninformative about which one applies. The
# offset was established by measurement instead: an offset scan against
# UTC-stamped Dukascopy ticks over the 2024-08-05 control window, which
# agrees to 0.01 pips. See docs/data-quality.md.
TZ_LABEL = "EDT"
UTC_OFFSET_HOURS = -4

# BOJ policy decision, 12:30 JST on 2024-07-31 = 03:30 UTC, expressed in
# the data's clock. Derived from UTC_OFFSET_HOURS rather than hard-coded
# so the two cannot drift apart. Plotting reference only: never passed to
# any model, and no estimator is given this date as a candidate.
BOJ_SHOCK = pd.Timestamp("2024-07-31 03:30:00") + pd.Timedelta(
    hours=UTC_OFFSET_HOURS)

# Daily rollover. Liquidity collapses, the synthetic leg goes stale and
# the basis widens mechanically. Excluded from episode detection.
ROLLOVER_START = "17:00"
ROLLOVER_END = "17:30"


# -------------------------------------------------------------- palette

# Four roles, and a colour means one role everywhere in the report.
#
# LEG_*   the three legs of the triangle. Unordered categories, so three
#         hues at matched lightness: no leg reads as more important.
# R1/R2   the two regimes. Ordered, so one hue at two lightnesses rather
#         than two hues; darker is later. Also carries uncertainty bands.
# FOUND   what the analysis found: episodes, tau-hat, event rules. Used
#         for nothing else, so its appearance always signals a result.
# MUTE    present in the data, excluded from estimation: rollover,
#         closures, gap-spanning rows.
# RULE    zero lines and axis furniture.
#
# Colour is never assigned from the calendar. A month is an assumption;
# a regime label is an estimate, and it comes from 02_regimes.parquet.

LEG_AUDUSD = "#2F6DA4"
LEG_USDJPY = "#4E9E8F"
LEG_AUDJPY = "#7E6AA6"

# Keyed by column name so a leg keeps its colour across every script
# without anyone maintaining list order.
LEG_COLOUR = {"audusd_mid": LEG_AUDUSD,
              "usdjpy_mid": LEG_USDJPY,
              "audjpy_direct": LEG_AUDJPY}

R1 = "#6D9AC6"      # first regime
R2 = "#1F4E79"      # second regime
REGIME_COLOUR = (R1, R2)

FOUND = "#D9922E"   # findings only
MUTE = "#98A2AB"    # excluded from estimation
RULE = "#333333"    # zero lines, axis furniture

BAND_ALPHA = 0.20   # uncertainty and envelope fills
SPAN_ALPHA = 0.18   # highlight spans


def make_dirs():
    for d in (FIG_DIR, TAB_DIR, TEX_DIR, DAT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ loading

def load_data(path=DATA_PATH, columns=None):
    """
    Load the synchronized grid indexed by timestamp.

    When `columns` is given the time column is requested explicitly: a
    projection that dropped it would otherwise fail at set_index with an
    error that does not name the cause.
    """
    if columns is not None:
        columns = list(dict.fromkeys(["grid_time", *columns]))
    df = pd.read_parquet(path, columns=columns)
    df = df.rename(columns={"grid_time": "t"})
    if "t" not in df.columns:
        raise KeyError(f"{path} has no 'grid_time' column")
    return df.set_index("t").sort_index()


def add_basis(df):
    """Basis in AUD/JPY pips, direct minus synthetic."""
    missing = {"audjpy_direct", "audjpy_synthetic"} - set(df.columns)
    if missing:
        raise KeyError(f"add_basis needs {sorted(missing)}")
    df["basis"] = (df["audjpy_direct"] - df["audjpy_synthetic"]) / PIP["AUDJPY"]
    return df


def add_log_prices(df):
    df["x_audusd"] = np.log(df["audusd_mid"])
    df["x_usdjpy"] = np.log(df["usdjpy_mid"])
    df["x_audjpy"] = np.log(df["audjpy_direct"])
    return df


# ------------------------------------------------------------ market state

def unchanged(df, cols=PAIRS):
    """
    True where the price equals the previous second's price.

    The first row has no predecessor and is reported False, matching
    closed_mask: an unknown predecessor is not evidence of staleness.
    """
    return pd.DataFrame({c: df[c].diff().eq(0) for c in cols}, index=df.index)


def _runs(flag):
    """Start and end (exclusive) indices of each True run in a bool array."""
    a = np.asarray(flag, dtype=bool).view(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], a, [0]))))
    return edges[::2], edges[1::2]


def closed_mask(df, cols=PAIRS, min_run=600):
    """
    Detect market closure from the data rather than from a calendar.

    A second is 'closed' if it sits inside a run of at least `min_run`
    consecutive seconds in which none of the three pairs printed a new
    price. Ten minutes of simultaneous silence across three majors does
    not happen while the market is open: it means the grid is padding.

    This cannot see a single-leg outage, which is why 00b_data_quality.py
    tests per-leg staleness separately.
    """
    flat = np.ones(len(df), dtype=bool)
    for c in cols:
        # The first row is not treated as flat: diff is undefined there.
        flat &= df[c].diff().eq(0).fillna(False).to_numpy()

    starts, ends = _runs(flat)
    mask = np.zeros(len(df), dtype=bool)
    keep = (ends - starts) >= min_run
    for s, e in zip(starts[keep], ends[keep]):
        mask[s:e] = True
    return pd.Series(mask, index=df.index)


def closure_report(mask):
    starts, ends = _runs(mask.to_numpy())
    return pd.DataFrame({
        "start": mask.index[starts],
        "end": mask.index[ends - 1],
        "hours": (ends - starts) / 3600.0,
    })


# ---------------------------------------------------------------- episodes

def in_rollover(index, start=None, end=None):
    """
    Half-open [start, end): the closing second belongs to the following
    window, so no second is counted in two regimes at once.
    """
    t = index.time
    lo = pd.Timestamp(start or ROLLOVER_START).time()
    hi = pd.Timestamp(end or ROLLOVER_END).time()
    return pd.Series((t >= lo) & (t < hi), index=index)


_EPISODE_COLUMNS = ["start", "end", "seconds", "peak", "sign", "month"]


def episodes(series, threshold, merge_gap="60s", min_seconds=5,
             exclude=None):
    """
    Contiguous stretches where |series| exceeds `threshold`.

    Dislocations in this data are plateaus, not isolated spikes, so the
    unit of analysis is the episode: when it started, how long the
    triangle stayed open, how wide it got, and in which direction.

    Two guards matter. Runs are split wherever consecutive rows are more
    than one grid step apart, so an episode can never span a market
    closure or a stretch removed as rollover. Rows flagged in `exclude`
    are removed first. Without the split, an exceedance either side of a
    48-hour weekend merges into one 48-hour episode; see
    tests/test_episodes.py.
    """
    gap = pd.Timedelta(merge_gap)
    s = series if exclude is None else series[
        ~exclude.reindex(series.index, fill_value=False)]
    if len(s) == 0:
        return pd.DataFrame(columns=_EPISODE_COLUMNS)

    t = s.index
    over = (s.abs() > threshold).to_numpy()

    jump = np.zeros(len(s), dtype=bool)
    jump[1:] = np.diff(t.asi8) > GRID_NS

    starts, ends = _runs(over)
    pieces = []
    for a, b in zip(starts, ends):
        cuts = np.flatnonzero(jump[a:b])
        prev = a
        for c in cuts:
            pieces.append((prev, a + c))
            prev = a + c
        pieces.append((prev, b))

    if not pieces:
        return pd.DataFrame(columns=_EPISODE_COLUMNS)

    merged = [list(pieces[0])]
    for a, b in pieces[1:]:
        contiguous = t[a] - t[merged[-1][1] - 1] <= gap
        if contiguous and not jump[a]:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    rows = []
    for a, b in merged:
        seg = s.iloc[a:b]
        secs = (t[b - 1] - t[a]).total_seconds() + GRID_SECONDS
        if secs < min_seconds:
            continue
        peak = seg.iloc[int(np.argmax(np.abs(seg.to_numpy())))]
        rows.append({"start": t[a], "end": t[b - 1], "seconds": secs,
                     "peak": peak, "sign": np.sign(peak)})

    if not rows:
        return pd.DataFrame(columns=_EPISODE_COLUMNS)

    ep = pd.DataFrame(rows)
    ep["month"] = ep["start"].dt.strftime("%B")
    return ep


# -------------------------------------------------------------- statistics

def robust_scale(x):
    """
    Median absolute deviation rescaled to standard-deviation units.

    The basis is leptokurtic, so the sample standard deviation is driven
    by a handful of episodes. Every dispersion figure in the report is
    reported on this scale for that reason.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return np.nan if a.size == 0 else 1.4826 * np.median(np.abs(a - np.median(a)))


def summary_stats(df, cols):
    """Kurtosis is excess kurtosis (pandas convention): 0 is Gaussian."""
    out = {}
    for c in cols:
        s = df[c].dropna()
        out[c] = {
            "n": len(s), "mean": s.mean(), "sd": s.std(),
            "MAD": robust_scale(s),
            "p99.9_abs": s.abs().quantile(0.999),
            "max_abs": s.abs().max(),
            "skew": s.skew(), "kurtosis": s.kurtosis(),
        }
    return pd.DataFrame(out).T


def acf1(x):
    """First-order autocorrelation; nan if the series is constant."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size <= 2 or np.ptp(a) == 0:
        return np.nan
    return float(np.corrcoef(a[1:], a[:-1])[0, 1])


def roll_noise(dx):
    """
    Roll (1984). Observed = efficient + iid noise of sd c implies
    Cov(dP_t, dP_{t-1}) = -c^2. Returns c, or nan where the covariance is
    non-negative, which rejects the model rather than admitting a complex
    root.
    """
    a = np.asarray(dx, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    cov1 = np.cov(a[1:], a[:-1])[0, 1]
    return float(np.sqrt(-cov1)) if cov1 < 0 else np.nan


def noise_share(dx):
    """
    Fraction of increment variance attributable to microstructure noise.
    Under Roll, rho_1 = -c^2 / (sigma_e^2 + 2c^2), so the noise share
    2c^2 / Var(dP) equals -2 * rho_1. Clipped to [0, 1]: the identity
    holds only under the model, and sampling error puts rho_1 outside
    [-0.5, 0] often enough to matter.
    """
    r = acf1(dx)
    return np.nan if not np.isfinite(r) else min(max(-2.0 * r, 0.0), 1.0)


def survival(x, n_grid=250, from_q=0.50):
    """
    Empirical P(|X| > u) on a log grid, for tail plots on log-log axes.
    Zeros are dropped because the grid is geometric.
    """
    a = np.abs(np.asarray(x, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return np.array([]), np.array([])
    lo = max(float(np.quantile(a, from_q)), 1e-6)
    hi = float(a.max())
    if hi <= lo:
        return np.array([lo]), np.array([0.0])
    grid = np.geomspace(lo, hi, n_grid)
    below = np.searchsorted(np.sort(a), grid, side="right")
    return grid, 1.0 - below / a.size


# ------------------------------------------------------------------ output

def save_table(df, name, caption="", label="", float_fmt="%.4g", index=True):
    """
    Write one table twice: CSV to read, LaTeX to \\input.

    No number is typed into the report by hand, so both files are written
    from the same frame in the same call and cannot drift apart.
    """
    make_dirs()
    df.to_csv(TAB_DIR / f"{name}.csv", float_format=float_fmt, index=index)
    df.to_latex(TEX_DIR / f"{name}.tex", float_format=float_fmt,
                caption=caption or None, label=label or None,
                escape=True, index=index)
    print(f"    -> {name}.csv / .tex")


def save_fig(fig, name):
    make_dirs()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"    -> {name}.png")


# ---------------------------------------------------------------- plotting

def set_style():
    """
    Agg is selected here so the pipeline renders identically headless,
    including under CI where no display exists.
    """
    matplotlib.use("Agg", force=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 8.5,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "axes.grid": True,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#777777",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(
            color=[LEG_AUDUSD, LEG_USDJPY, LEG_AUDJPY]),
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.dpi": 220,
    })


def envelope(series, rule="1min"):
    """
    Per-bucket min, max and median.

    The extremes are the point of these figures, so downsampling with
    .last() would delete them; the median gives the eye a line to follow
    through the band.
    """
    r = series.resample(rule)
    return r.min(), r.max(), r.median()


def plot_envelope(ax, series, colour, label=None, rule="12h",
                  alpha=BAND_ALPHA, lw=1.2):
    """
    Range as a band, median as a line.

    Use on full-sample and multi-week panels only. The 2.4 before/after
    figure must plot raw seconds: the argument there is a collapse from
    +43.38 to +0.20 pips in one second, and any resampling destroys the
    evidence the figure exists to show.
    """
    lo, hi, mid = envelope(series, rule)
    ax.fill_between(lo.index, lo, hi, color=colour, alpha=alpha, lw=0,
                    zorder=1)
    ax.plot(mid.index, mid, color=colour, lw=lw, label=label, zorder=3)


def plot_by_regime(ax, series, labels, colours=REGIME_COLOUR, lw=1.0):
    """
    Colour a series by estimated regime.

    `labels` comes from 02_regimes.parquet. No date is passed in, so the
    figure cannot disagree with the changepoint result. Masking with
    where() keeps the NaNs, so the line breaks at the boundary instead of
    being drawn across it.
    """
    lab = labels.reindex(series.index)
    for k, colour in enumerate(colours):
        ax.plot(series.index, series.where(lab == k), color=colour, lw=lw,
                zorder=3)


def plot_legs(ax, df, cols=PAIRS, lw=1.0):
    """Each leg keeps its colour across every script via LEG_COLOUR."""
    for c in cols:
        ax.plot(df.index, df[c], color=LEG_COLOUR[c], lw=lw,
                label=PAIR_LABEL[c])


def highlight_span(ax, start, end, colour=FOUND, alpha=SPAN_ALPHA):
    """
    Shade a time range behind the series. Spans are background: at full
    opacity the fill competes with the series it is meant to mark.
    """
    ax.axvspan(start, end, color=colour, alpha=alpha, lw=0, zorder=0)


def excluded_span(ax, start, end):
    """Rollover, closures, anything no estimator saw. Grey means only this."""
    highlight_span(ax, start, end, colour=MUTE, alpha=SPAN_ALPHA)


def zero_line(ax):
    ax.axhline(0, color=RULE, lw=0.6, zorder=2)


LABEL_GREY = "#6B7680"   # annotation text; quieter than the rule it labels


def annotate_event(ax, when, label, estimated=False, top=True):
    """
    Hairline rule with a label, drawn behind the data.

    Solid rather than dashed: the rule is already unambiguous, so a dash
    pattern is decoration. zorder=0 puts it under the series, which keeps
    the data continuous where the two cross.

    `estimated` selects the colour, and the distinction is the point.
    A reference event (the BOJ decision) is an input to the figure and
    draws in MUTE; an estimated quantity (tau-hat) is a result and draws
    in FOUND. Marking a reference in the accent colour would claim the
    analysis found something it was told.

    `when` must be in the data's clock (UTC-4), not UTC: a UTC timestamp
    places the rule four hours late.
    """
    colour = FOUND if estimated else MUTE
    ax.axvline(when, color=colour, lw=0.7, zorder=0)
    ax.annotate(label, xy=(when, 1.0 if top else 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -2 if top else 2), textcoords="offset points",
                color=colour if estimated else LABEL_GREY, fontsize=7.5,
                ha="left", va="top" if top else "bottom",
                annotation_clip=False)


def annotate_interval(ax, lo, hi, label=None, estimated=True, top=True):
    """
    Credible or confidence interval around an estimated instant, as a
    band rather than two rules. Use for the changepoint posterior: the
    width is the result, and two separate rules read as two events.
    """
    colour = FOUND if estimated else MUTE
    ax.axvspan(lo, hi, color=colour, alpha=SPAN_ALPHA, lw=0, zorder=0)
    if label:
        mid = lo + (hi - lo) / 2
        ax.annotate(label, xy=(mid, 1.0 if top else 0.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -2 if top else 2), textcoords="offset points",
                    color=colour, fontsize=7.5,
                    ha="center", va="top" if top else "bottom",
                    annotation_clip=False)