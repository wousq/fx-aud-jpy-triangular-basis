"""
05_execution_bounds.py

Was any of it executable?

Sections 01 to 04 measure the triangle on mid quotes. That is the right
series for asking how prices adjust, and the wrong one for asking whether a
dislocation was an opportunity: nobody trades at the mid. A round trip
around the triangle crosses three spreads, and the no-arbitrage condition
the earlier sections test is not z = 0 but z inside a band whose width is
those three spreads. This script measures the band and asks how often the
basis left it.

The two round trips
    Working in AUD/JPY price units and writing b and a for bid and ask,

        e_A = AJ_bid - AU_ask * UJ_ask      sell direct, buy synthetic
        e_B = AU_bid * UJ_bid - AJ_ask      buy direct, sell synthetic

    Each is the profit per unit of AUD from one complete circuit at prices
    that were quoted. Both are executable when positive, and they cannot be
    positive together: e_A > 0 and e_B > 0 would require AJ_bid > AJ_ask.

The band, and why it is an identity
    Let z be the mid basis section 01 measured. Then

        e_A = z - c_A,      c_A = (AJ_mid - AJ_bid)
                                + (AU_ask * UJ_ask - AU_mid * UJ_mid)
        e_B = -z - c_B,     c_B = (AJ_ask - AJ_mid)
                                + (AU_mid * UJ_mid - AU_bid * UJ_bid)

    so nothing is executable while z lies in [-c_B, +c_A], and the width of
    that band is

        W = c_A + c_B = s_AJ + AU_mid * s_UJ + UJ_mid * s_AU

    where s is a quoted spread. The cross terms cancel exactly, not to
    first order: the band decomposes additively into the three legs with no
    interaction. Which leg's spread sets the bound is therefore an identity
    question, answered by arithmetic, in the same way section 04 answers
    which leg opens a gap. Nothing is estimated and no cost parameter is
    assumed anywhere in this script.

What is measured, and against what
    Section 02 found the basis dispersion rose by a factor of about three
    under stress. That is a statement about the numerator of a ratio whose
    denominator this project had not measured. The quantity of interest
    here is

        W / MAD(z)      the band in units of the gap

    per regime. If W grew by more than the gap did, the market became
    harder to arbitrage while looking more dislocated, and the widening
    section 02 detected is a widening of the bound rather than a failure of
    it. If W grew by less, the opposite. The comparison is the point of the
    script and it needs both numbers on the same seconds, which is why the
    row admission below is taken from 01_clean.parquet rather than rebuilt.

Staleness, and why the gate is not optional
    The grid is forward-filled. A quote that has not printed for thirty
    seconds still appears in the row, and a band built from two stale legs
    and one fresh one is not a band anyone could have traded inside. Most
    apparent opportunities on a filled grid are this artefact. Freshness is
    read from the pipeline's own tick flags rather than inferred from price
    changes, because a repeated identical quote is fresh and a diff cannot
    see that. The headline requires all three legs to have printed within
    MAX_QUOTE_AGE seconds; the full ladder from that gate to no gate at all
    is reported so its effect can be read instead of argued.

Latency
    A signal at t cannot be traded at t. Every executability count is
    reported at execution delays of 0, 1, 2 and 5 seconds: the direction is
    chosen on quotes at t, the profit is booked at quotes at t + delta, and
    windows that span a closure, a rollover exclusion or a regime boundary
    are dropped rather than differenced across. delta = 0 is retained as
    the unattainable upper bound, not as a result.

Retail quotes, and the haircut
    Both vendors publish aggregator quotes, which are wider than what a
    bank arbitrage desk sees. This biases the exercise toward finding
    nothing, so a null result here is weaker than it looks and is not
    reported on its own. 05_haircut.csv instead scales every spread by a
    factor k and reports the k at which opportunities first appear, which
    turns the caveat into a measurement: the reader can decide whether a
    desk plausibly sees a band that much tighter.

Outputs
    output/tables/05_execution.csv      band, gap and executability by regime
    output/tables/05_band_vs_gap.csv    the ratio-of-ratios comparison
    output/tables/05_decomposition.csv  the band split across the three legs
    output/tables/05_breakeven.csv      per-episode cost required to clear
    output/tables/05_latency.csv        delay x freshness gate
    output/tables/05_haircut.csv        executability against a spread haircut
    output/tables/05_sensitivity.csv
    output/figures/05_bands.png
    output/figures/05_executable.png
    output/figures/05_decomposition.png
    output/data/05_bounds.parquet       per-second band and edge
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    make_dirs, set_style, load_data, adjacent, check_grid, in_rollover,
    robust_scale, save_table, save_fig, zero_line, annotate_event,
    regime_colours, survival,
    PIP, FOUND, RULE, BOJ_SHOCK, DAT_DIR,
    LEG_AUDUSD, LEG_USDJPY, LEG_AUDJPY, TZ_LABEL,
    ROLLOVER_START, ROLLOVER_END, BAND_ALPHA,
)

# All three legs must have printed within this many seconds for a second to
# count toward the headline. Zero means every leg printed in that second.
MAX_QUOTE_AGE = 1

# Execution delays, in seconds. Zero is reported as an upper bound only.
DELAYS = (0, 1, 2, 5)

# Freshness gates for the ladder. None means no gate at all.
GATES = (0, 1, 2, 5, 60, None)

# Spread haircuts. 1.0 is the quoted band; smaller is a tighter market than
# these vendors show.
HAIRCUTS = (1.00, 0.75, 0.50, 0.33, 0.25, 0.10, 0.05)

# Matches 03: a second joins a regime only where the posterior is at least
# this confident, so the buffer around each changepoint enters nothing.
MIN_POSTERIOR = 0.90

# Bid and ask must reproduce the mid the rest of the project used. Tolerance
# is a hundredth of a pip, which is below the quote increment of every leg.
MID_TOLERANCE_PIPS = 0.01

QUOTE_COLUMNS = ["audusd_bid", "audusd_ask", "usdjpy_bid", "usdjpy_ask",
                 "audjpy_bid", "audjpy_ask"]
FRESH_COLUMNS = ["audusd_fresh", "usdjpy_fresh", "audjpy_fresh"]

LEG_NAMES = ["AUD/USD", "USD/JPY", "AUD/JPY"]
LEG_BAND_COLOUR = {"AUD/USD": LEG_AUDUSD, "USD/JPY": LEG_USDJPY,
                   "AUD/JPY": LEG_AUDJPY}


# ------------------------------------------------------------------ loading

def load_quotes():
    """
    The quote columns, on exactly the rows sections 03 and 04 used.

    01_clean.parquet is the project's record of which seconds survived
    closure removal. Reapplying closed_mask here would recompute it, and a
    recomputation that disagreed by one row would put this script's
    dispersion figures on a different sample from section 02's without
    anything failing. So the index is read, not derived.
    """
    clean_path = DAT_DIR / "01_clean.parquet"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run 01 before 05: this script compares "
            f"a band against the dispersion 01 and 02 measured, and the "
            f"comparison is only meaningful on identical rows.")
    clean = pd.read_parquet(clean_path, columns=["basis"]).sort_index()

    full = load_data(columns=QUOTE_COLUMNS + FRESH_COLUMNS)
    missing = [c for c in QUOTE_COLUMNS if c not in full.columns]
    if missing:
        raise KeyError(
            f"synchronized_rates.parquet has no {missing}. The ingest step "
            f"collapsed each tick to a mid and discarded the quotes; rerun "
            f"00_ingest_and_sync.py with the bid/ask columns carried "
            f"through, then rerun 00b before this script.")

    d = clean.join(full, how="left")
    absent = int(d[QUOTE_COLUMNS].isna().any(axis=1).sum())
    if absent:
        raise ValueError(
            f"{absent:,} of {len(d):,} rows in 01_clean.parquet have no "
            f"quote in synchronized_rates.parquet. The two files come from "
            f"different runs; rerun 00 and 01.")
    return d


def check_quotes(d):
    """
    Two gates, in the style of 00b: the quotes must not be crossed, and
    their midpoint must be the mid the earlier sections analysed.

    The second gate is the informative one. It is the only thing standing
    between a column swap at ingest and a no-arbitrage band that is wrong
    by the width of a spread in every table below, which nothing downstream
    would flag because a wrong band is still a plausible band.
    """
    crossed = int(((d.audusd_bid >= d.audusd_ask)
                   | (d.usdjpy_bid >= d.usdjpy_ask)
                   | (d.audjpy_bid >= d.audjpy_ask)).sum())
    if crossed:
        raise ValueError(f"{crossed:,} seconds carry a crossed or locked "
                         f"quote; 00 should have rejected these")

    worst = 0.0
    for leg, lo, hi, ref, pip in [
        ("AUD/USD", "audusd_bid", "audusd_ask", "audusd_mid", PIP["AUDUSD"]),
        ("USD/JPY", "usdjpy_bid", "usdjpy_ask", "usdjpy_mid", PIP["USDJPY"]),
        ("AUD/JPY", "audjpy_bid", "audjpy_ask", "audjpy_direct", PIP["AUDJPY"]),
    ]:
        if ref not in d.columns:
            continue
        gap = ((d[lo] + d[hi]) / 2.0 - d[ref]).abs().max() / pip
        worst = max(worst, float(gap))
        print(f"    {leg}: quotes reproduce the mid to {gap:.4f} pips")
    if worst > MID_TOLERANCE_PIPS:
        raise ValueError(
            f"the midpoint of bid and ask departs from the mid used by 01 to "
            f"04 by up to {worst:.4f} pips, above the {MID_TOLERANCE_PIPS} "
            f"pip tolerance. Either the columns are misaligned or the two "
            f"were built from different ticks.")
    return worst


def quote_age(d):
    """
    Seconds since each leg last printed, from the pipeline's tick flags.

    Not from price changes: 00's docstring makes the point that a repeated
    identical quote is a genuine print and a diff calls it stale. Age is
    reset at a grid discontinuity, because the number of seconds since the
    last print is not defined across a closure.
    """
    n = len(d)
    pos = np.arange(n)
    adj = adjacent(d.index)
    out = {}
    for col in FRESH_COLUMNS:
        # A discontinuity restarts the clock: the row after a gap is treated
        # as having no measurable age rather than an age spanning the gap.
        anchor = d[col].to_numpy(dtype=bool) | ~adj
        last = np.maximum.accumulate(np.where(anchor, pos, 0))
        out[col.replace("_fresh", "")] = pos - last
    age = pd.DataFrame(out, index=d.index)
    return age.max(axis=1)


# -------------------------------------------------------------------- bands

def add_bounds(d):
    """
    The two round trips, the band around the mid basis and its decomposition.

    Everything here is in AUD/JPY pips. `basis` arrives from 01_clean and is
    recomputed from the mids as a cross-check: the identity e_A = z - c_A
    only holds if the z in the file is the z these quotes imply.
    """
    pip = PIP["AUDJPY"]
    au_mid = (d.audusd_bid + d.audusd_ask) / 2.0
    uj_mid = (d.usdjpy_bid + d.usdjpy_ask) / 2.0
    aj_mid = (d.audjpy_bid + d.audjpy_ask) / 2.0

    d["e_rich"] = (d.audjpy_bid - d.audusd_ask * d.usdjpy_ask) / pip
    d["e_cheap"] = (d.audusd_bid * d.usdjpy_bid - d.audjpy_ask) / pip
    # Mutually exclusive by construction, so the maximum is the profit of
    # the only circuit that could have been on.
    d["edge"] = np.maximum(d.e_rich, d.e_cheap)
    d["direction"] = np.where(d.e_rich >= d.e_cheap, 1, -1)

    z = (aj_mid - au_mid * uj_mid) / pip
    d["c_rich"] = z - d.e_rich
    d["c_cheap"] = -z - d.e_cheap
    d["band"] = d.c_rich + d.c_cheap

    # The additive decomposition. Exact: see the docstring.
    d["band_audjpy"] = (d.audjpy_ask - d.audjpy_bid) / pip
    d["band_usdjpy"] = au_mid * (d.usdjpy_ask - d.usdjpy_bid) / pip
    d["band_audusd"] = uj_mid * (d.audusd_ask - d.audusd_bid) / pip

    residual = float((d.band - (d.band_audjpy + d.band_usdjpy
                                + d.band_audusd)).abs().max())
    if residual > 1e-9:
        raise ValueError(f"the band decomposition leaves {residual:.3g} pips "
                         f"unattributed; it should be exact")

    drift = float((z - d["basis"]).abs().max())
    if drift > MID_TOLERANCE_PIPS:
        raise ValueError(
            f"the basis implied by these quotes departs from the one 01 "
            f"wrote by up to {drift:.4f} pips; the band would be measured "
            f"against a different series from the one 02 found a "
            f"changepoint in")
    print(f"    band decomposition exact to {residual:.2g} pips; "
          f"basis agrees with 01 to {drift:.4f} pips")
    return d


# ------------------------------------------------------------------ regimes

def load_regimes(d):
    """
    Per-second segment probabilities from 02.

    A compact restatement of 03's loader rather than an import, because a
    module whose name begins with a digit is not importable and duplicating
    forty lines is preferable to renaming four scripts. The invariants it
    checks are the same ones, and it fails on the same conditions.
    """
    path = DAT_DIR / "02_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 02 before 05: the pre/stress/post split "
            f"is an estimate, and substituting a calendar month for it is "
            f"what 02 exists to prevent.")

    r = pd.read_parquet(path).sort_index().reindex(d.index)
    prob_cols = [c for c in r.columns if c.startswith("p_")]
    if not prob_cols:
        raise KeyError(f"{path} carries no p_* columns, only {list(r.columns)}")
    if int(r[prob_cols].isna().any(axis=1).sum()):
        raise ValueError(f"{path} does not cover every second of "
                         f"01_clean.parquet; the two are from different runs")

    prob = r[prob_cols].to_numpy(dtype=float)
    labels = prob.argmax(axis=1).astype(np.int8)
    best = prob.max(axis=1)

    # The stressed segment is identified by dispersion, never by the
    # calendar and never by the decision date. The decision is then used
    # only to check the answer, exactly as in 03.
    n = len(prob_cols)
    mad = np.array([robust_scale(d["basis"].to_numpy()[labels == k])
                    if (labels == k).any() else np.nan for k in range(n)])
    stress_k = int(np.nanargmax(mad))
    pos = int(pd.DatetimeIndex(d.index).get_indexer([BOJ_SHOCK],
                                                    method="nearest")[0])
    if int(labels[pos]) != stress_k:
        print(f"    note: the widest segment is {stress_k} but the decision "
              f"falls in segment {int(labels[pos])}; dispersion decides")

    if n <= 2:
        names = ["stress" if k == stress_k else "calm" for k in range(n)]
    else:
        names = ["stress" if k == stress_k else ("pre" if k < stress_k
                 else "post") for k in range(n)]
    return labels, best, names


# ------------------------------------------------------------ executability

def contiguous_for(index, labels, delta):
    """
    True where the next `delta` seconds are an unbroken, same-regime run.

    Without this a delayed edge could be booked at a price on the far side
    of a weekend. The adjacency test is cumulative rather than a rolling
    window so the cost does not grow with delta.
    """
    n = len(index)
    ok = np.zeros(n, dtype=bool)
    if delta == 0:
        ok[:] = True
        return ok
    if n <= delta:
        return ok
    adj = adjacent(index).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(adj)))
    # positions i..i+delta are contiguous iff every adjacency flag in
    # (i, i+delta] is set.
    unbroken = (cs[1 + delta:] - cs[1:n - delta + 1]) == delta
    same = labels[delta:] == labels[:n - delta]
    ok[:n - delta] = unbroken & same
    return ok


def realised_edge(d, delta):
    """
    Profit booked at t + delta on a direction chosen at t.

    The signal and the fill are deliberately taken from different rows. A
    scan that reads the edge at t and books it at t is not measuring an
    opportunity, it is measuring a quote.
    """
    if delta == 0:
        return d["edge"].to_numpy()
    rich = d["e_rich"].to_numpy()
    cheap = d["e_cheap"].to_numpy()
    chose_rich = d["direction"].to_numpy() == 1
    out = np.full(len(d), np.nan)
    out[:len(d) - delta] = np.where(chose_rich[:len(d) - delta],
                                    rich[delta:], cheap[delta:])
    return out


def executable_counts(d, labels, names, admit, gate, delta, haircut=1.0):
    """
    Seconds on which a complete circuit would have paid, by regime.

    A haircut scales the band, which is equivalent to scaling every quoted
    spread: edge(k) = z - k * c on the rich side and -z - k * c on the
    cheap side, so the mid basis is held fixed and only the cost moves.
    """
    age_ok = np.ones(len(d), dtype=bool) if gate is None \
        else (d["max_age"].to_numpy() <= gate)
    run_ok = contiguous_for(d.index, labels, delta)

    if haircut == 1.0:
        edge = realised_edge(d, delta)
    else:
        z = d["basis"].to_numpy()
        rich = z - haircut * d["c_rich"].to_numpy()
        cheap = -z - haircut * d["c_cheap"].to_numpy()
        chose_rich = rich >= cheap
        if delta == 0:
            edge = np.maximum(rich, cheap)
        else:
            edge = np.full(len(d), np.nan)
            edge[:len(d) - delta] = np.where(chose_rich[:len(d) - delta],
                                             rich[delta:], cheap[delta:])

    live = admit & age_ok & run_ok & np.isfinite(edge)
    rows = {}
    for k, name in enumerate(names):
        sel = live & (labels == k)
        n_sel = int(sel.sum())
        hit = sel & (edge > 0)
        rows[name] = {
            "eligible seconds": n_sel,
            "executable seconds": int(hit.sum()),
            "share": float(hit.sum()) / n_sel if n_sel else np.nan,
            "best edge (pips)": float(edge[sel].max()) if n_sel else np.nan,
            "median shortfall (pips)": float(np.median(-edge[sel]))
                                       if n_sel else np.nan,
        }
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def titled(ax, title, sub):
    """
    Title and grey subtitle, set together.

    Two reasons this is a helper rather than two calls at each site. The
    pad has to clear the subtitle, and the subtitle is pinned to the axes
    top in offset points rather than at an axes fraction: elsewhere in the
    project it sits at y = 1.012, which is 2 points on a short panel and 3
    on a tall one, so whether it collides with the title depends on the
    height of the panel it happens to be drawn in.

    The title is passed in rather than read back with get_title(). This
    project sets axes.titlelocation to "left", and get_title() reads the
    centre title, so a helper that round-trips the text through it silently
    erases the title it was meant to keep.
    """
    ax.set_title(title, pad=20)
    ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points",
                fontsize=7.5, color="#666666", va="bottom", ha="left",
                annotation_clip=False)


def figure_bands(d, daily):
    fig = plt.figure(figsize=(10, 6.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.52)
    ax_top = fig.add_subplot(grid[0])
    ax_bot = fig.add_subplot(grid[1])

    minute = d[["basis", "c_rich", "c_cheap"]].resample("1min")
    lo, hi = minute["basis"].min(), minute["basis"].max()
    up, dn = minute["c_rich"].median(), -minute["c_cheap"].median()

    ax_top.fill_between(up.index, dn, up, color=FOUND, alpha=BAND_ALPHA,
                        lw=0, zorder=1, label="no-arbitrage band")
    ax_top.vlines(lo.index, lo.values, hi.values, color=RULE, lw=0.45,
                  zorder=3, label="basis, per-minute range")
    # The fill alone is not enough. Where the basis is dense it covers the
    # band, and the bound is the subject of the panel, so its edges are
    # drawn over the data rather than under it.
    ax_top.plot(up.index, up, color=FOUND, lw=0.9, zorder=4)
    ax_top.plot(dn.index, dn, color=FOUND, lw=0.9, zorder=4)
    zero_line(ax_top)

    # Symlog for the same reason 01 uses it: a handful of episodes reach
    # twenty pips and would otherwise flatten the region the figure is
    # about, which is the few pips either side of zero where the band sits.
    lin = max(1.0, float(np.ceil(np.nanmedian(up.to_numpy()) * 2)))
    ax_top.set_yscale("symlog", linthresh=lin, linscale=0.9)
    ax_top.set_ylabel("pips")
    titled(ax_top,
           "The mid basis against the band it would have had to cross",
           f"the band is quoted, not assumed: it is the sum of the three "
           f"legs' spreads at that minute. Symlog scale, linear within "
           f"±{lin:.0f} pips")
    ax_top.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_top.legend(loc="lower left", ncol=2)
    annotate_event(ax_top, BOJ_SHOCK, "BOJ")

    ax_bot.plot(daily.index, daily["band"], lw=1.5, color=FOUND, marker="D",
                ms=2.6, label="band width")
    ax_bot.plot(daily.index, daily["mad"], lw=1.5, color=RULE, marker="o",
                ms=2.6, label="basis dispersion (MAD)")
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("pips (log scale)")
    titled(ax_bot,
           "Both widened under stress. The question is which widened more",
           "on a log axis a parallel shift means the triangle was no closer "
           "to arbitrageable than before, only noisier")
    ax_bot.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_bot.legend(loc="upper left", ncol=2)
    annotate_event(ax_bot, BOJ_SHOCK, "BOJ", top=False)
    save_fig(fig, "05_bands")


def figure_executable(d, labels, names, colours, breakeven):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4),
                           gridspec_kw={"wspace": 0.28})

    for k, name in enumerate(names):
        sel = labels == k
        if not sel.any():
            continue
        shortfall = -d["edge"].to_numpy()[sel]
        values, probabilities = survival(shortfall[shortfall > 0])
        ax[0].loglog(values, probabilities, lw=1.8, color=colours[k],
                     label=name, solid_capstyle="round")
    ax[0].set_xlabel("shortfall to the nearest bound $u$ (pips)")
    ax[0].set_ylabel(r"$P\,(\,\mathrm{shortfall} > u\,)$")
    titled(ax[0], "How far inside the band the triangle sat",
           "left is closer to executable")
    ax[0].legend(loc="lower left")
    ax[0].grid(which="both", alpha=0.12)

    if len(breakeven):
        for k, name in enumerate(names):
            mine = breakeven[breakeven["regime"] == name]
            if not len(mine):
                continue
            ax[1].scatter(mine["required"], mine["actual"], s=30,
                          c=colours[k], alpha=0.85, lw=0.4,
                          edgecolor="white", label=f"{name} (n={len(mine)})")
        limit = max(float(breakeven[["required", "actual"]].max().max()) * 1.3,
                    1.0)
        floor = max(float(breakeven[["required", "actual"]].min().min()) * 0.7,
                    1e-2)
        ax[1].plot([floor, limit], [floor, limit], color=RULE, lw=0.8, ls=":")
        ax[1].annotate("above the line: the band was wider than the gap",
                       xy=(0.97, 0.05), xycoords="axes fraction",
                       ha="right", va="bottom", fontsize=7, color=RULE)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlim(floor, limit)
        ax[1].set_ylim(floor, limit)
    ax[1].set_xlabel("cost that would have broken even (pips, log)")
    ax[1].set_ylabel("cost actually quoted (pips, log)")
    ax[1].set_title("Every episode 01 detected, at its widest second")
    ax[1].legend(loc="upper left")
    ax[1].grid(which="both", alpha=0.12)
    save_fig(fig, "05_executable")


def figure_decomposition(names, decomposition, daily_legs):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.8),
                           gridspec_kw={"hspace": 0.46,
                                        "height_ratios": [1.1, 1]})

    bottom = np.zeros(len(daily_legs))
    for leg, col in [("AUD/JPY", "band_audjpy"), ("USD/JPY", "band_usdjpy"),
                     ("AUD/USD", "band_audusd")]:
        values = daily_legs[col].to_numpy()
        ax[0].fill_between(daily_legs.index, bottom, bottom + values,
                           color=LEG_BAND_COLOUR[leg], lw=0, alpha=0.85,
                           label=leg, step="mid")
        bottom = bottom + values
    ax[0].set_ylabel("pips")
    titled(ax[0], "What the no-arbitrage band is made of, day by day",
           "the three contributions sum to the band exactly; there is no "
           "interaction term")
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[0].legend(loc="upper left", ncol=3)
    annotate_event(ax[0], BOJ_SHOCK, "BOJ")

    width = 0.26
    spots = np.arange(len(names))
    for i, (leg, col) in enumerate([("AUD/USD", "AUD/USD"),
                                    ("USD/JPY", "USD/JPY"),
                                    ("AUD/JPY", "AUD/JPY")]):
        heights = [decomposition.loc[f"share, {col}", name] for name in names]
        ax[1].bar(spots + (i - 1) * width, heights, width * 0.92,
                  color=LEG_BAND_COLOUR[leg], label=leg)
    ax[1].set_xticks(spots)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("share of the band")
    ax[1].set_title("The same split, by estimated regime")
    tallest = max(decomposition.loc[f"share, {leg}", name]
                  for leg in LEG_NAMES for name in names)
    ax[1].set_ylim(0, tallest * 1.38)
    ax[1].legend(loc="upper center", ncol=3)
    save_fig(fig, "05_decomposition")


# --------------------------------------------------------------------- main

def main():
    make_dirs()
    set_style()

    print("load")
    d = load_quotes()
    info = check_grid(d.index, "05 quote grid")
    print(f" {len(d):,} rows, {info['adjacent_share']:.1%} adjacent, "
          f"{d.index[0]} to {d.index[-1]}")

    print("quote checks")
    check_quotes(d)

    print("bounds")
    d = add_bounds(d)
    d["max_age"] = quote_age(d)

    print("regimes")
    labels, posterior, names = load_regimes(d)
    colours = regime_colours(len(names))
    roll = in_rollover(d.index).to_numpy()
    admit = (posterior >= MIN_POSTERIOR) & ~roll
    print(f" {len(names)} segments: {', '.join(names)}")
    print(f" admitting {admit.mean():.1%} of rows "
          f"(posterior >= {MIN_POSTERIOR:.0%}, "
          f"{ROLLOVER_START}-{ROLLOVER_END} {TZ_LABEL} removed)")
    fresh = (d["max_age"].to_numpy() <= MAX_QUOTE_AGE)
    print(f" all three legs quoting within {MAX_QUOTE_AGE}s on "
          f"{fresh.mean():.1%} of rows")

    # ------------------------------------------------------- headline table
    print("band against gap")
    headline = {}
    for k, name in enumerate(names):
        sel = admit & fresh & (labels == k)
        rows = d.loc[sel]
        gap = robust_scale(rows["basis"])
        band = float(rows["band"].median())
        # The gated dispersion is the one the ratio uses, because a ratio
        # whose numerator and denominator come from different seconds is not
        # a ratio of anything. The ungated figure is carried beside it so the
        # comparison with 02, which gated on neither freshness nor posterior,
        # can be made rather than assumed.
        headline[name] = {
            "eligible seconds": int(sel.sum()),
            "hours": sel.sum() / 3600.0,
            "basis dispersion, MAD (pips)": gap,
            "basis dispersion, ungated (pips)": robust_scale(
                d.loc[labels == k, "basis"]),
            "band width, median (pips)": band,
            "band width, p10 (pips)": float(rows["band"].quantile(0.10)),
            "band width, p90 (pips)": float(rows["band"].quantile(0.90)),
            "cost to sell direct, median (pips)": float(rows["c_rich"].median()),
            "cost to buy direct, median (pips)": float(rows["c_cheap"].median()),
            "band / dispersion": band / gap if gap else np.nan,
            "widest |basis| (pips)": float(rows["basis"].abs().max()),
            "best edge (pips)": float(rows["edge"].max()),
            "executable seconds": int((rows["edge"] > 0).sum()),
        }
    headline = pd.DataFrame(headline)
    save_table(
        headline, "05_execution",
        caption=(f"The no-arbitrage band and the basis on the same seconds. "
                 f"Rows are restricted to seconds where all three legs quoted "
                 f"within {MAX_QUOTE_AGE}s and the regime posterior is at "
                 f"least {MIN_POSTERIOR:.0%}. A positive edge is a complete "
                 f"circuit at quoted prices; the band is the sum of the three "
                 f"quoted spreads and is not a fitted or assumed cost."),
        label="tab:execution")
    print(headline.to_string(float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------- the ratio comparison
    if len(names) >= 2 and "stress" in names:
        stress = "stress"
        others = [n for n in names if n != stress]
        rows = {}
        for other in others:
            gap_ratio = (headline.loc["basis dispersion, MAD (pips)", stress]
                         / headline.loc["basis dispersion, MAD (pips)", other])
            band_ratio = (headline.loc["band width, median (pips)", stress]
                          / headline.loc["band width, median (pips)", other])
            rows[f"stress / {other}"] = {
                "dispersion ratio": gap_ratio,
                "band ratio": band_ratio,
                "band ratio / dispersion ratio": band_ratio / gap_ratio,
                "reading": ("the bound widened faster than the gap"
                            if band_ratio > gap_ratio else
                            "the gap widened faster than the bound"),
            }
        band_vs_gap = pd.DataFrame(rows)
        save_table(
            band_vs_gap, "05_band_vs_gap",
            caption=("Section 02 measured the numerator of this ratio and "
                     "not the denominator. A band ratio above the dispersion "
                     "ratio means the market became harder to arbitrage as "
                     "it became more dislocated, so the widening 02 detected "
                     "is a widening of the no-arbitrage bound rather than a "
                     "failure of it."),
            label="tab:band_vs_gap")
        print(band_vs_gap.to_string())

    # ------------------------------------------------------- decomposition
    print("decomposition")
    decomposition = {}
    for k, name in enumerate(names):
        sel = admit & fresh & (labels == k)
        rows = d.loc[sel]
        total = float(rows["band"].median())
        cell = {}
        for leg, col in [("AUD/USD", "band_audusd"), ("USD/JPY", "band_usdjpy"),
                         ("AUD/JPY", "band_audjpy")]:
            median = float(rows[col].median())
            cell[f"median, {leg} (pips)"] = median
            cell[f"share, {leg}"] = median / total if total else np.nan
        cell["band width, median (pips)"] = total
        decomposition[name] = cell
    decomposition = pd.DataFrame(decomposition)
    save_table(
        decomposition, "05_decomposition",
        caption=("The band splits additively into the three legs' quoted "
                 "spreads, weighted into AUD/JPY pips. The identity is exact, "
                 "so the shares are arithmetic rather than an estimate. "
                 "Medians are taken per leg and need not sum exactly to the "
                 "median band."),
        label="tab:band_decomposition")
    print(decomposition.to_string(float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------------------ latency
    print("latency and freshness")
    ladder = []
    for gate in GATES:
        for delta in DELAYS:
            counts = executable_counts(d, labels, names, admit, gate, delta)
            row = {"quote age gate (s)": "none" if gate is None else gate,
                   "execution delay (s)": delta}
            for name in names:
                row[f"{name}, seconds"] = counts.loc["executable seconds", name]
                row[f"{name}, share"] = counts.loc["share", name]
            ladder.append(row)
    ladder = pd.DataFrame(ladder)
    save_table(
        ladder, "05_latency", index=False,
        caption=("Executable seconds against the freshness gate and the "
                 "execution delay. The direction is chosen on quotes at t "
                 "and the profit booked at quotes at t plus the delay; "
                 "windows spanning a closure, the rollover exclusion or a "
                 "regime boundary are dropped. A zero delay is unattainable "
                 "and is reported as an upper bound."),
        label="tab:latency")
    print(ladder.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------------------ haircut
    print("haircut")
    haircut_rows = []
    for k in HAIRCUTS:
        counts = executable_counts(d, labels, names, admit,
                                   MAX_QUOTE_AGE, 1, haircut=k)
        row = {"spread haircut": k,
               "band as quoted": f"{k:.0%}"}
        for name in names:
            row[f"{name}, seconds"] = counts.loc["executable seconds", name]
            row[f"{name}, share"] = counts.loc["share", name]
        haircut_rows.append(row)
    haircut = pd.DataFrame(haircut_rows)
    save_table(
        haircut, "05_haircut", index=False,
        caption=("Both vendors publish retail aggregator quotes, which are "
                 "wider than interdealer. Rather than assert that the "
                 "difference does not matter, every spread is scaled by the "
                 "haircut and the scan repeated at a one-second delay. The "
                 "haircut at which opportunities first appear is the amount "
                 "by which a desk would have to beat these quotes before the "
                 "dislocations in this sample became trades."),
        label="tab:haircut")
    print(haircut.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ----------------------------------------------------------- breakeven
    print("breakeven, episode by episode")
    ep_path = DAT_DIR / "01_episodes.parquet"
    breakeven = pd.DataFrame(columns=["regime", "required", "actual", "ratio"])
    if ep_path.exists():
        ep = pd.read_parquet(ep_path)
        rows = []
        for _, e in ep.iterrows():
            window = d.loc[e["start"]:e["end"]]
            if not len(window):
                continue
            at = window["basis"].abs().idxmax()
            row = d.loc[at]
            k = int(labels[d.index.get_loc(at)])
            # The cost that would have broken even is the gap itself, on the
            # side the gap opened; the cost actually quoted is that side's
            # half of the band.
            required = abs(float(row["basis"]))
            actual = float(row["c_rich"] if row["basis"] > 0 else row["c_cheap"])
            rows.append({"regime": names[k], "start": at,
                         "required": required, "actual": actual,
                         "ratio": actual / required if required else np.nan,
                         "quote age (s)": int(row["max_age"])})
        breakeven = pd.DataFrame(rows)

        summary = {}
        for name in names:
            mine = breakeven[breakeven["regime"] == name]
            if not len(mine):
                continue
            summary[name] = {
                "episodes": len(mine),
                "median gap at peak (pips)": mine["required"].median(),
                "median cost quoted (pips)": mine["actual"].median(),
                "median cost / gap": mine["ratio"].median(),
                "episodes that cleared": int((mine["ratio"] < 1).sum()),
                "tightest ratio": mine["ratio"].min(),
            }
        summary = pd.DataFrame(summary)
        save_table(
            summary, "05_breakeven",
            caption=("Each episode 01 detected, evaluated at its widest "
                     "second. The gap at that second is the round-trip cost "
                     "that would exactly have broken even; the quoted cost is "
                     "what the three spreads actually came to on the side the "
                     "gap opened. A ratio above one means the episode was "
                     "never a trade."),
            label="tab:breakeven")
        print(summary.to_string(float_format=lambda v: f"{v:.4g}"))
    else:
        print(f"    {ep_path} not found; skipping the episode comparison")

    # --------------------------------------------------------- sensitivity
    print("sensitivity")
    variants = [
        ("baseline (gate 1s, delay 1s, rollover out)", admit, MAX_QUOTE_AGE, 1),
        ("rollover seconds included",
         posterior >= MIN_POSTERIOR, MAX_QUOTE_AGE, 1),
        ("posterior at least 50% (hard boundary)",
         (posterior >= 0.50) & ~roll, MAX_QUOTE_AGE, 1),
        ("posterior at least 99%",
         (posterior >= 0.99) & ~roll, MAX_QUOTE_AGE, 1),
        ("no freshness gate", admit, None, 1),
        ("execution delay 5s", admit, MAX_QUOTE_AGE, 5),
        ("execution delay 0s (upper bound)", admit, MAX_QUOTE_AGE, 0),
    ]
    sens_rows = []
    for label, mask, gate, delta in variants:
        counts = executable_counts(d, labels, names, mask, gate, delta)
        row = {"variant": label}
        for k, name in enumerate(names):
            row[f"{name}, band (pips)"] = float(
                d.loc[mask & (labels == k), "band"].median())
            row[f"{name}, eligible"] = counts.loc["eligible seconds", name]
            row[f"{name}, executable"] = counts.loc["executable seconds", name]
        sens_rows.append(row)
    sens = pd.DataFrame(sens_rows)
    save_table(
        sens, "05_sensitivity", index=False,
        caption=("Every choice this script makes, and what it costs. The "
                 "freshness gate is the one that matters: without it the "
                 "forward fill supplies quotes that had not printed for "
                 "minutes and the scan counts opportunities in a market that "
                 "was not quoting."),
        label="tab:execution_sensitivity")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------------------ figures
    print("figures")
    eligible = d.loc[admit & fresh]
    eday = eligible.index.floor("D")
    daily = pd.DataFrame({
        "band": eligible["band"].groupby(eday).median(),
        "mad": eligible["basis"].groupby(eday).apply(robust_scale),
    }).dropna()
    daily_legs = eligible[["band_audjpy", "band_usdjpy",
                           "band_audusd"]].groupby(eday).median().dropna()

    figure_bands(d.loc[admit], daily)
    figure_executable(d.loc[admit & fresh], labels[admit & fresh], names,
                      colours, breakeven)
    figure_decomposition(names, decomposition, daily_legs)

    # --------------------------------------------------------------- write
    print("data")
    out = d[["basis", "edge", "e_rich", "e_cheap", "c_rich", "c_cheap",
             "band", "band_audusd", "band_usdjpy", "band_audjpy",
             "max_age"]].astype(np.float32)
    out.insert(0, "regime", labels)
    out["admitted"] = admit
    out.index.name = "t"
    out.to_parquet(DAT_DIR / "05_bounds.parquet")
    print(f" -> 05_bounds.parquet ({len(out):,} rows)")

    # ---------------------------------------------------------------- audit
    print("checks")
    ever = int((d.loc[admit & fresh, "edge"] > 0).sum())
    if ever == 0:
        print("  [INFO] no second in the eligible sample offered a complete "
              "circuit at quoted prices")
        first = haircut[haircut[[c for c in haircut.columns
                                 if c.endswith(', seconds')]].sum(axis=1) > 0]
        if len(first):
            k = first.iloc[0]["spread haircut"]
            print(f"  [INFO] the first haircut at which any second clears is "
                  f"{k:.0%} of the quoted band")
        else:
            print("  [INFO] no haircut in the ladder produces an executable "
                  "second; widen HAIRCUTS if a tighter bound is of interest")
    else:
        share = ever / int((admit & fresh).sum())
        print(f"  [INFO] {ever:,} executable seconds ({share:.3%} of the "
              f"eligible sample) before any latency allowance")
    worst_age = int(d.loc[admit & fresh, "max_age"].max())
    print(f"  [INFO] worst quote age inside the gate: {worst_age}s "
          f"(gate is {MAX_QUOTE_AGE}s)")


if __name__ == "__main__":
    main()