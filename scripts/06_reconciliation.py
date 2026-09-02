"""
06_reconciliation.py

Four questions the earlier sections raised and none of them answered.

    1. Filter boundaries. The rollover window 17:00-17:30 is excluded from
       episode detection, so the detector sees a session that opens at
       17:30:00 and closes at 16:59:59. An excursion already in progress
       when the window opens is recorded as starting at 17:30:00, and one
       still in progress when it closes is recorded as ending at 16:59:59.
       Neither is a start or an end. This counts how many of the widest
       episodes terminate on a boundary and compares that with the rate
       expected if starts were spread evenly over admissible seconds.

    2. Normalisation. Basis dispersion rises 2.8x on MAD and 4.0x on sd
       under stress, but the legs themselves got noisier too, so part of
       that is arithmetic rather than a change in how the triangle
       behaves. Section 03 reports per-leg residual scales and their
       correlations, which is enough to reconstruct the scale of the
       basis innovation directly and split the level change into a noise
       part and a persistence part.

    3. Closure speed. Five routes to the same quantity are scattered
       across 02, 03 and 04 and they do not agree. Collected into one
       table with what each one actually estimates, because a reader who
       computes the disagreement themselves and finds it unacknowledged
       will not believe the rest.

    4. Daily extremes. 31 July fits at 1,003 s, 27x any other day, and
       5 August fits at 16 s on the day of the deleveraging climax. Both
       need a sanity check against session length and episode count
       before either is quoted.

What this script is not
    It re-reads what the pipeline wrote and reconciles it. It estimates
    nothing, and it reaches no raw data. That is deliberate: every number
    it consumes is a number the report cites, so a disagreement found
    here is a disagreement a reader can find.

Inputs (tables, not data)
    01_episodes_top.csv       the twenty widest episodes
    01_closures.csv           market closures detected in 01
    02_regime_stats.csv       per-regime dispersion, AR(1), episode counts
    03_var_fit.csv            per-leg residual sd and residual correlations
    03_error_correction.csv   one-step and full-system half-lives
    03_closure_speed.csv      daily lambda and half-life
    03_persistence.csv        episode ruler and day-by-day summaries

Outputs
    06_boundary_audit.csv         one row per episode, with censoring verdict
    06_boundary_summary.csv       observed against expected boundary rate
    06_normalisation.csv          noise / persistence split of the level change
    06_closure_reconciliation.csv every closure estimate the project produced
    06_daily_extremes.csv         slowest and fastest days, with context

Every table is printed to the run log, so the log alone is a complete
record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils import (BOJ_SHOCK, ROLLOVER_START, ROLLOVER_END,
                       TAB_DIR, TEX_DIR)
except ImportError:  # running against a table directory outside the repo
    from utils import BOJ_SHOCK, ROLLOVER_START, ROLLOVER_END
    TAB_DIR = TEX_DIR = None


# ------------------------------------------------------------------ config

# convention. Where the tables live. Defaults to the repo's own table
# directory; overridable on the command line so the script can be pointed
# at an archived run without editing it.
IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/tables")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else IN_DIR

# derived. The first and last second the episode detector is allowed to
# see, taken from the rollover window in utils rather than typed, so the
# two cannot drift apart.
OPEN_SECOND = pd.Timestamp(ROLLOVER_END).time()
CLOSE_SECOND = (pd.Timestamp(ROLLOVER_START) - pd.Timedelta(seconds=1)).time()

# derived. Agreement tolerance for the normalisation identity. The AR(1)
# amplification is a one-parameter approximation to a 21-lag system, so
# exact agreement is not expected and would be suspicious; the band is
# wide enough to pass an approximation and narrow enough to fail an error.
NORM_TOL = (0.70, 1.40)

# convention. utils.save_table defaults to "%.4g", which is right for tables
# of estimates and wrong here: these carry six-figure second counts beside
# three-figure ratios, and %.4g renders the former in scientific notation and
# rounds 1002.67 to 1003. "%g" drops trailing zeros without doing either.
FLOAT_FMT = "%g"


# --------------------------------------------------------------- reporting

def emit(frame, name, caption, index=True):
    """
    Write one table twice, CSV to read and LaTeX to input, and print it.

    Mirrors utils.save_table. Kept local because this script may be run
    against a directory that is not the repo's output tree, in which case
    the module-level TAB_DIR is not the right destination.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / f"{name}.csv", index=index, float_format=FLOAT_FMT)

    # The report \inputs from report/tables, not from output/tables, so the
    # two halves of a table go to different places when this runs inside the
    # repo. Outside it they go together, which is what makes the script
    # runnable against an archived table directory.
    tex_dir = OUT_DIR
    if TEX_DIR is not None:
        # Resolved, not compared as written: the default OUT_DIR is the
        # relative "output/tables" while TAB_DIR from utils is absolute, so
        # comparing them literally never matches and the LaTeX silently
        # lands beside the CSVs instead of in the report tree.
        if OUT_DIR.resolve() == Path(TAB_DIR).resolve():
            tex_dir = Path(TEX_DIR)
    Path(tex_dir).mkdir(parents=True, exist_ok=True)
    frame.to_latex(Path(tex_dir) / f"{name}.tex", caption=caption, escape=True,
                   index=index, label=f"tab:{name[3:]}", float_format=FLOAT_FMT)
    print(f"\n{caption}")
    print(frame.to_string(index=index))
    print(f"    -> {name}.csv / .tex")


def read(name, index_col=0):
    path = IN_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Pass the table directory as the first "
            f"argument, e.g. python 06_reconciliation.py output/tables")
    return pd.read_csv(path, index_col=index_col)


def val(frame, row, col):
    """One cell as a float, tolerating the thousands separators in 03."""
    x = frame.loc[row, col]
    if isinstance(x, str):
        x = x.replace(",", "")
    return float(x)


def num(x, spec=".3f"):
    """
    Format a number for a table, or an em dash if it is not defined.

    The same helper 02 and 03 carry, and for the same reason: a NaN written
    into a LaTeX table renders as a blank cell or the literal 'nan'. An em
    dash says 'not applicable' and is the only thing that should reach the
    report in place of a number.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else format(v, spec)


# ------------------------------------------------------------- 1. boundary

def boundary_audit():
    """
    Which of the widest episodes terminate on a filter boundary.

    An episode that starts in the same second the filter opens was not
    observed starting: it was observed already open. Its start time is a
    property of the filter and its duration is a lower bound. The same
    argument runs backwards at the closing edge.
    """
    ep = read("01_episodes_top.csv", index_col=None)
    ep["start"] = pd.to_datetime(ep["start"])
    ep["end"] = pd.to_datetime(ep["end"])

    clo = read("01_closures.csv", index_col=None)
    clo["end"] = pd.to_datetime(clo["end"])
    reopen_dates = set(clo["end"].dt.date)

    ep["starts_at_open"] = ep["start"].dt.time == OPEN_SECOND
    ep["ends_at_close"] = ep["end"].dt.time == CLOSE_SECOND
    ep["after_closure"] = ep["start"].dt.date.isin(reopen_dates)

    def verdict(r):
        if r["starts_at_open"] and r["after_closure"]:
            return "left-censored, weekend reopen"
        if r["starts_at_open"]:
            return "left-censored"
        if r["ends_at_close"]:
            return "right-censored"
        return "interior"

    ep["verdict"] = ep.apply(verdict, axis=1)
    ep["|peak|"] = ep["peak"].abs()
    ep = ep.sort_values("|peak|", ascending=False).reset_index(drop=True)
    ep.index = np.arange(1, len(ep) + 1)

    out = ep[["start", "end", "seconds", "peak", "verdict"]].copy()
    out["start"] = out["start"].dt.strftime("%d %b %H:%M:%S")
    out["end"] = out["end"].dt.strftime("%d %b %H:%M:%S")
    emit(out, "06_boundary_audit",
         "The twenty widest episodes, ranked by absolute peak, with each "
         "one's relation to the rollover filter. A censored episode did "
         "not begin or end when the table says it did; the filter did.")
    return ep


def boundary_summary(ep):
    """
    Observed boundary terminations against the rate chance would give.

    Sessions are classified from the fitted second counts in 03 rather
    than from a calendar. A full day carries both boundary seconds, a
    Friday carries only the closing one because the week ends at 17:00,
    and a Sunday carries only the opening one because the week starts at
    17:30. The three counts differ by exactly the arithmetic that implies,
    which is the check that the classification is right.
    """
    day = read("03_closure_speed.csv", index_col=None)
    n = day["seconds"]
    full, short = n.max(), n.min()
    friday = n[(n != full) & (n != short)].iloc[0]

    has_open = int((n != friday).sum())
    has_close = int((n != short).sum())
    total = int(n.sum())
    k = len(ep)

    n_full = int((n == full).sum())
    n_fri = int((n == friday).sum())
    n_sun = int((n == short).sum())

    rows = [
        ("sessions fitted", f"{len(day)}"),
        ("of which full / short-close / late-open",
         f"{n_full} / {n_fri} / {n_sun}"),
        ("admissible seconds", f"{total:,}"),
        ("sessions with an opening second (17:30:00)", f"{has_open}"),
        ("sessions with a closing second (16:59:59)", f"{has_close}"),
        ("episodes examined", f"{k}"),
        ("expected starts at an opening second",
         f"{k * has_open / total:.4f}"),
        ("observed starts at an opening second",
         f"{int(ep['starts_at_open'].sum())}"),
        ("expected ends at a closing second",
         f"{k * has_close / total:.4f}"),
        ("observed ends at a closing second",
         f"{int(ep['ends_at_close'].sum())}"),
        ("episodes touching either boundary",
         f"{int((ep['starts_at_open'] | ep['ends_at_close']).sum())}"),
    ]
    out = pd.DataFrame(rows, columns=["", "value"]).set_index("")
    emit(out, "06_boundary_summary",
         "Boundary terminations against the rate expected if episode "
         "starts were spread evenly over admissible seconds. The expected "
         "counts are of order one ten-thousandth of an episode.")
    return out, (n_full, n_fri, n_sun, total)


def hour_concentration(ep, session_counts):
    """
    Where in the day the widest episodes start, against where the
    admissible seconds are.

    The single-second test above asks whether the filter's edge is
    over-represented. This asks the wider question the edge test raises:
    the rollover window was excluded because liquidity collapses and the
    synthetic leg goes stale, and there is no reason that condition should
    stop at 17:30:00 merely because the filter does. If the half hour
    after the window carries far more than its share of the widest
    episodes, the filter is ending too early rather than the market
    changing at 17:30.

    Admissible seconds per hour are reconstructed from the session
    classification: an hour is fully admissible on any session that spans
    it, except hour 17, which contributes half an hour on the sessions
    that reach it at all. This ignores the ~22 seconds per session that
    the lag-window guard removes, which is 0.05% of the total and cannot
    move any comparison below.
    """
    n_full, n_fri, n_sun, total = session_counts

    # Hours 00-16 exist on full and short-close sessions; hours 18-23 on
    # full and late-open ones; hour 17 on the latter pair but half-length.
    secs = {}
    for h in range(0, 17):
        secs[h] = (n_full + n_fri) * 3600
    secs[17] = (n_full + n_sun) * 1800
    for h in range(18, 24):
        secs[h] = (n_full + n_sun) * 3600

    tab = pd.DataFrame({"admissible seconds": pd.Series(secs)})
    tab["share of seconds"] = tab["admissible seconds"] / tab["admissible seconds"].sum()
    tab["episodes"] = ep["start"].dt.hour.value_counts().reindex(
        tab.index).fillna(0).astype(int)
    tab["expected"] = tab["share of seconds"] * len(ep)
    tab["observed / expected"] = tab["episodes"] / tab["expected"]
    tab.index.name = "hour"

    hot = tab[tab["episodes"] > 0].copy()
    emit(hot.round(3), "06_hour_concentration",
         "Start hour of the twenty widest episodes against the share of "
         "admissible seconds each hour holds. Hour 17 is half-length "
         "because its first thirty minutes are the excluded rollover "
         "window, so every episode counted there starts after 17:30:00.")

    h17 = tab.loc[17]
    print("\nchecks")
    print(f"    hour 17 holds {h17['share of seconds']:.2%} of admissible "
          f"seconds and {h17['episodes']}/{len(ep)} of the widest episodes, "
          f"{h17['observed / expected']:.0f}x its share")
    return tab


def freshness_crosscheck():
    """
    The widest basis reading in each regime, with and without the
    requirement that all three legs printed within one second.

    02 measures the basis on every labelled open-market second, forward
    filled. 05 measures it only where all three legs are fresh. A peak
    that appears in the first and not the second was recorded on a grid
    row where at least one leg had not reprinted, which is the exact
    condition under which a forward-filled triangle manufactures a gap.
    This does not prove any particular peak is spurious; it says which
    ones cannot be verified on simultaneous quotes.
    """
    reg = read("02_regime_stats.csv")
    ex = read("05_execution.csv")
    seg = {"pre": "segment 1", "stress": "segment 2", "post": "segment 3"}

    rows = {}
    for name, s in seg.items():
        ungated = val(reg, "max |basis| (pips)", s)
        gated = val(ex, "widest |basis| (pips)", name)
        rows[name] = {
            "widest |basis|, forward-filled grid (pips)": ungated,
            "widest |basis|, all three legs fresh (pips)": gated,
            "survives the freshness gate": "yes" if abs(gated - ungated) < 1e-9
                                           else "no",
            "MAD, forward-filled grid (pips)": val(reg, "basis MAD (pips)", s),
            "MAD, all three legs fresh (pips)":
                val(ex, "basis dispersion, MAD (pips)", name),
        }

    tab = pd.DataFrame(rows)[["pre", "stress", "post"]]
    emit(tab, "06_freshness_crosscheck",
         "The widest basis reading per regime with and without the "
         "one-second freshness requirement of 05. Dispersion is shown "
         "beneath it: the gate barely moves the middle of the "
         "distribution and removes two of the three extremes.")
    return tab


# -------------------------------------------------------- 2. normalisation

def normalisation():
    """
    Split the change in basis dispersion into a noise part and a
    persistence part.

    The basis is z = x_AJ - x_AU - x_UJ, so its innovation is the same
    combination of the three equation residuals and its scale follows
    from the three residual standard deviations and their correlations:

        var(e_z) = s_AJ^2 + s_AU^2 + s_UJ^2
                   - 2 s_AJ s_AU r(AU,AJ)
                   - 2 s_AJ s_UJ r(UJ,AJ)
                   + 2 s_AU s_UJ r(AU,UJ)

    Under an AR(1) in the level with coefficient rho, the level scale is
    the innovation scale amplified by 1/sqrt(1 - rho^2). 02 estimated rho
    per regime. Multiplying the two gives a predicted level dispersion
    that can be checked against the one 02 measured directly, and the
    check is the reason to trust the split rather than assert it.

    The amplification is a one-parameter stand-in for a 21-lag system, so
    it is an approximation. It is reported as one, and the agreement
    column is the evidence for how good an approximation it is.
    """
    fit = read("03_var_fit.csv")
    reg = read("02_regime_stats.csv")
    seg = {"pre": "segment 1", "stress": "segment 2", "post": "segment 3"}

    rows = {}
    for name, s in seg.items():
        s_au = val(fit, "residual sd, AUD/USD (pips)", name)
        s_uj = val(fit, "residual sd, USD/JPY (pips)", name)
        s_aj = val(fit, "residual sd, AUD/JPY (pips)", name)
        r_au_uj = val(fit, "residual corr, AUD/USD vs USD/JPY", name)
        r_au_aj = val(fit, "residual corr, AUD/USD vs AUD/JPY", name)
        r_uj_aj = val(fit, "residual corr, USD/JPY vs AUD/JPY", name)

        var = (s_aj ** 2 + s_au ** 2 + s_uj ** 2
               - 2 * s_aj * s_au * r_au_aj
               - 2 * s_aj * s_uj * r_uj_aj
               + 2 * s_au * s_uj * r_au_uj)
        if var <= 0:
            raise ValueError(
                f"{name}: implied basis innovation variance is {var:.3e}, "
                f"which is impossible. The residual correlation matrix in "
                f"03_var_fit.csv is not consistent with the basis identity.")
        s_z = np.sqrt(var)

        rho = val(reg, "AR(1) rho of level", s)
        amp = 1.0 / np.sqrt(1.0 - rho ** 2)
        obs = val(reg, "basis sd (pips)", s)

        rows[name] = {
            "residual sd, AUD/USD (pips)": s_au,
            "residual sd, USD/JPY (pips)": s_uj,
            "residual sd, AUD/JPY (pips)": s_aj,
            "implied basis innovation sd (pips)": s_z,
            "AR(1) rho of the level": rho,
            "amplification 1/sqrt(1-rho^2)": amp,
            "predicted level sd (pips)": s_z * amp,
            "measured level sd (pips)": obs,
            "predicted / measured": s_z * amp / obs,
            "measured level MAD (pips)": val(reg, "basis MAD (pips)", s),
        }

    tab = pd.DataFrame(rows)[["pre", "stress", "post"]]
    tab["stress / pre"] = tab["stress"] / tab["pre"]
    tab["stress / post"] = tab["stress"] / tab["post"]
    # A ratio of ratios is meaningless on the two agreement rows.
    for r in ("predicted / measured",):
        tab.loc[r, ["stress / pre", "stress / post"]] = np.nan

    shown = tab.map(lambda v: num(v, ".4f"))
    emit(shown, "06_normalisation",
         "Basis dispersion split into the scale of its innovation, which "
         "follows from the three legs, and the amplification that "
         "persistence applies to it. The predicted / measured row is the "
         "check: it is not fitted to anything.")

    # ------ the split, stated as the two ratios the report needs
    innov = tab.loc["implied basis innovation sd (pips)"]
    amp = tab.loc["amplification 1/sqrt(1-rho^2)"]
    meas_sd = tab.loc["measured level sd (pips)"]
    meas_mad = tab.loc["measured level MAD (pips)"]

    split = pd.DataFrame({
        "stress / pre": [
            innov["stress"] / innov["pre"],
            amp["stress"] / amp["pre"],
            (innov["stress"] / innov["pre"]) * (amp["stress"] / amp["pre"]),
            meas_sd["stress"] / meas_sd["pre"],
            meas_mad["stress"] / meas_mad["pre"],
        ],
        "stress / post": [
            innov["stress"] / innov["post"],
            amp["stress"] / amp["post"],
            (innov["stress"] / innov["post"]) * (amp["stress"] / amp["post"]),
            meas_sd["stress"] / meas_sd["post"],
            meas_mad["stress"] / meas_mad["post"],
        ]},
        index=["noise: innovation scale",
               "persistence: amplification",
               "product, predicted level sd",
               "measured level sd",
               "measured level MAD"])

    emit(split.round(3), "06_normalisation_split",
         "The same numbers as ratios. Persistence contributes more than "
         "noise, and the product reproduces the measured widening without "
         "being fitted to it.")

    print("\nchecks")
    for name in ("pre", "stress", "post"):
        r = tab.loc["predicted / measured", name]
        ok = NORM_TOL[0] <= r <= NORM_TOL[1]
        print(f"    {name:>6}: predicted / measured = {r:.3f} "
              f"{'ok' if ok else 'OUTSIDE TOLERANCE'}")
    return tab, split


# --------------------------------------------------------- 3. closure speed

def closure_reconciliation():
    """
    Every route to closure speed the project has produced, in one table.

    They disagree by a factor of seven, and the disagreement is the
    finding. The routes do not estimate the same thing: the episode ruler
    measures how long a large excursion takes to halve, the AR(1) and the
    VECM measure how fast an average deviation decays, and the day-by-day
    median measures the typical day rather than the typical second. A
    slowdown that shows in the first two and not the last says the damage
    was to large dislocations, not to routine adjustment.
    """
    per = read("03_persistence.csv")
    reg = read("02_regime_stats.csv")
    ec = read("03_error_correction.csv")

    rows = [
        ("episode ruler, median s to half peak",
         "how long a large excursion takes to halve",
         val(per, "episodes, median seconds to half peak", "pre"),
         val(per, "episodes, median seconds to half peak", "stress"),
         val(per, "episodes, median seconds to half peak", "post")),
        ("AR(1) on the level (02)",
         "decay of an average deviation, one parameter",
         val(reg, "half-life (s)", "segment 1"),
         val(reg, "half-life (s)", "segment 2"),
         val(reg, "half-life (s)", "segment 3")),
        ("VECM, one-step",
         "decay implied by lambda alone",
         val(ec, "half-life, one-step (s)", "pre"),
         val(ec, "half-life, one-step (s)", "stress"),
         val(ec, "half-life, one-step (s)", "post")),
        ("VECM, full system",
         "decay of the 21-lag system run down",
         val(ec, "half-life, full system (s)", "pre"),
         val(ec, "half-life, full system (s)", "stress"),
         val(ec, "half-life, full system (s)", "post")),
        ("VECM, day by day, median",
         "the typical day rather than the typical second",
         val(per, "day by day, median half-life (s)", "pre"),
         val(per, "day by day, median half-life (s)", "stress"),
         val(per, "day by day, median half-life (s)", "post")),
    ]

    tab = pd.DataFrame(rows, columns=["route", "estimand",
                                      "pre (s)", "stress (s)", "post (s)"])
    tab["calmest (s)"] = tab[["pre (s)", "post (s)"]].min(axis=1)
    tab["stress / calmest"] = tab["stress (s)"] / tab["calmest (s)"]
    tab = tab.set_index("route")

    emit(tab.round(2), "06_closure_reconciliation",
         "Five routes to closure speed. They agree on the sign and "
         "disagree on the size by a factor of seven, because they do not "
         "estimate the same quantity. The estimand column is the "
         "explanation, not an excuse.")
    return tab


# -------------------------------------------------------- 4. daily extremes

def daily_extremes(ep):
    """
    The slowest and fastest days, against session length and episode count.

    A day fitted on a third of the usual seconds is a weaker estimate than
    one fitted on all of them, and a day whose slow fit coincides with a
    large episode is explained by the episode. Neither is true of the two
    days the report leans on, which is why they are worth quoting.
    """
    day = read("03_closure_speed.csv", index_col=None)
    day["day"] = pd.to_datetime(day["day"])

    # Episodes are attributed to the day their peak window opens. The one
    # episode that spans midnight therefore belongs to the day it started
    # on, which is the day whose fit could have been affected by it.
    counts = ep.groupby(ep["start"].dt.normalize()).size()
    day["top20 episodes"] = day["day"].map(counts).fillna(0).astype(int)
    widest = ep.groupby(ep["start"].dt.normalize())["|peak|"].max()
    day["widest |peak| (pips)"] = day["day"].map(widest)

    full = day["seconds"].max()
    day["full session"] = day["seconds"] == full

    ranked = day.sort_values("half_life_s", ascending=False)
    show = pd.concat([ranked.head(5), ranked.tail(5)])
    show = show.assign(day=show["day"].dt.strftime("%a %d %b")).set_index("day")
    show = show[["regime", "seconds", "full session", "half_life_s",
                 "top20 episodes", "widest |peak| (pips)"]]
    show["half_life_s"] = show["half_life_s"].map(lambda v: num(v, ".2f"))
    show["widest |peak| (pips)"] = show["widest |peak| (pips)"].map(
        lambda v: num(v, ".3f"))

    emit(show, "06_daily_extremes",
         "The five slowest and five fastest days. Session length and "
         "episode count are the two things that would explain an extreme "
         "daily fit without the market having changed.")

    # ------ the two days the report names
    named = day[day["day"].isin(pd.to_datetime(["2024-07-31", "2024-08-05",
                                                "2024-08-04"]))]
    rank = day["half_life_s"].rank(ascending=False, method="min")
    named = named.assign(**{"rank, slowest first": rank[named.index]})
    named = named.assign(day=named["day"].dt.strftime("%a %d %b")).set_index("day")
    named = named[["regime", "seconds", "half_life_s", "rank, slowest first",
                   "top20 episodes", "widest |peak| (pips)"]]
    named["half_life_s"] = named["half_life_s"].map(lambda v: num(v, ".2f"))
    named["rank, slowest first"] = named["rank, slowest first"].map(
        lambda v: num(v, ".0f"))
    named["widest |peak| (pips)"] = named["widest |peak| (pips)"].map(
        lambda v: num(v, ".3f"))

    emit(named, "06_named_days",
         "The three dates the write-up names, with the context needed to "
         "read their daily fits.")

    print("\nchecks")
    print(f"    slowest day       {ranked.iloc[0]['day']:%d %b} at "
          f"{ranked.iloc[0]['half_life_s']:.1f} s, next slowest "
          f"{ranked.iloc[1]['half_life_s']:.1f} s, "
          f"ratio {ranked.iloc[0]['half_life_s'] / ranked.iloc[1]['half_life_s']:.1f}x")
    print(f"    BOJ reference     {BOJ_SHOCK:%d %b %H:%M} (estimated instant, "
          f"not a published one)")
    return day


# ------------------------------------------------------------------- main

def main():
    print(f"reading tables from {IN_DIR.resolve()}")
    print(f"writing tables to   {OUT_DIR.resolve()}")

    print("\n1. filter boundaries")
    ep = boundary_audit()
    _, sessions = boundary_summary(ep)
    hour_concentration(ep, sessions)
    freshness_crosscheck()

    print("\n2. normalisation")
    normalisation()

    print("\n3. closure speed")
    closure_reconciliation()

    print("\n4. daily extremes")
    daily_extremes(ep)


if __name__ == "__main__":
    main()