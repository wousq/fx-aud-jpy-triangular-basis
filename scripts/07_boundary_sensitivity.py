"""
07_boundary_sensitivity.py

The headline of this project is a decomposition: pooled dispersion over the
stress window is about three times its calm level, within-hour dispersion is
about a fifth higher, and the gap between them is day-to-day movement in the
basis level. That decomposition is computed over a window whose opening
boundary is contested. 02's onset posterior is bimodal, its mode at 24 July
and its median at 30 July, and 11 of 17 same-feature specifications select
the mode while the per-second labels encode the median.

Choosing between the two is a researcher degree of freedom. Forking the
pipeline and reporting both doubles every table and still evaluates the
decomposition at two arbitrary points. This script does neither. The
decomposition is a closed-form function of the boundary, so it is evaluated
across the whole range the posterior supports and reported as a curve. If
the day-to-day share is flat across that range the headline does not depend
on the choice, and saying so is a measurement rather than an assurance.

Nothing here is refitted. No VECM, no changepoint model, no episodes. The
quantities swept are the ones that follow from per-hour sufficient
statistics, which is exactly why the sweep is cheap:

    day-to-day share      Var_d(E[z|d]) / Var(z), by the law of total variance
    within-day sd         sqrt(E_d[Var(z|d)])
    pooled sd             sqrt(Var(z)), the statistic that mixes the two
    within-hour MAD       median over hours of the per-hour MAD

Slicing, not masking. 01_clean.parquet is sorted and a candidate window is
contiguous in time, so stress is one slice and calm is the two slices either
side of it. That is what keeps a few hundred evaluations over 3.8M rows in
the range of seconds.

Boundaries are read, never typed. The median pair comes from the label
transitions in 02_regimes.parquet; the modal pair is parsed out of
02_changepoint.csv. Neither date appears as a literal below, for the same
reason 03 refuses to type them.

Inputs
    output/data/01_clean.parquet      open-market seconds, basis
    output/data/02_regimes.parquet    per-second regime labels
    output/tables/02_changepoint.csv  the modal changepoint dates

Outputs
    output/tables/07_boundary_named.csv   the two rules side by side
    output/tables/07_boundary_sweep.csv   the full sweep, one row per boundary
    output/figures/07_boundary_sweep.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    make_dirs, set_style, in_rollover, robust_scale, save_table, save_fig,
    panel_title, annotate_event,
    DAT_DIR, TAB_DIR, BOJ_SHOCK, FOUND, MUTE, RULE, R1, R2, TZ_LABEL,
)

# convention. Sweep resolution. The changepoint model works on hourly
# buckets, so anything finer than an hour is false precision; 3h keeps the
# curve readable and the runtime in seconds.
SWEEP_STEP = pd.Timedelta("3h")

# convention. How far either side of the two candidate onsets to sweep, so
# the curve has context rather than stopping exactly where the argument is.
SWEEP_PAD = pd.Timedelta("24h")

# convention. A day thinner than this cannot support a mean, and the two
# boundary days are partial by construction. Both sides of the split are
# treated identically, so the floor cannot bias the comparison.
MIN_DAY_SECONDS = 3600


def _parse_stamp(text, year):
    """'24 Jul 17:00' -> Timestamp, using the sample's year."""
    return pd.to_datetime(f"{year} {text.strip()}", format="%Y %d %b %H:%M")


def read_boundaries(index):
    """
    The two summarisation rules 02 produced, read rather than typed.

    Median pair: the first and last second carrying the widest-dispersion
    label in 02_regimes.parquet. Which label that is comes from the data,
    not from an index, because the widest segment is not always the middle
    one.

    Modal pair: the posterior modes recorded in 02_changepoint.csv.
    """
    reg = pd.read_parquet(DAT_DIR / "02_regimes.parquet")
    basis = pd.read_parquet(DAT_DIR / "01_clean.parquet", columns=["basis"])
    lab = reg["regime"].reindex(basis.index).to_numpy()
    spread = {k: robust_scale(basis["basis"].to_numpy()[lab == k])
              for k in np.unique(lab[~pd.isna(lab)])}
    stress_k = max(spread, key=spread.get)
    hit = np.flatnonzero(lab == stress_k)
    med = (reg.index[hit[0]], reg.index[hit[-1]] + pd.Timedelta(seconds=1))

    cp = pd.read_csv(TAB_DIR / "02_changepoint.csv", index_col=0)["value"]
    year = index[0].year
    mod = (_parse_stamp(cp.loc["onset, posterior mode"], year),
           _parse_stamp(cp.loc["return, posterior mode"], year))
    return med, mod, stress_k


def decompose(z, day_codes, n_days, slices, mad_h=None, hour_ok=None):
    """
    Law of total variance over trading days, for a union of slices.

        Var(z) = E_d[Var(z|d)]  +  Var_d(E[z|d])

    Per-day counts and power sums are accumulated with bincount over the
    slices, so the cost is linear in the seconds admitted and independent of
    how many candidate boundaries are evaluated.
    """
    nd = np.zeros(n_days)
    s1 = np.zeros(n_days)
    s2 = np.zeros(n_days)
    for a, b in slices:
        if b <= a:
            continue
        dc = day_codes[a:b]
        zz = z[a:b]
        nd += np.bincount(dc, minlength=n_days)
        s1 += np.bincount(dc, weights=zz, minlength=n_days)
        s2 += np.bincount(dc, weights=zz * zz, minlength=n_days)

    keep = nd >= MIN_DAY_SECONDS
    if keep.sum() < 2:
        return None
    nd, s1, s2 = nd[keep], s1[keep], s2[keep]

    mean_d = s1 / nd
    # Clipped at zero: the identity is exact, the subtraction is not, and a
    # day with almost no dispersion can land a few ulp below.
    var_d = np.maximum(s2 / nd - mean_d ** 2, 0.0)

    w = nd / nd.sum()
    grand = float((w * mean_d).sum())
    within = float((w * var_d).sum())
    between = float((w * (mean_d - grand) ** 2).sum())
    total = within + between

    out = {
        "seconds": float(nd.sum()),
        "days": int(len(nd)),
        "pooled sd": np.sqrt(total),
        "within-day sd": np.sqrt(within),
        "day-to-day sd": np.sqrt(between),
        "day-to-day share": between / total if total > 0 else np.nan,
    }
    if mad_h is not None:
        m = mad_h[hour_ok]
        out["within-hour MAD"] = float(np.median(m[np.isfinite(m)])) if m.size else np.nan
    return out


def main():
    make_dirs()
    set_style()

    print("load")
    d = pd.read_parquet(DAT_DIR / "01_clean.parquet").sort_index()
    roll = in_rollover(d.index).to_numpy()
    d = d.loc[~roll]
    z = d["basis"].to_numpy(dtype=float)
    idx = d.index
    print(f" {len(d):,} open-market seconds outside rollover")

    day_codes, day_uniq = pd.factorize(idx.floor("D"), sort=True)
    n_days = len(day_uniq)

    # Per-hour MAD, computed once. It does not depend on where the boundary
    # sits, only on which side of it each hour falls.
    hour_key = idx.floor("h")
    hour_codes, hour_uniq = pd.factorize(hour_key, sort=True)
    mad_h = (pd.Series(z).groupby(hour_codes).apply(robust_scale)
             .reindex(range(len(hour_uniq))).to_numpy(dtype=float))
    print(f" {len(hour_uniq):,} hourly buckets summarised")

    (med_on, med_ret), (mod_on, mod_ret), stress_k = read_boundaries(idx)
    print(f" median rule  {med_on:%d %b %H:%M} -> {med_ret:%d %b %H:%M}")
    print(f" modal rule   {mod_on:%d %b %H:%M} -> {mod_ret:%d %b %H:%M}")

    def evaluate(onset, ret):
        """Stress against calm for one candidate window."""
        a = int(np.searchsorted(idx, onset))
        b = int(np.searchsorted(idx, ret))
        if b - a < 3600 or a < 3600 or len(idx) - b < 3600:
            return None
        ha = int(np.searchsorted(hour_uniq, onset))
        hb = int(np.searchsorted(hour_uniq, ret))
        in_win = np.zeros(len(hour_uniq), dtype=bool)
        in_win[ha:hb] = True

        s = decompose(z, day_codes, n_days, [(a, b)], mad_h, in_win)
        c = decompose(z, day_codes, n_days, [(0, a), (b, len(idx))],
                      mad_h, ~in_win)
        if s is None or c is None:
            return None
        row = {"onset": onset, "return": ret,
               "stress hours": (b - a) / 3600.0}
        for k in ("day-to-day share", "within-day sd", "pooled sd",
                  "within-hour MAD"):
            row[f"stress {k}"] = s[k]
        row["calm day-to-day share"] = c["day-to-day share"]
        row["pooled sd ratio"] = s["pooled sd"] / c["pooled sd"]
        row["within-day sd ratio"] = s["within-day sd"] / c["within-day sd"]
        row["within-hour MAD ratio"] = s["within-hour MAD"] / c["within-hour MAD"]
        return row

    # ------------------------------------------------ the two named rules
    print("named boundaries")
    named = {}
    for name, (on, ret) in [("median rule (the labels)", (med_on, med_ret)),
                            ("modal rule (02's posterior mode)", (mod_on, mod_ret))]:
        r = evaluate(on, ret)
        if r is None:
            raise ValueError(f"{name}: window too short to evaluate")
        # Pooled MAD is not available from power sums, so it is computed
        # directly here and not in the sweep.
        a = int(np.searchsorted(idx, on))
        b = int(np.searchsorted(idx, ret))
        mad_s = robust_scale(z[a:b])
        mad_c = robust_scale(np.concatenate([z[:a], z[b:]]))
        named[name] = {
            "onset": f"{on:%d %b %H:%M}",
            "return": f"{ret:%d %b %H:%M}",
            "stress hours": round(r["stress hours"], 1),
            "day-to-day share of stress level variance": r["stress day-to-day share"],
            "day-to-day share, calm": r["calm day-to-day share"],
            "pooled sd ratio": r["pooled sd ratio"],
            "pooled MAD ratio": mad_s / mad_c,
            "within-day sd ratio": r["within-day sd ratio"],
            "within-hour MAD ratio": r["within-hour MAD ratio"],
        }
    named = pd.DataFrame(named)
    save_table(
        named, "07_boundary_named",
        caption=("The headline decomposition under both summarisation rules "
                 "02 supports. The median rule is what the per-second labels "
                 "encode and what every other section conditions on; the "
                 "modal rule is the posterior mode and is selected by 11 of "
                 "17 same-feature specifications. Nothing is refitted."),
        label="tab:boundary_named")
    print(named.to_string(float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------ the sweep
    print("sweep")
    lo = min(med_on, mod_on) - SWEEP_PAD
    hi = max(med_on, mod_on) + SWEEP_PAD
    grid = pd.date_range(lo.ceil("h"), hi.floor("h"), freq=SWEEP_STEP)

    rows = []
    for ret, tag in [(med_ret, "median"), (mod_ret, "modal")]:
        for on in grid:
            r = evaluate(on, ret)
            if r is not None:
                r["return rule"] = tag
                rows.append(r)
    sweep = pd.DataFrame(rows)
    print(f" {len(sweep)} boundary pairs evaluated")

    out = sweep.copy()
    out["onset"] = out["onset"].dt.strftime("%Y-%m-%d %H:%M")
    out["return"] = out["return"].dt.strftime("%Y-%m-%d %H:%M")
    save_table(
        out, "07_boundary_sweep", index=False,
        caption=("The decomposition as a function of the stress window, "
                 "swept over the range 02's onset posterior supports, at "
                 "both candidate return dates. A flat day-to-day share "
                 "means the headline does not depend on which onset is "
                 "chosen."),
        label="tab:boundary_sweep")

    span = sweep["stress day-to-day share"]
    print(f" day-to-day share ranges {span.min():.1%} to {span.max():.1%} "
          f"across the swept boundaries")
    for k in ("pooled sd ratio", "within-day sd ratio", "within-hour MAD ratio"):
        print(f" {k:<24} {sweep[k].min():.2f} to {sweep[k].max():.2f}")

    # ------------------------------------------------ figure
    print("figure")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                           gridspec_kw={"hspace": 0.34})

    for tag, colour in [("median", R2), ("modal", R1)]:
        g = sweep[sweep["return rule"] == tag]
        ax[0].plot(g["onset"], 100 * g["stress day-to-day share"], lw=1.6,
                   color=colour, label=f"return at the {tag} changepoint")
    ax[0].set_ylabel("% of stress level variance")
    ax[0].set_ylim(0, 100)
    panel_title(ax[0], "Day-to-day share of the stress window's level variance",
                "flat across the swept range means the headline does not turn on the boundary")
    ax[0].legend(loc="lower left", ncol=2)

    g = sweep[sweep["return rule"] == "median"]
    for k, colour, lab in [
            ("pooled sd ratio", RULE, "pooled sd (mixes both components)"),
            ("within-day sd ratio", FOUND, "within-day sd"),
            ("within-hour MAD ratio", MUTE, "within-hour MAD")]:
        ax[1].plot(g["onset"], g[k], lw=1.6, color=colour, label=lab)
    ax[1].axhline(1.0, color=RULE, lw=0.6)
    ax[1].set_ylabel("stress / calm")
    ax[1].set_xlabel(f"onset boundary ({TZ_LABEL})")
    panel_title(ax[1], "Dispersion ratios against the same boundary",
                "the gap between the top line and the other two is the pooling effect")
    ax[1].legend(loc="upper right")

    for a in ax:
        for when, lab in [(mod_on, "02 mode"), (med_on, "02 median · the labels")]:
            a.axvline(when, color=FOUND, lw=0.7, zorder=0)
        annotate_event(a, BOJ_SHOCK, "BOJ", top=False)
        a.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[0].annotate("02 mode", xy=(mod_on, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(3, -2), textcoords="offset points", color=FOUND,
                   fontsize=7.5, ha="left", va="top", annotation_clip=False)
    ax[0].annotate("02 median (the labels)", xy=(med_on, 1.0),
                   xycoords=("data", "axes fraction"), xytext=(3, -2),
                   textcoords="offset points", color=FOUND, fontsize=7.5,
                   ha="left", va="top", annotation_clip=False)

    save_fig(fig, "07_boundary_sweep")


if __name__ == "__main__":
    main()