"""
01_exploratory_analysis.py

Descriptive characterisation of the AUD/JPY triangular basis.

This version is aligned with the shared API and palette in utils.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import (
    make_dirs, set_style, load_data, add_basis, add_log_prices,
    unchanged, closed_mask, closure_report, episodes, in_rollover,
    summary_stats, robust_scale, acf1, roll_noise, noise_share, survival,
    envelope, save_table, save_fig, zero_line, panel_title, annotate_event,
    PAIRS, PAIR_LABEL, LEG_COLOUR, FOUND, MUTE, RULE, BOJ_SHOCK, DAT_DIR,
    TZ_LABEL, ROLLOVER_START, ROLLOVER_END,
)

THRESH_Q = 0.9995
MERGE_GAP = "60s"
MIN_SECONDS = 5
ZOOM_H = 12


def main():
    make_dirs()
    set_style()

    print("load")
    df = add_log_prices(add_basis(load_data()))
    span = (df.index[-1] - df.index[0]).total_seconds() + 1
    print(f" {len(df):,} rows over {span:,.0f} seconds")
    if abs(len(df) - span) < 1:
        print(" grid is gapless: every calendar second present, weekends included as forward-filled padding")

    print("closures")
    closed = closed_mask(df, min_run=600)
    rep = closure_report(closed)
    print(
        f" {len(rep)} closures, {closed.mean():.1%} of rows, "
        f"longest {rep['hours'].max():.1f} h, "
        f"shortest {rep['hours'].min():.2f} h"
    )
    save_table(
        rep.assign(start=rep["start"].astype(str), end=rep["end"].astype(str)),
        "01_closures",
        index=False,
        caption=("Detected market closures: maximal runs of at least ten minutes "
                 "in which no pair printed a new price. Identified from the data, "
                 "not from a trading calendar."),
        label="tab:closures",
    )

    d = df.loc[~closed].copy()
    jul = d.loc["2024-07"]
    aug = d.loc["2024-08"]
    print(f" keeping {len(d):,} open-market rows")

    print("moments")
    stats = pd.concat([
        summary_stats(g, ["basis"]).rename(index={"basis": name})
        for name, g in [("Full sample", d), ("July", jul), ("August", aug)]
    ])
    save_table(
        stats,
        "01_summary_stats",
        caption=("AUD/JPY basis in pips, open-market seconds. MAD is the median "
                 "absolute deviation in sd units; the gap between sd and MAD measures "
                 "how far the moments are driven by the tail."),
        label="tab:summary_stats",
    )
    print(stats[["sd", "MAD", "p99.9_abs", "max_abs", "kurtosis"]].to_string(
        float_format=lambda value: f"{value:.3f}"
    ))

    print("microstructure")
    stale = unchanged(d)
    quality = pd.DataFrame(index=["July", "August"])
    quality["stale share, AUD/JPY"] = [
        stale.loc["2024-07", "audjpy_direct"].mean(),
        stale.loc["2024-08", "audjpy_direct"].mean(),
    ]
    quality["quote rate, AUD/JPY"] = 1 - quality["stale share, AUD/JPY"]
    quality["rho_1 of increments"] = [
        acf1(jul["basis"].diff()),
        acf1(aug["basis"].diff()),
    ]
    quality["Roll noise c (pips)"] = [
        roll_noise(jul["basis"].diff()),
        roll_noise(aug["basis"].diff()),
    ]
    quality["noise share of variance"] = [
        noise_share(jul["basis"].diff()),
        noise_share(aug["basis"].diff()),
    ]
    quality["basis MAD (pips)"] = [robust_scale(jul["basis"]), robust_scale(aug["basis"])]
    save_table(
        quality.T,
        "01_data_quality",
        caption=("Microstructure diagnostics. Roll's $c$ is implied by the first-order "
                 "autocovariance of basis increments; the noise share is $-2\\rho_1$."),
        label="tab:data_quality",
    )
    print(quality.T.to_string(float_format=lambda value: f"{value:.4f}"))

    print("episodes")
    roll = in_rollover(d.index)
    print(
        f" excluding {roll.mean():.2%} of rows as "
        f"{ROLLOVER_START}-{ROLLOVER_END} {TZ_LABEL} rollover"
    )
    threshold = jul.loc[~roll.loc[jul.index], "basis"].abs().quantile(THRESH_Q)
    ep = episodes(d["basis"], threshold, MERGE_GAP, MIN_SECONDS, exclude=roll)
    ep_j = ep[ep.month == "July"]
    ep_a = ep[ep.month == "August"]
    print(f" threshold {threshold:.2f} pips; {len(ep)} episodes ({len(ep_j)} July, {len(ep_a)} August)")

    esum = pd.DataFrame({
        "July": [
            len(ep_j), ep_j.seconds.median(), ep_j.seconds.max(),
            ep_j.peak.abs().median(), ep_j.peak.abs().max(), ep_j.seconds.sum() / 3600,
        ],
        "August": [
            len(ep_a), ep_a.seconds.median(), ep_a.seconds.max(),
            ep_a.peak.abs().median(), ep_a.peak.abs().max(), ep_a.seconds.sum() / 3600,
        ],
    }, index=[
        "episodes", "median duration (s)", "longest (s)",
        "median |peak| (pips)", "widest |peak| (pips)", "total time dislocated (h)",
    ])
    save_table(
        esum,
        "01_episode_stats",
        caption=(f"Dislocation episodes: contiguous stretches with |basis| above "
                 f"{threshold:.2f} pips, the July {THRESH_Q:.2%} quantile. Runs "
                 f"separated by under {MERGE_GAP} are merged; runs under "
                 f"{MIN_SECONDS}s are discarded."),
        label="tab:episode_stats",
    )
    print(esum.to_string(float_format=lambda value: f"{value:.2f}"))

    top = (
        ep.reindex(ep.peak.abs().sort_values(ascending=False).index)
        .head(20)
        .assign(start=lambda frame: frame.start.astype(str), end=lambda frame: frame.end.astype(str))
        [["start", "end", "seconds", "peak"]]
    )
    save_table(top, "01_episodes_top", index=False, caption="The twenty widest dislocation episodes.", label="tab:episodes_top")

    peak_ep = ep.loc[ep.peak.abs().idxmax()]
    peak_t = peak_ep.start + (peak_ep.end - peak_ep.start) / 2
    print(
        f" widest episode {peak_ep.start:%d %b %H:%M} -> {peak_ep.end:%H:%M}, "
        f"{peak_ep.seconds:.0f}s, peak {peak_ep.peak:+.1f} pips"
    )

    print("figures")
    lin = max(1.0, float(np.ceil(3 * robust_scale(d["basis"]))))
    lo, hi, _ = envelope(d["basis"], "1min")

    def window(centre, half_hours):
        start = centre - pd.Timedelta(hours=half_hours)
        end = centre + pd.Timedelta(hours=half_hours)
        return start, end, d.loc[start:end, "basis"]

    b0, b1, b_boj = window(BOJ_SHOCK, ZOOM_H)
    p0, p1, b_pk = window(peak_t, ZOOM_H)

    fig = plt.figure(figsize=(10, 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.42, wspace=0.14)
    ax_top = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_left)

    ax_top.axvspan(b0, b1, color=MUTE, alpha=0.18, lw=0, zorder=0)
    ax_top.axvspan(p0, p1, color=FOUND, alpha=0.18, lw=0, zorder=0)
    ax_top.vlines(lo.index, lo.values, hi.values, color=RULE, lw=0.45)
    zero_line(ax_top)
    ax_top.set_yscale("symlog", linthresh=lin, linscale=0.7)
    ax_top.set_ylabel("basis (pips)") 
    panel_title(ax_top, "AUD/JPY basis, direct minus synthetic · per-minute range", f"symlog scale, linear within ±{lin:.0f} pips")
    ax_top.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    annotate_event(ax_top, BOJ_SHOCK, "BOJ")

    for ax, (start, end, series), title, shade in [
        (ax_left, (b0, b1, b_boj), f"BOJ decision · {BOJ_SHOCK:%d %b} ±{ZOOM_H}h", MUTE),
        (ax_right, (p0, p1, b_pk), f"Widest episode · {peak_t:%d %b} ±{ZOOM_H}h", FOUND),
    ]:
        zlo, zhi, _ = envelope(series, "10s")
        ax.vlines(zlo.index, zlo.values, zhi.values, color=RULE, lw=0.45)
        zero_line(ax)
        ax.axhline(threshold, color=FOUND, lw=0.7, ls=":")
        ax.axhline(-threshold, color=FOUND, lw=0.7, ls=":")
        ax.set_yscale("symlog", linthresh=lin, linscale=0.7)
        ax.set_title(title)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel(TZ_LABEL)
        ax.spines["bottom"].set_color(shade)
        ax.spines["bottom"].set_linewidth(2.2)

    ax_left.set_ylabel("basis (pips)")
    ax_right.tick_params(labelleft=False)
    ax_right.annotate(
        f"episode threshold {threshold:.1f} pips",
        xy=(0.98, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7, color=FOUND,
    )
    save_fig(fig, "01_basis_episodes")

    exceed = (d["basis"].abs() > threshold).resample("1h").sum()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [0.85, 1.15], "hspace": 0.42})

    pct = exceed / 36.0
    ax[0].fill_between(pct.index, 0, pct.values, color=FOUND, lw=0, step="mid")
    ax[0].set_ylabel("% of the hour")
    ax[0].set_ylim(0, min(100, max(5.0, pct.max() * 1.18)))
    ax[0].set_title(f"Share of each hour spent dislocated (|basis| > {threshold:.2f} pips)")
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    annotate_event(ax[0], BOJ_SHOCK, "BOJ")

    for group, name, colour in [(ep_j, "July", MUTE), (ep_a, "August", FOUND)]:
        for sign, marker in [(1, "^"), (-1, "v")]:
            selected = group[group["sign"] == sign]
            ax[1].scatter(
                selected.seconds, selected.peak.abs(), s=26, c=colour, marker=marker,
                alpha=0.8, lw=0.4, edgecolor="white",
                label=f"{name} (n={len(group)})" if sign == 1 else None,
            )

    ax[1].axhline(threshold, color=RULE, lw=0.7, ls=":")
    ax[1].annotate(
        f"threshold {threshold:.2f} pips", xy=(0.995, threshold),
        xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
        ha="right", va="bottom", fontsize=7, color=RULE,
    )
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ticks = [5, 15, 60, 300, 900, 3600, 10800]
    labels = ["5s", "15s", "1min", "5min", "15min", "1h", "3h"]
    limit = ep.seconds.max() * 1.6
    selected_ticks = [(tick, label) for tick, label in zip(ticks, labels) if tick <= limit]
    ax[1].set_xticks([tick for tick, _ in selected_ticks])
    ax[1].set_xticklabels([label for _, label in selected_ticks])
    ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[1].set_xlabel("episode duration (log scale)")
    ax[1].set_ylabel("peak |basis| (pips, log scale)")
    panel_title(ax[1], "Every episode: duration against peak dislocation", "up markers: direct rich; down markers: synthetic rich")
    ax[1].legend(loc="upper left")
    save_fig(fig, "01_episode_structure")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for group, name, colour in [(jul, "July", MUTE), (aug, "August", FOUND)]:
        values, probabilities = survival(group["basis"])
        ax.loglog(values, probabilities, lw=1.8, color=colour, label=name, solid_capstyle="round")
    ax.axvline(threshold, color=RULE, lw=0.7, ls=":")
    ax.annotate(
        f"episode threshold\n{threshold:.2f} pips", xy=(threshold, 3e-5),
        xytext=(6, 0), textcoords="offset points", fontsize=7.5,
        color=RULE, va="bottom",
    )
    ax.set_xlabel("threshold $u$ (pips)")
    ax.set_ylabel(r"$P\,(\,|\mathrm{basis}| > u\,)$")
    ax.set_title("Tail of the basis distribution")
    ax.legend(loc="lower left")
    ax.grid(which="both", alpha=0.12)
    save_fig(fig, "01_tail")

    day = d.index.floor("D")
    daily = pd.DataFrame({
        "roll_c": d["basis"].diff().groupby(day).apply(roll_noise),
        "mad": d["basis"].groupby(day).apply(robust_scale),
        "share": d["basis"].diff().groupby(day).apply(noise_share),
    })
    quote_rate = (~stale).groupby(day).mean()

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"hspace": 0.42})
    ax[0].plot(daily.index, daily["mad"], lw=1.5, color=RULE, marker="o", ms=2.6, label="basis dispersion (MAD)")
    ax[0].plot(daily.index, daily["roll_c"], lw=1.5, color=FOUND, marker="D", ms=2.6, label="microstructure noise (Roll $c$)")
    ax[0].set_ylabel("pips")
    panel_title(ax[0], "Daily dispersion versus microstructure noise", "a staleness artefact would move both series together")
    ax[0].legend(loc="upper right", ncol=2)
    annotate_event(ax[0], BOJ_SHOCK, "BOJ", top=False)

    for column in PAIRS:
        ax[1].plot(quote_rate.index, quote_rate[column], lw=1.3, color=LEG_COLOUR[column], label=PAIR_LABEL[column])
    ax[1].set_ylabel("share of seconds")
    ax[1].set_ylim(0, 1)
    panel_title(ax[1], "Share of seconds with a new quote, by pair", "higher rates mean less forward-filling")
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax[1].legend(loc="lower right", ncol=3)
    save_fig(fig, "01_microstructure")

    keep = [
        "audusd_mid", "usdjpy_mid", "audjpy_direct", "audjpy_synthetic",
        "basis", "x_audusd", "x_usdjpy", "x_audjpy",
    ]
    d[keep].to_parquet(DAT_DIR / "01_clean.parquet")
    ep.to_parquet(DAT_DIR / "01_episodes.parquet")
    print(f" -> 01_clean.parquet ({len(d):,} rows), 01_episodes.parquet ({len(ep)} episodes)")


if __name__ == "__main__":
    main()