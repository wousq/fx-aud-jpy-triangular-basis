"""
00b_data_quality.py

Data-quality gate. Runs after ingest, before any analysis. Exits non-zero
if a check fails, so it can sit in CI or in a Makefile chain.

The audit inside the ingest pipeline inspects every tick that exists:
nulls, zeros, negatives, crossed quotes. It cannot detect ticks that are
absent, because a missing tick leaves no row to inspect. Forward-filling
then silently substitutes a stale price, which is correct when a quote is
late and wrong when a feed is dead. Nothing in the filled series
distinguishes the two cases.

These five checks close that gap.

  1. Grid continuity    the second grid has no holes
  2. Coverage           each leg prints often enough, per day
  3. Single-leg outage  no leg goes silent while the others quote
  4. Reconciliation     each leg agrees with the value implied by the
                        other two, in the tail
  5. Bias               and agrees on average, day by day

Checks 4 and 5 both use the over-identification: USD/JPY implied by
AUD/JPY divided by AUD/USD must agree with quoted USD/JPY if all three
legs are sound. They are separate checks because they detect different
failures, and neither sees the other's.

    Check 4 takes the 99.9th percentile of the *absolute* gap. That is a
    tail statistic, and it catches a frozen or wildly wrong leg.

    Check 5 takes the *median signed* gap within each day. A leg biased by
    a fraction of a pip moves this immediately and does not move check 4
    at all, because the tail is set by ordinary noise an order of
    magnitude larger. A persistent bias is the more dangerous of the two
    failures: everything downstream still looks plausible, and any
    analysis that measures dispersion, takes differences, or centres per
    day removes it before anyone sees it.

Closures are excluded from both. The FX week ends 17:00 Friday and reopens
17:00 Sunday. Across that stretch the grid is forward-filled padding, so
every second carries the same frozen quote and therefore the same frozen
gap. One weekend is roughly 172,000 identical values, over one percent of
a two-month sample, which is enough to occupy the upper percentiles of
anything computed on the full grid. Before this was fixed, check 4's p99
and p99.9 were bit-identical for all three legs and both equal to a single
stale weekend value: the check was reporting a property of the padding.

closed_mask is imported rather than reimplemented, so the rows this gate
inspects are the rows 01 goes on to analyse.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import closed_mask, load_data, DATA_PATH, LEGACY_DATA_PATH, TAB_DIR

PAIRS = ["audusd_mid", "usdjpy_mid", "audjpy_direct"]
LABEL = {"audusd_mid": "AUD/USD", "usdjpy_mid": "USD/JPY",
         "audjpy_direct": "AUD/JPY"}
PIP = {"audusd_mid": 1e-4, "usdjpy_mid": 1e-2, "audjpy_direct": 1e-2}

MAX_STALE_SECONDS = 600      # a leg silent this long while others quote
MIN_DAILY_COVERAGE = 0.05    # fraction of seconds with a fresh tick
MAX_RECON_GAP_PIPS = 25.0    # implied vs quoted, 99.9th percentile
CLOSURE_MIN_RUN = 600        # matches 01; a run this long is not a market

# A day whose median signed gap exceeds this is carrying a bias rather than
# noise. Clean days in this sample sit under 0.05 pips, so the threshold is
# about twice the noise floor. It is deliberately not loose: the failure it
# exists to catch was a shade under one AUD/USD pip and survived a 25-pip
# tail threshold for two full trading days.
MAX_DAILY_BIAS_PIPS = 0.10
MIN_SECONDS_FOR_BIAS = 3600  # a day thinner than this cannot support a median

failures: list[str] = []
warnings: list[str] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (warnings if warn_only else failures).append(f"{name}: {detail}")


def stale_age(s: pd.Series) -> pd.Series:
    """Seconds since this series last printed a new price."""
    changed = s.diff().ne(0).to_numpy().copy()
    changed[0] = True
    last = np.maximum.accumulate(np.where(changed, np.arange(len(s)), 0))
    return pd.Series(np.arange(len(s)) - last, index=s.index)


def freshness(df: pd.DataFrame, col: str) -> pd.Series:
    """Prefer explicit tick flags from the pipeline; fall back to diffs."""
    flag = col.replace("_mid", "").replace("_direct", "") + "_fresh"
    if flag in df.columns:
        return df[flag].astype(bool)
    return df[col].diff().ne(0)


def implied(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Each leg as the other two imply it."""
    return {
        "usdjpy_mid": df["audjpy_direct"] / df["audusd_mid"],
        "audusd_mid": df["audjpy_direct"] / df["usdjpy_mid"],
        "audjpy_direct": df["audusd_mid"] * df["usdjpy_mid"],
    }


def main() -> int:
    if not DATA_PATH.exists() and not LEGACY_DATA_PATH.exists():
        print(f"missing {DATA_PATH} — run 00_ingest_and_sync.py first")
        return 2

    df = load_data()
    print(f"{len(df):,} rows, {df.index[0]} to {df.index[-1]}\n")

    # ---------------------------------------------- 1. grid continuity
    print("1. Grid continuity")
    span = int((df.index[-1] - df.index[0]).total_seconds()) + 1
    check("no holes in the second grid", len(df) == span,
          f"{len(df):,} rows for {span:,} seconds")

    # ---------------------------------------------- 2. coverage
    print("\n2. Per-leg coverage")
    day = df.index.floor("D")
    cov = pd.DataFrame({LABEL[c]: freshness(df, c).groupby(day).mean()
                        for c in PAIRS})
    # Weekend rows are forward-filled padding and legitimately have none.
    weekday = cov.index.dayofweek < 5
    worst = cov[weekday].min()
    for pair, v in worst.items():
        check(f"{pair} daily coverage", v >= MIN_DAILY_COVERAGE,
              f"worst weekday {v:.1%}")

    # ---------------------------------------------- 3. single-leg outages
    print("\n3. Single-leg outages")
    age = pd.DataFrame({c: stale_age(df[c]) for c in PAIRS})
    others_live = pd.DataFrame(
        {c: (age.drop(columns=c).max(axis=1) < 60) for c in PAIRS})

    rows = []
    for c in PAIRS:
        isolated = (age[c] > MAX_STALE_SECONDS) & others_live[c]
        n = int(isolated.sum())
        rows.append({"pair": LABEL[c], "isolated_seconds": n,
                     "max_stale_seconds": int(age[c].max()),
                     "share": n / len(df)})
        detail = f"{n:,} seconds silent while the others quoted"
        if n:
            first = df.index[isolated.to_numpy().argmax()]
            detail += f", first at {first}"
        check(f"{LABEL[c]} never isolated", n == 0, detail)
    outages = pd.DataFrame(rows)

    # ---------------------------------------------- open-market rows
    # Everything below is computed on these. A gap measured on padding is a
    # property of the last quote before the close, repeated.
    closed = closed_mask(df, cols=PAIRS, min_run=CLOSURE_MIN_RUN)
    live = df.loc[~closed]
    print(f"\nExcluding {closed.mean():.1%} of rows as market closure; "
          f"{len(live):,} open-market seconds remain")

    imp = implied(live)
    gaps = pd.DataFrame({c: (imp[c] - live[c]) / PIP[c] for c in PAIRS},
                        index=live.index)

    # ---------------------------------------------- 4. reconciliation, tail
    print("\n4. Triangular reconciliation — dispersion")
    recon = []
    for c in PAIRS:
        g = gaps[c].abs()
        p999 = float(g.quantile(0.999))
        recon.append({"pair": LABEL[c], "median_gap_pips": float(g.median()),
                      "p99_gap_pips": float(g.quantile(0.99)),
                      "p999_gap_pips": p999, "max_gap_pips": float(g.max())})
        check(f"{LABEL[c]} agrees with its implied value",
              p999 <= MAX_RECON_GAP_PIPS,
              f"99.9th percentile gap {p999:.2f} pips")
    recon = pd.DataFrame(recon)

    # ---------------------------------------------- 5. reconciliation, bias
    print("\n5. Triangular reconciliation — daily bias")
    names = [LABEL[c] for c in PAIRS]
    lday = live.index.floor("D")
    bias = gaps.groupby(lday).median()
    bias.columns = names
    bias["seconds"] = gaps.groupby(lday).size()
    bias = bias[bias["seconds"] >= MIN_SECONDS_FOR_BIAS]

    for pair in names:
        offenders = bias.index[bias[pair].abs() > MAX_DAILY_BIAS_PIPS]
        detail = f"largest daily median {bias[pair].abs().max():.3f} pips"
        if len(offenders):
            span = (f"{offenders[0]:%d %b}" if len(offenders) == 1
                    else f"{offenders[0]:%d %b} to {offenders[-1]:%d %b}")
            detail += f"; {len(offenders)} day(s), {span}"
        check(f"{pair} unbiased day to day", not len(offenders), detail)

    flagged = bias[(bias[names].abs() > MAX_DAILY_BIAS_PIPS).any(axis=1)]
    if len(flagged):
        print("\n  days carrying a bias (median signed gap, each leg's own "
              "pips):")
        for line in flagged.to_string(
                float_format=lambda v: f"{v:+.3f}").split("\n"):
            print(f"    {line}")
        print("\n  The triangle cannot say which leg is at fault: these "
              "three columns are")
        print("  one discrepancy written three ways, and any of the three "
              "reproduces it")
        print("  exactly. Identifying the leg needs a source outside the "
              "triangle, so")
        print("  compare each leg against Dukascopy over the flagged days.")

    # ---------------------------------------------- 6. provenance
    if "usdjpy_source" in df.columns:
        print("\n6. Source provenance")
        for c in ["audusd_source", "usdjpy_source", "audjpy_source"]:
            if c in df.columns:
                vc = df[c].value_counts()
                parts = ", ".join(f"{k} {v:,}" for k, v in vc.items())
                print(f"  [INFO] {c}: {parts}")

    # ---------------------------------------------- report
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    outages.to_csv(TAB_DIR / "00_outages.csv", index=False)
    recon.to_csv(TAB_DIR / "00_reconciliation.csv", index=False)
    cov.to_csv(TAB_DIR / "00_coverage.csv")
    bias.to_csv(TAB_DIR / "00_daily_bias.csv")
    print(f"\nwrote 00_outages.csv, 00_reconciliation.csv, 00_coverage.csv, "
          f"00_daily_bias.csv to {TAB_DIR}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        print("\nDo not run the analysis on this dataset until these are "
              "resolved or explicitly documented.")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())