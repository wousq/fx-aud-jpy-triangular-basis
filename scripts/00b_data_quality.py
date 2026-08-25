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

These four checks close that gap.

  1. Grid continuity    the second grid has no holes
  2. Coverage           each leg prints often enough, per day
  3. Single-leg outage  no leg goes silent while the others quote
  4. Reconciliation     each leg agrees with the value implied by the
                        other two

Check 4 is the informative one. In an over-identified price system the
redundancy that creates the arbitrage relationship also creates a
validation instrument: USD/JPY implied by AUD/JPY divided by AUD/USD must
agree with quoted USD/JPY if all three legs are sound. A frozen leg fails
this immediately, and no threshold tuning is required to see it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "synchronized_rates.parquet"
TAB_DIR = ROOT / "output" / "tables"

PAIRS = ["audusd_mid", "usdjpy_mid", "audjpy_direct"]
LABEL = {"audusd_mid": "AUD/USD", "usdjpy_mid": "USD/JPY",
         "audjpy_direct": "AUD/JPY"}

MAX_STALE_SECONDS = 600      # a leg silent this long while others quote
MIN_DAILY_COVERAGE = 0.05    # fraction of seconds with a fresh tick
MAX_RECON_GAP_PIPS = 25.0    # implied vs quoted, 99.9th percentile

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


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA} — run 00_ingest_and_sync.py first")
        return 2

    df = pd.read_parquet(DATA).rename(columns={"grid_time": "t"})
    df = df.set_index("t").sort_index()
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

    # ---------------------------------------------- 4. reconciliation
    print("\n4. Triangular reconciliation")
    implied = {
        "usdjpy_mid": df["audjpy_direct"] / df["audusd_mid"],
        "audusd_mid": df["audjpy_direct"] / df["usdjpy_mid"],
        "audjpy_direct": df["audusd_mid"] * df["usdjpy_mid"],
    }
    recon = []
    for c in PAIRS:
        pip = 1e-4 if c == "audusd_mid" else 1e-2
        gap = (implied[c] - df[c]).abs() / pip
        p999 = float(gap.quantile(0.999))
        recon.append({"pair": LABEL[c], "median_gap_pips": float(gap.median()),
                      "p99_gap_pips": float(gap.quantile(0.99)),
                      "p999_gap_pips": p999, "max_gap_pips": float(gap.max())})
        check(f"{LABEL[c]} agrees with its implied value",
              p999 <= MAX_RECON_GAP_PIPS,
              f"99.9th percentile gap {p999:.2f} pips")
    recon = pd.DataFrame(recon)

    # ---------------------------------------------- provenance
    if "usdjpy_source" in df.columns:
        print("\n5. Source provenance")
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
    print(f"\nwrote 00_outages.csv, 00_reconciliation.csv, 00_coverage.csv "
          f"to {TAB_DIR}")

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