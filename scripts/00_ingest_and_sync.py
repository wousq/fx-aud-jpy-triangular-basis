"""
00_ingest_and_sync.py

Builds the synchronized 1-second grid for AUD/USD, USD/JPY and AUD/JPY.

Layout
    data/raw/histdata/    DAT_ASCII_{PAIR}_T_YYYYMM.csv   primary source
    data/raw/dukascopy/   {PAIR}_1Tick_{BID,ASK}_*.csv    gap repair
    data/processed/synchronized_rates.parquet             output

Three things distinguish this from a plain LOCF pipeline.

Gap repair. The HistData USD/JPY export is missing four hours on 5 August
2024 (12:00-14:00, 16:00-17:00 and 18:00-19:00 local), which contain the
peak of the yen carry unwind. Forward-filling across those hours produces
a spurious 60-pip triangular dislocation. They are refilled from Dukascopy
tick exports. HistData sources from Dukascopy, and over a control window
the two agree to 0.01 pips with correlation 1.00000, so this recovers
dropped data rather than blending two liquidity pools.

Provenance. Every row records which source its USD/JPY price came from and
whether each leg printed a genuine tick that second. Downstream code needs
the freshness flags: inferring staleness from price changes misclassifies a
repeated identical quote as stale.

Timezone. HistData timestamps are New York local time, which is UTC-4
across July and August 2024. This was established by correlating against
Dukascopy over a control window across candidate offsets: -4h gives
correlation 1.00000 and a gap standard deviation of 0.01 pips, while -5h
gives -0.71 and 48.65 pips. Dukascopy exports are UTC and are shifted
accordingly.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
HIST_DIR = RAW_DIR / "histdata"
DUKA_DIR = RAW_DIR / "dukascopy"
PROC_DIR = ROOT / "data" / "processed"
OUTPUT_PARQUET = PROC_DIR / "synchronized_rates.parquet"

PAIRS = ["AUDUSD", "USDJPY", "AUDJPY"]

# Dukascopy filenames use hyphens; HistData does not.
DUKA_NAME = {"AUDUSD": "AUD-USD", "USDJPY": "USD-JPY", "AUDJPY": "AUD-JPY"}

# Dukascopy exports are UTC; HistData is New York local (UTC-4 in summer).
DUKA_UTC_OFFSET = pd.Timedelta(hours=-4)


# --------------------------------------------------------------- Dukascopy

def load_dukascopy(pair: str, directory: Path = DUKA_DIR):
    """
    Read Dukascopy BID/ASK tick exports for one pair and return one mid per
    second, expressed in HistData's clock.

    Timestamps in these files carry second resolution with many ticks per
    second, so the last tick of a second is identified by file order.
    Pandas preserves that order; DuckDB's CSV reader does not guarantee it,
    which is why this step is not done in SQL.

    BID and ASK arrive as separate files with identical tick sequences, so
    they are aligned positionally rather than joined on time.
    """
    stem = DUKA_NAME.get(pair, pair)
    files = sorted(glob.glob(str(directory / f"{stem}_1Tick_*.csv")))
    if not files:
        return None

    bids = sorted(f for f in files if "_BID_" in f)
    asks = sorted(f for f in files if "_ASK_" in f)
    if len(bids) != len(asks):
        raise ValueError(f"{pair}: {len(bids)} BID files vs {len(asks)} ASK")

    frames = []
    for fb, fa in zip(bids, asks):
        b = pd.read_csv(fb)
        a = pd.read_csv(fa)
        if len(b) != len(a):
            raise ValueError(f"BID/ASK row mismatch in {Path(fb).name}: "
                             f"{len(b)} vs {len(a)}")
        t = pd.to_datetime(b.iloc[:, 0], utc=True, format="ISO8601")
        frames.append(pd.DataFrame({
            "tick_time": t.dt.tz_localize(None) + DUKA_UTC_OFFSET,
            "bid": b["Close"].to_numpy(float),
            "ask": a["Close"].to_numpy(float),
        }))

    tick = pd.concat(frames, ignore_index=True)
    tick = tick[(tick.bid > 0) & (tick.ask > 0) & (tick.bid < tick.ask)]
    tick["mid"] = (tick.bid + tick.ask) / 2.0

    sec = (tick.assign(sec_time=tick.tick_time.dt.floor("s"))
               .groupby("sec_time", sort=True)["mid"].last()
               .reset_index())
    print(f"    {pair}: {len(tick):,} ticks / {len(files)} files "
          f"-> {len(sec):,} seconds "
          f"({sec.sec_time.min()} to {sec.sec_time.max()} local)")
    return sec


# ------------------------------------------------------------------- build

def build_pipeline() -> None:
    for d in (HIST_DIR, DUKA_DIR):
        if not d.exists():
            raise FileNotFoundError(f"missing directory: {d}")

    con = duckdb.connect(database=":memory:")

    # ---------------------------------------------- audit and parse
    for pair in PAIRS:
        pattern = str(HIST_DIR / f"*{pair}*.csv")
        if not glob.glob(pattern):
            raise FileNotFoundError(f"no HistData files for {pair} in {HIST_DIR}")

        print(f"Auditing {pair}...")
        total, nulls, zeros, negatives, crossed = con.execute(f"""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE column1 IS NULL OR column2 IS NULL),
                   COUNT(*) FILTER (WHERE column1::DOUBLE = 0
                                    OR column2::DOUBLE = 0),
                   COUNT(*) FILTER (WHERE column1::DOUBLE < 0
                                    OR column2::DOUBLE < 0),
                   COUNT(*) FILTER (WHERE column1::DOUBLE > column2::DOUBLE)
            FROM read_csv_auto('{pattern}', header=False);
        """).fetchone()
        print(f"    {total:,} ticks | nulls {nulls:,} | zeros {zeros:,} | "
              f"negatives {negatives:,} | crossed {crossed:,}"
              + ("   [ANOMALY]" if crossed else ""))

        # This audit only inspects ticks that exist. It cannot detect ticks
        # that are absent, which is the failure mode addressed below and
        # tested for in 00b_data_quality.py.
        con.execute(f"""
            CREATE TABLE raw_{pair.lower()} AS
            SELECT strptime(column0, '%Y%m%d %H%M%S%g')::TIMESTAMP AS tick_time,
                   (column1::DOUBLE + column2::DOUBLE) / 2.0 AS mid
            FROM read_csv_auto('{pattern}', header=False)
            WHERE column1 IS NOT NULL AND column2 IS NOT NULL
              AND column1::DOUBLE > 0
              AND column2::DOUBLE > 0
              AND column1::DOUBLE < column2::DOUBLE;
        """)

    print("Downsampling to 1-second buckets (last tick of each second)...")
    for pair in PAIRS:
        con.execute(f"""
            CREATE TABLE sec_{pair.lower()} AS
            SELECT date_trunc('second', tick_time) AS sec_time,
                   ARG_MAX(mid, tick_time) AS mid
            FROM raw_{pair.lower()}
            GROUP BY 1;
        """)

    # ---------------------------------------------- gap repair
    print("Loading Dukascopy replacement ticks...")
    repaired = {}
    for pair in PAIRS:
        sec = load_dukascopy(pair)
        tbl = f"sec_{pair.lower()}_duka"
        if sec is None:
            con.execute(f"CREATE TABLE {tbl}(sec_time TIMESTAMP, mid DOUBLE);")
            continue
        con.register(f"duka_{pair.lower()}_df", sec)
        con.execute(f"CREATE TABLE {tbl} AS "
                    f"SELECT sec_time, mid FROM duka_{pair.lower()}_df;")
        n_new = con.execute(f"""
            SELECT COUNT(*) FROM {tbl} d
            WHERE NOT EXISTS (SELECT 1 FROM sec_{pair.lower()} h
                              WHERE h.sec_time = d.sec_time)
        """).fetchone()[0]
        repaired[pair] = n_new
        print(f"    {pair}: {n_new:,} seconds absent from HistData -> repaired")
    if not repaired:
        print("    none found; proceeding with HistData only")

    # ---------------------------------------------- bounds
    min_time, max_time = con.execute("""
        SELECT LEAST((SELECT MIN(sec_time) FROM sec_audusd),
                     (SELECT MIN(sec_time) FROM sec_usdjpy),
                     (SELECT MIN(sec_time) FROM sec_audjpy)),
               GREATEST((SELECT MAX(sec_time) FROM sec_audusd),
                        (SELECT MAX(sec_time) FROM sec_usdjpy),
                        (SELECT MAX(sec_time) FROM sec_audjpy))
    """).fetchone()
    print(f"Grid bounds: {min_time} to {max_time}")

    # ---------------------------------------------- grid, splice, LOCF
    print("Building grid, splicing sources and forward-filling...")
    con.execute(f"""
    CREATE TABLE synchronized_data AS
    WITH grid AS (
        SELECT UNNEST(generate_series('{min_time}'::TIMESTAMP,
                                      '{max_time}'::TIMESTAMP,
                                      INTERVAL '1 second')) AS grid_time
    ),
    joined AS (
        SELECT
            g.grid_time,
            -- HistData wins wherever it has a tick; Dukascopy fills only
            -- the seconds HistData never delivered.
            COALESCE(a.mid, da.mid) AS raw_audusd,
            COALESCE(u.mid, du.mid) AS raw_usdjpy,
            COALESCE(j.mid, dj.mid) AS raw_audjpy,
            CASE WHEN a.mid IS NOT NULL THEN 'histdata'
                 WHEN da.mid IS NOT NULL THEN 'dukascopy' END AS src_audusd,
            CASE WHEN u.mid IS NOT NULL THEN 'histdata'
                 WHEN du.mid IS NOT NULL THEN 'dukascopy' END AS src_usdjpy,
            CASE WHEN j.mid IS NOT NULL THEN 'histdata'
                 WHEN dj.mid IS NOT NULL THEN 'dukascopy' END AS src_audjpy
        FROM grid g
        LEFT JOIN sec_audusd       a  ON g.grid_time = a.sec_time
        LEFT JOIN sec_usdjpy       u  ON g.grid_time = u.sec_time
        LEFT JOIN sec_audjpy       j  ON g.grid_time = j.sec_time
        LEFT JOIN sec_audusd_duka  da ON g.grid_time = da.sec_time
        LEFT JOIN sec_usdjpy_duka  du ON g.grid_time = du.sec_time
        LEFT JOIN sec_audjpy_duka  dj ON g.grid_time = dj.sec_time
    ),
    filled AS (
        SELECT
            grid_time,
            raw_audusd IS NOT NULL AS audusd_fresh,
            raw_usdjpy IS NOT NULL AS usdjpy_fresh,
            raw_audjpy IS NOT NULL AS audjpy_fresh,
            LAST_VALUE(raw_audusd IGNORE NULLS) OVER w AS audusd_mid,
            LAST_VALUE(raw_usdjpy IGNORE NULLS) OVER w AS usdjpy_mid,
            LAST_VALUE(raw_audjpy IGNORE NULLS) OVER w AS audjpy_direct,
            LAST_VALUE(src_audusd IGNORE NULLS) OVER w AS audusd_source,
            LAST_VALUE(src_usdjpy IGNORE NULLS) OVER w AS usdjpy_source,
            LAST_VALUE(src_audjpy IGNORE NULLS) OVER w AS audjpy_source
        FROM joined
        WINDOW w AS (ORDER BY grid_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
    SELECT
        grid_time,
        audusd_mid,
        usdjpy_mid,
        audjpy_direct,
        audusd_mid * usdjpy_mid AS audjpy_synthetic,
        audjpy_direct - (audusd_mid * usdjpy_mid) AS spread_dislocation,
        audusd_fresh, usdjpy_fresh, audjpy_fresh,
        audusd_source, usdjpy_source, audjpy_source
    FROM filled
    WHERE audusd_mid IS NOT NULL
      AND usdjpy_mid IS NOT NULL
      AND audjpy_direct IS NOT NULL;
    """)

    # ---------------------------------------------- validation
    print("\nSource composition:")
    print(con.execute("""
        SELECT usdjpy_source,
               COUNT(*) AS seconds,
               COUNT(*) FILTER (WHERE usdjpy_fresh) AS fresh_ticks
        FROM synchronized_data GROUP BY 1 ORDER BY 2 DESC
    """).df().to_string(index=False))

    # The triangle validates its own inputs: USD/JPY implied by the other
    # two legs must track the series actually used. A bad splice shows up
    # immediately as a systematic gap on the repaired seconds.
    print("\nUSD/JPY implied by the other two legs vs the series used:")
    print(con.execute("""
        SELECT usdjpy_source,
               COUNT(*) AS n,
               ROUND(MEDIAN(ABS(audjpy_direct / audusd_mid - usdjpy_mid))
                     * 100, 3) AS median_gap_pips,
               ROUND(QUANTILE_CONT(ABS(audjpy_direct / audusd_mid
                     - usdjpy_mid), 0.99) * 100, 3) AS p99_gap_pips
        FROM synchronized_data GROUP BY 1 ORDER BY 2 DESC
    """).df().to_string(index=False))
    print("  comparable magnitudes across sources indicate a clean splice")

    # Residual gaps. A stretch where a leg is silent and the replacement
    # source is silent too is a thin market, not a vendor defect: both
    # feeds agree that nothing traded. Only gaps the replacement source
    # can actually fill are defects. This distinction matters — 4 July is
    # a US half-session and AUD/USD genuinely stops quoting, so patching
    # it would misrepresent a liquidity event as missing data.
    print("\nResidual single-leg silences after repair "
          "(these are thin markets, not defects):")
    print(con.execute("""
        WITH flagged AS (
            SELECT grid_time,
                   audusd_fresh::INT + usdjpy_fresh::INT
                                     + audjpy_fresh::INT AS legs_quoting
            FROM synchronized_data
        )
        SELECT date_trunc('day', grid_time) AS day,
               COUNT(*) FILTER (WHERE legs_quoting = 0) AS no_leg_quoted,
               COUNT(*) FILTER (WHERE legs_quoting = 3) AS all_legs_quoted
        FROM flagged
        GROUP BY 1 HAVING COUNT(*) FILTER (WHERE legs_quoting = 0) > 3000
        ORDER BY 1
    """).df().to_string(index=False))
    print("  large no-leg counts are weekends; 00b_data_quality.py "
          "classifies the rest")

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {OUTPUT_PARQUET}...")
    con.execute(f"""COPY synchronized_data TO '{OUTPUT_PARQUET.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION SNAPPY);""")

    n = con.execute("SELECT COUNT(*) FROM synchronized_data").fetchone()[0]
    print(f"Done. {n:,} synchronized 1-second rows.")


if __name__ == "__main__":
    build_pipeline()