"""
00_ingest_and_sync.py

Builds the synchronized 1-second grid for AUD/USD, USD/JPY and AUD/JPY
from HistData tick exports.

Layout
    data/raw/histdata/    DAT_ASCII_{PAIR}_T_YYYYMM.csv
    data/processed/synchronized_rates.parquet

Method
    1. Audit each raw file for nulls, zeros, negatives and crossed quotes.
    2. Parse surviving ticks and take the mid.
    3. Reduce to one observation per second (last tick of the second).
    4. Generate a gapless 1-second grid over the full span and
       forward-fill each leg.
    5. Compute the synthetic cross and the triangular dislocation.

Forward-filling carries the last observed price into seconds with no tick.
This is the standard treatment for irregularly-spaced quote data and is
correct whenever a quote is merely late.
"""

from __future__ import annotations

import glob
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "data" / "raw" / "histdata"
OUTPUT_PARQUET = ROOT / "data" / "processed" / "synchronized_rates.parquet"

PAIRS = ["AUDUSD", "USDJPY", "AUDJPY"]


def build_pipeline() -> None:
    if not HIST_DIR.exists():
        raise FileNotFoundError(f"missing directory: {HIST_DIR}")

    con = duckdb.connect(database=":memory:")

    # ------------------------------------------------ audit and parse
    for pair in PAIRS:
        pattern = str(HIST_DIR / f"*{pair}*.csv")
        if not glob.glob(pattern):
            raise FileNotFoundError(f"no files for {pair} in {HIST_DIR}")

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

    # ------------------------------------------------ per-second reduction
    print("Downsampling to 1-second buckets (last tick of each second)...")
    for pair in PAIRS:
        con.execute(f"""
            CREATE TABLE sec_{pair.lower()} AS
            SELECT date_trunc('second', tick_time) AS sec_time,
                   ARG_MAX(mid, tick_time) AS mid
            FROM raw_{pair.lower()}
            GROUP BY 1;
        """)

    min_time, max_time = con.execute("""
        SELECT LEAST((SELECT MIN(sec_time) FROM sec_audusd),
                     (SELECT MIN(sec_time) FROM sec_usdjpy),
                     (SELECT MIN(sec_time) FROM sec_audjpy)),
               GREATEST((SELECT MAX(sec_time) FROM sec_audusd),
                        (SELECT MAX(sec_time) FROM sec_usdjpy),
                        (SELECT MAX(sec_time) FROM sec_audjpy))
    """).fetchone()
    print(f"Grid bounds: {min_time} to {max_time}")

    # ------------------------------------------------ grid and LOCF
    print("Building grid and forward-filling...")
    con.execute(f"""
    CREATE TABLE synchronized_data AS
    WITH grid AS (
        SELECT UNNEST(generate_series('{min_time}'::TIMESTAMP,
                                      '{max_time}'::TIMESTAMP,
                                      INTERVAL '1 second')) AS grid_time
    ),
    joined AS (
        SELECT g.grid_time,
               a.mid AS raw_audusd,
               u.mid AS raw_usdjpy,
               j.mid AS raw_audjpy
        FROM grid g
        LEFT JOIN sec_audusd a ON g.grid_time = a.sec_time
        LEFT JOIN sec_usdjpy u ON g.grid_time = u.sec_time
        LEFT JOIN sec_audjpy j ON g.grid_time = j.sec_time
    ),
    filled AS (
        SELECT grid_time,
               raw_audusd IS NOT NULL AS audusd_fresh,
               raw_usdjpy IS NOT NULL AS usdjpy_fresh,
               raw_audjpy IS NOT NULL AS audjpy_fresh,
               LAST_VALUE(raw_audusd IGNORE NULLS) OVER w AS audusd_mid,
               LAST_VALUE(raw_usdjpy IGNORE NULLS) OVER w AS usdjpy_mid,
               LAST_VALUE(raw_audjpy IGNORE NULLS) OVER w AS audjpy_direct
        FROM joined
        WINDOW w AS (ORDER BY grid_time
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
    SELECT grid_time,
           audusd_mid,
           usdjpy_mid,
           audjpy_direct,
           audusd_mid * usdjpy_mid AS audjpy_synthetic,
           audjpy_direct - (audusd_mid * usdjpy_mid) AS spread_dislocation,
           audusd_fresh, usdjpy_fresh, audjpy_fresh
    FROM filled
    WHERE audusd_mid IS NOT NULL
      AND usdjpy_mid IS NOT NULL
      AND audjpy_direct IS NOT NULL;
    """)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUTPUT_PARQUET}...")
    con.execute(f"""COPY synchronized_data TO '{OUTPUT_PARQUET.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION SNAPPY);""")

    n = con.execute("SELECT COUNT(*) FROM synchronized_data").fetchone()[0]
    print(f"Done. {n:,} synchronized 1-second rows.")


if __name__ == "__main__":
    build_pipeline()