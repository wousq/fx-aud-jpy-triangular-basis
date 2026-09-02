# data/

Nothing under this directory is committed. Tick data is redistributable only from the
vendor, and the processed grid is a few hundred megabytes that `make data` rebuilds in a
few minutes. This file records exactly what the pipeline expects to find, so the ingest
stage can be reproduced from a fresh clone.

## Layout

```
data/
├── raw/
│   ├── histdata/     DAT_ASCII_{PAIR}_T_YYYYMM.csv          primary source
│   └── dukascopy/    {PAIR}_1Tick_{BID,ASK}_*.csv           gap repair only
└── processed/
    └── synchronized_rates.parquet                           written by make data
```

`00_ingest_and_sync.py` raises `FileNotFoundError` if `raw/histdata` or `raw/dukascopy` is
missing, so create both before the first run even if you only have one vendor's files.

## What to obtain

Three pairs, `AUDUSD`, `USDJPY` and `AUDJPY`, for July and August 2024.

**HistData** supplies the primary series: ASCII tick exports, one file per pair per month.
The script globs `*{PAIR}*.csv` inside `raw/histdata`, so the vendor's own filenames work
unchanged and the two monthly files per pair can simply be dropped in together.

**Dukascopy** supplies replacement ticks for one defect only. The USD/JPY export from
HistData is missing four hours on 5 August 2024, which contain the peak of the carry
unwind, and forward-filling across them manufactures a spurious 60-pip dislocation. Export
those hours from Dukascopy, or the whole month if that is easier: the splice takes only
seconds HistData never delivered, so extra coverage changes nothing.

Dukascopy filenames use hyphens where HistData does not. The script looks for
`AUD-USD_1Tick_*.csv`, `USD-JPY_1Tick_*.csv` and `AUD-JPY_1Tick_*.csv`, and BID and ASK
must be exported as separate files with identical tick sequences, which is what the
exporter produces by default.

If `raw/dukascopy` is empty the pipeline still runs and says so. The 5 August gap is then
forward-filled, and the resulting dislocation is an artefact rather than a measurement.

## Formats the parser assumes

**HistData**, no header row:

| column | contents |
|---|---|
| 0 | timestamp, `YYYYMMDD HHMMSSmmm` |
| 1 | bid |
| 2 | ask |
| 3 | volume, unused |

Timestamps are New York local time. The vendor documents EST, but the sample lies entirely
inside US daylight saving, so the operative offset is UTC−4. See the clock note in the top
level README: the offset was measured rather than assumed.

**Dukascopy**, with a header row. The first column is an ISO 8601 timestamp in UTC and the
price is read from the `Close` column. The loader shifts these by −4 hours to match
HistData's clock.

Ticks are dropped where bid or ask is null, zero or negative, or where the quote is crossed.
Both vendors are aggregators, so the quoted spreads are wider than interdealer, which
matters for the execution bounds in section 05 of the top level README.

## Checking you have the right files

Run the ingest, then the gate:

```bash
make data
make check
```

Four numbers to compare against:

- roughly **3.86 million** open-market seconds survive after closures are removed
- **8** closures are detected, all 48 hours to within two minutes
- the triangular reconciliation gap sits at **3.21**, **7.29** and **4.83** pips at the
  99.9th percentile for AUD/USD, USD/JPY and AUD/JPY
- the source composition printed by the ingest shows Dukascopy supplying only a small
  number of USD/JPY seconds, in the low thousands rather than the millions

A Dukascopy share in the millions means the splice is replacing HistData rather than
filling its gaps, and the clock is the first thing to check.

## Expect the gate to fail

`make check` exits non-zero on this sample by design. Seven of 53 days breach the daily
bias threshold, and 1 and 2 August breach it by a factor of fifteen. This is documented in
section 2 of the top level README, and is a property of the vendor data rather than of the
download. Dukascopy carries the same level on those days, but HistData sources from
Dukascopy, so that agreement is not an independent check.

To build past it, recording that the failure is known:

```bash
make all ACCEPT_KNOWN_BIAS=1
```
