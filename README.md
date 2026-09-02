# AUD/JPY triangular basis

**One-second measurement across the August 2024 yen carry unwind. 1 July to 30 August 2024.**

**[Full report (PDF, 25 pages)](report/main.pdf)**

## Overview

AUD/JPY should equal AUD/USD times USD/JPY. The residual of that relation is the triangular
basis:

```
z_t  =  AJ_t  −  AU_t × UJ_t          (AUD/JPY pips)
```

This repository measures `z` every second for two months around the yen carry unwind, on
3.86 million observations spliced from two tick vendors. It estimates when the statistical
properties of `z` changed, how fast `z` returns to zero, which leg does the returning, and
how the size of `z` compares with the cost of a round trip around the triangle.

`z` stays inside that cost band throughout the sample. No arbitrage is found and none is
claimed.

## Headline result

Measured within the hour, basis dispersion under stress is **1.21×** its level in the calm
segments. Pooled across every second in the stress window, the same quantity reads
**2.91×** on a robust scale and **4.22×** on a non-robust one.

The difference is day-to-day movement in the basis level, which accounts for **80.6%** of
the stress window's level variance against 11.6% in the calm segments. Two days inside the
window fail the pipeline's daily-bias check by a factor of fifteen and account for the
whole of the regime's mean level shift.

The within-hour figure is the defensible one, and it does not depend on where the regime
boundary is drawn. Section 1 sets out the decomposition, Section 2 documents the defect,
and Section 3 shows the boundary sweep that settles the choice.

## Results at a glance

Hourly dispersion is best described by a bounded excursion that returns, not a permanent
shift: posterior mass 1.000 on the two-changepoint models, observed log Bayes factor +97.1
against a null 95th percentile of −2.8 (Section 3). Regime boundaries are estimated rather
than assigned by calendar; the stress window runs 30 July 15:00 to 9 August 09:59 under the
segmentation in use, 183 hours.

| | pre | stress | post |
|---|---|---|---|
| **Dispersion** (Section 1) | | | |
| Within-hour MAD (pips) | 0.093 | 0.111 | 0.091 |
| Pooled MAD (pips) | 0.109 | 0.303 | 0.098 |
| **Execution** (Section 4) | | | |
| Transaction-cost band, median (pips) | 2.61 | 3.35 | 2.57 |
| Band / dispersion | 23.3× | 12.7× | 27.1× |
| **Episodes** (Section 5) | | | |
| Count | 24 | 22 | 12 |
| Median seconds to close | 3 | 24 | 4 |
| **Closure speed** (Section 6) | | | |
| Half-life, day-by-day median (s) | 43.6 | 38.4 | 27.6 |
| Half-life, pooled system (s) | 56.5 | 155.4 | 38.3 |

---

## 1. The widening depends entirely on the scope of the measurement

The same widening, measured at four nested scopes. Calm means the pre and post segments
pooled.

| Scope | Stress vs calm |
|---|---|
| Within the hour (MAD) | **1.21×** |
| Within the day (sd) | 1.98× |
| Pooled across the window, robust (MAD) | 2.91× |
| Pooled across the window, non-robust (sd) | 4.22× |

Each step up the ladder admits another layer of level movement into the statistic. Nothing
about the market changes between the rows.

### Variance decomposition

Splitting the level variance on the trading day `d`:

```
Var(z)  =  E_d[ Var(z | d) ]   +   Var_d( E[z | d] )
          └── within-day ──┘      └── day-to-day ──┘
```

| | stress | calm |
|---|---|---|
| Day-to-day share of level variance | **80.6%** | 11.6% |

Four fifths of what a pooled statistic calls dispersion under stress is the level moving
between days. Removing that component leaves a within-day widening of 1.98×, and
restricting further to a single hour leaves 1.21×. The changepoint model measured the
within-hour figure independently from hourly buckets and returned 1.19× to 1.22×, which is
the cross-check that the decomposition and the fit describe the same object.

### Effect on the persistence estimates

Write `z_t = μ_d + u_t` with `μ_d` constant within a day and `u_t` an AR(1) with
coefficient `ρ_w`. Almost every lag-one pair lies inside a day, so:

```
ρ_pooled  =  ( σ²_day  +  ρ_w · σ²_within ) / ( σ²_day + σ²_within )
```

The stress regime's pooled AR(1) of 0.976 is therefore dominated by its daily-mean
component. This is corroborated by an independent route: the VECM refitted day by day
removes the daily level by construction and returns median half-lives of 43.6, 38.4 and
27.6 seconds, placing the stress window between the other two.

### Summary

Estimators that remove the daily level return a small effect: within-hour MAD (1.21×), the
day-by-day VECM (1.4×). Estimators that retain it return a large one: pooled AR(1) (9.7×),
pooled full-system VECM (4.1×), episode ruler (9.3×). The choice of estimand moves the
reported effect by a factor of seven.

Files: `07_boundary_named.csv`, `06_closure_reconciliation.csv`, `03_var_fit.csv`,
`02_changepoint_posterior.png`

---

## 2. Two days fail the data-quality gate

`00b_data_quality.py` tests whether each leg agrees on average with the value the other two
imply, using the median signed gap within each day. Seven of 53 days exceed the 0.10 pip
threshold.

| Day | AUD/USD | USD/JPY | AUD/JPY | Open-market seconds |
|---|---|---|---|---|
| 11 Jul | +0.143 | +0.338 | −0.228 | 86,400 |
| 12 Jul | +0.137 | +0.320 | −0.217 | 61,200 |
| 31 Jul | −0.045 | −0.104 | +0.068 | 86,400 |
| **1 Aug** | **−0.977** | **−2.247** | **+1.464** | 86,400 |
| **2 Aug** | **−0.991** | **−2.252** | **+1.468** | 61,200 |
| 8 Aug | +0.176 | +0.393 | −0.258 | 86,400 |
| 9 Aug | +0.180 | +0.402 | −0.265 | 61,200 |

The gate returns a non-zero exit code on this sample. It exists to catch a persistent
sub-pip bias, which is invisible to a tail statistic: the 99.9th percentile of the absolute
gap passes at 3.21, 7.29 and 4.83 pips against a 25-pip threshold.

The triangle cannot attribute the discrepancy to a leg. The three columns are one
discrepancy expressed three ways, and any single leg reproduces it exactly.

**The defect is not resolvable with free data.** Dukascopy carries the same level on those
days, but HistData sources from Dukascopy, so agreement between them confirms the level is
present in a shared upstream feed rather than that it is real. Settling it requires a feed
from a different liquidity pool. The offset is therefore documented and carried, not fixed.

### Effect on downstream estimates

1 and 2 August carry a standing basis offset near −1.47 pips across 147,600 open-market
seconds, which is 22% to 24% of the stress window depending on the admission rule. The
implied contribution to the window's mean basis is −0.33 to −0.35 pips. The measured stress
mean basis is −0.27 pips, against +0.016 pre and +0.005 post. The two days account for the
entire mean level shift attributed to the stress window.

Estimates that are insensitive to this:

- **Changepoint dates.** A level offset constant within a day leaves within-hour dispersion
  unchanged, and within-hour dispersion is the fitted feature. Neither estimated changepoint
  falls near 1 or 2 August.
- **Error-correction coefficients.** The VECM centres on the day, which removes a daily
  constant by construction. The sensitivity variant that centres on the regime instead
  returns a stress half-life of 967 seconds against a baseline of 155.

Estimates that are sensitive to it: the regime-level standard deviation, the pooled AR(1)
of the level, and the positive-edge counts in Section 05.

Files: `00_daily_bias.csv`, `00_reconciliation.csv`

---

## 3. Structural change: an excursion, not a permanent shift

Four segmentations of hourly basis dispersion, compared by exact marginal likelihood.

| Model | Changepoints | log evidence | Posterior |
|---|---|---|---|
| M0 no change | 0 | 605.8 | 2.7 × 10⁻⁴³ |
| M1 permanent shift | 1 | 645.4 | 4.5 × 10⁻²⁶ |
| M2 excursion | 2 | 702.8 | 0.384 |
| M3 two free changes | 2 | 703.3 | 0.616 |

Posterior mass on the two-changepoint models is 1.000 to three decimals. Dispersion rose
and then returned to its previous level.

M0 is included because a one-changepoint model always returns a changepoint. Without a
no-change alternative in the comparison, the selected date carries no information about
whether a change occurred.

### Calibration under the null

A Bayes factor measures how much better M2 fits than M0 on this series. It does not
establish that a series without a changepoint would have looked different. One hundred
replicates were simulated with the sample's length, session structure, fitted AR(1)
coefficient and residual scale, and no change anywhere, then refitted with the same code.

| | |
|---|---|
| P(any change selected \| no change) | 0.020 |
| P(excursion selected \| no change) | 0.010 |
| Null log Bayes factor, 95th percentile | −2.8 |
| Observed log Bayes factor (M2 vs M0) | **+97.1** |
| Fraction of null replicates exceeded | 1.000 |

The AR(1) error term produces that calibration. Under independent errors the same code
selects an excursion on noise in roughly four series out of five.

All 20 specifications in `02_sensitivity.csv` select a two-changepoint model. Varied: the
summary statistic, bucket width, prior scale, minimum segment length, both admission
filters, the deseasonalisation, the rollover boundary, and the error model. None selects M0
or M1.

### The onset is bimodal, and the boundary does not drive the result

The onset posterior has two modes. One sits at 24 July 17:00, the other at the policy
decision on 30 July, and the 95% window between them spans a weekend closure. Across the 17
same-feature specifications the split is 11 at 24 July, 4 at 30 July, 1 at 29 July and 1 at
1 July. The boundary in use is 30 July, which is the median segmentation and therefore what
the per-second labels encode, but it is not the date the estimator selects most often.

Rather than choose, `07_boundary_sweep.py` re-aggregates the basis under every hourly onset
from 23 to 31 July, under both candidate returns, and reports the decomposition as a
function of the boundary. Nothing is refitted.

| | median rule (in use) | modal rule | all 128 boundaries |
|---|---|---|---|
| Onset | 30 Jul 15:00 | 24 Jul 17:00 | 23 to 31 Jul |
| Return | 09 Aug 10:00 | 13 Aug 19:00 | both |
| Stress window (hours) | 183 | 331 | 160 to 354 |
| **Day-to-day share, stress** | **80.6%** | **78.7%** | **78.7% to 87.3%** |
| Day-to-day share, calm | 11.6% | 11.8% | 10.2% to 12.9% |
| Within-hour MAD ratio | 1.21× | 1.13× | 1.12× to 1.21× |
| Pooled MAD ratio | 2.91× | **1.63×** | |

The decomposition holds across every boundary tested. The within-hour widening moves by 8%
across the entire sweep, which makes it the most stable quantity in the project.

The pooled MAD ratio is the exception. It falls from 2.91× to 1.63× when the boundary moves
to the modal onset. The quantity this project reports as its headline is stable under the
boundary choice; the quantity it argues against reporting is the one that is not.

Files: `02_model_evidence.csv`, `02_null_calibration.csv`, `07_boundary_named.csv`,
`07_boundary_sweep.png`

---

## 4. The transaction-cost band exceeds the basis by 12.7× at its tightest

A round trip around the triangle crosses three spreads. Writing `z` for the mid basis and
`s` for a quoted spread, the two circuits rearrange into an exact identity:

```
W  =  s_AJ  +  AU_mid · s_UJ  +  UJ_mid · s_AU
```

The quadratic terms cancel in the difference, so the band decomposes additively across the
three legs with no interaction. Nothing is estimated and no cost parameter is assumed.

| | pre | stress | post |
|---|---|---|---|
| Band width, median (pips) | 2.61 | 3.35 | 2.57 |
| Basis dispersion, MAD (pips) | 0.112 | 0.264 | 0.095 |
| Band / dispersion | 23.3× | 12.7× | 27.1× |

Rows are restricted to seconds where all three legs quoted within one second and the regime
posterior is at least 90%.

Both quantities widened. The dispersion grew by 2.35× and the band by 1.29×, so the
band-to-gap ratio approximately halved. The basis moved closer to the band and remained an
order of magnitude inside it.

### Sensitivity to the spread level

Both vendors publish aggregator quotes, which are wider than interdealer. This biases the
comparison toward finding no executable dislocation. Scaling every spread by a factor `k`
and recording the share of seconds with a positive edge:

| Spreads scaled to | pre | stress | post |
|---|---|---|---|
| 100% (as quoted) | 0.004% | 12.7% | 0.003% |
| 50% | 0.05% | 23.5% | 0.05% |
| 25% | 2.0% | 26.1% | 0.4% |
| 10% | 22.6% | 45.7% | 14.4% |

Halving every quoted spread leaves 99.95% of calm seconds inside the band.

The 12.7% stress column is an upper bound, not an opportunity rate. See Limitations.

Files: `05_execution.csv`, `05_band_vs_gap.csv`, `05_haircut.csv`, `05_bands.png`

---

## 5. Leg attribution: the dollar legs open, the cross closes

The basis is a fixed linear combination of three log prices, so its change over any window
is exactly the sum of three leg contributions. Applied to the opening and closing window of
each of the 58 detected episodes, this splits both by arithmetic. No model and no lag order
are involved.

| Share of the move | pre | stress | post |
|---|---|---|---|
| Opening: AUD/USD + USD/JPY | 0.86 | 1.12 | 1.05 |
| Closing: AUD/JPY | 0.64 | 0.71 | 0.48 |
| Median seconds to open | 2 | 19 | 4 |
| Median seconds to close | 3 | 24 | 4 |

Shares can exceed one because leg contributions can oppose each other.

A second, independent line agrees. The error-correction model carries a sign restriction
fixed before estimation: α positive for AUD/USD and USD/JPY, negative for AUD/JPY. All
three hold in all three regimes at |t| between 2.4 and 11.5.

### Predictability

Cross-leg Granger effects clear their placebo by 3.5× to 10.6×. The placebo redraws one
leg's increments from a different day, which preserves that leg's autocorrelation and
volatility clustering and destroys only the cross-leg timing.

Leg-to-basis effects behave differently. Conditioning on the basis's own memory, each leg
explains between 0.01% and 0.07% of basis increment variance. AUD/JPY fails Holm correction
in all three regimes and USD/JPY in two, on samples of half a million seconds.

The basis is close to unforecastable from its own components while those components remain
highly forecastable from each other.

At these sample sizes a p-value carries no information, so every test in Section 04 reports
an effect size in pips per second beside a placebo threshold computed from the same
statistic.

Files: `04_attribution.csv`, `04_weak_exogeneity.csv`, `04_basis_causality.csv`

---

## 6. Closure speed: five estimands spanning 1.4× to 9.7×

| Route | What it estimates | Stress / calmest |
|---|---|---|
| Episode ruler | Time for a large excursion to halve | 9.3× |
| AR(1) on the level | Decay of an average deviation, one parameter | 9.7× |
| VECM, one-step | Decay implied by λ alone | 2.9× |
| VECM, full system | Decay of the 21-lag system run down | 4.1× |
| VECM, day by day, median | The typical day rather than the typical second | 1.4× |

The daily fits resolve the spread. Eight of the nine stress days fit at half-lives between
14 and 67 seconds, inside the pre-stress range of 10 to 144 seconds. The ninth is 31 July,
the session following the policy announcement at 30 July 23:30 in the data's clock, at
1,003 seconds: seven times slower than the next slowest day in the sample and fifteen times
slower than the next slowest stress day. The pooled stress mean of 155 seconds is above
every stress day except that one.

Across the conventions in `03_sensitivity.csv` the ratio spans 1.5× to 24.8×. The direction
is stable across all of them. The magnitude is not determined by the data alone.

The supportable statement is that one day was extraordinary. A uniformly slower regime is
not supported.

Files: `03_persistence.csv`, `03_closure_speed.csv`, `03_closure_speed.png`

---

## Method

### Basis and cointegration

`z = AJ − AU × UJ` in AUD/JPY pips. In logs, with `x` the vector of scaled log prices and
`w = (−1, −1, +1)`, the basis is `w'x`. Triangular parity is a cointegrating restriction
with a known vector, so the rank and the vector are both fixed by arithmetic. The Johansen
procedure is not required and is not used.

### Episodes

Contiguous stretches with |z| above 2.42 pips, the July 99.95% quantile. Runs are split
wherever consecutive rows are more than one grid step apart, so an episode cannot span a
market closure. Without that guard, exceedances either side of a weekend merge into a
single 48-hour episode. 58 episodes are detected.

### Changepoint model

Each hour is summarised by the median absolute deviation of the basis on a log scale. The
basis has excess kurtosis of 277, so the hourly standard deviation is driven by a small
number of seconds and the MAD is not; the log makes a shift multiplicative.

The diurnal profile and a session-reopen ramp are subtracted, both estimated over the full
sample. The diurnal amplitude is roughly fourfold peak to trough. The reopen term matters
because a weekend closure is followed by a thin Sunday evening, so the first buckets of each
week sit systematically high, and a regime ending during a closure would otherwise be
recorded a few hours after the reopen.

Segments carry a Normal-Inverse-Gamma prior on (μ, σ²), so each marginal likelihood is
available in closed form. With at most two discrete changepoints the posterior is computed
by enumerating every admissible configuration rather than sampled. There are no chains, no
convergence diagnostics, and the output is identical on every run.

Within-segment errors are AR(1). Whitening uses the one-step-ahead form where the previous
bucket is adjacent in clock time and the stationary marginal where it is not, which keeps
the likelihood correct across weekly closures. The coefficient is marginalised over a grid
under a uniform prior rather than profiled. The grid includes zero, so the independent-error
model is nested.

No candidate date is supplied to any model. The BOJ decision appears in figures as a
reference line and enters no likelihood.

### Error-correction model

```
Δx_t  =  c  +  α z_{t−1}  +  Σ_{i=1..p} Γ_i Δx_{t−i}  +  e_t
```

fitted separately within each estimated regime at p = 21 seconds. That is the cap of the
tested grid (1, 2, 3, 5, 8, 13, 21); BIC is still improving at the cap.
`03_sensitivity.csv` refits at p = 1 and p = 5.

Since `z_t = z_{t−1} + w'Δx_t` is an identity, the basis closure rate is `λ = w'α`,
obtained from the system without a separate regression.

Two half-lives are reported. The one-step figure uses λ alone and describes the first
second. The full-system figure runs a unit dislocation through the companion matrix of the
state `(z_t, Δx_t, …, Δx_{t−p+1})`, whose first row is the identity above. Its spectral
radius is the stationarity check the error-correction reading requires; observed values are
0.9889, 0.9957 and 0.9841.

The three components of α are estimated far less precisely than their combination, because
each leg's own volatility is orders of magnitude larger than the basis while λ is a
combination in which the common moves cancel. The closure-rate claim is therefore stronger
than the leg-split claim, and the tables do not present them as equally reliable.

### Standard errors

Increments carry first-order autocorrelation between −0.30 and −0.42 from bid-ask bounce,
so OLS standard errors are wrong by a non-negligible factor. Standard errors are Newey-West.

The full 3K × 3K HAC matrix is not formed. Only the error-correction coefficient is of
interest, and for a single linear functional the sandwich collapses to the long-run
covariance of one scalar series per equation. That covariance is computed from overlapping
block sums, which reproduces the Bartlett kernel exactly up to the first and last m−1 rows
of each contiguous run. Blocks that would straddle a break are refused.

### Row admission

A lag window is admitted only where the whole span is contiguous in clock time, outside
rollover, inside one regime and one trading day, and labelled with regime posterior at
least 90%. Nothing is differenced across a weekend, a closure, a rollover window or a
regime boundary. The row set is fixed once at the lag cap and reused for every lag order
and every regime.

The error-correction term is centred on the day. Centring on the regime leaves drift in the
daily mean inside the residual, where the estimator reads it as persistence.

### Why no Hasbrouck or Gonzalo-Granger shares

Those measures are defined for two prices of one asset: one cointegrating vector, one
common trend, and loadings that split into shares summing to one. This system has three
prices and one restriction, so it has two common trends and α's orthogonal complement is
3 × 2 rather than a vector. No scalar share per leg is identified without a normalisation
the data does not supply. The weak-exogeneity tests are the part of that framework this
system supports.

---

## Limitations

**Two days fail the data-quality gate, and the defect is not resolvable with free data.**
Covered in Section 2, including which estimates are affected and which are not.

**The magnitude of the closure slowdown is not determined by the data alone.** Covered in
Section 6.

**A one-second grid cannot establish simultaneity.** Triangular parity requires three legs
at once. The grid records the last print of each leg inside a second, not three prices known
to have stood together. A computed positive edge is therefore an upper bound and not a
measurement. When prices travel far enough inside a second, the separation between prints
produces an apparent edge. No finer data exists for these pairs from either vendor.

Two diagnostics bear on the 51,280 positive-edge stress seconds. The count is flat in
execution delay, 0.1277 at zero delay against 0.1277 at five seconds, whereas a real
short-lived opportunity would decay. It is also not concentrated in the fastest quintiles of
within-second travel, which is the pattern non-simultaneity produces. Both point away from
reading the count as opportunity. Files: `05_latency.csv`, `05_edge_velocity.csv`

**Censoring at the filter boundary.** Episode detection excludes 17:00 to 17:30, so the
detector sees a session opening at 17:30:00 and closing at 16:59:59. Nine of the twenty
widest episodes start in hour 17, which holds 2.1% of admissible seconds, at 21.5 times the
expected rate. Two begin exactly at 17:30:00 and three end exactly at 16:59:59, against an
expectation of 0.0002 for each. These are censored observations rather than starts and ends,
and the widest episode in the sample, 8.911 pips on 4 August, is one of them. Moving the
boundary to 18:00 leaves the selected segmentation unchanged. File: `06_boundary_audit.csv`

**Scope.** One event, two months, one currency triangle. The results describe a single
episode and are not distributional claims about FX stress in general.

---

## Data

Primary source is HistData tick exports for the three pairs.

**Gap repair.** The HistData USD/JPY export is missing four hours on 5 August 2024, which
contain the peak of the carry unwind. Forward-filling across them produces a spurious 60-pip
dislocation. Those hours are refilled from Dukascopy tick exports. HistData sources from
Dukascopy, and over a control window the two agree to 0.01 pips at correlation 1.00000, so
this recovers dropped data rather than blending two liquidity pools. Every row records which
source its price came from. The same shared lineage is why Dukascopy cannot be used as an
independent check on the 1 to 2 August level in Section 2.

**Clock.** Vendor timestamps are documented as EST, but the sample lies entirely inside US
daylight saving, so the operative offset is UTC−4. The Friday-to-Sunday week boundary does
not establish this, since FX closes at 17:00 New York local year-round under either offset.
The offset was determined by scanning candidate offsets against UTC-stamped Dukascopy ticks
over a control window: at −4h the correlation is 1.00000 with a gap standard deviation of
0.01 pips, at −5h it is −0.71 and 48.65 pips.

**Quote pairing.** Bid and ask are carried through the grid beside the mid and are always
taken from a single tick. Three `ARG_MAX` calls over one ordering key resolve to the same
winning row, so the bid and ask columns are null in identical rows and the forward fill
cannot pair a bid from one second with an ask from another. Taking max(ask) with min(bid)
across a second would produce a spread no participant was shown. Section 4 depends on this
pairing, so it is asserted after the splice.

**Closures.** A second is classified as closed if it sits inside a run of at least ten
minutes in which none of the three pairs printed. No trading calendar is consulted. Eight
closures are found, all 48 hours to within two minutes. About 3.86 million open-market
seconds remain.

---

## Running it

```bash
pip install -r requirements.txt
make all
```

Make targets are files rather than phony aliases, so a stage is skipped when its inputs
have not changed. Editing `03_var_analysis.py` rebuilds 03, 04, 06 and the PDF, and leaves
01, 02, 05 and 07 alone. Editing `utils.py` rebuilds everything downstream of the tick
parse.

| Target | Does |
|---|---|
| `make` | print the target list |
| `make all` | gate, analysis, report |
| `make data` | the 1-second grid only |
| `make check` | the data-quality gate; the exit code propagates, for CI |
| `make analysis` | 01 to 07, skipping the gate |
| `make report` | compile the PDF |
| `make test` | pytest |
| `make fast` | analysis with 02's null calibration switched off |
| `make clean` | remove generated output, keep the parsed grid |

**The gate fails on this sample.** `00b_data_quality.py` exits non-zero because two days
carry a triangular bias the triangle cannot attribute to a leg, which is Section 2 above.
`make all` stops there and says so. To build anyway, recording that the failure is known:

```bash
make all ACCEPT_KNOWN_BIAS=1
```

The stages in order, for running them by hand:

```bash
python scripts/00_ingest_and_sync.py       # tick parse, gap repair, 1s grid
python scripts/00b_data_quality.py         # gate; returns 1 on this sample, see Section 2
python scripts/01_exploratory_analysis.py  # closures, moments, episodes
python scripts/02_changepoint_detection.py
python scripts/03_var_analysis.py
python scripts/04_granger_causality.py
python scripts/05_execution_bounds.py
python scripts/06_reconciliation.py output/tables output/tables
python scripts/07_boundary_sweep.py        # re-aggregates 01 under every boundary
pytest tests -q
```

Null calibration in `02` is roughly 70% of that script's runtime and nothing downstream
depends on it. `make fast` sets `CP_NULL_DRAWS=0` to skip it while iterating; a reported
figure is never produced that way.

Every table is written twice, as CSV and as LaTeX, from the same frame in the same call. No
number is typed into the report by hand, and a rebuilt table rebuilds the PDF.

### Tests

18 tests in `test_granger.py`. `make_synthetic.py` builds a series with a known lead-lag
structure. A Granger test is a claim about direction, and the only way to verify an
implementation gets direction right is to build data where the direction is known in
advance.

The generator injects a lead from AUD/USD into the basis and nothing in the reverse
directions, then adds the features the real data has: Roll noise producing first-order
autocorrelation near −0.35, forward-filled quotes, a wandering daily mean, weekend closures,
and a mid-week single-leg outage. The tests assert that the estimator finds the injected
lead at the right lag, ranks the legs correctly, clears its own placebo, finds nothing when
nothing is injected, and refuses to run without its upstream inputs.

---

## Repository

Most of this tree is not in git. Vendor ticks cannot be redistributed, and everything under
`output/` is reproducible, so a fresh clone contains the code and the report and little
else. `make all` fills in the rest.

```
.
├── Makefile
├── requirements.txt
├── scripts/
│   ├── utils.py                      shared constants, palette, episode and grid helpers
│   ├── 00_ingest_and_sync.py
│   ├── 00b_data_quality.py
│   ├── 01_exploratory_analysis.py
│   ├── 02_changepoint_detection.py
│   ├── 03_var_analysis.py
│   ├── 04_granger_causality.py
│   ├── 05_execution_bounds.py
│   ├── 06_reconciliation.py
│   └── 07_boundary_sweep.py
├── tests/
│   ├── test_granger.py               18 tests
│   └── make_synthetic.py             a series with a known lead-lag structure
├── data/                             not in git: vendor terms
│   ├── raw/histdata/                 DAT_ASCII_{PAIR}_T_YYYYMM.csv
│   ├── raw/dukascopy/                {PAIR}_1Tick_{BID,ASK}_*.csv
│   └── processed/
│       └── synchronized_rates.parquet
├── output/                           not in git: rebuilt by make
│   ├── data/                         intermediate parquet passed between stages
│   ├── tables/                       *.csv
│   └── figures/                      *.png
└── report/
    ├── main.tex
    ├── tables/                       *.tex, written by utils.save_table
    └── main.pdf                      built by make report
```

| Stage | Script | Question |
|---|---|---|
| 00 | `00_ingest_and_sync.py` | Tick parse, gap repair, splice, 1s grid |
| 00b | `00b_data_quality.py` | Continuity, coverage, outages, reconciliation, bias |
| 01 | `01_exploratory_analysis.py` | Closures, moments, microstructure, episodes |
| 02 | `02_changepoint_detection.py` | When the properties of the basis changed |
| 03 | `03_var_analysis.py` | How fast the basis closes, and which leg closes it |
| 04 | `04_granger_causality.py` | Which leg opens a dislocation, which leg repairs it |
| 05 | `05_execution_bounds.py` | Band width against basis size |
| 06 | `06_reconciliation.py` | Where the earlier sections disagree |
| 07 | `07_boundary_sweep.py` | Whether the headline turns on the regime boundary |

`Makefile` encodes the dependency graph above, including the two edges that are not obvious
from the run order: `04` depends on `03`'s source file because it loads it by path, and `05`
reads the quoted grid rather than `01`'s output because it needs bid and ask.

Four figures carry most of the argument. `07_boundary_sweep.png` shows the decomposition
holding across every candidate boundary. `03_closure_speed.png` shows the pooled stress mean
sitting above every stress day but one. `02_changepoint_posterior.png` shows the excursion
and the bimodal onset. `05_bands.png` puts the band against the basis on a log axis, where a
parallel shift indicates the band-to-gap ratio did not change.

Figures use four colour roles: three matched hues for the legs, one hue at three lightnesses
for regimes, one accent reserved for estimated quantities, and grey for data excluded from
estimation. Colour is never assigned from the calendar, since a month is an assumption and a
regime label is an estimate.

---

## Summary of what is supported

- Hourly dispersion is best described by a bounded excursion that returns, not a permanent
  shift. Posterior mass 1.000 on two changepoints; observed log Bayes factor +97.1 against a
  null 95th percentile of −2.8.
- Within-hour dispersion rose by 1.21×. The pooled figures of 2.91× and 4.22× are dominated
  by day-to-day movement in the level, which is 80.6% of the stress window's level variance
  against 11.6% in the calm segments.
- That decomposition holds across all 128 boundaries tested, spanning stress windows from
  160 to 354 hours.
- The transaction-cost band exceeded the median basis by 12.7× at its tightest.
- The two dollar legs opened the observed dislocations; the cross leg closed them. Confirmed
  by arithmetic decomposition and by a sign restriction fixed before estimation.
- The basis is close to unforecastable from its own components, while those components are
  highly forecastable from each other.
- 31 July had a closure half-life of 1,003 seconds, seven times the next slowest day.

Not supported: a permanent regime change, a uniformly slower stress regime, a threefold
widening of the intraday basis, or any executable dislocation.
