"""
make_synthetic.py

Build a synthetic 01/02 output pair with a *known* lead-lag structure, so 04
can be tested against a truth instead of against the one dataset it will ever
see.

A Granger test is a claim about direction, and the only way to know an
implementation gets the direction right is to build a series where the
direction is known in advance. The same generator also forces code paths the
single real dataset never takes.

What is injected
    Two random walks, AUD/USD and USD/JPY, plus a basis that closes at an
    AR(1) rate. The truth has three parts, all switched on only in the middle
    segment:

      kappa   an AUD/USD move at t-1 opens the basis at t. So AUD/USD leads
              the triangle: its lagged increments predict the next move in
              the basis, and - since the direct leg is the sum of the other
              two and the basis - they predict AUD/JPY as well.
      c21     USD/JPY follows AUD/USD with a one-second lag.
      phi     the basis closes slowly under stress and fast either side.

    Nothing is injected in the reverse directions. AUD/JPY's lagged
    increments must not predict AUD/USD, and the basis must not lead
    AUD/USD. A test that finds those has found the estimator, not the market.

What is layered on top, because the real data has it
    Roll noise on every leg, so increments carry rho_1 near -0.35 and a
    Newey-West correction is not optional; forward-filled quotes, so a
    fraction of seconds carry no new price; a wandering daily mean basis,
    which an estimator centred on a whole regime will report as persistence; a mechanically wide rollover window; weekend closures, which
    are removed from the file exactly as 01 removes them; and a mid-week
    single-leg outage, so the gap guards face a hole that is not a weekend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import (  # noqa: E402
    DAT_DIR, PIP, ROLLOVER_START, ROLLOVER_END, episodes, in_rollover,
    make_dirs, robust_scale,
)

# Levels chosen so AUD/JPY sits near 100 and the scaled log basis is the pip
# basis to within a few parts in a thousand, as it is in the real sample.
AUDUSD_0 = 0.6600
USDJPY_0 = 152.00

SIGMA_LEG = 1.7e-5      # log return per second, ~0.5% a day
ROLL_C = 0.45           # Roll noise as a multiple of the efficient sd
QUOTE_P = 0.55          # probability a leg prints a new price in a second

# Common volatility factor: log-AR(1) at one second, with a half-life of about
# twenty minutes. Gives the clustered variance the real sample has.
VOL_RHO = 0.9994
VOL_SD = 0.45

THRESH_Q = 0.9995       # 01's episode threshold quantile
MERGE_GAP = "60s"
MIN_SECONDS = 5


def segment_plan(n_days, stress_from, stress_to):
    """Per-day parameters. Index is the day number, not a date."""
    plan = []
    for i in range(n_days):
        stress = stress_from <= i < stress_to
        plan.append({
            "stress": stress,
            # Basis persistence. 0.80 halves in ~3.1 s, 0.975 in ~27 s, which
            # is the contrast the real sample shows.
            "phi": 0.975 if stress else 0.80,
            "eta": 0.090 if stress else 0.085,
            # The wandering level a regime-wide centring would absorb.
            "level": 0.0,
            # The truth this harness exists to test.
            "kappa": 0.35 if stress else 0.0,
            "c21": 0.25 if stress else 0.0,
        })
    return plan


def build(start="2024-07-15", n_days=21, stress_from=9, stress_to=16, seed=11,
          outage=True, null=False):
    rng = np.random.default_rng(seed)

    t0 = pd.Timestamp(start)
    index = pd.date_range(t0, t0 + pd.Timedelta(days=n_days), freq="1s",
                          inclusive="left")
    n = len(index)
    day_no = (index - t0).days.to_numpy()
    plan = segment_plan(n_days, stress_from, stress_to)
    if null:
        # Everything else identical: the regimes still differ in dispersion and
        # in persistence, and the volatility still clusters. Only the
        # cross-leg lead is removed. This is the series on which 04 must
        # find nothing, and it is the only way to know what "nothing" looks
        # like at a million rows.
        for p in plan:
            p["kappa"] = 0.0
            p["c21"] = 0.0

    # Per-day mean basis. Calm days sit at zero, stressed days are displaced
    # and move about from one day to the next; both are what 03 measured.
    for i, p in enumerate(plan):
        p["level"] = (rng.normal(-0.27, 0.30) if p["stress"]
                      else rng.normal(0.0, 0.05))

    phi = np.array([plan[d]["phi"] for d in day_no])
    eta_sd = np.array([plan[d]["eta"] for d in day_no])
    level = np.array([plan[d]["level"] for d in day_no])
    kappa = np.array([plan[d]["kappa"] for d in day_no])
    c21 = np.array([plan[d]["c21"] for d in day_no])

    # Rollover: liquidity goes, the basis widens mechanically and closes more
    # slowly. Excluded by every estimator, so it is here to be excluded.
    roll = in_rollover(index).to_numpy()
    eta_sd = np.where(roll, eta_sd * 4.0, eta_sd)
    phi = np.where(roll, np.minimum(phi + 0.015, 0.995), phi)

    # --- efficient prices -------------------------------------------------
    #
    # A common, slowly varying volatility factor multiplies every shock in the
    # system. This is not decoration. Volatility clusters in this data, and an
    # estimator that assumes it does not will mistake a run of busy hours for
    # a level shift; clustered variance is exactly what makes a Wald test on a million
    # rows over-reject. A harness without it would report that 04's tests are
    # well sized when they are not.
    lv = np.empty(n)
    lv[0] = 0.0
    shock = rng.normal(0.0, VOL_SD * np.sqrt(1 - VOL_RHO ** 2), n)
    for i in range(1, n):
        lv[i] = VOL_RHO * lv[i - 1] + shock[i]
    vol = np.exp(lv - 0.5 * VOL_SD ** 2)

    e1 = rng.normal(0.0, SIGMA_LEG, n) * vol              # AUD/USD
    e2 = rng.normal(0.0, SIGMA_LEG, n) * vol              # USD/JPY

    # USD/JPY follows AUD/USD by one second, under stress only.
    d2 = e2.copy()
    d2[1:] += c21[1:] * e1[:-1]

    x1 = np.log(AUDUSD_0) + np.cumsum(e1)
    x2 = np.log(USDJPY_0) + np.cumsum(d2)

    # Basis in pips: AR(1) around the day's level, pushed open by yesterday's
    # second of AUD/USD. 1e4 * e1 puts the leg return on the basis's scale.
    eta = rng.normal(0.0, 1.0, n) * eta_sd * vol
    push = np.zeros(n)
    push[1:] = kappa[1:] * (1e4 * e1[:-1])
    z = np.empty(n)
    z[0] = level[0]
    for i in range(1, n):
        z[i] = level[i] + phi[i] * (z[i - 1] - level[i]) + push[i] + eta[i]

    # x3 is the identity: direct = synthetic * exp(basis). Any lead in
    # AUD/USD therefore reaches AUD/JPY through the basis, which is the
    # asymmetry the test has to recover.
    x3 = x1 + x2 + z / 1e4

    # --- observation: Roll noise, then forward-filled quotes ---------------
    def observe(x, key):
        obs = x + rng.normal(0.0, ROLL_C * SIGMA_LEG, n)
        fresh = rng.random(n) < QUOTE_P
        fresh[0] = True
        idx = np.maximum.accumulate(np.where(fresh, np.arange(n), 0))
        return pd.Series(np.exp(obs[idx]), index=index, name=key)

    audusd = observe(x1, "audusd_mid")
    usdjpy = observe(x2, "usdjpy_mid")
    audjpy = observe(x3, "audjpy_direct")

    df = pd.concat([audusd, usdjpy, audjpy], axis=1)

    # --- closures ---------------------------------------------------------
    # FX closes 17:00 Friday to 17:00 Sunday, New York local. 01 removes
    # these rows, so the file 04 reads has holes in it and every adjacency
    # guard has to survive them.
    wd, tod = index.dayofweek, index.hour + index.minute / 60.0
    closed = (((wd == 4) & (tod >= 17)) | (wd == 5)
              | ((wd == 6) & (tod < 17)))

    # A single mid-week hole that is not a weekend: the shape of the 5 August
    # outage, kept short. Guards that only ever meet weekends are guards that
    # have only been tested on weekends.
    if outage:
        hole = ((index >= t0 + pd.Timedelta(days=4, hours=12))
                & (index < t0 + pd.Timedelta(days=4, hours=14)))
        closed = closed | hole

    df = df.loc[~closed].copy()

    df["audjpy_synthetic"] = df["audusd_mid"] * df["usdjpy_mid"]
    df["basis"] = (df["audjpy_direct"] - df["audjpy_synthetic"]) / PIP["AUDJPY"]
    df["x_audusd"] = np.log(df["audusd_mid"])
    df["x_usdjpy"] = np.log(df["usdjpy_mid"])
    df["x_audjpy"] = np.log(df["audjpy_direct"])
    df.index.name = "t"

    truth = {"stress_days": (stress_from, stress_to), "plan": plan,
             "kappa": max(p["kappa"] for p in plan),
             "c21": max(p["c21"] for p in plan)}
    return df, truth


def write_regimes(df, stress_from, stress_to, start, n_segments=3,
                  sharpness=6.0):
    """
    The regime labels, without the changepoint model that produces them.

    Probabilities rather than a hard label, because that is what the
    changepoint stage writes and what everything downstream conditions on. They are made soft near the boundaries so the
    posterior threshold that buffers the changepoints has something to bite
    on; a harness that wrote 1.0 everywhere would never exercise it.
    """
    t0 = pd.Timestamp(start)
    hours = (df.index - t0).total_seconds().to_numpy() / 3600.0
    onset = stress_from * 24.0
    ret = stress_to * 24.0

    def logistic(u):
        return 1.0 / (1.0 + np.exp(-u / sharpness))

    p_stress = logistic(hours - onset) * (1.0 - logistic(hours - ret))
    if n_segments == 2:
        cols = {"p_calm": 1.0 - p_stress, "p_stress": p_stress}
    else:
        rest = 1.0 - p_stress
        w_pre = 1.0 - logistic(hours - (onset + ret) / 2)
        cols = {"p_pre": rest * w_pre, "p_stress": p_stress,
                "p_post": rest * (1.0 - w_pre)}

    out = pd.DataFrame(cols, index=df.index).astype(np.float32)
    out = out.div(out.sum(axis=1), axis=0)
    out.insert(0, "regime", out.to_numpy().argmax(axis=1).astype(np.int8))
    out.index.name = "t"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-15")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--stress-from", type=int, default=9)
    ap.add_argument("--stress-to", type=int, default=16)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-outage", action="store_true")
    ap.add_argument("--null", action="store_true",
                    help="no cross-leg lead; everything else unchanged")
    args = ap.parse_args()

    make_dirs()
    df, truth = build(args.start, args.days, args.stress_from, args.stress_to,
                      args.seed, outage=not args.no_outage, null=args.null)

    roll = in_rollover(df.index)
    calm = df.loc[df.index < pd.Timestamp(args.start)
                  + pd.Timedelta(days=args.stress_from)]
    threshold = calm.loc[~roll.reindex(calm.index).to_numpy(),
                         "basis"].abs().quantile(THRESH_Q)
    ep = episodes(df["basis"], threshold, MERGE_GAP, MIN_SECONDS, exclude=roll)

    reg = write_regimes(df, args.stress_from, args.stress_to, args.start,
                        args.segments)

    df.to_parquet(DAT_DIR / "01_clean.parquet")
    ep.to_parquet(DAT_DIR / "01_episodes.parquet")
    reg.to_parquet(DAT_DIR / "02_regimes.parquet")

    z = 1e4 * (df["x_audjpy"] - df["x_audusd"] - df["x_usdjpy"]).to_numpy()
    print(f"01_clean.parquet   {len(df):,} rows, {df.index[0]} .. "
          f"{df.index[-1]}")
    print(f"01_episodes.parquet {len(ep)} episodes at |basis| > "
          f"{threshold:.2f} pips")
    print(f"02_regimes.parquet  {list(reg.columns)}")
    print(f"basis MAD {robust_scale(df['basis']):.3f} pips; scaled log basis "
          f"is {robust_scale(z) / robust_scale(df['basis']):.4f} x it")
    print(f"injected: kappa {truth['kappa']}, c21 {truth['c21']}, "
          f"stress days {truth['stress_days']}")
    print(f"rollover share {in_rollover(df.index).mean():.2%}")


if __name__ == "__main__":
    main()