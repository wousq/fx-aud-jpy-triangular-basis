"""
test_granger_causality.py

Tests for scripts/04_granger_causality.py.

There is one real dataset, and a bug that lives in a code path it never
takes is a bug no amount of running the pipeline will find. On top of that, a
causality test is a claim about *direction*, and there is no way to know an
implementation gets direction right by running it on data whose direction is
unknown. So the tests below fall into three groups:

    recovery    a series with a lead injected into one leg in one regime. The
                script has to find it, in that leg, in that regime, and at
                roughly that lag.
    calibration the same series with the lead removed and nothing else
                changed. The script has to find nothing. This is the test that
                caught the placebo overclaiming, and it is the one to run
                first when anything here changes.
    branches    two segments instead of three, and each upstream input
                missing in turn. None of these occurs on the real sample,
                which is exactly why they are here.

The pipeline tests run the real entry point against a temporary tree laid out
like the repository, so utils.ROOT resolves inside it and nothing writes to
output/. They are slow by the standards of a unit test and quick by the
standards of the thing they are testing; the unit tests above them are
instant and are where a failure should be diagnosed from.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
NEEDED = ("utils.py", "03_var_analysis.py", "04_granger_causality.py")

# Small enough to run in the time a test should take, large enough that every
# floor in the script is cleared and no regime is skipped for being short.
DAYS = 12
STRESS_FROM, STRESS_TO = 5, 9
SMALL_DAYS = 8
SMALL_STRESS = (3, 6)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tree(tmp_path_factory):
    """A repository-shaped tree with the scripts in it, and 04 imported."""
    root = tmp_path_factory.mktemp("fx")
    (root / "scripts").mkdir()
    for sub in ("output/figures", "output/tables", "output/data",
                "report/tables"):
        (root / sub).mkdir(parents=True)
    for name in NEEDED:
        shutil.copy(SCRIPTS / name, root / "scripts" / name)
    shutil.copy(Path(__file__).with_name("make_synthetic.py"),
                root / "scripts" / "make_synthetic.py")

    sys.path.insert(0, str(root / "scripts"))
    gen = _load(root / "scripts" / "make_synthetic.py", "make_synthetic")
    g = _load(root / "scripts" / "04_granger.py", "granger04")

    # Trimmed so the suite runs in a minute rather than ten. Two placebo
    # offsets still give the maximum something to be a maximum of, and the lag
    # grid only matters for the branch where 03 has not run.
    g.PLACEBO_OFFSETS = (-2, 2)
    g.LAG_GRID = (1, 2, 3)
    g.LAG_CAP = 3
    g.SAMPLE_STEPS = (5, 15)
    return {"root": root, "g": g, "gen": gen}


def write_inputs(tree, *, days=DAYS, stress=(STRESS_FROM, STRESS_TO),
                 segments=3, null=False, episodes=True, var_model=True,
                 seed=11, start="2024-07-15"):
    """Lay down the upstream inputs, optionally the fitted model, then clear."""
    g, gen = tree["g"], tree["gen"]
    dat = g.DAT_DIR
    for old in list(dat.glob("*.parquet")):
        old.unlink()
    for folder in (g.TAB_DIR, g.TEX_DIR):
        for old in folder.glob("04_*"):
            old.unlink()

    d, truth = gen.build(start, days, stress[0], stress[1], seed, null=null)
    roll = gen.in_rollover(d.index)
    calm = d.loc[d.index < pd.Timestamp(start) + pd.Timedelta(days=stress[0])]
    threshold = calm.loc[~roll.reindex(calm.index).to_numpy(),
                         "basis"].abs().quantile(gen.THRESH_Q)
    ep = gen.episodes(d["basis"], threshold, gen.MERGE_GAP, gen.MIN_SECONDS,
                      exclude=roll)
    reg = gen.write_regimes(d, stress[0], stress[1], start, segments)

    d.to_parquet(dat / "01_clean.parquet")
    reg.to_parquet(dat / "02_regimes.parquet")
    if episodes:
        ep.to_parquet(dat / "01_episodes.parquet")
    if var_model:
        # Only the lag order is ever read out of this file, so a stub with the
        # right term names is a faithful stand-in for 03 having run.
        terms = ["const", "z(-1)"] + [f"d AUD/USD (-{i})" for i in (1, 2, 3)]
        pd.DataFrame({"regime": "pre", "equation": "AUD/USD", "term": terms,
                      "coef": 0.0}).to_parquet(dat / "03_var_model.parquet",
                                               index=False)
    return truth


def read_tests(g):
    return pd.read_parquet(g.DAT_DIR / "04_granger_tests.parquet")


def read_lead_lag(g):
    return pd.read_csv(g.TAB_DIR / "04_lead_lag.csv")


def asym_at(ll, regime, leg, horizon=1):
    row = ll[(ll["regime"] == regime) & (ll["leads"] == leg)
             & (ll["follows"] == "basis") & (ll["horizon (s)"] == horizon)]
    assert len(row) == 1, f"no asymmetry row for {regime}/{leg}"
    return float(row["asymmetry"].iloc[0])


def band(tests, regime):
    """
    2/sqrt(n) for a regime, the reference the script itself reports against.

    Expressed in these units rather than in absolute correlation because the
    branch datasets are a third the length of the recovery one, and a fixed
    cutoff that suits one is either vacuous or false on the other.
    """
    n = tests[tests["regime"] == regime]["n"]
    assert len(n), f"no fitted rows recorded for {regime}"
    return 2.0 / np.sqrt(float(n.iloc[0]))


# --------------------------------------------------------------- unit tests

def test_design_reproduces_03(tree):
    """
    The extended design, cut back to its first 2+3p columns, must be 03's.

    This is the assertion that lets the script claim it tests the model the
    error-correction stage fitted. It is checked inside the script's own audit
    as well; having it here too means a failure names the function rather than
    the pipeline.
    """
    g = tree["g"]
    rng = np.random.default_rng(0)
    n, p = 200, 3
    DX = rng.normal(size=(n, 3))
    rows = np.arange(p + 1, n, dtype=np.int64)
    zc = rng.normal(size=rows.size)
    y_off = rng.normal(size=(rows.size, 3))

    Xa, Ya = g.design(rows, DX, DX, p, zc, y_off, ext=False)
    Xb, Yb = g.V._design(rows, DX, p, zc, y_off)
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(Ya, Yb)

    # And the extended block really is the basis increment at each lag.
    Xe, _ = g.design(rows, DX, DX, p, zc, y_off)
    for i in range(1, p + 1):
        assert np.allclose(Xe[:, 2 + 3 * (i - 1):2 + 3 * i] @ g.W,
                           Xe[:, 2 + 3 * p + (i - 1)])


def test_cross_corr_finds_a_known_lead(tree):
    """b leads a by three seconds, and nothing else."""
    g = tree["g"]
    rng = np.random.default_rng(1)
    n = 40_000
    b = rng.normal(size=n)
    a = np.empty(n)
    a[:3] = rng.normal(size=3)
    a[3:] = 0.8 * b[:-3] + 0.6 * rng.normal(size=n - 3)
    run = np.zeros(n, dtype=np.int64)

    lags = np.arange(-10, 11)
    cc = g.cross_corr(a, b, run, lags)
    assert int(lags[int(np.nanargmax(np.abs(cc)))]) == 3
    # The mirror image must be empty: a does not lead b.
    assert abs(cc[int(np.flatnonzero(lags == -3)[0])]) < 0.05


def test_cross_corr_will_not_pair_across_a_break(tree):
    """
    Two runs whose junction would manufacture a correlation.

    The second run is the negation of the first, so a pair formed across the
    boundary is strongly correlated while no pair inside either run is. If the
    run guard is dropped the lag-1 correlation moves; with it, nothing does.
    """
    g = tree["g"]
    rng = np.random.default_rng(2)
    half = 5_000
    x = rng.normal(size=half)
    a = np.concatenate([x, -x])
    b = np.concatenate([x, -x])
    run = np.concatenate([np.zeros(half, dtype=np.int64),
                          np.ones(half, dtype=np.int64)])
    lags = np.array([1])
    guarded = g.cross_corr(a, b, run, lags)[0]
    unguarded = g.cross_corr(a, b, np.zeros(2 * half, dtype=np.int64),
                             lags)[0]
    assert np.isfinite(guarded)
    # The junction is one pair in ten thousand, so the two differ only a
    # little; what matters is that the guarded version excludes it at all.
    assert guarded != unguarded


def test_block_test_matches_a_brute_force_refit(tree):
    """
    The effect size is computed from accumulated cross-products rather than by
    refitting, because refitting a million rows nine times is not free. That
    shortcut is exact, and this is where that is checked: against an explicit
    two-model OLS on a design small enough to hold whole.
    """
    g = tree["g"]
    rng = np.random.default_rng(3)
    n, K = 4_000, 8
    X = rng.normal(size=(n, K))
    X[:, 0] = 1.0
    beta = rng.normal(size=K)
    y = X @ beta + rng.normal(size=n)
    Y = np.column_stack([y, rng.normal(size=n), rng.normal(size=n)])

    XtX, XtY, YtY = X.T @ X, X.T @ Y, Y.T @ Y
    w = np.array([1.0, 0.0, 0.0])
    cols = np.arange(K)
    block = np.array([3, 4])

    got = g.block_test("a", "b", "test", XtX, XtY @ w, float(w @ YtY @ w),
                       cols, block, np.eye(block.size),
                       np.zeros(K), n, K)

    keep = np.setdiff1d(cols, block)
    rss_u = float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
    Xr = X[:, keep]
    rss_r = float(np.sum((y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]) ** 2))
    assert got["gain_pips"] == pytest.approx(
        np.sqrt((rss_r - rss_u) / (n - K)), rel=1e-9)
    assert got["delta_r2"] == pytest.approx(
        (rss_r - rss_u) / float(y @ y), rel=1e-9)
    assert got["df"] == 2


def test_holm_is_monotone_and_bounded(tree):
    g = tree["g"]
    p = np.array([0.001, 0.02, 0.5, 0.9])
    adj = g.holm(p)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj <= 1.0)
    assert np.all(adj >= p - 1e-12)
    assert np.isnan(g.holm(np.array([np.nan]))[0])


def test_the_cross_check_against_03_actually_fires(tree):
    """
    A check nobody has watched fail is a check nobody has tested.

    `check_against_03` compares every fitted coefficient against the ones 03
    wrote out. On the real sample it passes, which is the outcome that proves
    nothing: a checker that returns an empty list unconditionally would look
    exactly the same. So one coefficient is moved by a part in ten thousand
    and the checker has to notice.
    """
    g = tree["g"]
    write_inputs(tree, days=SMALL_DAYS, stress=SMALL_STRESS)

    p_lags = 3
    terms = ["const", "z(-1)"] + [f"d {g.leg_label(j)} (-{i})"
                                  for i in range(1, p_lags + 1)
                                  for j in range(3)]
    rows = [{"regime": "pre", "equation": g.leg_label(j), "term": t,
             "coef": 0.25} for j in range(3) for t in terms]
    ref = pd.DataFrame(rows)
    ref.to_parquet(g.DAT_DIR / "03_var_model.parquet", index=False)

    class Fit:
        p = p_lags
        fcols = np.arange(2 + 3 * p_lags)
        coef = np.full((2 + 3 * p_lags, 3), 0.25)

    fits = {"pre": Fit()}
    assert g.check_against_03(fits, ["pre"]) == [], "clean fit was flagged"

    Fit.coef = Fit.coef.copy()
    Fit.coef[1, 2] *= 1.0001
    problems = g.check_against_03(fits, ["pre"])
    assert len(problems) == 1 and "differ from 03" in problems[0], problems

    # And a lag order that does not match 03's has to be named, not ignored.
    class Wrong(Fit):
        p = p_lags + 1
        fcols = np.arange(2 + 3 * (p_lags + 1))
        coef = np.full((2 + 3 * (p_lags + 1), 3), 0.25)

    assert "lag order" in g.check_against_03({"pre": Wrong()}, ["pre"])[0]

    # A file too thin to have come from a completed 03 run is a note, not a
    # disagreement: it says nothing about whether this script fitted the
    # right model, and treating it as evidence would make the harness's own
    # lag-order stub look like a failure of the pipeline.
    thin = ref[ref["equation"] == g.leg_label(0)]
    thin.to_parquet(g.DAT_DIR / "03_var_model.parquet", index=False)
    assert g.check_against_03(fits, ["pre"]) == []


def test_attribution_windows_mirror_each_other(tree, injected):
    """
    Each episode should contribute an opening and a closing window spanning
    the same half of the same peak. A window running from the episode's first
    second instead is long enough for the legs' own moves to swamp the gap,
    and the shares then leave the unit interval; the travel diagnostic is what
    makes that visible.
    """
    g = tree["g"]
    att = pd.read_csv(g.TAB_DIR / "04_attribution.csv", index_col=0)
    counts = att.loc["episodes"].astype(str).str.replace(",", "").astype(int)
    for regime in {c.split(",")[0] for c in att.columns}:
        opening = counts[f"{regime}, opening"]
        closing = counts[f"{regime}, closing"]
        assert abs(opening - closing) <= max(2, 0.2 * closing), (
            f"{regime}: {opening} opening windows against {closing} closing")
    travel = pd.to_numeric(att.loc["median leg travel per pip of gap"],
                           errors="coerce").dropna()
    assert (travel < 3.0).all(), (
        f"legs travelled {travel.max():.1f}x the gap; shares are ratios of "
        f"offsetting numbers, not an attribution")


# ------------------------------------------------------------ recovery

@pytest.fixture(scope="session")
def injected(tree):
    truth = write_inputs(tree)
    tree["g"].main()
    return {"truth": truth, "tests": read_tests(tree["g"]),
            "ll": read_lead_lag(tree["g"])}


def test_finds_the_injected_lead_in_the_right_place(tree, injected):
    """
    AUD/USD was made to lead the basis by one second, in the stressed regime
    only. That is the whole truth of the series, and all three of the script's
    routes have to agree on it.
    """
    ll, t = injected["ll"], injected["tests"]
    stress = asym_at(ll, "stress", "AUD/USD")
    # Five, against the three the null test allows. On these two datasets the
    # measured separation is roughly eight against roughly one, so the gap is
    # wide; the thresholds sit inside it rather than on the observed values,
    # so a change to the harness that halved the effect would still pass and a
    # change that removed it would not.
    assert stress > 5 * band(t, "stress"), (
        f"the injected lead was not recovered ({stress})")
    for regime in ("pre", "post"):
        quiet = asym_at(ll, regime, "AUD/USD")
        assert abs(quiet) < 3 * band(t, regime), (
            f"a lead was found in {regime}, where none was injected ({quiet})")
    # USD/JPY was given no lead over the basis in any regime.
    for regime in ("pre", "stress", "post"):
        assert abs(asym_at(ll, regime, "USD/JPY")) < 3 * band(t, regime)


def test_the_effect_size_ranks_the_legs_correctly(tree, injected):
    t = injected["tests"]
    pair = t[(t["family"] == "measured") & (t["to"] == "basis")
             & (t["conditioning"] == "basis memory")]
    stress = pair[pair["regime"] == "stress"].set_index("from")["gain_pips"]
    pre = pair[pair["regime"] == "pre"].set_index("from")["gain_pips"]
    assert stress["AUD/USD"] == max(stress), "the driving leg is not the largest"
    assert stress["AUD/USD"] > 3 * pre["AUD/USD"], (
        "the lead does not separate the stressed regime from the calm one")


def test_measured_effect_clears_its_own_placebo(tree, injected):
    t = injected["tests"]
    key = ((t["to"] == "basis") & (t["conditioning"] == "basis memory")
           & (t["regime"] == "stress") & (t["from"] == "AUD/USD"))
    measured = float(t[key & (t["family"] == "measured")]["gain_pips"].iloc[0])
    placebo = t[key & (t["family"] == "placebo")]["gain_pips"]
    assert len(placebo) >= 1, "the placebo produced no draws at all"
    assert measured > 5 * float(placebo.max())


def test_attribution_shares_sum_to_one(tree, injected):
    """The decomposition is an identity, so this cannot be approximately true."""
    g = tree["g"]
    att = pd.read_csv(g.TAB_DIR / "04_attribution.csv", index_col=0)
    rows = [i for i in att.index if str(i).startswith("share, ")]
    assert rows, "the attribution table carries no shares"
    for column in att.columns:
        values = pd.to_numeric(att.loc[rows, column], errors="coerce").dropna()
        if len(values) == 3:
            assert float(values.sum()) == pytest.approx(1.0, abs=2e-3)


def test_every_output_is_labelled_04(tree, injected):
    g = tree["g"]
    import utils
    produced = (list(g.TAB_DIR.glob("*.csv")) + list(g.TEX_DIR.glob("*.tex"))
                + list(utils.FIG_DIR.glob("*.png"))
                + list(g.DAT_DIR.glob("04*.parquet")))
    mine = [f for f in produced if f.name.startswith("04_")]
    assert len(mine) >= 10, "04 wrote fewer outputs than it claims to"
    for path in g.DAT_DIR.glob("04*"):
        assert path.name.startswith("04_")


def test_no_generated_table_reaches_the_report_with_a_nan(tree, injected):
    """
    The audit inside the script raises on this, so reaching here means it
    passed; the assertion is repeated because it is the failure that would
    otherwise be found by a reader of the compiled PDF.
    """
    g = tree["g"]
    for path in list(g.TEX_DIR.glob("04_*.tex")) + list(
            g.TAB_DIR.glob("04_*.csv")):
        text = path.read_text(encoding="utf-8").lower()
        assert " nan" not in text and ",nan" not in text, path.name


# --------------------------------------------------------- calibration

def test_finds_nothing_when_nothing_is_there(tree):
    """
    The same series with the lead removed and everything else - the regimes,
    the volatility clustering, the forward filling, the wandering daily mean -
    left in place.

    This is the test that changed the script. Before it, every block cleared
    its placebo in every regime and the tables said "yes"; running it showed
    the same "yes" on a series with no lead in it, which is how the placebo
    came to be described as bounding sampling variability rather than
    bounding a finding.
    """
    g = tree["g"]
    write_inputs(tree, null=True)
    g.main()
    ll = read_lead_lag(g)
    t = read_tests(g)
    for regime in ("pre", "stress", "post"):
        for leg in ("AUD/USD", "USD/JPY", "AUD/JPY"):
            a = asym_at(ll, regime, leg)
            assert abs(a) < 3 * band(t, regime), (
                f"{leg} appears to lead the basis in {regime} ({a}) on a "
                f"series built with no lead in it")

    pair = t[(t["family"] == "measured") & (t["to"] == "basis")
             & (t["conditioning"] == "basis memory")
             & (t["regime"] == "stress")]
    measured = pair.set_index("from")["gain_pips"]
    placebo = (t[(t["family"] == "placebo") & (t["to"] == "basis")
                 & (t["conditioning"] == "basis memory")
                 & (t["regime"] == "stress")]
               .groupby("from")["gain_pips"].max())
    # Some excess over the placebo is expected and is not a finding: the basis
    # carries each leg's own quotation noise. What must not happen is the
    # ten-to-fifteen-fold excess the injected series produces.
    for leg in measured.index:
        assert measured[leg] < 6 * placebo[leg], (
            f"{leg} reads {measured[leg] / placebo[leg]:.1f}x its placebo on a "
            f"null series; the threshold is not calibrated")


# ------------------------------------------------------------- branches

def test_two_segments_instead_of_three(tree):
    """
    The number of regimes is estimated, not given, and the winning model won
    narrowly. Had it gone the other way every downstream script would meet a
    shape it had never been run on.
    """
    g = tree["g"]
    write_inputs(tree, days=SMALL_DAYS, stress=SMALL_STRESS, segments=2)
    g.main()
    ll = read_lead_lag(g)
    assert set(ll["regime"]) <= {"calm", "stress"}, set(ll["regime"])
    assert "stress" in set(ll["regime"])
    t = read_tests(g)
    assert len(t) > 0
    assert (g.TAB_DIR / "04_basis_causality.csv").exists()


def test_runs_without_an_episode_file_or_03s_output(tree):
    """
    Degraded inputs. The attribution needs the episode file and the lag order
    prefers the fitted model file; neither is required, and the script has to
    say so rather than fail on an absent path.
    """
    g = tree["g"]
    write_inputs(tree, days=SMALL_DAYS, stress=SMALL_STRESS, episodes=False,
                 var_model=False)
    g.main()
    assert not (g.TAB_DIR / "04_attribution.csv").exists(), (
        "an attribution table was written with no episodes to attribute")
    assert (g.TAB_DIR / "04_basis_causality.csv").exists()
    assert (g.DAT_DIR / "04_granger_tests.parquet").exists()


def test_refuses_to_run_without_02(tree):
    """
    The regime labels are this script's only source of a split, and inventing
    one from the calendar would defeat the point of estimating them. The
    failure has to name that rather than surface as a missing-column error.
    """
    g = tree["g"]
    write_inputs(tree, days=SMALL_DAYS, stress=SMALL_STRESS)
    (g.DAT_DIR / "02_regimes.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="02"):
        g.main()


def test_refuses_to_run_without_01(tree):
    g = tree["g"]
    write_inputs(tree, days=SMALL_DAYS, stress=SMALL_STRESS)
    (g.DAT_DIR / "01_clean.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="01"):
        g.main()