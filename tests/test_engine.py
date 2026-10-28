import numpy as np
import pandas as pd
from backtest.engine import tsmom_weights, run, stats


def synthetic(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = rng.normal(0.0003, 0.01, (n, 3))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=list("ABC"))


def test_no_lookahead():
    """Changing the final day's price must not change any earlier weight."""
    px = synthetic()
    w1 = tsmom_weights(px)
    px2 = px.copy(); px2.iloc[-1] *= 5
    w2 = tsmom_weights(px2)
    pd.testing.assert_frame_equal(w1.iloc[:-1], w2.iloc[:-1])


def test_weights_lag_one_day():
    """First non-NaN weights must appear only after the lookback window has filled."""
    px = synthetic()
    w = tsmom_weights(px, lookback_m=12).dropna()
    assert w.index[0] > px.index[252]


def test_leverage_cap_respected():
    w = tsmom_weights(synthetic(), vol_target=0.5, max_gross=2.0).dropna()
    assert (w.abs().sum(axis=1) <= 2.0 + 1e-9).all()


def test_costs_reduce_returns():
    px = synthetic(); w = tsmom_weights(px)
    free, _ = run(px, w, cost_bps=0)
    paid, turn = run(px, w, cost_bps=20)
    assert paid.sum() < free.sum() and turn.sum() > 0


def test_stats_on_known_series():
    r = pd.Series([0.01, -0.01] * 126)
    s = stats(r)
    assert abs(s["ann_vol"] - 0.01 * np.sqrt(252) * np.sqrt(252 / 251)) < 1e-2 and s["max_drawdown"] < 0
