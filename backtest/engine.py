"""Time-series momentum (trend following) backtest on a multi-asset ETF universe.

No look-ahead: signals use data through day t-1 and are applied to day t returns.
Positions: sign of trailing (lookback - skip) return, scaled to a per-asset volatility target,
rebalanced monthly, with proportional transaction costs on turnover.
"""
import numpy as np
import pandas as pd

TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]


def load_prices(start="2007-01-01", end="2026-01-01"):
    import yfinance as yf
    px = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    return px.dropna(how="any")


def tsmom_weights(px, lookback_m=12, skip_m=1, vol_target=0.10, vol_window=60, max_gross=2.0):
    rets = px.pct_change()
    lb, sk = int(lookback_m * 21), int(skip_m * 21)
    mom = px.shift(sk) / px.shift(lb) - 1
    sig = np.sign(mom)
    vol = rets.rolling(vol_window).std() * np.sqrt(252)
    w = sig * (vol_target / vol) / px.shape[1]
    gross = w.abs().sum(axis=1)
    w = w.div(np.maximum(gross / max_gross, 1.0), axis=0)  # cap leverage
    # monthly rebalance: hold last month-end weights
    month_end = w.groupby([w.index.year, w.index.month]).tail(1).index
    wm = w.loc[month_end].reindex(w.index).ffill()
    return wm.shift(1)  # trade on next day: no look-ahead


def run(px, weights, cost_bps=5.0):
    rets = px.pct_change()
    w = weights.dropna()
    r = rets.loc[w.index]
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    net = gross - turnover * cost_bps / 1e4
    return net, turnover


def stats(r, rf=0.0):
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(ann_return=ann_ret, ann_vol=vol, sharpe=(r.mean() - rf / 252) / r.std() * np.sqrt(252),
                max_drawdown=dd, calmar=ann_ret / abs(dd) if dd < 0 else np.nan)


def block_bootstrap_sharpe_diff(a, b, n=5000, block=21, seed=0):
    """CI for Sharpe(a) - Sharpe(b) using a circular block bootstrap on aligned daily returns."""
    rng = np.random.default_rng(seed)
    df = pd.concat([a, b], axis=1).dropna().values
    T = len(df)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        s = df[idx]
        sh = s.mean(0) / s.std(0, ddof=1) * np.sqrt(252)
        diffs[i] = sh[0] - sh[1]
    return np.percentile(diffs, [2.5, 97.5]), float((diffs <= 0).mean())


def walk_forward(px, lookbacks=(3, 6, 9, 12), train_years=5, cost_bps=5.0):
    """Each calendar year: pick the lookback with best Sharpe over the prior `train_years`, trade it that year."""
    nets = {lb: run(px, tsmom_weights(px, lookback_m=lb), cost_bps)[0] for lb in lookbacks}
    years = sorted(set(px.index.year))
    out, chosen = [], {}
    for y in years:
        past = [nets[lb][(nets[lb].index.year < y) & (nets[lb].index.year >= y - train_years)] for lb in lookbacks]
        if min(len(s.dropna()) for s in past) < 252 * 2:
            continue
        best = lookbacks[int(np.argmax([stats(s)["sharpe"] for s in past]))]
        chosen[y] = best
        out.append(nets[best][nets[best].index.year == y])
    return pd.concat(out), chosen, nets
