"""Reproduces every number in the README. Run: python run_experiments.py (downloads data via yfinance)."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backtest.engine import (load_prices, tsmom_weights, run, stats, walk_forward,
                             block_bootstrap_sharpe_diff)

px = load_prices()
rets = px.pct_change().dropna()
res = {"data": dict(start=str(px.index[0].date()), end=str(px.index[-1].date()), days=len(px), assets=list(px.columns))}

net, turn = run(px, tsmom_weights(px, 12), cost_bps=5)
bench_ew = rets.mean(axis=1)                              # equal-weight, daily rebalanced
bench_6040 = 0.6 * rets["SPY"] + 0.4 * rets["IEF"]
common = net.dropna().index
curves = {"TSMOM 12-1 (net 5bps)": net.loc[common], "Equal-weight buy&hold": bench_ew.loc[common],
          "60/40 SPY/IEF": bench_6040.loc[common]}
res["full_sample"] = {k: {a: float(b) for a, b in stats(v).items()} for k, v in curves.items()}
res["annual_turnover_x"] = float(turn.loc[common].sum() / (len(common) / 252))

# cost sensitivity
res["sharpe_by_cost_bps"] = {c: float(stats(run(px, tsmom_weights(px, 12), cost_bps=c)[0].loc[common])["sharpe"])
                             for c in [0, 5, 10, 25, 50]}
# lookback sensitivity (all reported, not just the best)
res["sharpe_by_lookback_m"] = {lb: float(stats(run(px, tsmom_weights(px, lb), 5)[0].loc[common])["sharpe"])
                               for lb in [3, 6, 9, 12]}
# sub-periods
sub = {}
for name, (a, b) in {"2008-2012": ("2008", "2012"), "2013-2019": ("2013", "2019"), "2020-2025": ("2020", "2025")}.items():
    s = net.loc[a:b]
    sub[name] = dict(tsmom_sharpe=float(stats(s)["sharpe"]), ew_sharpe=float(stats(bench_ew.loc[s.index])["sharpe"]),
                     tsmom_maxdd=float(stats(s)["max_drawdown"]), ew_maxdd=float(stats(bench_ew.loc[s.index])["max_drawdown"]))
res["subperiods"] = sub

# walk-forward lookback selection (out-of-sample)
wf, chosen, nets = walk_forward(px)
res["walk_forward"] = dict(oos_stats={a: float(b) for a, b in stats(wf).items()},
                           ew_same_period_sharpe=float(stats(bench_ew.loc[wf.index])["sharpe"]),
                           lookback_chosen_by_year=chosen)
# bootstrap: is TSMOM Sharpe different from equal-weight?
ci, p_le0 = block_bootstrap_sharpe_diff(curves["TSMOM 12-1 (net 5bps)"], curves["Equal-weight buy&hold"])
res["bootstrap_sharpe_diff_tsmom_minus_ew"] = dict(ci95=[float(ci[0]), float(ci[1])], frac_bootstrap_le_zero=p_le0)
json.dump(res, open("results/results.json", "w"), indent=2, default=str)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for k, v in curves.items():
    ax[0].plot((1 + v).cumprod(), label=k)
ax[0].set(title="Growth of $1", yscale="log"); ax[0].legend()
for k, v in curves.items():
    eq = (1 + v).cumprod(); ax[1].plot(eq / eq.cummax() - 1, label=k)
ax[1].set(title="Drawdown"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/summary.png", dpi=130)
print(json.dumps(res, indent=2, default=str))
