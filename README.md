# Options Pricing & Delta-Hedging Simulation

> **Status: 2025 teaching script, superseded.** This is a single-path
> (seed 42) daily delta-hedging walkthrough of one short ATM call under GBM.
> The multi-path, cost-aware version with expiry settlement, bootstrap
> standard errors and real-price replays lives in
> [neural-options-lab](https://github.com/Ronak-Mahajan/neural-options-lab):
> the hedging module
> [`backend/quant/hedging.py`](https://github.com/Ronak-Mahajan/neural-options-lab/blob/main/backend/quant/hedging.py)
> and the SPY/BTC replay script
> [`scripts/hedge_real_paths.py`](https://github.com/Ronak-Mahajan/neural-options-lab/blob/main/scripts/hedge_real_paths.py).
> This repo is kept as a readable 170-line reference and is not developed further.

This script illustrates the practical limitations of the Black-Scholes model by simulating dynamic hedging in discrete time: it implements the pricing formula, generates one geometric-Brownian-motion path, and tracks the P&L of a daily-rebalanced delta hedge of a short call through settlement at expiry.

## Mathematical Framework

The simulator calculates European Call prices using the standard Black-Scholes-Merton formula:

$$C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)$$

Where:
* $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$
* $d_2 = d_1 - \sigma\sqrt{T-t}$

### Dynamic Hedging
The strategy maintains a **Delta Neutral** portfolio by holding $\Delta$ shares of the underlying asset against a short option position:

$$\Pi_t = \text{Cash}_t + \Delta_t S_t - C(S_t, t)$$

In theory (continuous time), $d\Pi = r\Pi dt$ (risk-free growth). In this simulation (discrete daily steps), we observe $d\Pi \neq r\Pi dt$ due to Gamma exposure between rebalance points.

## Simulation Methodology

1.  **Market Generation:** One asset price path is generated using **Geometric Brownian Motion (GBM)** with a fixed seed (`seed=42`), $S_0 = K = 100$, $T = 1$, $r = 5\%$, $\sigma = 20\%$, 252 daily steps.
2.  **Execution:**
    * $t=0$: Sell ATM Call, Buy $\Delta_0$ shares.
    * $t=1 \ldots T$: accrue interest on cash, re-calculate $\Delta$, buy/sell shares to re-hedge.
    * $t=T$: the option is settled against its payoff $\max(S_T - K, 0)$ and the hedge closed at $\Delta_T = \mathbf{1}\{S_T > K\}$.
3.  **Tracking:** The total portfolio value is recorded at every step.

## What the script reports (and what it does not)

The script prints two numbers for the single simulated path:

* **Terminal hedging P&L** after settlement at expiry.
* **Standard deviation of the cumulative-P&L time series along that one path.**
  This is a time-series statistic of a single realisation. It is *not* the
  cross-path standard deviation of terminal hedging error, which is the
  quantity usually meant by "gamma risk" from discrete rebalancing and which
  requires a Monte Carlo over many paths (and, to compare rebalancing
  frequencies, a sweep over step counts). Neither exists in this script.

No unhedged (naked) benchmark, transaction costs, rebalancing-frequency
sweep or vol-misspecification study is implemented here; all of those are in
the neural-options-lab hedging module linked above.

`hedging_results.png` was produced by the original version of the script,
which stopped one step before expiry; rerun `python options_hedging.py` to
regenerate it with settlement included.

## Running

```bash
pip install -r requirements.txt
python options_hedging.py
```

## Dependencies
* Python 3.9+
* NumPy, SciPy, Matplotlib, Seaborn (unpinned in `requirements.txt`; the
  original exact pins did not install on Python 3.12+).
