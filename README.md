# Options Pricing & Delta-Hedging Simulation

This project explores the practical limitations of the Black-Scholes model by simulating dynamic hedging in discrete time. It implements a pricing engine from scratch and a backtester to quantify the variance ("Gamma Risk") introduced when rebalancing is daily rather than continuous.

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

1.  **Market Generation:** Asset price paths are generated using **Geometric Brownian Motion (GBM)**.
2.  **Execution:**
    * $t=0$: Sell ATM Call, Buy $\Delta_0$ shares.
    * $t=1...T$: Re-calculate $\Delta$, buy/sell shares to re-hedge.
3.  **Attribution:** The PnL of the total portfolio is tracked to measure the slippage caused by discrete rebalancing.

## Results

The simulation demonstrates that while Delta Hedging significantly reduces directional risk compared to a naked position, it does not eliminate it entirely. The residual variance in the PnL path represents **Gamma Risk**—the impact of large price moves occurring between the daily rebalancing intervals.

## Dependencies
* Python 3.8+
* NumPy (Vectorization)
* SciPy (Statistical functions)
* Matplotlib / Seaborn (Visualization)