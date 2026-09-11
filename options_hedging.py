import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import seaborn as sns

# Set visualization style
plt.style.use('dark_background')
sns.set_palette("husl")

class BlackScholesModel:
    """
    Black-Scholes option pricing engine.
    Implemented from scratch using standard normal distribution.
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = S        # Underlying Price
        self.K = K        # Strike Price
        self.T = T        # Time to Maturity (years)
        self.r = r        # Risk-free Rate
        self.sigma = sigma # Volatility (Annualized)

    def d1(self, t: float = 0):
        time_left = self.T - t
        if time_left <= 0: return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * time_left) / (self.sigma * np.sqrt(time_left))

    def d2(self, t: float = 0):
        time_left = self.T - t
        if time_left <= 0: return 0
        return self.d1(t) - self.sigma * np.sqrt(time_left)

    def call_price(self, t: float = 0):
        time_left = self.T - t
        if time_left <= 0: return max(self.S - self.K, 0)

        d1 = self.d1(t)
        d2 = self.d2(t)

        return (self.S * si.norm.cdf(d1, 0.0, 1.0) -
                self.K * np.exp(-self.r * time_left) * si.norm.cdf(d2, 0.0, 1.0))

    def delta(self, t: float = 0):
        """Delta: Sensitivity to underlying price dC/dS"""
        time_left = self.T - t
        if time_left <= 0:
            # At expiry the call is worth max(S - K, 0), so delta is 1{S > K}.
            # d1() returns 0 here, which would otherwise give 0.5 for any moneyness.
            return 1.0 if self.S > self.K else 0.0
        return si.norm.cdf(self.d1(t), 0.0, 1.0)

    def gamma(self, t: float = 0):
        """Gamma: Sensitivity of Delta to underlying price d2C/dS2"""
        time_left = self.T - t
        if time_left <= 0: return 0
        return si.norm.pdf(self.d1(t), 0.0, 1.0) / (self.S * self.sigma * np.sqrt(time_left))

def simulate_gbm_path(S0, mu, sigma, T, steps, seed=42):
    """Generate a stock price path using Geometric Brownian Motion."""
    np.random.seed(seed)
    dt = T / steps
    t = np.linspace(0, T, steps + 1)

    # Brownian increments
    W = np.random.standard_normal(size=steps)
    W = np.cumsum(W) * np.sqrt(dt)

    # Geometric Brownian Motion dynamics
    # S_t = S_0 * exp((mu - 0.5*sigma^2)t + sigma*W_t)
    # W[i-1] is the Brownian motion at t[i], so the drift is evaluated at t[1:].
    X = (mu - 0.5 * sigma ** 2) * t[1:] + sigma * W
    S = np.concatenate(([S0], S0 * np.exp(X)))

    return t, S

def run_hedging_simulation(S_path, t_steps, K, T, r, sigma):
    """
    Simulates a Delta-Neutral hedging strategy over the life of the option.
    Rebalances the hedge ratio at every time step (daily) and settles the
    option against its payoff at expiry.
    """
    dt = T / (len(t_steps) - 1)
    portfolio_values = []

    # Initial Setup
    S0 = S_path[0]
    bs = BlackScholesModel(S0, K, T, r, sigma)

    # Short 1 Call Option
    option_premium = bs.call_price(0)
    cash = option_premium

    # Long Delta Shares
    shares = bs.delta(0)
    cash -= shares * S0 # Pay for shares

    portfolio_values.append(cash + (shares * S0) - option_premium)

    # Simulation Loop. Runs through i = len(S_path) - 1, where t_curr == T:
    # there call_price() returns the payoff max(S_T - K, 0) and delta()
    # returns 1{S_T > K}, so the hedge is settled against the payoff.
    # (The original loop broke at time_left <= 1e-5 and never reached expiry.)
    for i in range(1, len(S_path)):
        S_curr = S_path[i]
        t_curr = t_steps[i]

        # Accrue Interest on the cash held over [t_{i-1}, t_i], before the
        # rebalance trade at t_i.
        cash *= np.exp(r * dt)

        # Update Market Data
        bs.S = S_curr

        # Mark to Market
        option_value = bs.call_price(t_curr)
        current_delta = bs.delta(t_curr)

        # Rebalance Hedge (Discrete hedging)
        # Buy/Sell shares to match new delta
        share_change = current_delta - shares
        cash -= share_change * S_curr
        shares = current_delta

        # Calculate PnL
        # Portfolio = Cash + Stock Value - Option Liability
        port_val = cash + (shares * S_curr) - option_value
        portfolio_values.append(port_val)

    return portfolio_values

def main():
    # Parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    steps = 252 # Daily rebalancing

    print("Running Options Delta-Hedging Simulation...")
    print(f"Parameters: S0={S0}, K={K}, Vol={sigma}, Rate={r}, Steps={steps}")

    # 1. Simulate Market
    t_grid, S_path = simulate_gbm_path(S0, 0.1, sigma, T, steps)

    # 2. Run Hedging
    pnl_path = run_hedging_simulation(S_path, t_grid, K, T, r, sigma)

    # 3. Analysis
    # This is ONE simulated path with a fixed seed. final_pnl is the terminal
    # hedging error of that path after settlement at expiry. pnl_std is the
    # standard deviation of the cumulative-PnL time series ALONG that single
    # path; it is NOT the cross-path dispersion of terminal hedging error,
    # which is the usual "gamma risk" statistic and needs many paths
    # (see neural-options-lab/backend/quant/hedging.py for that).
    final_pnl = pnl_path[-1]
    pnl_std = np.std(pnl_path)

    print("\nResults (single GBM path, seed=42):")
    print(f"Terminal hedging PnL after settlement at expiry: ${final_pnl:.2f}")
    print(f"Std dev of the PnL time series along this one path: ${pnl_std:.2f}")
    print("Note: the std above is a time-series statistic of a single path,")
    print("      not the cross-path standard deviation of terminal hedging error.")

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t_grid, S_path, color='#4ECDC4', label='Underlying Asset')
    ax1.axhline(K, color='white', linestyle='--', alpha=0.5, label='Strike')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Simulated Asset Price Path (GBM)')
    ax1.legend()
    ax1.grid(True, alpha=0.1)

    # pnl_path now has one entry per grid point (settlement step included).
    ax2.plot(t_grid, pnl_path, color='#FF6B6B', label='Hedged Portfolio PnL')
    ax2.axhline(0, color='white', linestyle='-', alpha=0.3)
    ax2.set_ylabel('PnL ($)')
    ax2.set_xlabel('Time (Years)')
    ax2.set_title('Delta Hedging Performance (one path)')
    ax2.legend()
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig('hedging_results.png')
    plt.show()

if __name__ == "__main__":
    main()
