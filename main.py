import warnings
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import calibration
from datetime import datetime, date
import QuantLib as ql

TICKER_SYMBOL = "AAPL"

MIN_VOLUME_CALIBRATION = 20
MIN_VOLUME_ANALYSIS = 10
STRIKE_RANGE_FACTOR = 0.20
MISPRICING_THRESHOLD_PCT = 10
MIN_MARKET_PRICE = 0.05

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.10f}".format)
pd.options.display.max_rows = 100

stock, S, q, calculation_date_ql = calibration.get_market_data(TICKER_SYMBOL)
risk_free_ts = calibration.get_yield_curve(calculation_date_ql)

risk_free_ts.enableExtrapolation()

target_date = date.today() + relativedelta(months=+6)

date_format = "%Y-%m-%d"

closest_date = min(stock.options, key=lambda d: abs(datetime.strptime(d, date_format).date() - target_date))

closest_index = stock.options.index(closest_date)

expiration_date_str = stock.options[closest_index]
exp_date_obj = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
exp_date_ql = ql.Date(exp_date_obj.day, exp_date_obj.month, exp_date_obj.year)

day_count = ql.Actual365Fixed()
T_years = day_count.yearFraction(calculation_date_ql, exp_date_ql)
T_days = (exp_date_obj - calculation_date_ql.to_date()).days

opt_chain = stock.option_chain(expiration_date_str)
calls = opt_chain.calls.assign(option_type="call")
puts = opt_chain.puts.assign(option_type="put")
full_chain = pd.concat([calls, puts])
full_chain["market_price"] = (full_chain["bid"] + full_chain["ask"]) / 2
full_chain["T_days"] = T_days
full_chain = full_chain[full_chain["market_price"] > 0].copy()

strike_min = S * (1 - STRIKE_RANGE_FACTOR)
strike_max = S * (1 + STRIKE_RANGE_FACTOR)

calibration_data = full_chain[
    (full_chain["volume"] >= MIN_VOLUME_CALIBRATION) &
    (full_chain["strike"] >= strike_min) & 
    (full_chain["strike"] <= strike_max)
].copy()

if len(calibration_data) < 10:
    print(f"Warning: Insufficient liquid market data for a reliable calibration: {TICKER_SYMBOL}")
calibrated_params, final_error = calibration.calibrate_heston(S, T_years, risk_free_ts, q, calibration_data, calculation_date_ql)

n_steps = T_days
n_sims = 10000
beta = stock.info.get("beta", 1.0)
equity_risk_premium = 0.055
r_long_term = risk_free_ts.zeroRate(10.0, ql.Continuous).rate()
mu = r_long_term + beta * equity_risk_premium

T_years = 0.5

simulated_paths = calibration.generate_bates_paths(S, T_years, mu, q, calibrated_params, n_steps, n_sims)

plt.figure(figsize=(10, 6))
plt.plot(simulated_paths[:, :10])
plt.title(f"Price Paths {TICKER_SYMBOL}")
plt.xlabel("T")
plt.ylabel("Price")
plt.grid(True)

risk_metrics = calibration.calculate_risk_metrics(simulated_paths, S, T_years, risk_free_ts)
    
print(f"\n--- Comprehensive Risk & Performance Analysis ---")
print(f"Horizon: {T_days} days | Simulations: {n_sims} | Ticker: {TICKER_SYMBOL}")

print("\n## Performance Ratios")
print("-" * 35)
print(f"{"Sharpe Ratio":<25}: {risk_metrics["Sharpe Ratio"]:.3f}")
print(f"{"Sortino Ratio":<25}: {risk_metrics["Sortino Ratio"]:.3f}")
print(f"{"Profit Factor":<25}: {risk_metrics["Profit Factor"]:.3f}")

print("\n## Return & Volatility Profile")
print("-" * 35)
print(f"{"Expected Return (Annual.)":<25}: {risk_metrics["Expected Return (Annualized)"]:.2%}")
print(f"{"Volatility (Annual.)":<25}: {risk_metrics["Volatility (Annualized)"]:.2%}")
print(f"{"Downside Vol. (Annual.)":<25}: {risk_metrics["Downside Volatility (Annualized)"]:.2%}")

print("\n## Probabilistic Metrics")
print("-" * 35)
print(f"{"Probability of Profit":<25}: {risk_metrics["Probability of Profit"]:.2%}")
print(f"{"Prob. of >20% Loss":<25}: {risk_metrics["Prob. of >20% Loss"]:.2%}")

print("\n## Tail Risk & Drawdown")
print("-" * 35)
print(f"{"VaR 95%":<25}: {risk_metrics["VaR_95%"]:.2%}")
print(f"{"CVaR 95%":<25}: {risk_metrics["CVaR_95%"]:.2%}")
print(f"{"Average Max Drawdown":<25}: {risk_metrics["Average Max Drawdown"]:.2%}")
print(f"{"Worst Max Drawdown":<25}: {risk_metrics["Worst Max Drawdown"]:.2%}")

print("\n## Return Distribution")
print("-" * 35)
print(f"{"Skewness":<25}: {risk_metrics["Skewness"]:.3f}")
print(f"{"Kurtosis":<25}: {risk_metrics["Kurtosis"]:.3f}")

plt.show()