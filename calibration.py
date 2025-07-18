import yfinance as yf
import numpy as np
import QuantLib as ql
from scipy.optimize import minimize
from scipy import stats

def heston_price_quantlib(S, K, T_years, r, q, v0, kappa, theta, sigma_v, rho, jump_lambda, jump_mean, jump_vol, calculation_date_ql, option_type="call"):
    ql.Settings.instance().evaluationDate = calculation_date_ql
    day_count = ql.Actual365Fixed()
    
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, K)
    
    exercise_date = calculation_date_ql + ql.Period(int(T_years * 365), ql.Days)
    exercise = ql.AmericanExercise(calculation_date_ql, exercise_date)
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, r, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date_ql, q, day_count))
    
    bates_process = ql.BatesProcess(risk_free_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma_v, rho, jump_lambda, jump_mean, jump_vol)
    
    model = ql.BatesModel(bates_process)
    engine = ql.FdHestonVanillaEngine(model, 101, 201, 51)
    
    option_instrument = ql.VanillaOption(payoff, exercise)
    option_instrument.setPricingEngine(engine)
    
    return option_instrument.NPV()

def get_market_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="5d")
    if hist.empty:
        raise ValueError("Could not fetch stock history for symbol.")
    
    S = hist["Close"].iloc[-1]
    calculation_date_dt = hist.index[-1].to_pydatetime()
    calculation_date_ql = ql.Date(calculation_date_dt.day, calculation_date_dt.month, calculation_date_dt.year)
    
    q = stock.info.get("dividendYield", 0.0)
    if q is None: q = 0.0

    return stock, S, q, calculation_date_ql

def get_yield_curve(calculation_date_ql):
    tsy_maturities = [
        ("^IRX", 13, ql.Weeks),
        ("^FVX", 5, ql.Years),
        ("^TNX", 10, ql.Years)
    ]

    spot_dates = []
    spot_rates = []

    for ticker, value, unit in tsy_maturities:
        try:
            hist = yf.Ticker(ticker).history(period="1mo")
            rate = hist["Close"].iloc[-1] / 100
            
            if rate <= 0: rate = 1e-9

            spot_dates.append(calculation_date_ql + ql.Period(value, unit))
            spot_rates.append(rate)
        except IndexError:
            print(f"Warning: Could not fetch rate for {ticker}. Skipping.")
    
    if not spot_rates:
        raise RuntimeError("Could not fetch any rates to build the yield curve.")

    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    return ql.YieldTermStructureHandle(
        ql.LogLinearZeroCurve(spot_dates, spot_rates, day_count, calendar)
    )

def calibration_error_function(params, S, T_years, risk_free_ts, dividend_q, market_data, calc_date_ql, otk="option_type"):
    v0, kappa, theta, sigma_v, rho, jump_lambda, jump_mean, jump_vol = params
    
    if v0 <= 1e-4 or kappa <= 1e-4 or theta <= 1e-4 or sigma_v <= 1e-4 or not -1 < rho < 1 or jump_lambda < 0 or jump_vol <= 1e-4:
        return 1e9
    if 2 * kappa * theta <= sigma_v**2:
        return 1e9
        
    squared_errors = []
    for _, row in market_data.iterrows():
        r_T = risk_free_ts.zeroRate(row["T_days"] / 365.0, ql.Continuous).rate()
        
        model_price = heston_price_quantlib(
            S, row["strike"], T_years, r_T, dividend_q, 
            v0, kappa, theta, sigma_v, rho,
            jump_lambda, jump_mean, jump_vol,
            calc_date_ql,
            option_type=row[otk]
        )
        squared_errors.append((model_price - row["market_price"])**2)
            
    return np.sqrt(np.mean(squared_errors))

def calibrate_heston(S, T_years, risk_free_ts, q, market_data, calc_date_ql, otk="option_type"):
    initial_params = [0.05, 1.5, 0.05, 0.3, -0.6, 0.1, 0.0, 0.2]
    bounds = [(1e-5, 1.0), (1e-3, 10.0), (1e-5, 1.0), (1e-3, 2.0), (-0.99, 0.99), (1e-3, 1.0), (-0.5, 0.5), (1e-3, 1.0)]
    
    result = minimize(
        calibration_error_function,
        initial_params,
        args=(S, T_years, risk_free_ts, q, market_data, calc_date_ql, otk),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 300, "disp": False, "ftol": 1e-6}
    )
    
    final_error = result.fun

    v0, kappa, theta, sigma_v, rho, jump_lambda, jump_mean, jump_vol = result.x
    return {"v0": v0, "kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "lambda": jump_lambda, "jump_mean": jump_mean, "jump_vol": jump_vol}, final_error

def generate_bates_paths(S, T, mu, q, params, n_steps, n_sims):
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma_v = params["sigma_v"]
    rho = params["rho"]
    jump_lambda = params["lambda"]
    jump_mean = params["jump_mean"]
    jump_vol = params["jump_vol"]

    dt = T / n_steps
    S_paths = np.zeros((n_steps + 1, n_sims))
    v_paths = np.zeros((n_steps + 1, n_sims))

    S_paths[0, :] = S
    v_paths[0, :] = v0

    Z1 = np.random.normal(size=(n_steps, n_sims))
    Z2 = np.random.normal(size=(n_steps, n_sims))
    W_s = Z1
    W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    J = np.random.poisson(jump_lambda * dt, size=(n_steps, n_sims))
    M = np.random.normal(jump_mean, jump_vol, size=(n_steps, n_sims))
    jump_comp = J * M

    for t in range(1, n_steps + 1):
        v_paths[t, :] = np.maximum(v_paths[t-1, :] + kappa * (theta - v_paths[t-1, :]) * dt +
                                 sigma_v * np.sqrt(v_paths[t-1, :]) * np.sqrt(dt) * W_v[t-1, :], 0)
        
        variance_term = 0.5 * v_paths[t-1, :]

        jump_compensator = jump_lambda * (np.exp(jump_mean + 0.5 * jump_vol**2) - 1)

        drift = (mu - q - variance_term - jump_compensator) * dt
        diffusion = np.sqrt(v_paths[t-1, :]) * np.sqrt(dt) * W_s[t-1, :]
        S_paths[t, :] = S_paths[t-1, :] * np.exp(drift + diffusion + jump_comp[t-1, :])

    return S_paths

def calculate_risk_metrics(simulated_paths, S, T_years, risk_free_ts, var_levels=[0.95, 0.99]):
    final_prices = simulated_paths[-1, :]
    returns = (final_prices - S) / S
    total_sims = len(returns)
    
    r_annual = risk_free_ts.zeroRate(T_years, ql.Continuous).rate()
    
    metrics = {}
    
    metrics["Expected Return (Period)"] = np.mean(returns)
    metrics["Expected Return (Annualized)"] = (1 + np.mean(returns))**(1/T_years) - 1
    metrics["Median Return (Period)"] = np.median(returns)
    metrics["Probability of Profit"] = np.sum(returns > 0) / total_sims
    metrics["Prob. of >20% Loss"] = np.sum(returns < -0.20) / total_sims

    annualized_vol = np.std(returns) / np.sqrt(T_years)
    metrics["Volatility (Annualized)"] = annualized_vol
    metrics["Skewness"] = stats.skew(returns)
    metrics["Kurtosis"] = stats.kurtosis(returns)

    risk_free_return_period = np.exp(r_annual * T_years) - 1
    downside_returns = returns[returns < risk_free_return_period]
    if len(downside_returns) > 1:
        downside_deviation_period = np.sqrt(np.mean((downside_returns - risk_free_return_period)**2))
        annualized_downside_vol = downside_deviation_period / np.sqrt(T_years)
    else:
        annualized_downside_vol = 0
    metrics["Downside Volatility (Annualized)"] = annualized_downside_vol

    for level in var_levels:
        metrics[f"VaR_{int(level*100)}%"] = -np.percentile(returns, (1 - level) * 100)
        cvar_returns = returns[returns <= -metrics[f"VaR_{int(level*100)}%"]]
        if cvar_returns.size > 0:
            metrics[f"CVaR_{int(level*100)}%"] = -np.mean(cvar_returns)
        else:
            metrics[f"CVaR_{int(level*100)}%"] = metrics[f"VaR_{int(level*100)}%"]

    annualized_return = metrics["Expected Return (Annualized)"]
    metrics["Sharpe Ratio"] = (annualized_return - r_annual) / annualized_vol if annualized_vol > 0 else 0
    metrics["Sortino Ratio"] = (annualized_return - r_annual) / annualized_downside_vol if annualized_downside_vol > 0 else np.inf
    
    gross_gains = np.sum(returns[returns > 0])
    gross_losses = np.abs(np.sum(returns[returns < 0]))
    metrics["Profit Factor"] = gross_gains / gross_losses if gross_losses > 0 else np.inf

    drawdowns = []
    for i in range(simulated_paths.shape[1]):
        path = simulated_paths[:, i]
        running_max = np.maximum.accumulate(path)
        path_drawdown = (path - running_max) / running_max
        drawdowns.append(np.min(path_drawdown))
        
    metrics["Average Max Drawdown"] = np.mean(drawdowns)
    metrics["Worst Max Drawdown"] = np.min(drawdowns)
    
    return metrics