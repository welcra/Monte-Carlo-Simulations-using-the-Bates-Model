# Monte-Carlo-Simulations-using-the-Bates-Model

First, using current options data for a stock, a Bates model is calibrated. Then, using those parameters, price paths are generated, offering much more accuracy with stochastic volatility and jump diffusion compared to a simple GBM. The monte carlo simulation of price paths leads to generation of risk management metrics.

Why is this useful: The bates model is an extension of the heston model, an extension of the black-scholes model. It performs well in practice, which is useful (of course). Additionally, this performs better than other price paths generators because of its calibration using options data. By calibrating the Bates model to live options data, the simulation is forward-looking (market-implied), instead of backward-looking (historical returns).

I created this to help my team in the Wharton Investment Competition
