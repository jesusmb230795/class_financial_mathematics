# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["legacy"]
# # Modern Portfolio Theory I
#
# Module: Modern Portfolio Theory
#
# ## Lesson summary
#
# This notebook introduces mean-variance portfolio construction through return estimates, covariance, simulated weights, efficient frontiers, and Sharpe-ratio comparisons.
#
# ## Learning objectives
# - Estimate asset returns, volatility, and covariance from market data.
# - Simulate portfolios and visualize the risk-return space.
# - Identify the minimum variance portfolio and efficient frontier.
# - Incorporate the risk-free asset, Sharpe ratio, and tangency portfolio concepts.
#
# ## Core equations
#
# For weights $w$, expected returns $\mu$, and covariance matrix $\Sigma$, portfolio return and variance are:
#
# $$
# \mu_p=w^\top\mu,\qquad \sigma_p^2=w^\top\Sigma w.
# $$
#
# The Sharpe ratio uses the excess return over the risk-free rate:
#
# $$
# S_p=\frac{\mu_p-r_f}{\sigma_p}.
# $$
#
# ## Lesson flow
# 1. Download and inspect price data.
# 2. Estimate return and risk inputs.
# 3. Simulate portfolios and locate efficient allocations.
# 4. Evaluate risk using both portfolio theory and VaR concepts.
#

# %% [markdown] id="LgJsKYlC9v1T" tags=["legacy"]
# ## Setup

# %% executionInfo={"elapsed": 1260, "status": "ok", "timestamp": 1664636683926, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="lOK2aQnk9twr" tags=["setup", "hide-input", "legacy"]
import pandas as pd
import numpy as np
import random
from datetime import date, timedelta


import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from scipy.stats import norm


sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636683926, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="sp_04aj690Mw" tags=["legacy"]
tickers = [
    "MAT",
    "DIS",
    "KO",
    "NVDA",
    "PFE",
    "AAPL",
    "META",
    "TSLA",
    "^GSPC",
    "MSFT",
]
rf = 0.06

yesterday = str(date.today() - timedelta(days=1))
print("Today's date:", yesterday)

# %% [markdown] id="vHJjSwD690pI" tags=["legacy"]
# ## Data

# %% executionInfo={"elapsed": 3636, "status": "ok", "timestamp": 1664636687560, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="x1Xt9xSW91hY" tags=["legacy"]
# %%time
start_date = "2018-01-01"
print(f"The number of stock to download are {len(tickers)}")
all_data = yf.download(tickers, start_date, yesterday)
price_data = all_data[
    "Adj Close"
].copy()  # pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
price_data.info()

# %% colab={"height": 237} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1664636687562, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="MS5QJXltxucY" outputId="55c3f318-2be3-4583-91c6-b72ae0853c73" tags=["legacy"]
price_data.head()

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1664636687562, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="T09HuZPC4OAI" outputId="e0b85a1d-034b-4c19-ef86-2dda7ec9bfcd" tags=["legacy"]
price_data.info()

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636687562, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="ASUc6zBLyRAF" outputId="de84a346-ead7-4784-95f1-fd38fe158b7e" tags=["legacy"]
price_data.isna().sum()

# %% executionInfo={"elapsed": 231, "status": "ok", "timestamp": 1664636687789, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="wf05x3Pj32Nw" outputId="1dcf5db0-0d0b-482d-e2d8-76b1b258b603" tags=["legacy"]
price_data.reset_index().Date.agg([min, max])

# %% colab={"height": 112} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636687789, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="dEi53zUOy80j" outputId="b8f84e2b-841e-4280-8b7e-ea13135efb89" tags=["legacy"]
price_data.reset_index().agg([min, max])

# %% colab={"height": 81} executionInfo={"elapsed": 538, "status": "ok", "timestamp": 1664636688325, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="ULRoXxZeyW_7" outputId="1c0cb02e-bb80-4d08-bd2f-7c252cdb5079" tags=["legacy"]
price_data.agg([min, max]).pct_change().dropna()

# %% tags=["legacy"]
price_data.iloc[[0, -1]]

# %% tags=["legacy"]
100000 * (164.309998 / 21.368668)

# %% colab={"height": 447} executionInfo={"elapsed": 169, "status": "ok", "timestamp": 1664636688493, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="CXMkTedD2Bw6" outputId="3dab8538-b558-4a0d-d8fd-46a41a21fdc1" tags=["legacy"]
price_data.iloc[[0, -1]].pct_change().dropna().reset_index(drop=True).T.sort_values(
    by=0, ascending=False
).plot(kind="bar", legend=False, figsize=(12, 7))
# %% colab={"height": 206} executionInfo={"elapsed": 190, "status": "ok", "timestamp": 1664636688682, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="vQie19bfITzu" outputId="94160858-cf0f-41bc-c48e-bbc8ded76cf2" tags=["legacy"]
price_data.iloc[[0, -1]].pct_change().dropna().reset_index(drop=True).T.sort_values(
    by=0, ascending=False
)

# %% colab={"height": 532} executionInfo={"elapsed": 1159, "status": "ok", "timestamp": 1664636689839, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="O-Kbh2nMzYrA" outputId="1cab8b53-8e0c-4131-e16a-0e70842c0894" tags=["legacy"]
price_data.plot(figsize=(14, 9))

# %% tags=["legacy"]
price_data.drop(columns="^GSPC").plot(figsize=(14, 9))
# %% colab={"height": 532} executionInfo={"elapsed": 830, "status": "ok", "timestamp": 1664636690667, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="0lM2BvQb1yXx" outputId="e8f47aba-fefb-4ee3-e8b8-5150789cb40d" tags=["legacy"]
price_data.MAT.plot(figsize=(14, 9))
# %% colab={"height": 237} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1664636690668, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="7czlYf87xgGA" outputId="92cedabf-82ec-4954-b3db-ee035675bd8f" tags=["legacy"]
df_log = np.log(price_data.pct_change().dropna() + 1)
df_log.head()

# %% colab={"height": 607} executionInfo={"elapsed": 3237, "status": "ok", "timestamp": 1664636693903, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="oRAT1A7U0MXK" outputId="38a054c1-8cff-435a-b995-8bf64746b7f0" tags=["legacy"]
df_log.hist(figsize=(10, 10), bins=25)
# %% colab={"height": 505} executionInfo={"elapsed": 1235, "status": "ok", "timestamp": 1664636695136, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="129YHvIv0SpV" outputId="321b787e-7a26-449a-88c8-a89f19dee146" tags=["legacy"]
df_log.plot(figsize=(18, 9), alpha=0.6)
# %% [markdown] id="Fls9bTxU0-uC" tags=["legacy"]
# ## Estimators

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1664636695137, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="LvlfxGLG0akx" tags=["legacy"]
avg_daily = df_log.mean()
std_daily = df_log.std()
cov_daily = df_log.cov()

avg_ann = avg_daily * 252
std_ann = std_daily * (252 ** (1 / 2))
cov_ann = cov_daily * 252

# %% executionInfo={"elapsed": 188, "status": "ok", "timestamp": 1664636695319, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="QLJsT6WZN8tX" outputId="dae9eb24-9ba8-4c42-b7ce-ee3d5e9fa510" tags=["legacy"]
avg_ann.sort_values()

# %% colab={"height": 284} executionInfo={"elapsed": 508, "status": "ok", "timestamp": 1664636695826, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="Z-kbAl_m1ig-" outputId="e1f94026-9cc2-4f57-a40c-79de26807c6d" tags=["legacy"]
avg_ann.sort_values().plot(kind="bar")
# %% colab={"height": 206} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1664636695826, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="6Hx8u0AB8d2w" outputId="ffbaedfb-79e5-40e0-c5a2-829a19026b86" tags=["legacy"]
cov_ann

# %% tags=["legacy"]
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(df_log.corr(method="pearson"), vmin=-1, vmax=1, annot=True, cmap="rocket_r", ax=ax)
plt.show()

# %% executionInfo={"elapsed": 214, "status": "ok", "timestamp": 1664636696036, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="w5jbS70m8lpt" outputId="3d028cf6-9270-4f60-f9a3-5dec0d01e875" tags=["legacy"]
df_log.var() * 252

# %% colab={"height": 265} executionInfo={"elapsed": 590, "status": "ok", "timestamp": 1664636696625, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="cIl2pGM8_uxc" outputId="f9ecb60b-34a3-4612-f2ec-2fa22492b876" tags=["legacy"]
df_parameters_stocks = pd.concat([avg_ann, (df_log.var() * 252) ** (1 / 2)], axis=1)

df_parameters_stocks.columns = ["return", "volatility"]

df_parameters_stocks["sharpe_ratio"] = (
    df_parameters_stocks["return"] - rf
) / df_parameters_stocks.volatility

df_parameters_stocks["sharpe_ratio"].sort_values().plot(kind="barh")
# %% colab={"height": 265} executionInfo={"elapsed": 406, "status": "ok", "timestamp": 1664636697030, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="lvA574mbAlNy" outputId="d0a98ca8-65b1-4c72-e406-1dbb83b9e1ca" tags=["legacy"]
df_parameters_stocks["return"].sort_values().plot(kind="barh")
# %% colab={"height": 206} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636697031, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="dA6hk86IAxW3" outputId="db6979e8-31cd-42ca-8e23-a8849e653b0e" tags=["legacy"]
df_parameters_stocks

# %% [markdown] id="STVS8Dju5WPw" tags=["legacy"]
# ## Random-weight portfolio search

# %% colab={"height": 242} executionInfo={"elapsed": 4356, "status": "ok", "timestamp": 1664636771507, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="ENjidyed5dIp" outputId="e2936ea5-2eea-43da-a2cf-9ee7a21996da" tags=["legacy"]
# %%time
port_returns = []
port_volatility = []
lst_weights = []

num_assets = len(tickers)
num_portfolios = 50000

for single_portfolio in range(num_portfolios):
    weights = np.array([random.randint(-1000, 1000) for x in range(num_assets)], dtype=float)
    weights = weights / np.sum(weights)

    returns = np.dot(weights, avg_ann)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_ann, weights)))
    if volatility <= 0.9:
        port_returns.append(returns)
        port_volatility.append(volatility)
        lst_weights.append([round(weight, 2) for weight in weights])


df_portfolios = pd.DataFrame(
    {"return": port_returns, "volatility": port_volatility, "weights": lst_weights}
)
df_portfolios.head()


# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1664636771507, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="D0-NjFLYOSRD" outputId="2134d648-ec19-4746-89bb-82cca40f3050" tags=["legacy"]
df_portfolios["return"].agg(["mean", "std"])

# %% colab={"height": 265} executionInfo={"elapsed": 383, "status": "ok", "timestamp": 1664636771888, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="fMOh8e4jOcpX" outputId="57859f2c-1ca2-43ce-a196-426146620826" tags=["legacy"]
df_portfolios["return"].hist(bins=40)
# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636771888, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="o5JUPI25-_Ph" outputId="8140c6d0-44ed-4d69-b58c-15942d8d216c" tags=["legacy"]
df_portfolios.info()

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636771888, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="nfr_HCmEAgXp" outputId="fa195b29-2e39-44df-fe8d-305e8b14f7f2" tags=["legacy"]
avg_ann.head()

# %% executionInfo={"elapsed": 218, "status": "ok", "timestamp": 1664636772104, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="pU_ZhidsAqey" outputId="f1c79ced-e37b-4dc4-985a-68b1982a11fe" tags=["legacy"]
avg_ann.loc["MAT"]

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636772104, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="3iMiokjYAh1H" outputId="0339acc4-3912-4aa1-f5f3-eefbd84f39f1" tags=["legacy"]
std_ann

# %% colab={"height": 513} executionInfo={"elapsed": 287, "status": "ok", "timestamp": 1664636772390, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="bYHfRG29_ODS" outputId="634b6b0f-38c4-424e-c4aa-6cc71ed47f78" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c="r",
    marker="o",
    label="random weights",
    alpha=0.3,
)

for stock in tickers:
    ax1.scatter(std_ann.loc[stock], avg_ann.loc[stock], s=40, marker="x", label=stock)

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
plt.axvline(0.2)
plt.axhline(df_portfolios["return"].max(), color="green")
plt.show()
# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1664636772390, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="lDKrpRkJAKff" outputId="d9c0473e-6457-4ec7-9524-c285de9f8028" tags=["legacy"]
df_portfolios.iloc[df_portfolios["return"].idxmax()]

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636772390, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="ZWOWDzKLQz0g" outputId="8da2206f-1d49-4f9d-fa30-3a16b59c0062" tags=["legacy"]
df_portfolios.iloc[df_portfolios["volatility"].idxmin()]

# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1664636772390, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="KUsFYtWVP9Sn" outputId="8ba79970-17ae-4742-959b-258ebb792843" tags=["legacy"]
tickers


# %% [markdown] id="9roN1RrEXQUZ" tags=["legacy"]
# ## Minimum variance portfolio (MVP)


# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636772675, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="hdokCWr9QtXG" tags=["legacy"]
def get_mvp_portfolio(Cov):
    inv_Cov = np.linalg.inv(Cov)
    M_1 = np.zeros((Cov.shape[0],)) + 1

    num = inv_Cov @ M_1
    den = M_1.T @ inv_Cov @ M_1

    W_mvp = num / den

    return W_mvp


# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636772675, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="RiBIKn2IZknz" outputId="f59f1903-de50-414a-f606-5f3dc5d68c3a" tags=["legacy"]
W_mvp = get_mvp_portfolio(cov_ann)
print(W_mvp)

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636772675, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="XADskkh1Zz4v" outputId="971a46e4-4972-433d-e879-491d7dbdef12" tags=["legacy"]
return_mvp = W_mvp.T @ avg_ann
volatility_mvp = (W_mvp.T @ cov_ann @ W_mvp) ** 0.5

print(f"Return mvp portfolio: {return_mvp}")
print(f"Volatility mvp portfolio: {volatility_mvp}")

# %% colab={"height": 513} executionInfo={"elapsed": 348, "status": "ok", "timestamp": 1664636773021, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="P9IlDbfNZq1Q" outputId="78a31173-5958-4701-e22b-4645a98f47f2" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c="r",
    marker="o",
    label="random weights",
    alpha=0.3,
)

for stock in tickers:
    ax1.scatter(std_ann.loc[stock], avg_ann.loc[stock], s=40, marker="x", label=stock)

ax1.scatter(volatility_mvp, return_mvp, s=40, marker="*", label="MVP Portfolio")

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
plt.axvline(0.2)
plt.axhline(df_portfolios["return"].max(), color="green")
plt.show()
# %% [markdown] id="nm3-eg6N2qab" tags=["legacy"]
# ## Efficient frontier

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636773021, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="Al_RUeV437W3" outputId="6fa55bbd-f030-41c3-e85c-39073147a70b" tags=["legacy"]
np.zeros((len(cov_ann),)) + 1


# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1664636773021, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="SiKdSc7FaxBP" tags=["legacy"]
def get_efficient_frontier(Cov, E, rp):
    """
    Calculate efficient frontier given the following parameters:
      - Cov: Covariance matrix
      - E: Vector of estimated returns
      - rp: Expected return of portfolio
    """
    M_1 = np.zeros((len(Cov),)) + 1
    inv_Cov = np.linalg.inv(Cov)
    A = M_1.T @ inv_Cov @ E
    B = E.T @ inv_Cov @ E
    C = M_1.T @ inv_Cov @ M_1
    D = B * C - A**2

    g = (1 / D) * ((B * inv_Cov @ M_1) - (A * inv_Cov @ E))
    h = (1 / D) * ((C * inv_Cov @ E) - (A * inv_Cov @ M_1))

    W_efficient_frontier = g + (h * rp)

    return W_efficient_frontier


# %% executionInfo={"elapsed": 200, "status": "ok", "timestamp": 1664636773218, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="jZ8x_F0h5sfl" outputId="0ad26d97-7064-45fd-d133-a90475b817e2" tags=["legacy"]
get_efficient_frontier(cov_ann, avg_ann, 0.25)

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1664636773218, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="F8HGfqGo6EuN" outputId="65d71251-32da-4519-d3cd-2eee9f7e6f72" tags=["legacy"]
W_15_mvp = get_efficient_frontier(cov_ann, avg_ann, 0.25)
return_15_mvp = W_15_mvp.T @ avg_ann
volatility_15_mvp = (W_15_mvp.T @ cov_ann @ W_15_mvp) ** 0.5

print(f"Return mvp portfolio: {return_15_mvp}")
print(f"Volatility mvp portfolio: {volatility_15_mvp}")

# %% colab={"height": 513} executionInfo={"elapsed": 500, "status": "ok", "timestamp": 1664636773716, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="E9sUT1U154Cj" outputId="88c1b6f9-45dd-43e6-c1dc-d126265c628f" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)
expected_return = 0.25

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c="r",
    marker="o",
    label="random weights",
    alpha=0.3,
)

for stock in tickers:
    ax1.scatter(std_ann.loc[stock], avg_ann.loc[stock], s=40, marker="x", label=stock)

ax1.scatter(volatility_mvp, return_mvp, s=40, marker="*", label="MVP Portfolio")
ax1.scatter(volatility_15_mvp, return_15_mvp, s=40, marker="*", label=f"EF {expected_return:.0%}")

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
plt.axvline(0.2)
plt.axhline(df_portfolios["return"].max(), color="green")
plt.axhline(0.15, color="yellow")
plt.show()
# %% colab={"height": 206} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1664636773716, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="AlWDwaHV6q6r" outputId="81c593bb-7b3c-4126-e515-f06d6ff5bc36" tags=["legacy"]
n_points = 100
max_return = 1.5


def optimal_weights(n_points, E, cov, r_max):
    w_mvp = get_mvp_portfolio(cov)
    r_min = w_mvp.T @ E
    expected_returns = np.linspace(r_min, r_max, n_points)
    weights = [get_efficient_frontier(cov, E, rp) for rp in expected_returns]
    return weights


weights_ef = optimal_weights(n_points, avg_ann, cov_ann, max_return)
portfolio_returns = []
portfolio_volatility = []
portfolio_weights = []
for w in weights_ef:
    returns = np.dot(w, avg_ann)
    volatility = np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))

    portfolio_returns.append(returns)
    portfolio_volatility.append(volatility)
    portfolio_weights.append([round(x, 2) for x in w])

df_ef = pd.DataFrame(
    {"return": portfolio_returns, "volatility": portfolio_volatility, "weights": portfolio_weights}
)

df_ef.head()


# %% colab={"height": 513} executionInfo={"elapsed": 828, "status": "ok", "timestamp": 1664636774540, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="P-QX-ghR9E_Q" outputId="3a5eed38-2d54-455a-8e30-18b78693a705" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c="r",
    marker="o",
    label="random weights",
    alpha=0.3,
)
ax1.scatter(
    df_ef["volatility"],
    df_ef["return"],
    s=10,
    c="green",
    marker="o",
    label="efficient frontier",
    alpha=0.3,
)

for stock in tickers:
    ax1.scatter(std_ann.loc[stock], avg_ann.loc[stock], s=40, marker="x", label=stock)

ax1.scatter(volatility_mvp, return_mvp, s=40, marker="*", label="MVP Portfolio")
# ax1.scatter(volatility_15_mvp, return_15_mvp, s=40, marker='*', label='EF 15%')

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
# plt.axvline(.2)
# plt.axhline(df_portfolios['return'].max(), color='green')
# plt.axhline(.15, color='yellow')
plt.show()


# %% [markdown] id="b5T7KyMCPYI5" tags=["legacy"]
# ## Efficient portfolio with risk free asset


# %% executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1664636774541, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="_ig4hdBHPXvH" tags=["legacy"]
def get_efficient_portfolio_with_rf(Cov, E, rp, rf):
    """
    Calculate the efficient portfolio with risk free asset given the following parameters:
      - Cov: Covariance matrix
      - E: Vector of estimated returns
      - rp: Expected return of portfolio
      - rf: Risk free asset return
    """

    M_1 = np.zeros(len(Cov)) + 1
    inv_Cov = np.linalg.inv(Cov)

    H = (E - rf).T @ inv_Cov @ (E - rf)

    W = inv_Cov @ (E - rf) * (rp - rf) / H
    Wf = 1 - W.sum()

    return W, Wf


# %% colab={"height": 206} executionInfo={"elapsed": 474, "status": "ok", "timestamp": 1664636775012, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="eVnpxUIjaw_n" outputId="258f5f58-1f17-4775-9911-6afa94a89109" tags=["legacy"]
n_points = 100
min_return = 0.18
max_return = 1.5


def optimal_weights_rf(n_points, E, cov, r_min, r_max, rf):
    expected_returns = np.linspace(r_min, r_max, n_points)
    weights = [get_efficient_portfolio_with_rf(cov, E, rp, rf) for rp in expected_returns]
    return weights


weights_ef_rf = optimal_weights_rf(n_points, avg_ann, cov_ann, min_return, max_return, rf)
portfolio_returns = []
portfolio_volatility = []
portfolio_weights = []
for w, wf in weights_ef_rf:
    returns = np.dot(w, avg_ann) + wf * rf
    volatility = np.sqrt(np.dot(w.T, np.dot(cov_ann, w)))

    portfolio_returns.append(returns)
    portfolio_volatility.append(volatility)
    portfolio_weights.append([round(x, 2) for x in w] + [round(wf, 2)])

df_ef_rf = pd.DataFrame(
    {"return": portfolio_returns, "volatility": portfolio_volatility, "weights": portfolio_weights}
)

df_ef_rf.head()

# %% colab={"height": 513} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1664636775012, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="4sYPFqQy_Zss" outputId="60fc0af3-e441-4c4e-96b8-e7348f34a458" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c="r",
    marker="o",
    label="Simulation",
    alpha=0.3,
)
ax1.scatter(
    df_ef["volatility"],
    df_ef["return"],
    s=10,
    c="green",
    marker="o",
    label="Efficient frontier",
    alpha=0.3,
)
ax1.scatter(
    df_ef_rf["volatility"],
    df_ef_rf["return"],
    s=10,
    c="blue",
    marker="o",
    label="Efficient frontier with risk free asset",
    alpha=0.3,
)

# for stock in tickers:
#   ax1.scatter(std_ann.loc[stock], avg_ann.loc[stock], s=40, marker='x', label=stock)

ax1.scatter(volatility_mvp, return_mvp, s=40, marker="*", label="MVP Portfolio")


plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
plt.show()
# %% [markdown] id="rVJYdYaU9bFX" tags=["legacy"]
# ## Sharpe ratio

# %% colab={"height": 206} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1664636775013, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="20jqipeu9fLI" outputId="9c7dcc3e-f739-4032-9025-f0b4b4ff2ad4" tags=["legacy"]
df_portfolios.head()

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1664636775013, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="bzqe6uUr9fIA" tags=["legacy"]
df_portfolios["sharpe_ratio"] = (df_portfolios["return"] - rf) / df_portfolios.volatility

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1664636775013, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="6Y8SYqNC-7Ep" outputId="a0c0f8a2-f782-4a1d-9892-5076507bd813" tags=["legacy"]
df_portfolios["sharpe_ratio"].agg([min, max])

# %% colab={"height": 283} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1664636775013, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="7cuWszuZ_bJe" outputId="a44ceadf-6ea7-44e9-9283-e0e7b048ea5c" tags=["legacy"]
df_portfolios["return"].hist(bins=25)
# %% colab={"height": 513} executionInfo={"elapsed": 1321, "status": "ok", "timestamp": 1664636776328, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="jKPhzY889fFJ" outputId="8b5f0caf-02b4-4d5f-c345-e47964cec16a" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c=df_portfolios["sharpe_ratio"],
    marker="o",
    label="Simulation",
    alpha=0.3,
)
ax1.scatter(
    df_ef["volatility"],
    df_ef["return"],
    s=10,
    c="red",
    marker="o",
    label="Efficient frontier",
    alpha=0.3,
)
ax1.scatter(
    df_ef_rf["volatility"],
    df_ef_rf["return"],
    s=10,
    c="blue",
    marker="o",
    label="Efficient frontier with risk free asset",
    alpha=0.3,
)

ax1.scatter(volatility_mvp, return_mvp, s=40, marker="*", label="MVP Portfolio")

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Random-weight portfolio search")

plt.legend(loc="upper left")
plt.show()
# %% tags=["legacy"]
df_portfolios.sharpe_ratio.hist(bins=50)


# %% [markdown] id="qleD52fEBGk5" tags=["legacy"]
# ## Tangency portfolio


# %% tags=["legacy"]
def get_tangency_portfolio(covariance_matrix, expected_returns, risk_free_rate):
    # The fully invested tangency portfolio is proportional to
    # Sigma^{-1}(mu-r_f 1) and normalized by 1' Sigma^{-1}(mu-r_f 1).
    inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)

    # Calculate the vector of excess returns
    excess_returns = expected_returns - risk_free_rate

    # Calculate the numerator of the tangency portfolio weights
    numerator = np.dot(inverse_covariance_matrix, excess_returns)

    # Calculate the denominator of the tangency portfolio weights
    denominator = np.ones(len(excess_returns)) @ numerator
    if np.isclose(denominator, 0):
        raise ValueError("tangency portfolio is undefined for these excess returns")

    # Calculate the tangency portfolio weights
    tangency_weights = numerator / denominator

    return tangency_weights


# %% tags=["legacy"]
# W_tg = compute_tangency_portfolio(
#     cov_ann, rf,avg_ann
# )

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1664636776328, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="DF0p5gNg-atj" tags=["legacy"]
# def get_tangency_portfolio(Cov, E, rf):
#     inv_Cov = np.linalg.inv(Cov)
#     M_1 = np.zeros((Cov.shape[0],)) + 1

#     num = inv_Cov @ (E - (rf * (np.zeros((Cov.shape[0],)) + 1) ))
#     den = M_1.T @ inv_Cov @ (E - (rf * (np.zeros((Cov.shape[0],)) + 1) ))

#     W_tg = num /den

#     return W_tg

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1664636776329, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="QkiAnXzvC6ur" tags=["legacy"]
W_tg = get_tangency_portfolio(cov_ann, avg_ann, rf)
W_tg

# %% tags=["legacy"]
np.sum(W_tg)

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1664636776329, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="RGptWWg2CTj7" outputId="21ae8d4d-5bc3-47b7-ad50-4ddfa7816a43" tags=["legacy"]
return_tg = W_tg.T @ avg_ann
volatility_tg = (W_tg.T @ cov_ann @ W_tg) ** 0.5

print(f"Return tangency portfolio: {return_tg}")
print(f"Volatility tangency portfolio: {volatility_tg}")
print(f"Sharpe ratio: {(return_tg - rf) / volatility_tg}")

# %% colab={"height": 513} executionInfo={"elapsed": 1123, "status": "ok", "timestamp": 1664636777449, "user": {"displayName": "Jesus Enrique Miranda Blanco", "userId": "18438776225727683385"}, "user_tz": 300} id="xKUgIJglCYr8" outputId="a11aee4b-0bda-41c2-9750-e341ca67ff4f" tags=["legacy"]
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(111)

ax1.scatter(
    df_portfolios["volatility"],
    df_portfolios["return"],
    s=3,
    c=df_portfolios["sharpe_ratio"],
    marker="o",
    label="Simulation",
    alpha=0.3,
)
ax1.scatter(
    df_ef["volatility"],
    df_ef["return"],
    s=10,
    c="red",
    marker="o",
    label="Efficient frontier",
    alpha=0.3,
)
ax1.scatter(
    df_ef_rf["volatility"],
    df_ef_rf["return"],
    s=10,
    c="blue",
    marker="o",
    label="Efficient frontier with risk free asset",
    alpha=0.3,
)

ax1.scatter(volatility_mvp, return_mvp, s=80, marker="*", label="MVP Portfolio")
ax1.scatter(volatility_tg, return_tg, s=80, marker="*", label="Tangency Portfolio")

plt.xlabel("Volatility (Annualized STD)")
plt.ylabel("Expected Return (Annualized AVG)")
plt.title("Mean-variance portfolio (MPT)")

plt.legend(loc="upper left")
plt.show()
# %% tags=["legacy"]
df_ef.max()

# %% [markdown] tags=["legacy"]
# ## Value at Risk (VaR)

# %% [markdown] tags=["legacy"]
# ### Parametric positive-loss convention
#
# VaR is reported as a non-negative loss threshold. A 95% one-day VaR uses the
# 5th percentile of the **daily return** distribution; annualized portfolio inputs
# are scaled back to the requested horizon before the quantile is computed
# {cite}`jorion2007var,mcneil2015quantitative`.

# %% tags=["legacy"]
initial_investment = 1000000
conf_level = 0.95


# %% tags=["legacy"]
def VaR_parametric(initial_investment, conf_level):
    return_quantile = norm.ppf(1 - conf_level, loc=avg_daily, scale=std_daily)
    VaR_daily = np.maximum(-initial_investment * return_quantile, 0.0)
    df_var = pd.DataFrame(
        {
            "stock": price_data.columns,
            "return_quantile": return_quantile,
            "daily_var": VaR_daily,
        }
    )
    return df_var


# %% tags=["legacy"]
VaR_param = VaR_parametric(initial_investment, conf_level)
VaR_param.sort_values(by="daily_var").set_index("stock").daily_var.plot(kind="barh")
# %% tags=["legacy"]
VaR_param

# %% tags=["legacy"]
df_var_horizons = pd.concat([VaR_param.daily_var * np.sqrt(x) for x in range(1, 31)], axis=1).T

df_var_horizons.columns = VaR_param.stock.to_list()

df_var_horizons = df_var_horizons.reset_index(drop=True)
df_var_horizons.index = df_var_horizons.index + 1
df_var_horizons.head()

# %% tags=["legacy"]
df_var_horizons.plot()


# %% tags=["legacy"]
def calculate_portfolio_var(
    portfolio_value,
    weights,
    annual_mean_returns,
    annual_covariance_matrix,
    confidence_level,
    horizon_days=1,
    periods_per_year=252,
):
    # Convert annualized moments to the requested daily horizon.
    annual_portfolio_return = np.dot(weights, annual_mean_returns)
    horizon_return = annual_portfolio_return * horizon_days / periods_per_year

    annual_portfolio_std = np.sqrt(np.dot(np.dot(weights, annual_covariance_matrix), weights.T))
    horizon_std = annual_portfolio_std * np.sqrt(horizon_days / periods_per_year)

    # Lower-tail return quantile converted to a positive monetary loss.
    return_quantile = horizon_return + norm.ppf(1 - confidence_level) * horizon_std
    portfolio_var = max(-portfolio_value * return_quantile, 0.0)

    return portfolio_var


# %% tags=["legacy"]
calculate_portfolio_var(10000, W_mvp, avg_ann, cov_ann, 0.95)

# %% tags=["legacy"]

# %% [markdown] tags=["legacy"]
# ## Model limitations
#
# - Mean-variance optimization is highly sensitive to expected-return and covariance estimates.
# - The efficient frontier assumes stable inputs, a single-period objective, and frictionless rebalancing.
# - Portfolio weights should be interpreted alongside concentration, turnover, and robustness checks.
#
