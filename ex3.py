import pandas as pd
import numpy as np
from statistics import NormalDist
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import cbrt
from scipy.stats import norm
from scipy.optimize import fsolve
from arch import arch_model
import scipy

data = pd.read_excel('data_Ass3_G2.xlsx', sheet_name='Returns', engine='openpyxl').drop('date', 1)

def ex1(data):
    print("EXERCISE 1")
    cov_matrix = data.iloc[:, 2:].cov()
    mean_vector = data.iloc[:, 2:].mean()
    weight_vector = pd.DataFrame({'w': ([0.2] * 5)})
    portfolio_value = 50000

    portfolio_my = weight_vector.values.flatten().T.dot(mean_vector)
    portfolio_sigma = (((weight_vector.values.flatten().T.dot(cov_matrix)).dot(weight_vector)) ** 0.5)[0]

    inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

    value_at_risk = (- portfolio_my * portfolio_value) + (portfolio_sigma * portfolio_value * inv_dist_98)

    print("VaR:", round(value_at_risk, 1))
    print(round(value_at_risk / portfolio_value, 4) * 100, "%\n")


# ex1(data)

def ex2(data):
    print("EXERCISE 2")
    portfolio_value = 50000
    weight_vector = pd.DataFrame({'w': ([0.2] * 5)})

    variables = []
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        residual_std = result.resid.std()
        variables.append([result.params[0], result.params[1], residual_std])

    var_matrix = pd.DataFrame(variables)
    var_matrix.columns = ['alpha', 'mkt_beta', 'sigma']
    var_matrix.index = data.iloc[:, 2:].columns
    print(var_matrix, "\n")

    market_my = data['mkt'].mean()
    market_sigma = data['mkt'].std()

    print("Market MY:", round(market_my, 5))
    print("Market SIGMA:", round(market_sigma, 5), "\n")

    predicted_returns = data[['mkt']].copy()
    predicted_returns = pd.concat([predicted_returns] * 5, axis=1, ignore_index=True)
    predicted_returns.columns = data.iloc[:, 2:].columns

    for i in range(predicted_returns.shape[0]):
        for j in range(predicted_returns.shape[1]):
            predicted_returns.iloc[i, j] = predicted_returns.iloc[i, j] * var_matrix.iloc[j, 1] + var_matrix.iloc[j, 0]

    cov_matrix_model_2 = predicted_returns.cov()

    print("CORRELATION MATRIX")
    pd.set_option('display.max_columns', None)
    print(cov_matrix_model_2, "\n")

    portfolio_beta = var_matrix['mkt_beta'].mean()
    portfolio_alpha = var_matrix['alpha'].mean()
    portfolio_sigma = (((var_matrix['sigma'] * 0.2) ** 2).sum()) ** 0.5

    inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

    potential_shock = (((market_sigma * portfolio_beta) ** 2 + portfolio_sigma ** 2) ** 0.5) * inv_dist_98

    var = (-(portfolio_alpha + (market_my * portfolio_beta)) + potential_shock) * 50000

    # MONTE CARLO
    def day_returns_MC(var_matrix, market_my, market_sigma, runs):
        asset_returns = []
        for i in tqdm(range(runs)):
            returns = 0
            market_ret = np.random.normal(market_my, market_sigma)
            for i in range(5):
                returns += 0.2 * (var_matrix.iloc[i, 0] + var_matrix.iloc[i, 1] * market_ret + np.random.normal(0,
                                                                                                                var_matrix.iloc[
                                                                                                                    i, 2]))
            asset_returns.append(returns)

        return np.array(asset_returns)

    print(np.percentile(day_returns_MC(var_matrix, market_my, market_sigma, 25000), 2) * 50000)
    print(var)
    print(market_sigma)


# ex2(data)


def ex3(data):
    print("EXERCISE 3(a)")
    return_market = data['mkt']
    delta = return_market.std() * cbrt(return_market.skew() / 2)
    omega = (return_market.var() - (delta ** 2)) ** 0.5
    eta = return_market.mean() - delta

    skewness_return_market = 2 * ((delta / (((omega ** 2) + (delta ** 2)) ** 0.5)) ** 3)
    print("Skew(Rm):", skewness_return_market)

    print("delta:", delta)
    print("omega:", omega)
    print("eta:", eta)

    print("EXERCISE 3(b)")
    variables = []
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        residual_std = result.resid.std()
        variables.append([result.params[0], result.params[1], residual_std])

    var_matrix = pd.DataFrame(variables)
    var_matrix.columns = ['alpha', 'mkt_beta', 'sigma']
    var_matrix.index = data.iloc[:, 2:].columns

    def day_returns_MC(var_matrix, runs):
        asset_returns = []
        market_returns = []
        for i in tqdm(range(runs)):
            returns = 0
            excess_return_mkt = eta + delta * np.random.exponential(1) + omega * np.random.normal(0, 1)
            market_returns.append(excess_return_mkt)
            for i in range(5):
                returns += 0.2 * (
                        var_matrix.iloc[i, 0] + var_matrix.iloc[i, 1] * excess_return_mkt + np.random.normal(0, var_matrix.iloc[i, 2]))
            asset_returns.append(returns)

        return [np.array(asset_returns), np.array(market_returns)]

    res = day_returns_MC(var_matrix, 1000000)
    print(np.percentile(res[0], 2) * 50000)
    return pd.DataFrame(res[1])


# ex3(data)


def ex4(data):
    variables = []
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        residual_std = result.resid.std()
        variables.append([result.params[0], result.params[1], residual_std])

    var_matrix = pd.DataFrame(variables)
    var_matrix.columns = ['alpha', 'mkt_beta', 'sigma']
    var_matrix.index = data.iloc[:, 2:].columns

    return_market = data['mkt']
    delta_m = return_market.std() * cbrt(return_market.skew() / 2)
    omega_m = (return_market.var() - (delta_m ** 2)) ** 0.5
    eta_m = return_market.mean() - delta_m

    eta_l = 0
    delta_l = 0
    omega_sum = 0
    for i in range (5):
        eta_l -= 0.2 * (var_matrix.iloc[i, 0] + var_matrix.iloc[i, 1] * eta_m) * 50000
        delta_l -= 0.2 * var_matrix.iloc[i, 1] * delta_m * 50000
        omega_sum += (0.04 * (var_matrix.iloc[i, 2] ** 2))

    omega_l = np.sqrt(((var_matrix.iloc[:, 1].mean() * omega_m) ** 2) + omega_sum) * 50000
    print("eta_l", eta_l)
    print("delta_1", delta_l)
    print("omega_l", omega_l)

# QUESTION 4(b)

    def cdfNE(x, omega, eta, delta):
        if (delta > 0):
            return norm.cdf((x-eta) / omega) - np.exp(((omega ** 2) / (2 * (delta ** 2))) - ((x-eta) / delta)) * norm.cdf(((x - eta) / omega) - (omega / delta))
        if (delta < 0):
            return norm.cdf((x - eta) / omega) + np.exp(
                ((omega ** 2) / (2 * delta ** 2)) - ((x - eta) / delta)) * norm.cdf(
                (omega / delta) - ((x - eta) / omega))

    def targetfunc(x, omega, eta, delta, alpha):
        return cdfNE(x, omega, eta, delta) - alpha

    print("L:", fsolve(targetfunc, 0, (omega_l, eta_l, delta_l, 0.98)))

# ex4(data)


def ex5(data):
    # Times 100 later divide omega by 100^2
    market_returns = data['mkt'] * 100
    model = arch_model(market_returns, vol="GARCH", mean="Constant", p=1, q=1)
    model_fit = model.fit()
    plt.plot(model_fit.conditional_volatility ** 2)
    var_mkt = np.var(market_returns)
    plt.axhline(y=var_mkt, color='r', linestyle='-')
    plt.savefig("5_variance.png", dpi=300)
    plt.show()
    # print(model_fit.summary())
    mu = model_fit.params[0]/100
    omega = model_fit.params[1] / 10000
    gamma = model_fit.params[2]
    theta = model_fit.params[3]
    print("mu", mu)
    print("omega", omega)
    print("gamma", gamma)
    print("theta", theta)

    variables = []
    for i in range(5):
        return_data = data[['mkt']].copy()
        return_data['ret'] = data.iloc[:, 2 + i]

        model = smf.ols("ret ~ mkt", data=return_data)
        result = model.fit()
        residual_std = result.resid.std()
        variables.append([result.params[0], result.params[1], residual_std])

    var_matrix = pd.DataFrame(variables)
    var_matrix.columns = ['alpha', 'mkt_beta', 'sigma']
    var_matrix.index = data.iloc[:, 2:].columns

    def day_returns_MC(var_matrix, market_mu, gamma, theta, runs=10000):
        asset_returns = []
        market_ret = 0.00287
        sigma = 0.0000314
        for i in tqdm(range(runs)):
            returns = 0
            sigma = mu + gamma * ((market_ret - market_mu) ** 2) + theta * (sigma)
            market_ret = market_mu + sigma * np.random.normal(0, 1)
            for i in range(5):
                returns += 0.2 * (var_matrix.iloc[i, 0] + var_matrix.iloc[i, 1] * market_ret + np.random.normal(0, var_matrix.iloc[i, 2]))
            asset_returns.append(returns)

        return np.array(asset_returns)

    # print(np.percentile(day_returns_MC(var_matrix, mu, gamma, theta, runs=50000), 2) * 50000)
    print(var_matrix)


# ex5(data)

def ex6(data):
    cov_matrix = data.iloc[:, 2:].cov()
    mean_vector = data.iloc[:, 2:].mean()
    weight_vector = [0.2, 0.2, 0.2, 0.2, 0.2]
    portfolio_value = 50000

    def constraint(weights):
        return 1 - sum(weights)

    cons = {'type': 'eq', 'fun': constraint}

    def var_calc(weight_vector):

        weight_vector = pd.DataFrame(weight_vector)
        portfolio_my = weight_vector.values.flatten().T.dot(mean_vector)
        portfolio_sigma = (((weight_vector.values.flatten().T.dot(cov_matrix)).dot(weight_vector)) ** 0.5)[0]
        inv_dist_98 = NormalDist(0, 1).inv_cdf(0.98)

        value_at_risk = (- portfolio_my * portfolio_value) + (portfolio_sigma * portfolio_value * inv_dist_98)

        return value_at_risk

    res = scipy.optimize.minimize(var_calc, weight_vector, constraints=cons)
    print(res.x)
    print(res.fun)
    print(round((res.fun / 50000) * 100, 2), "%")

# ex6(data)

def ex7(data):

