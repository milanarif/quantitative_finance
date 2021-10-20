import pandas as pd
import numpy as np
from statistics import NormalDist
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import cbrt
from sympy import Symbol, solve
from scipy.stats import norm
from scipy.optimize import fsolve

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


# ex2(data)


def ex3(data):
    return_market = data['mkt']
    delta = return_market.std() * cbrt(return_market.skew() / 2)
    omega = (return_market.var() - (delta ** 2)) ** 0.5
    eta = return_market.mean() - delta

    print("delta:", delta)
    print("omega:", omega)
    print("eta:", eta)

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
                        var_matrix.iloc[i, 0] + var_matrix.iloc[i, 1] * excess_return_mkt + np.random.normal(0,
                                                                                                             var_matrix.iloc[
                                                                                                                 i, 2]))
            asset_returns.append(returns)

        return [np.array(asset_returns), np.array(market_returns)]

    res = day_returns_MC(var_matrix, 10000)
    print(np.percentile(res[0], 2) * 50000)
    return pd.DataFrame(res[1])


# ex3(data)


# def ex4(data):
#     loss_return = data.iloc[:, 2:].mean(axis=1) * -1
#     expected_return_loss = loss_return.mean()
#     std_loss_return = loss_return.std()
#     skew_loss_return = loss_return.skew()
#
#     delta_L = std_loss_return * cbrt(skew_loss_return / 2)
#     omega_L = (std_loss_return ** 2 - delta_L ** 2) ** 0.5
#     eta_L = (expected_return_loss - delta_L)
#
#     # print(delta_L, omega_L, eta_L)
#
#     def NEcdf(x, delta=delta_L, omega=omega_L, eta=eta_L):
#         part_1 = norm.cdf((x - eta) / omega)
#         part_2 = np.exp(((omega ** 2) / (2 * (delta ** 2))) - ((x - eta) / delta))
#         part_3 = norm.cdf((omega / delta) - ((x - eta) / omega))
#         return part_1 + (part_2 * part_3)
#
#     def optimize_goal(x):
#         return NEcdf(x) - 0.98
#
#     opt_x = fsolve(optimize_goal, 0, maxfev=10000)
#     print(NEcdf(opt_x), opt_x)


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
    delta = return_market.std() * cbrt(return_market.skew() / 2)
    omega = (return_market.var() - (delta ** 2)) ** 0.5
    eta = return_market.mean() - delta







ex4(data)
