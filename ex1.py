import datetime

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt

returns_sheet = pd.read_excel('data_Ass1_G2.xlsx', sheet_name='Returns').drop('month', 1)
BM_sheet = pd.read_excel('data_Ass1_G2.xlsx', sheet_name='BM').drop('month', 1)


# start is the first period we want to predict.
def mv_weights(data, start, end, intervals):
    weights = []
    column_ones = np.ones((10, 1))
    row_ones = np.ones(10)
    # Go through each period (row) and calculate using the covariance matrix from the interval before
    for i in range(start, end):
        cov_matrix = data.iloc[i - intervals: i].cov().to_numpy()
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        numerator = np.matmul(inv_cov_matrix, column_ones)
        denominator = np.matmul(row_ones, np.matmul(inv_cov_matrix, column_ones))

        # Add the weights to the array to be returned at the end
        weights.append(np.divide(numerator, denominator))

    return np.array(weights)[:, :, 0]


def constrained_mv_weights(data, start, end, intervals):
    # Define the function that we aim to minimize
    def min_objective(weights, cov_matrix):
        return np.matmul(np.transpose(weights), np.matmul(cov_matrix, weights))

    # The constraint that our minimization is subject to
    def constraint(weights):
        return 1 - np.matmul(np.transpose(weights), column_ones)

    weights = []
    column_ones = np.ones((10, 1))
    # We define the constraint as an equality constraint and set the constraining function to constraint above
    cons = ({'type': 'eq', 'fun': constraint})

    # We set the bounds for the weights and give the function starting values for the optimization
    def optimize_function(cov_matrix):
        bounds = [(0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25),
                  (0, 0.25)]
        x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        maximal = optimize.minimize(min_objective, x0, args=cov_matrix, constraints=cons, method="SLSQP",
                                    bounds=bounds)
        opt = maximal.x
        return opt

    # It is here that we actually call the above function and append the results after each iteration to an array
    # The below for-loop has the same indexing as the unconstrained example. Only difference is the numeric optimization
    for i in range(start, end):
        cov_matrix = data.iloc[i - intervals: i].cov().to_numpy()
        weights.append(optimize_function(cov_matrix))

    return np.array(weights)

# Asked at Q&A and indexed returns and bm equal following answer
# Thetas are passed in as a list, start is first period to be calculated, end is the second to last.
def param_weights(thetas, returns, bm, start, end):
    weights = []

    for i in range(start, end):
        t_minus_1_return = returns.iloc[i - 1].to_numpy()
        t_bm = bm.iloc[i - 1].to_numpy()

        std_return = ((t_minus_1_return - t_minus_1_return.mean()) / t_minus_1_return.std())
        std_bm = ((t_bm - t_bm.mean()) / t_bm.std())

        row_weights = []
        for x in range(10):
            row_weights.append((0.1 * (1 + (thetas[0] * std_return[x]) + (thetas[1] * std_bm[x]))))
        weights.append(row_weights)

    return np.asarray(weights)


# Just builds on the param_weights function. Could be more clean but it works...
def constrained_param_weights(parametric_weights):
    row_sum = 0
    max_array = []
    restricted_parametric_weights = np.empty_like(parametric_weights)
    for i in range(len(parametric_weights)):
        for j in range(len(parametric_weights[i])):
            row_sum += max(parametric_weights[i][j], 0)
        max_array.append(row_sum)
        row_sum = 0

    for i in range(len(parametric_weights)):
        for j in range(len(parametric_weights[i])):
            restricted_parametric_weights[i][j] = (max(parametric_weights[i][j], 0)) / max_array[i]

    return restricted_parametric_weights


# ChopChop the returns to match weights period and then multiply the elements and sum them up to get total returns
def portfolio_returns(weights, returns, start, end):
    returns = returns.iloc[start: end].to_numpy()
    weighted_returns = np.multiply(weights, returns)
    return_list = []
    for row in weighted_returns:
        return_list.append(np.sum(row))

    return return_list


# QUESTION 1
w1 = mv_weights(returns_sheet, 120, 121, 120)
w2 = constrained_mv_weights(returns_sheet, 120, 121, 120)
w3 = param_weights([0.9, 0.8], returns_sheet, BM_sheet, 120, 121)
w4 = constrained_param_weights(w3)
print(" QUESTION 1 ")
print("\nP1: ", w1)
print("\nP2: ", w2)
print("\nP3: ", w3)
print("\nP4: ", w4)

# QUESTION 2
print("\nP1")
w1 = mv_weights(returns_sheet, 120, returns_sheet.shape[0], 120)
returns_p1 = portfolio_returns(w1, returns_sheet, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(returns_p1))
print("STD: ", np.std(returns_p1))
print("SHARPE: ", (np.power((1 + np.mean(returns_p1)), 12) - 1) / (np.std(returns_p1) * np.sqrt(12)))

print("\nP2")
w2 = constrained_mv_weights(returns_sheet, 120, returns_sheet.shape[0], 120)
returns_p2 = portfolio_returns(w2, returns_sheet, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(returns_p2))
print("STD: ", np.std(returns_p2))
print("SHARPE: ", (np.power((1 + np.mean(returns_p2)), 12) - 1) / (np.std(returns_p2) * np.sqrt(12)))

print("\nP3")
w3 = param_weights([0.9, 0.8], returns_sheet, BM_sheet, 120, returns_sheet.shape[0])
returns_p3 = portfolio_returns(w3, returns_sheet, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(returns_p3))
print("STD: ", np.std(returns_p3))
print("SHARPE: ", (np.power((1 + np.mean(returns_p3)), 12) - 1) / (np.std(returns_p3) * np.sqrt(12)))

print("\nP4")
w4 = constrained_param_weights(w3)
returns_p4 = portfolio_returns(w4, returns_sheet, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(returns_p4))
print("STD: ", np.std(returns_p4))
print("SHARPE: ", (np.power((1 + np.mean(returns_p4)), 12) - 1) / (np.std(returns_p4) * np.sqrt(12)))

# QUESTION 3
cumulative_returns_p1 = np.cumsum(returns_p1)
cumulative_returns_p2 = np.cumsum(returns_p2)
cumulative_returns_p3 = np.cumsum(returns_p3)
cumulative_returns_p4 = np.cumsum(returns_p4)
#
dates = map(str, pd.read_excel('data_Ass1_G2.xlsx', sheet_name='Returns').iloc[120:,0].to_numpy())
x_values = [datetime.datetime.strptime(d, "%Y%m").date() for d in dates]
plt.figure(dpi=200)
plt.plot(x_values, cumulative_returns_p1, label="MV (P1)")
plt.plot(x_values, cumulative_returns_p2, label="Constrained MV (P2)")
plt.plot(x_values, cumulative_returns_p3, label="Parametric (P3)")
plt.plot(x_values, cumulative_returns_p4, label="Constrained Parametric (P4)")
plt.title("Cumulative returns on the portfolios")
plt.ylabel("Cumulative return")
plt.legend()
plt.show()
plt.savefig("plot.png")

# QUESTION 4

# QUESTION 5
def max_p3_sr(returns, bm, start, end):

    # The objective is to maximize this function that calculates portfolio returns for parametric portfolio
    # Instead of set thetas, we let the optimization function calling this function to set the optimal thetas
    def min_objective(thetas, returns, bm, start, end):
        w = param_weights(thetas, returns, bm, start, end)
        r = portfolio_returns(w, returns, start, end)

        # Negative sign to make minimum objective an actual maximum
        return -(np.power((1 + np.mean(r)), 12) - 1) / (np.std(r) * np.sqrt(12))

    # Give optimization starting values and bounds
    def optimization(returns, bm, start, end):
        x0 = np.array([0.75, 0.75])
        bounds = [(0, 1.5), (0, 1.5)]

        maximum = optimize.minimize(min_objective, x0, args=(returns, bm, start, end), method="SLSQP", bounds=bounds)

        opt = maximum.x

        max_sharpe = -maximum.fun
        theta_1 = float(opt[0])
        theta_2 = float(opt[1])

        return [max_sharpe, theta_1, theta_2]

    # Call the contained optimization function and return the resulting array with sharpe and thetas.
    return optimization(returns, bm, start, end)


print("\nOPTIMIZED THETAS P3:")
# Get array with optimal thetas
opt_sharpe_theta_res = max_p3_sr(returns_sheet, BM_sheet, 120, returns_sheet.shape[0])

# Use thetas from optimization to get weights given those thetas
w3_opt = param_weights(opt_sharpe_theta_res[1:3], returns_sheet, BM_sheet, 120, returns_sheet.shape[0])

# Calculate the returns of the optimized portfolio
returns_p3_opt = portfolio_returns(w3_opt, returns_sheet, 120, returns_sheet.shape[0])
print("THETA 1: ", opt_sharpe_theta_res[1])
print("THETA 2: ", opt_sharpe_theta_res[2])
print("MEAN: ", np.mean(returns_p3_opt))
print("STD: ", np.std(returns_p3_opt))
print("SHARPE: ", (np.power((1 + np.mean(returns_p3_opt)), 12) - 1) / (np.std(returns_p3_opt) * np.sqrt(12)))


# QUESTION 6
# Calculate hold portfolio weights, returns an array for one time index for all assets in a portfolio
def w_tilde_t(asset_ret_t_minus_1, port_ret_t_minus_1, weights_t_minus_1):
    w_tilde_t = []
    for i in range(len(weights_t_minus_1)):
        w_tilde_t.append(weights_t_minus_1[i] * ((1 + asset_ret_t_minus_1[i]) / (1 + port_ret_t_minus_1)))
    return w_tilde_t

# Returns the turnover of an asset given the weights and hold portfolio weights
def turnover_t(weights, tilde_weights):
    sum = 0
    for i in range(10):
        sum += np.absolute(weights[i] - tilde_weights[i])
    return sum / 2

# Returns an array of each months total turnover for all assets in a portfolio
def turnover_series(returns, port_returns, weights, start, end):
    turnover = []
    returns = returns.iloc[start : end]
    for i in range(len(port_returns)):
        if i > 0:
            w_tilde = w_tilde_t(returns.to_numpy()[i - 1], port_returns[i - 1], weights[i - 1])
            turnover.append(turnover_t(weights[i], w_tilde))

        else:
            turnover.append(turnover_t(weights[i], weights[i]))

    return np.asarray(turnover)

# Calculate the turnover arrays of each portfolio
t1 = turnover_series(returns_sheet, returns_p1, w1, 120, returns_sheet.shape[0])
t2 = turnover_series(returns_sheet, returns_p2, w2, 120, returns_sheet.shape[0])
t3 = turnover_series(returns_sheet, returns_p3, w3, 120, returns_sheet.shape[0])
t4 = turnover_series(returns_sheet, returns_p4, w4, 120, returns_sheet.shape[0])

# Calculate mean turnover of each portfolio.
print("\nTURNOVER RATES:")
print("P1: ", np.mean(t1))
print("P2: ", np.mean(t2))
print("P3: ", np.mean(t3))
print("P4: ", np.mean(t4))


# QUESTION 7
# Create an array and append to it the returns for each time index. Subtract a cost of 50 basis points of turnover
def adj_portfolio_returns(returns, port_returns, weights, start, end):
    sum = 0
    return_list = []
    returns = returns.iloc[start: end]
    for i in range(len(port_returns)):
        if i > 0:
            weights_tilde = w_tilde_t(returns.to_numpy()[i - 1], port_returns[i - 1], weights[i - 1])
        else:
            weights_tilde = weights[i]
        for j in range(0, 10):
            sum += weights[i][j] * returns.to_numpy()[i][j] - (0.005 * (np.absolute(weights[i][j] - weights_tilde[j])))
        return_list.append(sum)
        sum = 0

    return return_list


print("\nAdjusted P1")
adj_returns_p1 = adj_portfolio_returns(returns_sheet, returns_p1, w1, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(adj_returns_p1))
print("STD: ", np.std(adj_returns_p1))
print("SHARPE: ", (np.power((1 + np.mean(adj_returns_p1)), 12) - 1) / (np.std(adj_returns_p1) * np.sqrt(12)))

print("\nAdjusted P2")
adj_returns_p2 = adj_portfolio_returns(returns_sheet, returns_p2, w2, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(adj_returns_p2))
print("STD: ", np.std(adj_returns_p2))
print("SHARPE: ", (np.power((1 + np.mean(adj_returns_p2)), 12) - 1) / (np.std(adj_returns_p2) * np.sqrt(12)))

print("\nAdjusted P3")
adj_returns_p3 = adj_portfolio_returns(returns_sheet, returns_p3, w3, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(adj_returns_p3))
print("STD: ", np.std(adj_returns_p3))
print("SHARPE: ", (np.power((1 + np.mean(adj_returns_p3)), 12) - 1) / (np.std(adj_returns_p3) * np.sqrt(12)))

print("\nAdjusted P4")
adj_returns_p4 = adj_portfolio_returns(returns_sheet, returns_p4, w4, 120, returns_sheet.shape[0])
print("MEAN: ", np.mean(adj_returns_p4))
print("STD: ", np.std(adj_returns_p4))
print("SHARPE: ", (np.power((1 + np.mean(adj_returns_p4)), 12) - 1) / (np.std(adj_returns_p4) * np.sqrt(12)))


# QUESTION 8
def max_p3_sr_w_tc(returns, bm, start, end):

    # Objective is the portfolio returns adjusted for transaction costs
    def min_objective(thetas, returns, bm, start, end):
        w = param_weights(thetas, returns, bm, start, end)
        r = portfolio_returns(w, returns, start, end)
        adj_r = adj_portfolio_returns(returns, r, w, 120, returns.shape[0])

        return -(np.power((1 + np.mean(adj_r)), 12) - 1) / (np.std(adj_r) * np.sqrt(12))

    # Define optimization function, set initial values and bounds for thetas
    def optimization(returns, bm, start, end):
        x0 = np.array([0.2, 0.2])
        bounds = [(0, 1.5), (0, 1.5)]

        maximum = optimize.minimize(min_objective, x0, args=(returns, bm, start, end), method="SLSQP", bounds=bounds)

        opt = maximum.x

        max_sharpe = -maximum.fun
        theta_1 = float(opt[0])
        theta_2 = float(opt[1])

        return [max_sharpe, theta_1, theta_2]

    return optimization(returns, bm, start, end)


print("\nOPTIMIZED THETAS P3 (ADJUSTED FOR TC):")
opt_sharpe_theta_res_tc = max_p3_sr_w_tc(returns_sheet, BM_sheet, 120, returns_sheet.shape[0])
w3_opt_tc = param_weights(opt_sharpe_theta_res_tc[1:3], returns_sheet, BM_sheet, 120, returns_sheet.shape[0])
returns_p3_opt_tc = portfolio_returns(w3_opt_tc, returns_sheet, 120, returns_sheet.shape[0])
print("THETA 1: ", opt_sharpe_theta_res_tc[1])
print("THETA 2: ", opt_sharpe_theta_res_tc[2])
print("MEAN: ", np.mean(returns_p3_opt_tc))
print("STD: ", np.std(returns_p3_opt_tc))
print("SHARPE: ", (np.power((1 + np.mean(returns_p3_opt_tc)), 12) - 1) / (np.std(returns_p3_opt_tc) * np.sqrt(12)))
