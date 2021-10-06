import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt

returns_sheet = pd.read_excel('data_Ass1_G2.xlsx', sheet_name='Returns').drop('month', 1)
BM_sheet = pd.read_excel('data_Ass1_G2.xlsx', sheet_name='BM').drop('month', 1)


def mv_weights(data, start, end, intervals):
    weights = []
    column_ones = np.ones((10, 1))
    row_ones = np.ones(10)
    for i in range(start, end - intervals + 1):
        cov_matrix = data.iloc[i: intervals + i].cov().to_numpy()
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        numerator = np.matmul(inv_cov_matrix, column_ones)
        denominator = np.matmul(row_ones, np.matmul(inv_cov_matrix, column_ones))

        weights.append(np.divide(numerator, denominator))

    return np.around(np.array(weights)[:, :, 0], 4)


def constrained_mv_weights(data, start, end, intervals):
    def min_objective(weights, cov_matrix):
        return np.matmul(np.transpose(weights), np.matmul(cov_matrix, weights))

    def constraint(weights):
        return 1 - np.matmul(np.transpose(weights), column_ones)

    weights = []
    column_ones = np.ones((10, 1))
    cons = ({'type': 'eq', 'fun': constraint})

    def optimize_function(cov_matrix):
        bounds = [(0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25), (0, 0.25),
                  (0, 0.25)]
        x0 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        maximal = optimize.minimize(min_objective, x0, args=cov_matrix, constraints=cons, method="SLSQP",
                                    bounds=bounds)
        opt = maximal.x
        return opt

    for i in range(start, end - intervals + 1):
        cov_matrix = data.iloc[i: intervals + i].cov().to_numpy()
        weights.append(optimize_function(cov_matrix))

    return np.around(np.array(weights), 4)


def param_weights(thetas, returns, bm, start, end):
    weights = []
    # TODO: ONLY ONE MONTH?
    # TODO: 120 or 121 ??

    for i in range(start, end):
        t_return = returns.iloc[i - 1].to_numpy()
        t_bm = bm.iloc[i].to_numpy()

        std_return = ((t_return - t_return.mean()) / t_return.std())
        std_bm = ((t_bm - t_bm.mean()) / t_bm.std())

        row_weights = []
        for x in range(10):
            row_weights.append((0.1 * (1 + (thetas[0] * std_return[x]) + (thetas[1] * std_bm[x]))))
        weights.append(row_weights)

    return np.asarray(weights)


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
            restricted_parametric_weights[i][j] = round((max(parametric_weights[i][j], 0)) / max_array[i], 4)

    return restricted_parametric_weights


def portfolio_returns(weights, returns):
    sum = 0
    return_list = []
    for i in range(120, 480):
        for j in range(0, 10):
            sum += weights[i - 120][j] * returns.iloc[i, j]
        return_list.append(sum)
        sum = 0

    return return_list


# QUESTION 1
# w1 = mv_weights(returns_sheet, 0, 120, 120)
# w2 = constrained_mv_weights(returns_sheet, 0, 120, 120)
# w3 = param_weights(returns_sheet, BM_sheet, 120, 121)
# w4 = constrained_param_weights(w3)
# print(" QUESTION 1 ")
# print("\nP1: ", w1)
# print("\nP2: ", w2)
# print("\nP3: ", w3)
# print("\nP4: ", w4)

# QUESTION 2
# print(mv_weights(returns_sheet, 0, returns_sheet.shape[0], 120))
# print(param_weights(returns_sheet, BM_sheet, 120, returns_sheet.shape[0]))


w1 = mv_weights(returns_sheet, 0, returns_sheet.shape[0], 120)
w2 = constrained_mv_weights(returns_sheet, 0, returns_sheet.shape[0], 120)
w3 = param_weights([0.9, 0.8], returns_sheet, BM_sheet, 120, returns_sheet.shape[0])
w4 = constrained_param_weights(w3)

returns_p1 = portfolio_returns(w1, returns_sheet)
returns_p2 = portfolio_returns(w2, returns_sheet)
returns_p3 = portfolio_returns(w3, returns_sheet)
returns_p4 = portfolio_returns(w4, returns_sheet)

# print(np.mean(returns_p1))
# print(np.mean(returns_p2))
# print(np.mean(returns_p3))
# print(np.mean(returns_p4))

# print(np.std(returns_p1))
# print(np.std(returns_p2))
# print(np.std(returns_p3))
# print(np.std(returns_p4))

# print((np.power((1 + np.mean(returns_p1)), 12) - 1) / (np.std(returns_p1)*np.sqrt(12)))
# print((np.power((1 + np.mean(returns_p2)), 12) - 1) / (np.std(returns_p2)*np.sqrt(12)))
print((np.power((1 + np.mean(returns_p3)), 12) - 1) / (np.std(returns_p3)*np.sqrt(12)))
# print((np.power((1 + np.mean(returns_p4)), 12) - 1) / (np.std(returns_p4)*np.sqrt(12)))

# QUESTION 3
# cumulative_returns_p1 = np.cumsum(returns_p1)
# cumulative_returns_p2 = np.cumsum(returns_p2)
# cumulative_returns_p3 = np.cumsum(returns_p3)
# cumulative_returns_p4 = np.cumsum(returns_p4)
# #
# plt.plot(cumulative_returns_p1, label="p1")
# plt.plot(cumulative_returns_p2, label="p2")
# plt.plot(cumulative_returns_p3, label="p3")
# plt.plot(cumulative_returns_p4, label="p4")
# plt.legend()
# plt.show()


# QUESTION 4

# QUESTION 5
def max_p3_sr(returns, bm, start, end):

    def min_objective(thetas, returns, bm, start, end):
        w = param_weights(thetas, returns, bm, start, end)
        r = portfolio_returns(w, returns)

        return -(np.power((1 + np.mean(r)), 12) - 1) / (np.std(r) * np.sqrt(12))

    def optimization(returns, bm, start, end):
        x0 = np.array([0.75, 0.75])
        bounds = [(0, 1.5), (0, 1.5)]

        maximum = optimize.minimize(min_objective, x0, args=(returns, bm, start, end), method="SLSQP", bounds=bounds)

        opt = maximum.x

        max_sharpe = round(-maximum.fun, 4)
        theta_1 = round(float(opt[0]), 4)
        theta_2 = round(float(opt[1]), 4)

        return [max_sharpe, theta_1, theta_2]

    return optimization(returns, bm, start, end)


print(max_p3_sr(returns_sheet, BM_sheet, 120, returns_sheet.shape[0]))
