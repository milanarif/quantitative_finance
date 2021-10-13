import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

announcementReturn = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='AnnouncementReturn').drop('month', 1)
returns = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='Returns').drop('month', 1)
me = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='ME').drop('month', 1)
factors = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='Factors').drop('month', 1)


def ex1(announcementReturn, returns):
    announcementReturn = announcementReturn.to_numpy()
    returns = returns.to_numpy()

    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], returns[i]))
        returns[i] = [element for _, element in sorted_zipped]

    for i in range(announcementReturn.shape[0]):
        announcementReturn[i] = np.sort(announcementReturn[i], axis=0)

    portfolio_return_matrix_eq_weighted = []

    for i in range(returns.shape[0]):
        row = []
        for j in range(0, returns.shape[1], 100):
            row.append(np.average(returns[i][j: j + 100]))
        portfolio_return_matrix_eq_weighted.append(row)

    average_returns = []
    for i in range(10):
        average_returns.append(np.mean(np.transpose(np.array(portfolio_return_matrix_eq_weighted))[i]))

    return average_returns


# print(ex1(announcementReturn, returns))


def ex2(announcementReturn, me, returns):
    announcementReturn = announcementReturn.to_numpy()
    me = me.to_numpy()
    returns = returns.to_numpy()
    portfolio_return_matrix_me_weighted = []

    # Zip together the two tables, sort based on announcementReturn and save the new order to ME-table.
    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], me[i]))
        me[i] = [element for _, element in sorted_zipped]

    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], returns[i]))
        returns[i] = [element for _, element in sorted_zipped]

    me_times_return = np.array(np.multiply(me, returns))

    sum_me_portfolio = []
    for i in range(me.shape[0]):
        row = []
        for j in range(0, me.shape[1], 100):
            row.append(np.sum(me[i][j: j + 100]))
        sum_me_portfolio.append(row)

    for i in range(me_times_return.shape[0]):
        x = 0
        for j in range(0, me_times_return.shape[1], 100):
            me_times_return[i][j: j + 100] /= sum_me_portfolio[i][x]
            x += 1

    for i in range(me_times_return.shape[0]):
        row = []
        for j in range(0, me_times_return.shape[1], 100):
            row.append(np.sum(me_times_return[i][j: j + 100]))
        portfolio_return_matrix_me_weighted.append(row)

    # average_returns = []
    # for i in range(10):
    #    average_returns.append(np.mean(np.transpose(np.array(portfolio_return_matrix_me_weighted))[i]))
    #
    # return average_returns

    return np.array(portfolio_return_matrix_me_weighted)


# print(ex2(announcementReturn, me, returns))


def ex3(announcementReturn, me, factors, returns):
    vw_returns = pd.DataFrame(ex2(announcementReturn, me, returns))
    betas = []

    for i in range(10):
        data = pd.DataFrame({'ret': (vw_returns[i])})
        data['mktrf'] = factors['Mktrf']

        model = smf.ols("ret ~ mktrf", data=data)
        result = model.fit()

        # Uncomment below for regression table
        # print(result.summary())

        betas.append(result.params[1])

    risk_free = factors['Rf']
    for i in range(360):
        for j in range(10):
            vw_returns.iloc[i, j] -= risk_free.iloc[i]

    mean_vw_ex_returns = vw_returns.mean(axis=0).tolist()
    market_excess = factors['Mktrf'].tolist()
    mean_mkt_ex_return = np.mean(market_excess)

    plt.scatter(betas, mean_vw_ex_returns)
    x = np.linspace(0, 1.6)
    plt.plot(x, mean_mkt_ex_return*x, color = 'red')

    plt.text(x=(betas[0] + 0.025), y=mean_vw_ex_returns[0], s='1st', weight="bold")
    plt.text(x=(betas[1] + 0.025), y=mean_vw_ex_returns[1], s='2nd', weight="bold")
    plt.text(x=(betas[8] + 0.025), y=mean_vw_ex_returns[8], s='9th', weight="bold")
    plt.text(x=(betas[9] + 0.025), y=mean_vw_ex_returns[9], s='10th', weight="bold")

    plt.show()


ex3(announcementReturn, me, factors, returns)

sys.exit()
