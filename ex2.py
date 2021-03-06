import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

announcementReturn = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='AnnouncementReturn', engine='openpyxl').drop(
    'month', 1)
returns = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='Returns', engine='openpyxl').drop('month', 1)
me = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='ME', engine='openpyxl').drop('month', 1)
factors = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='Factors', engine='openpyxl').drop('month', 1)


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
    rf = factors['Rf']
    vw_excess_returns = []
    for i in range(360):
        row = []
        for j in range(10):
            row.append(vw_returns.iloc[i, j] - rf.iloc[i])
        vw_excess_returns.append(row)

    vw_excess_returns = pd.DataFrame(vw_excess_returns)

    betas = []

    for i in range(10):
        data = pd.DataFrame({'ret': (vw_excess_returns[i])})
        data['mktrf'] = factors['Mktrf']

        model = smf.ols("ret ~ mktrf", data=data)
        result = model.fit()

        # Uncomment below for regression table
        # print(result.summary())

        betas.append(result.params[1])

    mean_vw_excess_returns = vw_excess_returns.mean(axis=0).tolist()
    market_excess = factors['Mktrf'].tolist()
    mean_mkt_ex_return = np.mean(market_excess)

    # Create and design plot
    plt.scatter(betas, mean_vw_excess_returns)
    x = np.linspace(0, 1.6)
    plt.plot(x, mean_mkt_ex_return * x, color='red')

    plt.text(x=(betas[0] + 0.02), y=mean_vw_excess_returns[0], s='1st', weight="bold")
    plt.text(x=(betas[1] + 0.02), y=mean_vw_excess_returns[1], s='2nd', weight="bold")
    plt.text(x=(betas[8] + 0.02), y=mean_vw_excess_returns[8], s='9th', weight="bold")
    plt.text(x=(betas[9] + 0.02), y=mean_vw_excess_returns[9], s='10th', weight="bold")
    plt.title('Announcement Return Portfolios')
    plt.ylabel('Average monthly excess return')
    plt.xlabel('CAPM Beta')
    plt.xlim([0, 1.6])
    plt.ylim([0, 0.018])
    plt.savefig("capm.png", dpi=300)
    plt.show()


# ex3(announcementReturn, me, factors, returns)


def ex4(announcementReturn, me, factors, returns):
    vw_returns = pd.DataFrame(ex2(announcementReturn, me, returns))
    rf = factors['Rf']
    vw_excess_returns = []
    for i in range(360):
        row = []
        for j in range(10):
            row.append(vw_returns.iloc[i, j] - rf.iloc[i])
        vw_excess_returns.append(row)

    vw_excess_returns = pd.DataFrame(vw_excess_returns)

    reg_results = []
    for i in range(10):
        data = pd.DataFrame({'ret': (vw_excess_returns[i])})
        data['mktrf'] = factors['Mktrf']
        data['smb'] = factors['SMB']
        data['hml'] = factors['HML']
        data['wml'] = factors['WML']

        model = smf.ols("ret ~ mktrf + smb + hml + wml", data=data)
        result = model.fit()

        # Uncomment below for regression table
        print("DECILE: ", i)
        print(result.summary())

        reg_results.append([result.params[1], result.params[2], result.params[3], result.params[4]])

    factor_means = data.mean(axis=0).drop('ret').tolist()

    carhart_portfolio_prediction = []
    for i in range(10):
        sum = 0
        for j in range(4):
            sum += reg_results[i][j] * factor_means[j]
        carhart_portfolio_prediction.append(sum)

    mean_vw_excess_returns = vw_excess_returns.mean(axis=0).tolist()

    plt.scatter(carhart_portfolio_prediction, mean_vw_excess_returns)
    x = np.linspace(0, 0.017)
    plt.plot(x, x, color='red')
    plt.text(x=(carhart_portfolio_prediction[0] + 0.0002), y=mean_vw_excess_returns[0], s='1st', weight="bold")
    plt.text(x=(carhart_portfolio_prediction[1] + 0.0002), y=mean_vw_excess_returns[1], s='2nd', weight="bold")
    plt.text(x=(carhart_portfolio_prediction[8] + 0.0002), y=mean_vw_excess_returns[8], s='9th', weight="bold")
    plt.text(x=(carhart_portfolio_prediction[9] + 0.0002), y=mean_vw_excess_returns[9], s='10th', weight="bold")
    plt.xlim([0, 0.017])
    plt.ylim([0, 0.017])
    plt.title('Carhart Four-Factor model')
    plt.ylabel('Average monthly excess return')
    plt.xlabel('Predicted monthly excess return')
    plt.savefig("carhart.png", dpi=300)
    plt.show()


# ex4(announcementReturn, me, factors, returns)


def ex5(announcementReturn, me, factors, returns):
    vw_returns = pd.DataFrame(ex2(announcementReturn, me, returns))
    port_returns = vw_returns[9] - vw_returns[0]
    data = pd.DataFrame({'ret': port_returns})
    data['mktrf'] = factors['Mktrf']
    data['smb'] = factors['SMB']
    data['hml'] = factors['HML']
    data['wml'] = factors['WML']

    capm = smf.ols("ret ~ mktrf", data=data)
    result = capm.fit()

    print(result.summary())

    carhart = smf.ols("ret ~ mktrf + smb + hml + wml", data=data)
    result = carhart.fit()

    print(result.summary())

    return port_returns


# ex5(announcementReturn, me, factors, returns)


# TODO: CAN WE ZIP-SORT LIKE THIS!?
def ex6(announcementReturn, me, returns):
    announcementReturn = announcementReturn.to_numpy()
    me = me.to_numpy()
    returns = returns.to_numpy()

    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], returns[i]))
        returns[i] = [element for _, element in sorted_zipped]

    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], me[i]))
        me[i] = [element for _, element in sorted_zipped]

    low = returns[:, :300]
    low_me = me[:, :300]

    for i in range(low_me.shape[0]):
        sorted_zipped = sorted(zip(low_me[i], low[i]))
        low[i] = [element for _, element in sorted_zipped]

    mid = returns[:, 300: 700]
    mid_me = me[:, 300: 700]

    for i in range(mid_me.shape[0]):
        sorted_zipped = sorted(zip(mid_me[i], mid[i]))
        mid[i] = [element for _, element in sorted_zipped]

    high = returns[:, 700: 1000]
    high_me = me[:, 700: 1000]

    for i in range(high_me.shape[0]):
        sorted_zipped = sorted(zip(high_me[i], high[i]))
        high[i] = [element for _, element in sorted_zipped]

    small_low = low[:, :150]
    small_low_me = low_me[:, :150]
    small_low_sum_me = small_low_me.sum(axis=1)

    vw_small_low_returns = []
    for i in range(360):
        sum = 0
        for j in range(150):
            sum += ((small_low_me[i][j] / small_low_sum_me[i]) * small_low[i][j])
        vw_small_low_returns.append(sum)

    big_low = low[:, 150:]
    big_low_me = low_me[:, 150:]
    big_low_sum_me = big_low_me.sum(axis=1)

    vw_big_low_returns = []
    for i in range(360):
        sum = 0
        for j in range(150):
            sum += ((big_low_me[i][j] / big_low_sum_me[i]) * big_low[i][j])
        vw_big_low_returns.append(sum)

    small_high = high[:, :150]
    small_high_me = high_me[:, :150]
    small_high_sum_me = small_high_me.sum(axis=1)

    vw_small_high_returns = []
    for i in range(360):
        sum = 0
        for j in range(150):
            sum += ((small_high_me[i][j] / small_high_sum_me[i]) * small_high[i][j])
        vw_small_high_returns.append(sum)

    big_high = high[:, 150:]
    big_high_me = high_me[:, 150:]
    big_high_sum_me = big_high_me.sum(axis=1)

    vw_big_high_returns = []
    for i in range(360):
        sum = 0
        for j in range(150):
            sum += ((big_high_me[i][j] / big_high_sum_me[i]) * big_high[i][j])
        vw_big_high_returns.append(sum)

    return_factor = []
    for i in range(360):
        return_factor.append(0.5 * (vw_small_high_returns[i] + vw_big_high_returns[i]) - 0.5 * (
                    vw_small_low_returns[i] + vw_big_low_returns[i]))

    cumulative_returns = np.cumsum(return_factor)

    dates = map(str, pd.read_excel('data_Ass2_G2.xlsx', sheet_name='Returns').iloc[:, 0].to_numpy())
    x_values = [datetime.datetime.strptime(d, "%Y%m").date() for d in dates]
    plt.plot(x_values, cumulative_returns)
    plt.title("Cumulative returns on double sorted portfolio (Hi - Lo)\n(Announcement Return and Market Equity)")
    plt.ylabel("Total net return")
    plt.savefig("ex6.png", dpi=300)
    plt.show()

    return return_factor


# ex6(announcementReturn, me, returns)


def ex7(announcementReturn, me, factors, returns):
    return_factor = ex6(announcementReturn, me, returns)
    vw_returns = pd.DataFrame(ex2(announcementReturn, me, returns))
    returns_ls = vw_returns[9] - vw_returns[0]

    data = pd.DataFrame({'ret': returns_ls})
    data['factor'] = return_factor
    data['mktrf'] = factors['Mktrf']

    model = smf.ols("ret ~ mktrf + factor", data=data)
    result = model.fit()

    print(result.summary())


ex7(announcementReturn, me, factors, returns)

sys.exit()
