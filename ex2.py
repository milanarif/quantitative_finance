import datetime
import sys

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt

announcementReturn = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='AnnouncementReturn').drop('month', 1)
me = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='ME').drop('month', 1)


def ex1(announcementReturn):
    announcementReturn = announcementReturn.to_numpy()
    for i in range(announcementReturn.shape[0]):
        announcementReturn[i] = np.sort(announcementReturn[i], axis=0)

    portfolio_return_matrix_eq_weighted = []

    for i in range(announcementReturn.shape[0]):
        row = []
        for j in range(0, announcementReturn.shape[1], 100):
            row.append(np.average(announcementReturn[i][j: j + 100]))
        portfolio_return_matrix_eq_weighted.append(row)

    average_returns = []
    for i in range(10):
        average_returns.append(np.mean(np.transpose(np.array(portfolio_return_matrix_eq_weighted))[i]))

    return average_returns


print(ex1(announcementReturn))


def ex2(announcementReturn, me):
    announcementReturn = announcementReturn.to_numpy()
    me = me.to_numpy()
    portfolio_return_matrix_me_weighted = []

    # Zip together the two tables, sort based on announcementReturn and save the new order to ME-table.
    for i in range(announcementReturn.shape[0]):
        sorted_zipped = sorted(zip(announcementReturn[i], me[i]))
        me[i] = [element for _, element in sorted_zipped]

    me_times_return = np.array(np.multiply(me, announcementReturn))

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

    average_returns = []
    for i in range(10):
        average_returns.append(np.mean(np.transpose(np.array(portfolio_return_matrix_me_weighted))[i]))

    return average_returns

print(ex2(announcementReturn, me))

sys.exit()
