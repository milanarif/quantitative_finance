import datetime

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt

announcementReturn = pd.read_excel('data_Ass2_G2.xlsx', sheet_name='AnnouncementReturn').drop('month', 1)

def ex1(announcementReturn):
    announcementReturn = announcementReturn.to_numpy()
    for i in range(announcementReturn.shape[0]):
        announcementReturn[i] = np.sort(announcementReturn[i], axis=0)

    portfolio_return_matrix = []

    for i in range(announcementReturn.shape[0]):
        row = []
        for j in range(0, announcementReturn.shape[1], 100):
            row.append(np.average(announcementReturn[i][j : j+100]))
        portfolio_return_matrix.append(row)

    average_returns = []
    for i in range(10):
        average_returns.append(np.mean(np.transpose(np.array(portfolio_return_matrix))[i]))

    return average_returns

print(ex1(announcementReturn))