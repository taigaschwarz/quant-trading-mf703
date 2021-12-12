#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:10:56 2021

@author: taigaschwarz

back-test functions
"""

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime as dt
import re 
import yfinance as yf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import requests
from get_all_tickers import get_tickers as gt
import bs4 as bs  # beautiful soups for web-scraping
import os
import pandas_datareader.data as pdr
import seaborn as sns

# user-defined functions
def clean_date_index(df):
    '''cleans date index of a dataframe such that the new format becomes year_period'''
    df.index = [str(x.year) + "_" + str(x.quarter) for x in df.index]
    return df

def calc_rets(df=None, nRows=None, nCols=None, index=None):
    '''calculates period returns given period price time series data within a dataframe'''
    rets = pd.DataFrame(np.zeros((nRows, nCols)), index = df.index, columns = index)
    for i in range(nCols):
        rets.iloc[:,i] = df.iloc[:,i] / df.iloc[:,i].shift(1) - 1
    rets = rets.shift(-1)  # shift n+1 period return into the n period row
    return rets

def qtr_hist_mkt_cap(ticker, start_qtr, end_qtr):
    '''outputs a dataframe with historical market cap data over quarterly periods from 2007-2020 given a ticker'''
    
    url_1 = "https://financialmodelingprep.com/api/v3/historical-market-capitalization/"
    url_2 = f"{ticker}?period=month&apikey=f0060c7a7b275396c21cb98f4985f3ae"
    mkt_cap = requests.get(url_1 + url_2)
    mkt_cap = mkt_cap.json()
    mkt_cap = pd.DataFrame(mkt_cap).set_index('date').drop(columns = ['symbol'], axis = 1)
    mkt_cap.index = pd.to_datetime(mkt_cap.index)
    mkt_cap = mkt_cap.resample('1M').mean()
    mkt_cap = mkt_cap.iloc[::3, :]  # quarterly data
    mkt_cap = clean_date_index(mkt_cap)
    mkt_cap = mkt_cap.loc[start_qtr:end_qtr,]  # slice data into sample timeframe
    mkt_cap.columns = pd.MultiIndex.from_tuples([(ticker, 'mkt_cap')])
    mkt_cap.index.names = ['qtr']
    return mkt_cap

def daily_hist_mkt_cap(ticker):
    '''outputs a dataframe with daily historical market cap data over entire history of the stock given a ticker'''
    
    url_1 = "https://financialmodelingprep.com/api/v3/historical-market-capitalization/"
    url_2 = f"{ticker}?apikey=f0060c7a7b275396c21cb98f4985f3ae"
    mkt_cap = requests.get(url_1 + url_2)
    mkt_cap = mkt_cap.json()
    mkt_cap = pd.DataFrame(mkt_cap).set_index('date').drop(columns = ['symbol'], axis = 1)
    mkt_cap.columns = pd.MultiIndex.from_tuples([(ticker, 'mkt_cap')])
    mkt_cap.index.names = ['qtr']
    return mkt_cap
    
def Piotroski_F_score(ticker, df):
    '''computes Piotroski F-score for a given ticker and dataframe of Bloomberg finanical data;
    outputs Series object of F-scores '''
    
    # factor components
    ROA = df[ticker]["RETURN_ON_ASSET"]  # return on assets
    CFO = df[ticker]["CF_CASH_FROM_OPER"]  # cashflow from operations
    dROA = ROA.diff(4)  # y/y change in ROA 
    dTURN = df[ticker]['ASSET_TURNOVER'].diff(4)  # y/y change in asset turnover ratio
    dLEVER = df[ticker]['TOT_DEBT_TO_TOT_ASSET'].diff(4)  # y/y change leveraged assets
    # accrual
    NET_INCOME = df[ticker]["NET_INCOME"]
    TOT_CURR_ASSET = df[ticker]["BS_CUR_ASSET_REPORT"]
    TOT_CURR_LIABILITIES = df[ticker]['BS_CUR_LIAB']
    ACCRUAL = (NET_INCOME - CFO)/TOT_CURR_ASSET.shift(4)
    # y/y change in liquidity
    dLIQUID = TOT_CURR_ASSET/TOT_CURR_LIABILITIES - TOT_CURR_ASSET.shift(4)/TOT_CURR_LIABILITIES.shift(4)  
    dMARGIN = df[ticker]['GROSS_MARGIN'].diff(4)  # y/y change in gross margin ratio

    # factors
    f_index = ROA.index
    F_ROA = pd.Series(np.where(ROA > 0, 1, 0))  # F_ROA
    F_ROA.index = f_index
    F_dROA = pd.Series(np.where(dROA > 0, 1, 0))  # F_dROA
    F_dROA.index = f_index
    F_CFO = pd.Series(np.where(ROA > 0, 1, 0))  # F_CFO
    F_CFO.index = f_index
    F_ACCRUAL = pd.Series(np.where(CFO > ROA, 1, 0))  # F_ACCRUAL
    F_ACCRUAL.index = f_index
    F_dLEVER = pd.Series(np.where(dLEVER < dLEVER.shift(1), 1, 0))  # F_dLEVER
    F_dLEVER.index = f_index
    F_dLIQUID = pd.Series(np.where(dLIQUID > 0, 1, 0))  # F_dLIQUID
    F_dLIQUID.index = f_index
    F_dMARGIN = pd.Series(np.where(dMARGIN > 0, 1, 0))  # F_dMARGIN
    F_dMARGIN.index = f_index
    F_dTURN = pd.Series(np.where(dTURN > 0, 1, 0))  # F_dTURN
    F_dTURN.index = f_index

    # F-score
    F_score = F_ROA + F_dROA + F_CFO + F_ACCRUAL + F_dLEVER + F_dLIQUID + F_dMARGIN + F_dTURN 
    F_score = F_score[4:]  # remove the first year
    
    # construct DataFrame
    f_score_data = pd.DataFrame(F_score, index = F_score.index, columns = [ticker])
    f_score_data.columns = pd.MultiIndex.from_tuples([(ticker, 'f_score')])
    f_score_data.index.names = ['qtr']
    
    return f_score_data

def back_test_data(X, mod, colname):
    '''inputs: X = predictors in back-test sample; mod = model; colname = name of the predicted values column'''
    X['const'] = 1  # add an intercept column
    pred_results = mod.predict(X)
    X.loc[:,colname] = pred_results
    return X