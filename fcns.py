import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def stationarity_test(ts):
    """
        Descr: performs augmented dickey-fuller test for stationarity
        
        Params:
            ts: timeseries data object
            
        returns: array of statistical measures indicating stationarity
    """
    from statsmodels.tsa.stattools import adfuller
    res = adfuller(ts, autolag='AIC')
    out = pd.Series(res[0:4], index=['statistic', 'p-value', '#lags', '#obs'])
    if(out['p-value'] < .05):
        res = 1
    else:
        res = 0
    return res, out


def plot_autocorrelation(ts, lags):
    """
        Descr: returns acf and pacf plots
        
        Params:
            ts: timeseries object
            lags: number of lags
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(ts, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(ts, lags=lags, ax=ax2)
    

def simple_smoother(ts, window):
    """
        Descr: only works with pandas timeseries or df objects
    """
    rol_mean = ts.rolling(window).mean()
    rol_std  = ts.rolling(window).std()
    
    return rol_mean, rol_std

def ewma(ts, alpha):
    """
        Descr: only works with pandas timeseries or df objects
    """
    return ts.ewm(alpha=alpha).mean()

def plot_norm(data, label, title, bins='auto', density=True, width=.8):
    from scipy.stats import norm
    plt.figure(figsize=(12,8))
    plt.hist(data, bins=bins, density=density, rwidth=width, label=label)
    mu,std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'm', linewidth=2)
    plt.grid(axis='y', alpha=.25)
    plt.xlabel(label)
    plt.ylabel("PDF")
    plt.title("{}, mu:{:.2f}, std:{:.2f}".format(title, mu, std))