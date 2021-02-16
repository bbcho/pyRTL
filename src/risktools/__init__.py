from . import data

import warnings
from matplotlib.pyplot import subplots
import pandas as pd
import numpy as np
import quandl
from math import sqrt
import warnings as _warnings
from . import data
import arch
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from scipy import interpolate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly
import matplotlib.pyplot as plt
from datetime import datetime
from math import ceil
from ._morningstar import *
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

us_swap = data.open_data("usSwapCurves")

#####################################################################
# TODO
# * Add legend to chart_eia_sd function
#####################################################################


def ir_df_us(quandl_key=None, ir_sens=0.01):
    """
    Extracts US Tresury Zero Rates using Quandl

    Parameters
    ----------
    quandl_key : str
        Your Quandl key "yourkey" as a string, by default None. Optional as Quandl allows up to 50
        calls per day for free
    ir_sens : float
        Creates plus and minus IR sensitivity scenarios with specified shock value.

    Returns
    -------
    A pandas data frame of zero rates

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us()
    """
    # Last 30 days
    sdt = pd.Timestamp.now().floor("D") - pd.DateOffset(days=30)

    if quandl_key is not None:
        quandl.ApiConfig.api_key = quandl_key

    fedsfund = quandl.get("FED/RIFSPFF_N_D", start_date=sdt).dropna()
    fedsfund["FedsFunds0"] = np.log((1 + fedsfund.Value / 360) ** 365)
    fedsfund.drop("Value", axis=1, inplace=True)

    zero_1yr_plus = quandl.get("FED/SVENY")

    zero_tb = quandl.get(
        ["FED/RIFLGFCM01_N_B", "FED/RIFLGFCM03_N_B", "FED/RIFLGFCM06_N_B"],
        start_date=sdt,
    ).dropna()
    zero_tb.columns = zero_tb.columns.str.replace(" - Value", "")

    # get most recent full curve (some more recent days will have NA columns)
    x = fedsfund.join(zero_tb).join(zero_1yr_plus).dropna().iloc[-1, :].reset_index()
    x.columns = ["maturity", "yield"]
    x["yield"] /= 100
    x["maturity"] = x.maturity.str.extract("(\d+)").astype("int")

    # change maturity numbers to year fraction for first four rows
    x.iloc[1:4, x.columns.get_loc("maturity")] /= 12
    x.iloc[0, x.columns.get_loc("maturity")] = 1 / 365
    # x.maturity[1:4] /= 12
    # x.maturity[0] = 1/365

    # add new row for today, same yield as tomorrow
    x = pd.concat(
        [pd.DataFrame({"maturity": [0], "yield": [x["yield"][0]]}), x],
        ignore_index=True,
    )

    x["discountfactor"] = np.exp(-x["yield"] * x.maturity)
    x["discountfactor_plus"] = np.exp(-(x["yield"] + ir_sens) * x.maturity)
    x["discountfactor_minus"] = np.exp(-(x["yield"] - ir_sens) * x.maturity)

    return x


def simGBM(s0=10, drift=0, sigma=0.2, T=1, dt=1 / 12):
    """
    Simulates a Geometric Brownian Motion process

    Parameters
    ----------
    s0 : spot price at time = 0
    drift : drift %
    sigma : standard deviation
    T : maturity in years
    dt : time step in period e.g. 1/250 = 1 business day

    Returns
    -------
    A list of simulated values

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simGBM(s0=5, drift=0, sigma=0.2, T=2, dt=0.25)
    """
    periods = T / dt
    s = [s0] * int(periods)

    for i in range(1, int(periods)):
        s[i] = s[i - 1] * np.exp(
            (drift - (sigma ** 2) / 2) * dt
            + sigma * np.random.normal(loc=0, scale=1) * sqrt(dt)
        )

    return s


def simOU_arr(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252, sims=1000):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk) with multiple
    simulations

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float, int or pandas Series
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods)
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU_arr()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    periods = int(T / dt)

    if isinstance(mu, list):
        assert len(mu) == (
            periods - 1
        ), "Time dependent mu used, but the length of mu is not equal to the number of periods calculated."

    # init df with zeros, rows are steps forward in time, columns are simulations
    out = np.zeros((periods, sims))
    out = pd.DataFrame(data=out)

    # set first row as starting value of sim
    out.loc[0, :] = s0

    # print half-life of theta
    print("Half-life of theta in days = ", np.log(2) / theta * bdays_in_year)

    if isinstance(mu, list):
        mu = pd.Series(mu)

    for i, _ in out.iterrows():
        if i == 0:
            continue  # skip first row

        # calc gaussian vector
        ep = pd.Series(np.random.normal(size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, pd.Series):
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu.iloc[i - 1] - out.iloc[i - 1, :]) * dt
                + sigma * ep * sqrt(dt)
            )
        else:
            out.iloc[i, :] = (
                out.iloc[i - 1, :]
                + theta * (mu - out.iloc[i - 1, :]) * dt
                + sigma * ep * sqrt(dt)
            )

    return out


def simOU(s0=5, mu=4, theta=2, sigma=1, T=1, dt=1 / 252):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk)

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float, int or pandas Series
        Mean that the function will revert to
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year

    Returns
    -------
    A numpy array of simulated values

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOU()
    """
    s = np.array(simOU_arr(s0, mu, theta, sigma, T, dt, sims=1).iloc[:, 0])

    return s


def simOUJ_arr(
    s0=5,
    mu=5,
    theta=0.5,
    sigma=0.2,
    jump_prob=0.05,
    jump_avgsize=3,
    jump_stdv=0.05,
    T=1,
    dt=1 / 12,
    sims=1000,
):
    """
    Function for calculating an Ornstein-Uhlenbeck Jump Mean Reversion stochastic process (random walk) with multiple
    simulations

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float, int or pandas Series
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods)
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    jump_prob : float
        Probablity of jumps
    jump_avgsize : float
        Average size of jumps
    jump_stdv : float
        Standard deviation of average jump size
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOUJ_arr()
    """
    # number of business days in a year
    bdays_in_year = 252

    # number of periods dt in T
    periods = int(T / dt)

    if isinstance(mu, list):
        assert len(mu) == (
            periods - 1
        ), "Time dependent mu used, but the length of mu is not equal to the number of periods calculated."

    # init df with zeros, rows are steps forward in time, columns are simulations
    s = np.zeros((periods, sims))
    s = pd.DataFrame(data=s)

    # set first row as starting value of sim
    s.loc[0, :] = s0

    # print half-life of theta
    print("Half-life of theta in days = ", np.log(2) / theta * bdays_in_year)

    if isinstance(mu, list):
        mu = pd.Series(mu)

    for i, _ in s.iterrows():
        if i == 0:
            continue  # skip first row

        # calc gaussian and poisson vectors
        ep = pd.Series(np.random.normal(size=sims))
        elp = pd.Series(
            np.random.lognormal(mean=np.log(jump_avgsize), sigma=jump_stdv, size=sims)
        )
        jp = pd.Series(np.random.poisson(lam=jump_prob * dt, size=sims))

        # calc step
        if isinstance(mu, list) | isinstance(mu, pd.Series):
            s.iloc[i, :] = (
                s.iloc[i - 1, :]
                + theta
                * (mu.iloc[i - 1] - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
                + sigma * s.iloc[i - 1, :] * ep * sqrt(dt)
                + jp * elp
            )
        else:
            s.iloc[i, :] = (
                s.iloc[i - 1, :]
                + theta
                * (mu - jump_prob * jump_avgsize - s.iloc[i - 1, :])
                * s.iloc[i - 1, :]
                * dt
                + sigma * s.iloc[i - 1, :] * ep * sqrt(dt)
                + jp * elp
            )

    return s


def simOUJ(
    s0=5,
    mu=5,
    theta=0.5,
    sigma=0.2,
    jump_prob=0.05,
    jump_avgsize=3,
    jump_stdv=0.05,
    T=1,
    dt=1 / 12,
):
    """
    Function for calculating an Ornstein-Uhlenbeck Mean Reversion stochastic process (random walk) with Jump

    From Wikipedia:

    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process

    The process is a stationary Gauss–Markov process, which means that it is a Gaussian process, a Markov process,
    and is temporally homogeneous. In fact, it is the only nontrivial process that satisfies these three conditions,
    up to allowing linear transformations of the space and time variables. Over time, the process tends to drift
    towards its mean function: such a process is called mean-reverting.

    Parameters
    ----------
    s0 : float
        Starting value for mean reverting random walk at time = 0
    mu : float, int or pandas Series
        Mean that the function will revert to. Can be either a scalar value (i.e. 5) or a pandas series for a
        time dependent mean. If array-like, it must be the same length as T/dt (i.e. the number of periods)
    theta : float
        Mean reversion rate, higher number means it will revert slower
    sigma : float
        Annualized volatility or standard deviation. To calculate, take daily volatility and multiply by sqrt(T/dt)
    jump_prob : float
        Probablity of jumps
    jump_avgsize : float
        Average size of jumps
    jump_stdv : float
        Standard deviation of average jump size
    T : float or int
        Period length in years (i.e. 0.25 for 3 months)
    dt : float
        Time step size in fractions of a year. So a day would be 1/252, where 252 is the number of business
        days in a year
    sims : int
        Number of simulations to run

    Returns
    -------
    A pandas dataframe with the time steps as rows and the number of simulations as columns

    Examples
    --------
    >>> import risktools as rt
    >>> rt.simOUJ()
    """
    s = np.array(
        simOUJ_arr(
            s0, mu, theta, sigma, jump_prob, jump_avgsize, jump_stdv, T, dt, sims=1
        ).iloc[:, 0]
    )

    return s


def fitOU(spread):
    """
    Parameter estimation for the Ornstein-Uhlenbeck process

    Parameters
    ----------
    spread : array-like
        OU process as a list or series to estimate parameters for

    Returns
    -------
    Dictionary of alpha, mu and theta

    Examples
    --------
    >>> import risktools as rt
    >>> spread = rt.simOU(mu=5, theta=0.5, signma=0.2, T=5, dt=1/250)
    >>> rt.fitOU(spread)

    """
    n = len(spread)
    delta = 1

    Sx = spread[:-1].sum()
    Sy = spread[1:].sum()
    Sxx = (spread[:-1] ** 2).sum()
    Syy = (spread[1:] ** 2).sum()
    Sxy = (spread[:-1] * spread[1:]).sum()

    mu = (Sy * Sxx - Sx * Sxy) / ((n - 1) * (Sxx - Sxy) - (Sx ** 2 - Sx * Sy))
    theta = (
        -np.log(
            (Sxy - mu * Sx - mu * Sy + (n - 1) * mu ** 2)
            / (Sxx - 2 * mu * Sx + (n - 1) * mu ** 2)
        )
        / delta
    )
    a = np.exp(-theta * delta)
    sigmah2 = (
        Syy
        - 2 * a * Sxy
        + a ** 2 * Sxx
        - 2 * mu * (1 - a) * (Sy - a * Sx)
        + (n - 1) * mu ** 2 * (1 - a) ** 2
    ) / (n - 1)
    print((sigmah2) * 2 * theta / (1 - a ** 2))
    sigma = sqrt((sigmah2) * 2 * theta / (1 - a ** 2))
    theta = {"theta": theta, "mu": mu, "sigma": sigma}

    return theta


def bond(ytm=0.05, c=0.05, T=1, m=2, output="price"):
    """
    Compute bond price, cash flow table and duration

    Parameters
    ----------
    ytm : float
        Yield to Maturity
    c : float
        Coupon rate per annum
    T : float
        Time to maturity in years
    m : float
        Periods per year for coupon payments e.g semi-annual = 2.
    output : string
        "price", "df" or "duration", by default "df"

    Returns
    -------
    Price scalar, cash flows data frame and/or duration scalar

    Examples
    --------
    >>> import risktools as rt
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "price")
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "df")
    >>> rt.bond(ytm = 0.05, c = 0.05, T = 1, m = 2, output = "duration")
    """
    assert output in [
        "df",
        "price",
        "duration",
    ], "output not a member of ['df','price','duration']"

    # init df
    df = pd.DataFrame(
        {"t_years": np.arange(1 / m, T + 1 / m, 1 / m), "cf": [c * 100 / m] * (T * m)}
    )
    df["t_periods"] = df.t_years * m
    df.loc[df.t_years == T, "cf"] = c * 100 / m + 100
    df["disc_factor"] = 1 / ((1 + ytm / m) ** df["t_periods"])
    df["pv"] = df.cf * df.disc_factor
    price = df.pv.sum()
    df["duration"] = df.pv * df.t_years / price

    if output == "price":
        ret = price
    elif output == "df":
        ret = df
    elif output == "duration":
        ret = df.duration.sum()
    else:
        ret = None

    return ret


def return_cumulative(r, geometric=True):
    """
    Based on the function Return.annualize from the R package PerformanceAnalytics
    by Peter Carl and Brian G. Peterson

    Calculate a compounded (geometric) cumulative return. Based om R's PerformanceAnalytics

    This is a useful function for calculating cumulative return over a period of
    time, say a calendar year.  Can produce simple or geometric return.

    product of all the individual period returns

    \deqn{(1+r_{1})(1+r_{2})(1+r_{3})\ldots(1+r_{n})-1=prod(1+R)-1}{prod(1+R)-1}

    Parameters
    ----------
    r : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True

    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> spy = data.DataReader("SPY",  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> spy = spy.pct_change()
    >>> return_cumulative(spy['Adj Close'])
    >>> return_cumulative(spy['Adj Close'],geometric=FALSE)
    """
    r = r.dropna()

    if geometric:
        r = r.add(1).prod() - 1
    else:
        r = r.sum()

    return r


def return_annualized(r, scale=None, geometric=True):
    """
    Based on the function Return.annualize from the R package PerformanceAnalytics
    by Peter Carl and Brian G. Peterson

    Calculate an annualized return for comparing instruments with different
    length history

    An average annualized return is convenient for comparing returns.

    Annualized returns are useful for comparing two assets.  To do so, you must
    scale your observations to an annual scale by raising the compound return to
    the number of periods in a year, and taking the root to the number of total
    observations:
    \deqn{prod(1+R_{a})^{\frac{scale}{n}}-1=\sqrt[n]{prod(1+R_{a})^{scale}}-1}{prod(1
    + Ra)^(scale/n) - 1}

    where scale is the number of periods in a year, and n is the total number of
    periods for which you have observations.

    For simple returns (geometric=FALSE), the formula is:

    \deqn{\overline{R_{a}} \cdot scale}{mean(R)*scale}

    Parameters
    ----------
    r : Series or DataFrame
        pandas Series with a datetime index with freq defined. Freq must be either 'D','W','M','Q' or 'Y'
    scale : int
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq),
        specify your own scale to use.
    geometric : bool
        utilize geometric chaining (True) or simple/arithmetic chaining (False) to aggregate returns,
        by default True

    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    References
    ----------
    Bacon, Carl. \emph{Practical Portfolio Performance Measurement
    and Attribution}. Wiley. 2004. p. 6

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> return_annualized(r=df[('Adj Close','SPY')], geometric=True)
    >>> return_annualized(r=df[('Adj Close','SPY')], geometric=False)
    >>> return_annualized(r=df['Adj Close'], geometric=True)
    >>> return_annualized(r=df['Adj Close'], geometric=False)
    """
    r, scale = _check_ts(r, scale, name="r")

    r = r.dropna()
    n = r.shape[0]

    if geometric:
        res = (r.add(1).cumprod() ** (scale / n) - 1).iloc[-1]
    else:
        res = r.mean() * scale
    return res


def return_excess(R, Rf=0):
    """
    Calculates the returns of an asset in excess of the given risk free rate

    Calculates the returns of an asset in excess of the given "risk free rate"
    for the period.

    Ideally, your risk free rate will be for each period you have returns
    observations, but a single average return for the period will work too.

    Mean of the period return minus the period risk free rate

    \deqn{\overline{(R_{a}-R_{f})}}{mean(Ra-Rf=0)}

    OR

    mean of the period returns minus a single numeric risk free rate

    \deqn{\overline{R_{a}}-R_{f}}{mean(R)-rf}

    Note that while we have, in keeping with common academic usage, assumed that
    the second parameter will be a risk free rate, you may also use any other
    timeseries as the second argument.  A common alteration would be to use a
    benchmark to produce excess returns over a specific benchmark, as
    demonstrated in the examples below.

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    Rf : float
        risk free rate, in same period as your returns, or as a single
        digit average

    References
    ----------
    Bacon, Carl. \emph{Practical Portfolio Performance Measurement
    and Attribution}. Wiley. 2004. p. 47-52

    Examples
    --------
    data(managers)
    head(Return.excess(managers[,1,drop=FALSE], managers[,10,drop=FALSE]))
    head(Return.excess(managers[,1,drop=FALSE], .04/12))
    head(Return.excess(managers[,1:6], managers[,10,drop=FALSE]))
    head(Return.excess(managers[,1,drop=FALSE], managers[,8,drop=FALSE]))
    """

    res = R - Rf
    return res


def sd_annualized(x, scale=None, *args):
    """
    calculate a multiperiod or annualized Standard Deviation

    Standard Deviation of a set of observations \eqn{R_{a}} is given by:

    \deqn{\sigma = variance(R_{a}) , std=\sqrt{\sigma} }{std = sqrt(var(R))}

    It should follow that the variance is not a linear function of the number of
    observations.  To determine possible variance over multiple periods, it
    wouldn't make sense to multiply the single-period variance by the total
    number of periods: this could quickly lead to an absurd result where total
    variance (or risk) was greater than 100%.  It follows then that the total
    variance needs to demonstrate a decreasing period-to-period increase as the
    number of periods increases. Put another way, the increase in incremental
    variance per additional period needs to decrease with some relationship to
    the number of periods. The standard accepted practice for doing this is to
    apply the inverse square law. To normalize standard deviation across
    multiple periods, we multiply by the square root of the number of periods we
    wish to calculate over. To annualize standard deviation, we multiply by the
    square root of the number of periods per year.

    \deqn{\sqrt{\sigma}\cdot\sqrt{periods}}

    Note that any multiperiod or annualized number should be viewed with
    suspicion if the number of observations is small.

    Parameters
    ----------

    x : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    scale : int (optional)
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq),
        specify your own scale to use.
    *args
        any other passthru parameters

    Returns
    -------
    floating point number if r is a Series, or pandas Series if r is a DataFrame

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> sd_annualized(x=df[('Adj Close','SPY')])
    >>> sd_annualized(x=df['Adj Close'])
    """
    if (~isinstance(x, pd.DataFrame) & ~isinstance(x, pd.Series)) == True:
        raise ValueError("x must be a pandas Series or DataFrame")

    if isinstance(x.index, pd.DatetimeIndex):
        if scale is None:
            if (x.index.freq == "D") | (x.index.freq == "B"):
                scale = 252
            elif x.index.freq == "W":
                scale = 52
            elif (x.index.freq == "M") | (x.index.freq == "MS"):
                scale = 12
            elif (x.index.freq == "Q") | (x.index.freq == "QS"):
                scale = 4
            elif (x.index.freq == "Y") | (x.index.freq == "YS"):
                scale = 1
            else:
                raise ValueError(
                    "parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'"
                )
    else:
        raise ValueError(
            "parameter x's index must be a datetime index with freq 'D','W','M','Q' or 'Y'"
        )

    x = x.dropna()
    res = x.std() * sqrt(scale)

    return res


def omega_sharpe_ratio(R, MAR, *args):
    """
    Omega-Sharpe ratio of the return distribution

    The Omega-Sharpe ratio is a conversion of the omega ratio to a ranking statistic
    in familiar form to the Sharpe ratio.

    To calculate the Omega-Sharpe ration we subtract the target (or Minimum
    Acceptable Returns (MAR)) return from the portfolio return and we divide
    it by the opposite of the Downside Deviation.

    \deqn{OmegaSharpeRatio(R,MAR) = \frac{r_p - r_t}{\sum^n_{t=1}\frac{max(r_t - r_i, 0)}{n}}}{OmegaSharpeRatio(R,MAR) = (Rp - Rt) / -DownsidePotential(R,MAR)}

    where \eqn{n} is the number of observations of the entire series

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    MAR : int or array_like
        Minimum Acceptable Return, in the same periodicity as your
        returns
    *args:
        any other passthru parameters

    Returns
    -------

    Examples
    --------
    >>> mar = 0.005
    >>> print(omega_sharpe_ratio(portfolio_bacon[,1], MAR))
    """
    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(
        R, (pd.Series, pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = np.mean(MAR)

    result = (
        upside_risk(R, MAR, stat="potential")
        - downside_deviation(R, MAR, potential=True)
    ) / downside_deviation(R, MAR, potential=True)
    return result


def upside_risk(R, MAR=0, method="full", stat="risk"):
    """
    upside risk, variance and potential of the return distribution

    Upside Risk is the similar of semideviation taking the return above the
    Minimum Acceptable Return instead of using the mean return or zero.

    To calculate it, we take the subset of returns that are more than the target
    (or Minimum Acceptable Returns (MAR)) returns and take the differences of
    those to the target.  We sum the squares and divide by the total number of
    returns and return the square root.

    \deqn{ UpsideRisk(R , MAR) = \sqrt{\sum^{n}_{t=1}\frac{
    max[(R_{t} - MAR), 0]^2}{n}}}{UpsideRisk(R, MAR) = sqrt(1/n * sum(t=1..n)
    ((max(R(t)-MAR, 0))^2))}

    \deqn{ UpsideVariance(R, MAR) = \sum^{n}_{t=1}\frac{max[(R_{t} - MAR), 0]^2} {n}}{UpsideVariance(R, MAR) = 1/n * sum(t=1..n)((max(R(t)-MAR, 0))^2)}

    \deqn{UpsidePotential(R, MAR) = \sum^{n}_{t=1}\frac{max[(R_{t} - MAR), 0]} {n}}{DownsidePotential(R, MAR) =  1/n * sum(t=1..n)(max(R(t)-MAR, 0))}

    where \eqn{n} is either the number of observations of the entire series or
    the number of observations in the subset of the series falling below the
    MAR.

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    MAR : int
        Minimum Acceptable Return, in the same periodicity as your
        returns
    method : str
        Either "full" or "subset", indicating whether to use the
        length of the full series or the length of the subset of the series below
        the MAR as the denominator, defaults to "full"
    stat : str
        one of "risk", "variance" or "potential" indicating whether
        to return the Upside risk, variance or potential. By default,
        'risk'
    *args :
        any other passthru parameters

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> print(upside_risk(df['Adj Close'], MAR=0))
    >>> print(upside_risk(df[('Adj Close','SPY')], MAR=0))
    >>> print(upside_risk(df['Adj Close'], MAR=0, stat='variance'))
    >>> print(upside_risk(df['Adj Close'], MAR=0, stat='potential'))
    """

    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.gt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(
        R, (pd.Series, pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = np.mean(MAR)

    if method == "full":
        length = len(R)
    else:
        length = len(r)

    if stat == "risk":
        result = np.sqrt(r.sub(MAR).pow(2).div(length).sum())
    elif stat == "variance":
        result = r.sub(MAR).pow(2).div(length).sum()
    else:
        result = r.sub(MAR).div(length).sum()

    return result


def downside_deviation(R, MAR=0, method="full", potential=False):
    """
    Downside deviation, similar to semi deviation, eliminates positive returns
    when calculating risk.  To calculate it, we take the returns that are less
    than the target (or Minimum Acceptable Returns (MAR)) returns and take the
    differences of those to the target.  We sum the squares and divide by the
    total number of returns to get a below-target semi-variance.

    This is also useful for calculating semi-deviation by setting
    MAR = mean(R)

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    MAR : int
        Minimum Acceptable Return, in the same periodicity as your
        returns
    method : str
        Either "full" or "subset", indicating whether to use the
        length of the full series or the length of the subset of the series below
        the MAR as the denominator, defaults to "full"
    potential : bool
        potential if True, calculate downside potential instead, by default False
    *args :
        any other passthru parameters

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> print(downside_deviation(df[('Adj Close','SPY')], MAR=0))
    >>> print(downside_deviation(df['Adj Close'], MAR=0))
    """

    if isinstance(R, (pd.Series, pd.DataFrame)):
        if isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if ~isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "MAR index must be a datatime index if MAR and R are a Dataframe or Series with a datetime index"
                )
        elif ~isinstance(R.index, pd.DatetimeIndex) & isinstance(
            MAR, (pd.Series, pd.DataFrame)
        ):
            if isinstance(MAR.index, pd.DatetimeIndex):
                raise ValueError(
                    "R does not have a datetime index but MAR does. If both DataFrames or Series, index types must be the same"
                )

    R = R.dropna()
    r = R[R.lt(MAR)]

    if isinstance(MAR, (pd.Series, pd.DataFrame)) & isinstance(
        R, (pd.Series, pd.DataFrame)
    ):
        # subset to the same dates as the R data. Already checked that if both are series or dataframes, that
        # indices are of the same type
        MAR = MAR[r.index]
    else:
        # works for any array_like MAR. Scalars will just return itself
        # if MAR is array_like, we have to assume that R and MAR both
        # cover the same time period
        MAR = np.mean(MAR)

    if method == "full":
        length = len(R)
    else:
        length = len(r)

    if potential:
        result = r.mul(-1).add(MAR).div(length).sum()
    else:
        result = np.sqrt(r.mul(-1).add(MAR).pow(2).div(length).sum())
        # result = r.mul(-1).add(MAR).pow(2).div(length).sum().apply(np.sqrt)

    return result


def _check_ts(R, scale, name="R"):
    """
    Function to check frequency of R, a time series Series or Dataframe

    Parameters
    ----------
    R : Series or DataFrame
        A time series of asset returns
    scale : int (optional)
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None. Note that if scale is None,
        the function will calculate a scale based on the index frequency. If however
        you wish to override this (becuase maybe there is no index freq),
        specify your own scale to use.

    Returns
    -------
    tuple with R as Series or Dataframe and scale as int
    """
    if (~isinstance(R, pd.DataFrame) & ~isinstance(R, pd.Series)) == True:
        raise ValueError(f"{name} must be a pandas Series or DataFrame")

    if isinstance(R.index, pd.DatetimeIndex):
        if scale is None:
            if (R.index.freq == "D") | (R.index.freq == "B"):
                scale = 252
            elif R.index.freq == "W":
                scale = 52
            elif (R.index.freq == "M") | (R.index.freq == "MS"):
                scale = 12
            elif (R.index.freq == "Q") | (R.index.freq == "QS"):
                scale = 4
            elif (R.index.freq == "Y") | (R.index.freq == "YS"):
                scale = 1
            else:
                raise ValueError(
                    f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'"
                )
    else:
        raise ValueError(
            f"parameter {name}'s index must be a datetime index with freq 'D','B','W','M','Q' or 'Y'"
        )

    return R, scale


def sharpe_ratio_annualized(R, Rf=0, scale=None, geometric=True):
    """
    calculate annualized Sharpe Ratio

    The Sharpe Ratio is a risk-adjusted measure of return that uses standard
    deviation to represent risk.

    The Sharpe ratio is simply the return per unit of risk (represented by
    variance).  The higher the Sharpe ratio, the better the combined performance
    of "risk" and return.

    This function annualizes the number based on the scale parameter.

    \deqn{\frac{\sqrt[n]{prod(1+R_{a})^{scale}}-1}{\sqrt{scale}\cdot\sqrt{\sigma}}}

    Using an annualized Sharpe Ratio is useful for comparison of multiple return
    streams.  The annualized Sharpe ratio is computed by dividing the annualized
    mean monthly excess return by the annualized monthly standard deviation of
    excess return.

    William Sharpe now recommends Information Ratio preferentially to the
    original Sharpe Ratio.

    Parameters
    ----------
    R : array_like
        an xts, vector, matrix, data frame, timeSeries or zoo object of
        asset returns
    Rf : float
        risk free rate, in same period as your returns. By default 0
    scale : int
        number of periods in a year (daily scale = 252, monthly scale =
        12, quarterly scale = 4). By default None
    geometric : bool
        utilize geometric chaining (True) or simple/arithmetic chaining (False) to aggregate returns,
        default True

    see also \code{\link{SharpeRatio}} \cr \code{\link{InformationRatio}} \cr
    \code{\link{TrackingError}} \cr \code{\link{ActivePremium}} \cr
    \code{\link{SortinoRatio}}

    References
    ----------
    Sharpe, W.F. The Sharpe Ratio,\emph{Journal of Portfolio
    Management},Fall 1994, 49-58.

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')

    data(managers)
    SharpeRatio.annualized(managers[,1,drop=FALSE], Rf=.035/12)
    SharpeRatio.annualized(managers[,1,drop=FALSE], Rf = managers[,10,drop=FALSE])
    SharpeRatio.annualized(managers[,1:6], Rf=.035/12)
    SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE])
    SharpeRatio.annualized(managers[,1:6], Rf = managers[,10,drop=FALSE],geometric=FALSE)
    """
    R, scale = _check_ts(R, scale)

    xR = return_excess(R, Rf)
    res = return_annualized(xR, scale=scale, geometric=geometric)
    res /= sd_annualized(R, scale=scale)

    return res


def drawdowns(R, geometric=True, *args):
    """
    Function to calculate drawdown levels in a timeseries

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True

    """

    if geometric:
        res = R.add(1).cumprod()
    else:
        res = R.cumsum() + 1

    res = res / res.clip(lower=1).cummax() - 1

    return res


def find_drawdowns(R, geometric=True, *args):
    """
    Find the drawdowns and drawdown levels in a timeseries.

    find_drawdowns() will find the starting period, the ending period, and
    the amount and length of the drawdown.

    Often used with sort_drawdowns() to get the largest drawdowns.

    drawdowns() will calculate the drawdown levels as percentages, for use
    in \code{\link{chart.Drawdown}}.

    Returns an dictionary: \cr
    \itemize{
      \item return depth of drawdown
      \item from starting period
      \item to ending period \item length length in periods
    }

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    geometric : bool
        utilize geometric chaining (TRUE) or simple/arithmetic chaining (FALSE) to aggregate returns, by default True
    *args
        any pass-through parameters

    Returns
    -------
    Nested dictionary with asset name(s) as the top level key(s). Nested below that as another
    dictionary are the following keys and values:
        'return': numpy array of minimum of returns below the risk free rate of return (Rf) for each
            trough. If returns are positive, array has 0 value
        'from' : array index positiion of beginning of each trough or recovery period corresponding to each
            element of 'return'
        'trough' : array index position of peak trough period corresponding to each
            element of 'return'. Returns beginning of recovery periods
        'to' : array index positiion of end of each trough or recovery period corresponding to each
            element of 'return'
        'length' : length of each trough period corresponding to each element of 'return' as given by
            the difference in to and from index positions
        'peaktotrough' : array index distance from the peak of each trough or recovery period from the
            beginning of each trough or recovery period, corresponding to each element of 'return'
        'recovery' : array index distance from the peak of each trough or recovery period to the
            end of each trough or recovery period, corresponding to each element of 'return'

    References
    ----------
    Bacon, C. \emph{Practical Portfolio Performance Measurement and
    Attribution}. Wiley. 2004. p. 88 \cr

    Examples
    --------
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> rs = find_drawdowns(df['Adj Close'])
    >>> print(rs['SPY']['peaktotrough'])
    """
    dd = drawdowns(R, geometric=geometric).dropna()

    # init result dict
    rs = dict()

    # convert series into dataframe for flexibility
    series_flag = False
    if isinstance(dd, pd.Series):
        dd = pd.DataFrame({"drawdown": dd})
        series_flag = True

    for lab, con in dd.iteritems():
        rs[lab] = dict()
        rs[lab]["return"] = np.array([]).astype(float)
        rs[lab]["from"] = np.array([]).astype(int)
        rs[lab]["to"] = np.array([]).astype(int)
        rs[lab]["length"] = np.array([]).astype(int)
        rs[lab]["trough"] = np.array([]).astype(int)

        if con[0] >= 0:
            prior_sign = 1
        else:
            prior_sign = 0

        frm = 0
        to = 0
        dmin = 0
        sofar = con[0]

        for i, r in enumerate(con):  # .iteritems():
            if r < 0:
                this_sign = 0
            else:
                this_sign = 1

            if this_sign == prior_sign:
                if r < sofar:
                    sofar = r
                    dmin = i
                to = i + 1
            else:
                rs[lab]["return"] = np.append(rs[lab]["return"], sofar)
                rs[lab]["from"] = np.append(rs[lab]["from"], frm)
                rs[lab]["trough"] = np.append(rs[lab]["trough"], dmin)
                rs[lab]["to"] = np.append(rs[lab]["to"], to)

                frm = i
                sofar = r
                to = i + 1
                dmin = i
                prior_sign = this_sign

        rs[lab]["return"] = np.append(rs[lab]["return"], sofar)
        rs[lab]["from"] = np.append(rs[lab]["from"], frm)
        rs[lab]["trough"] = np.append(rs[lab]["trough"], dmin)
        rs[lab]["to"] = np.append(rs[lab]["to"], to)

        rs[lab]["length"] = rs[lab]["to"] - rs[lab]["from"] + 1
        rs[lab]["peaktotrough"] = rs[lab]["trough"] - rs[lab]["from"] + 1
        rs[lab]["recovery"] = rs[lab]["to"] - rs[lab]["trough"]

        # if original parameter was a series, remove top layer of
        # results dictionary
        if series_flag == True:
            rs = rs["drawdown"]

    return rs


def trade_stats(R, Rf=0):
    """
    Compute list of risk reward metrics

    Parameters
    ----------
    R : Series or DataFrame
        pandas Series or DataFrame with a datetime index. If DataFrame, function will calculate
        cummmulative returns on each column
    Rf : float
        risk free rate, in same period as your returns, or as a single
        digit average

    Returns
    -------
    Dictionary with cummulative returns, annual returns, Annualized Sharpe ratio, Omega Sharpe ratio,
    Win &, % in the market, and drawdown specs

    Examples
    --------
    >>> from pandas_datareader import data
    >>> df = data.DataReader(["SPY","AAPL"],  "yahoo", '2000-01-01', '2012-01-01')
    >>> df = df.pct_change()
    >>> df = df.asfreq('B')
    >>> rt.trade_stats(df[('Adj Close','SPY')])
    """

    # r = R.dropna() # don't use, messes up freq check
    r = R.copy()

    rs = dict()

    # convert series into dataframe for flexibility
    series_flag = False
    if isinstance(r, pd.Series):
        r = pd.DataFrame({"trade_stats": r})
        series_flag = True

    for lab, con in r.iteritems():
        rs[lab] = dict()
        y = find_drawdowns(con)
        rs[lab]["cum_ret"] = return_cumulative(con, geometric=True)
        rs[lab]["ret_ann"] = return_annualized(con, scale=252)
        rs[lab]["sd_ann"] = sd_annualized(con, scale=252)
        rs[lab]["omega"] = omega_sharpe_ratio(con, MAR=Rf) * sqrt(252)
        rs[lab]["sharpe"] = sharpe_ratio_annualized(con, Rf=Rf)

        # need to dropna to calc perc_win properly
        con_clean = con.dropna()
        rs[lab]["perc_win"] = (
            con_clean[con_clean > 0].shape[0] / con_clean[con_clean != 0].shape[0]
        )
        rs[lab]["perc_in_mkt"] = con_clean[con_clean != 0].shape[0] / con_clean.shape[0]

        rs[lab]["dd_length"] = max(y["length"])
        rs[lab]["dd_max"] = min(y["return"])

    if series_flag == True:
        rs = rs["trade_stats"]

    return rs


def returns(df, ret_type="abs", period_return=1, spread=False):
    """
    Computes periodic returns from a dataframe ordered by date

    Parameters
    ----------
    df : Series or DataFrame
        If pandas series, series must have a single datetime index or a multi-index in the same of ('asset name','date'). Values must be asset prices
        If pandas dataframe, dataframe must have a single datetime index with asset prices as columns
    ret_type : str
        "abs" for absolute, "rel" for relative, or "log" for log returns. By default "abs"
    period_return : int
        Number of rows over which to compute returns. By default 1
    spread : bool
        True if you want to spread into a wide dataframe. By default False

    Returns
    -------
    A pandas series of returns.

    Examples
    --------
    >>> import risktools as rt
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = True)
    >>> rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
    """
    df = _check_df(df)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({df.name: df})
    elif isinstance(df, pd.DataFrame) == False:
        raise ValueError("df is not a pandas Series or DataFrame")

    index_flag = False
    if len(df.index.names) == 1:
        # convert single index into multi-index by adding dummy index in position 0 to allow for groupby
        df["dummy"] = "dummy"
        df = df.set_index("dummy", append=True)
        df = df.swaplevel()
        index_flag = True

    if (spread == True) & (len(df.index.names) == 1):
        raise ValueError(
            "You have set spread to True but only passed a single index, not a multi-index. There is nothing to spread"
        )

    # df = df.dropna()

    # calc return type
    if ret_type == "abs":
        df = df.groupby(level=0).apply(lambda x: x.diff())
    elif ret_type == "rel":
        df = df.groupby(level=0).apply(lambda x: x / x.shift(period_return) - 1)
    elif ret_type == "log":
        if df[df < 0].count().sum() > 0:
            warnings.warn(
                "Negative values passed to log returns. You will likely get NaN values using log returns",
                RuntimeWarning,
            )
        df = df.groupby(level=0).apply(lambda x: np.log(x / x.shift(period_return)))
    else:
        raise ValueError("ret_type is not valid")

    if spread:
        # return spread df, don't dropna or it will remove entire rows
        df = df.unstack(level=0).droplevel(level=0, axis=1)
        return df
    else:
        if index_flag == True:
            # drop dummy index to return single index
            df = df.droplevel(0)
        if len(df.columns) == 1:
            return df.dropna().iloc[:, 0]
        else:
            return df.dropna()


def roll_adjust(
    df, commodity_name="cmewti", roll_type="Last_Trade", roll_sch=None, *args
):
    """
    Returns a pandas series adjusted for contract roll. The methodology used to adjust returns is to remove the daily returns on
    the day after expiry and for prices to adjust historical rolling front month contracts by the size of the roll at
    each expiry. This is conducive to quantitative trading strategies as it reflects the PL of a financial trader.

    Note that this will apply a single expiry schedule to all assets passed to this function. To apply different roll schedules
    by asset type, perform a separate function call.

    Parameters
    ----------
    df : Series | DataFrame
        pandas series or dataframe with a with a datetime index and which columns are asset prices or with a dataframe with a multi-index in the shape ('asset name','date')
    commodity_name : str
        Name of commodity in expiry_table. See example below for values.
    roll_type : str
        Type of contract roll: "Last_Trade" or "First_Notice". By default "Last_Trade"
    roll_sch : Series
        For future capability. Optional
    *args: Other parms to pass to function

    Returns
    -------
    Roll-adjusted pandas series object of returns with datetime index or multi-index with the shape ('asset name','date')

    Examples
    --------
    >>> import risktools as rt
    >>> dflong = rt.data.open_data('dflong')
    >>> rt.data.open_data('expiry_table').cmdty.unique() # for list of commodity names
    >>> ret = rt.returns(df=dflong, ret_type="abs", period_return=1, spread=True)
    >>> ret = ret.iloc[:,0]
    >>> rt.roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade")
    """

    df = _check_df(df)

    if isinstance(df, pd.Series):
        df = pd.DataFrame({df.name: df})
    elif isinstance(df, pd.DataFrame) == False:
        raise ValueError("df is not a pandas Series or DataFrame")

    if roll_sch is None:
        roll_sch = data.open_data("expiry_table")
        roll_sch = roll_sch[roll_sch.cmdty == commodity_name]
        roll_sch = roll_sch[roll_type]

    df["expiry"] = df.index.isin(roll_sch, level=-1)
    df["expiry"] = df["expiry"].shift()

    df = df[df.expiry == False].drop("expiry", axis=1)

    if len(df.columns) == 1:
        return df.dropna().iloc[:, 0]
    else:
        return df.dropna()


def garch(df, out="data", scale=None, show_fig=True, forecast_horizon=1, **kwargs):
    """
    Computes annualised Garch(1,0,1) volatilities using arch package. Note that the
    RTL package uses a sGarch model, but there is no similiar implementation in
    python.

    Parameters
    ----------
    df : DataFrame
        Wide dataframe with date column and single series (univariate).
    out : str
        "plotly" or "matplotlib" to return respective chart types. "data" to return data or "fit" for garch fit output. By default 'data'
    show_fig : bool
        Only used if out is 'matplotlib' or 'plotly'. If True, function will display plots as well as return their respective fig object
    **kwargs
        key-word parameters to pass tp arch_model. if none, sets p=1, o=0, q=1

    Returns
    -------
    out='data' returns a pandas df
    out='fit' returns an arch_model object from the package arch
    out='matplotlib' returns matplotlib figure and axes objects and displays the plot
    out='plotly' returns a plotly figure object and displays the plot

    Examples
    --------
    >>> import risktools as rt
    >>> dflong = rt.data.open_data('dflong')
    >>> dflong = dflong['CL01']
    >>> df = rt.returns(df=dflong, ret_type="rel", period_return=1)
    >>> df = rt.roll_adjust(df=df, commodity_name="cmewti", roll_type="Last_Trade")
    >>> rt.garch(df, out="data")
    >>> rt.garch(df, out="fit")
    >>> rt.garch(df, out="plotly")
    >>> rt.garch(df, out="matplotlib")
    """
    df = _check_df(df).sort_index()

    # if kwargs is None:
    #     kwargs = {"p": 1, "o": 0, "q": 1}

    df = df - df.mean()

    # find a more robust way to do this, rolladjust may break it
    freq = pd.infer_freq(df.index[-10:])

    if scale is None:
        if (freq == "B") or (freq == "D"):
            scale = 252
        elif freq[0] == "W":
            scale = 52
        elif (freq == "M") or (freq == "MS"):
            scale = 12
        elif (freq == "Q") or (freq == "QS"):
            scale = 4
        elif (freq == "Y") or (freq == "YS"):
            scale = 1
        else:
            raise ValueError(
                "Could not infer frequency of timeseries, please provide scale parameter instead"
            )

    # a standard GARCH(1,1) model
    garch = arch.arch_model(df, **kwargs)
    garch_fitted = garch.fit()

    # # calc annualized volatility from variance
    yhat = np.sqrt(
        garch_fitted.forecast(horizon=forecast_horizon, start=0).variance
    ) * sqrt(scale)

    if out == "data":
        return yhat
    elif out == "fit":
        return garch_fitted
    elif out == "plotly":
        fig = px.line(yhat)
        fig.show()
        return fig
    elif out == "matplotlib":
        fig, ax = plt.subplots(1, 1)
        yhat.plot(ax=ax)
        fig.show()
        return fig, ax


def _beta(y, x, subset=None):
    """
    Function to perform a linear regression on y = f(x) -> y = beta*x + const and returns
    beta

    Parameters
    ----------
    y : array-like
        dependent variable
    x : array-like
        independent variable
    subset : array-like of type bool
        array/series/list of bool to subset x and Y

    Examples
    --------
    >>> import numpy as np
    >>> import risktools as rt
    >>> x = np.array([1,2,3,4,5,6,7,8])
    >>> y = x + 5
    >>> rt._beta(x,y)
    """
    x = x.dropna()
    y = y.dropna()

    if subset is None:
        subset = np.repeat(True, len(x))
    else:
        subset = subset.dropna().astype(bool)

    if (
        (isinstance(x, (np.ndarray, pd.Series)) == False)
        & (isinstance(y, (np.ndarray, pd.Series)) == False)
        & (isinstance(subset, (np.ndarray, pd.Series)) == False)
    ):
        raise ValueError(
            "all arguements of _beta must be pandas Series or numpy arrays"
        )

    # convert to arrays
    x = np.array(x)
    y = np.array(y)
    subset = np.array(subset)

    # subset
    x = x[subset]
    y = y[subset]

    model = LinearRegression()
    model.fit(x.reshape((-1, 1)), y)
    beta = model.coef_

    return beta[0]


def CAPM_beta(Ra, Rb, Rf=0, kind="all"):
    """
    calculate single factor model (CAPM) beta

    The single factor model or CAPM Beta is the beta of an asset to the variance
    and covariance of an initial portfolio.  Used to determine diversification potential.

    This function uses a linear intercept model to achieve the same results as
    the symbolic model used by \code{\link{BetaCoVariance}}

    \deqn{\beta_{a,b}=\frac{CoV_{a,b}}{\sigma_{b}}=\frac{\sum((R_{a}-\bar{R_{a}})(R_{b}-\bar{R_{b}}))}{\sum(R_{b}-\bar{R_{b}})^{2}}}{beta
    = cov(Ra,Rb)/var(R)}

    Ruppert(2004) reports that this equation will give the estimated slope of
    the linear regression of \eqn{R_{a}}{Ra} on \eqn{R_{b}}{Rb} and that this
    slope can be used to determine the risk premium or excess expected return
    (see Eq. 7.9 and 7.10, p. 230-231).

    kind='bull' and kind='bear' apply the same notion of best fit to positive and
    negative market returns, separately. kind='bull' is a
    regression for only positive market returns, which can be used to understand
    the behavior of the asset or portfolio in positive or 'bull' markets.
    Alternatively, kind='bear' provides the calculation on negative
    market returns.

    The function \code{TimingRatio} may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio, which is calculated as
    \deqn{TimingRatio =\frac{\beta^{+}}{\beta^{-}}}{Timing Ratio = beta+/beta-}
    is best when greater than one in a rising market and less than one in a
    falling market.

    While the classical CAPM has been almost completely discredited by the
    literature, it is an example of a simple single factor model,
    comparing an asset to any arbitrary benchmark.

    Parameters
    ----------
    Ra : array-like or DataFrame
        Array-like or DataFrame with datetime index of asset returns to be tested vs benchmark
    Rb : array-like
        Benchmark returns to use to test Ra
    Rf : array-like | float
        risk free rate, in same period as your returns, or as a single
        digit average
    kind : str
        Market type to return, by default 'all'
        'all' : returns beta for all market types
        'bear' : returns beta for bear markets
        'bull' : returns beta for bull markets

    Returns
    -------
    If Ra is array-like, function returns beta as a scalar. If Ra is a DataFrame, it will return
    a series indexed with asset names from df columns

    References
    ----------
    Sharpe, W.F. Capital Asset Prices: A theory of market
    equilibrium under conditions of risk. \emph{Journal of finance}, vol 19,
    1964, 425-442.
    Ruppert, David. \emph{Statistics and Finance, an Introduction}. Springer. 2004.
    Bacon, Carl. \emph{Practical portfolio performance measurement and attribution}. Wiley. 2004. \cr

    Examples
    --------
    >>> import risktools as rt
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["XOM","AAPL","SPY"],  "yahoo", datetime(2010,1,1), datetime(2017,12,31))
    >>> df = df.pct_change()['Adj Close']
    >>> df = df.asfreq('B')
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0, kind='bear'))
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0, kind='bull'))
    >>> print(rt.CAPM_beta(df[['XOM','AAPL']], df['SPY'], Rf=0))
    """

    xRa = return_excess(Ra, Rf)
    xRb = return_excess(Rb, Rf)

    if kind == "bear":
        subset = xRb.lt(0).mask(
            xRb.isna(), np.nan
        )  # need to include the mask, otherwise lt/gt returns False for NaN
    elif kind == "bull":
        subset = xRb.gt(0).mask(
            xRb.isna(), np.nan
        )  # need to include the mask, otherwise lt/gt returns False for NaN
    else:
        subset = None

    if isinstance(xRa, pd.DataFrame):
        # applies function _beta to each columns of df
        rs = xRa.apply(lambda x: _beta(x, xRb, subset), axis=0)
    else:
        rs = _beta(xRa, xRb, subset)

    return rs


def timing_ratio(Ra, Rb, Rf=0):
    """
    The function \code{TimingRatio} may help assess whether the manager is a good timer
    of asset allocation decisions.  The ratio, which is calculated as
    \deqn{TimingRatio =\frac{\beta^{+}}{\beta^{-}}}{Timing Ratio = beta+/beta-}
    is best when greater than one in a rising market and less than one in a
    falling market.

    Parameters
    ----------
    Ra : array-like or DataFrame
        Array-like or DataFrame with datetime index of asset returns to be tested vs benchmark
    Rb : array-like
        Benchmark returns to use to test Ra
    Rf : array-like | float
        risk free rate, in same period as your returns, or as a single
        digit average

    Returns
    -------
    If Ra is array-like, function returns beta as a scalar. If Ra is a DataFrame, it will return
    a series indexed with asset names from df columns

    Examples
    --------
    >>> import risktools as rt
    >>> from pandas_datareader import data, wb
    >>> from datetime import datetime
    >>> df = data.DataReader(["XOM","AAPL","SPY"],  "yahoo", datetime(2010,1,1), datetime(2017,12,31))
    >>> df = df.pct_change()['Adj Close']
    >>> df = df.asfreq('B')
    >>> rt.timing_ratio(df[['XOM','AAPL']], df['SPY'], Rf=0)
    """

    beta_bull = CAPM_beta(Ra, Rb, Rf=Rf, kind="bull")
    beta_bear = CAPM_beta(Ra, Rb, Rf=Rf, kind="bear")

    result = beta_bull / beta_bear

    return result


def prompt_beta(df, period="all", beta_type="all", output="chart"):
    """
    Returns array/dataframe of betas for futures contract returns of a commodity
    with it's front contract (i.e. next most expirying contract). For use with futures
    contracts (i.e. NYMEX WTI: CL01, CL02, CL03 and so forth) with standardized expiry periods.

    Using the WTI example, the betas will represent the covariance of daily returns of
    CL02, CL03, CL04 with CL01. The covariance of CL01 with CL01 is also returned, but is
    always 1.

    This function uses the single factor model (CAPM) beta. See the function CAPM_beta
    for more details

    Parameters
    ----------
    df : DataFrame
        Wide dataframe with datetime index and multiple series columns for each futures contract.
        Always use continuous contracts for columns.
    period : str | int | float
        Timeframe to use to calculate beta. "all" to use all data available or scalar number
        n to only use the last n periods/rows of data from the last, by default 'all'. i.e.
        for WTI contracts (CL), 30 would be the last 30 days. Recommend running roll_adjust
        function prior to using prompt_beta to remove swings from contract expiry
    beta_type : str
        'all', 'bull', or 'bear', by default 'all'
    output : str
        'betas', 'chart', 'stats', by default 'chart'

    Returns
    -------
    output='chart' : A plotly figure with the beta lines charted for 'all', 'bear' and 'bull' markets
    output='betas' : A dataframe of betas by contract order for 'all', 'bear' and 'bull' markets
    output='stats' : A scipy object from a least_squares fit of the betas by market type. Model used
                        to fit betas was of the form:
                            \{beta} = x0 * exp(x1*t) + x2
                        where t is the contract order (1, 2, 3 etc..., lower for expirying sooner)

    chart, df of betas or stats

    Examples
    --------
    >>> import risktools as rt
    >>> dfwide = rt.data.open_data('dfwide')
    >>> col_mask = dfwide.columns[dfwide.columns.str.contains('CL')]
    >>> dfwide = dfwide[col_mask]
    >>> x = rt.returns(df=dfwide, ret_type="abs", period_return=1)
    >>> x = rt.roll_adjust(df=x, commodity_name="cmewti", roll_type="Last_Trade")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="chart")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="betas")
    >>> rt.prompt_beta(df=x, period="all", beta_type="all", output="stats")
    >>> rt.prompt_beta(df=x, period=30, beta_type="all", output="plotly")
    """
    df = df.copy()
    # this assumes that the numeric component of the column name represents
    # an order to the asset contract
    term = df.columns.str.replace("[^0-9]", "").astype(int)

    if isinstance(period, (int, float)):
        df = df.sort_index()
        df = df.iloc[-period:]

    # leave only contract # in column names
    df.columns = term
    df = df.sort_index()

    # calculate betas by market type (mkt is all types) using front contract as the benchmark
    mkt = CAPM_beta(df, df.iloc[:, 0])
    bull = CAPM_beta(df, df.iloc[:, 0], kind="bull")
    bear = CAPM_beta(df, df.iloc[:, 0], kind="bear")

    # create array for non-linear least squares exponential
    prompt = np.arange(0, mkt.shape[0]) + 1

    # proposed model for beta as a function of prompt
    def beta_model(x, prompt):
        return x[0] * np.exp(x[1] * prompt) + x[2]

    # cost function for residuals. Final equation that we're trying to minimize is
    # beta - x0*exp(x1*prompt) + x2
    def beta_residuals(x, beta, prompt):
        return beta - beta_model(x, prompt)

    # run least squares fit. Note that I ignore the first row of mkt and prompt arrays since
    # correlation of a var with itself should always be 1. Also, the beta of the second contract
    # will likely be a lot less then 1, and so ignoring the 1st contract will allow for a better fit
    r = least_squares(
        beta_residuals, x0=[-1, -1, -1], args=(np.array(mkt[1:]), prompt[1:])
    )

    # construct output df
    out = pd.DataFrame()
    out["all"] = mkt
    out["bull"] = bull
    out["bear"] = bear

    if output == "stats":
        return r
    elif output == "betas":
        return out
    elif output == "chart":
        out = out[
            1:
        ]  # exclude beta of front contract with itself (always 1, no information)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=out.index, y=out["all"], mode="lines", name="all"))
        fig.add_trace(go.Scatter(x=out.index, y=out["bear"], mode="lines", name="bear"))
        fig.add_trace(go.Scatter(x=out.index, y=out["bull"], mode="lines", name="bear"))
        fig.update_xaxes(range=[out.index.min() - 1, out.index.max() + 1])
        fig.update_layout(
            title="Contract Betas vs Front Contract: Bear (Bull) = Beta in Down (Up) Moves",
            xaxis_title="Contract",
            yaxis_title="Beta",
            legend_title="Market Type",
        )
        return fig


def custom_date_range(start, end, freq):
    if freq == "M":
        mul = 1
    elif freq == "Q":
        mul = 3
    elif (freq == "H") | (freq == "6M"):
        mul = 6
    elif freq == "Y":
        mul = 12
    else:
        raise ValueError("freq is not in ['M','Q','H' or 'Y']")

    dr = [pd.to_datetime(start)]

    i = 1
    while dr[-1] < pd.to_datetime(end):
        dr.append(dr[0] + pd.DateOffset(months=mul * i))
        i += 1

    return pd.Index(dr)


def swap_irs(
    trade_date=pd.Timestamp.now().floor("D"),
    eff_date=pd.Timestamp.now().floor("D") + pd.DateOffset(days=2),
    mat_date=pd.Timestamp.now().floor("D")
    + pd.DateOffset(days=2)
    + pd.DateOffset(years=2),
    notional=1000000,
    pay_rec="rec",
    fixed_rate=0.05,
    float_curve=us_swap,
    reset_freq="Q",
    disc_curve=us_swap,
    days_in_year=360,
    convention="act",
    bus_calendar="NY",  # not implemented
    output="price",
):
    """
    Commodity swap pricing from exchange settlement

    Parameters
    ----------
    trade_date : Timestamp | str
        Defaults to today().
    eff_date : Timestamp | str
        Defaults to today() + 2 days.
    mat_date : Timestamp | str
        Defaults to today() + 2 years.
    notional : long int
        Numeric value of notional. Defaults to 1,000,000.
    pay_rec : str
        "pay" for pyement (positive pv) or "rec" for receivables (negative pv).
    fixed_rate : float
        fixed interest rate. Defaults to 0.05.
    float_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. Defaults to rt.data.open_data("usSwapCurves")
        which is a dictionary based on the R object DiscountCurve. Column times is year fractions going forward in time.
        So today is 0, tomorrow is 1/365, a year from today is 1 etc...
    reset_freq : str
        Pandas/datetime Timestamp frequency (allowable values are 'M', 'Q', '6M', or 'Y')
    disc_curve : DataFrame | Dict
        DataFrame of interest rate curves with columns 'times' and 'discounts'. Defaults to rt.data.open_data("usSwapCurves")
        which is a dictionary based on the R object DiscountCurve. Column times is year fractions going forward in time.
        So today is 0, tomorrow is 1/365, a year from today is 1 etc...
    days_in_year : int
        360 or 365
    convention : list
        List of convention e.g. ["act",360], [30,360],...
    bus_calendar : str
        Banking day calendar. Not implemented.
    output : str
        "price" for swap price or "all" for price, cash flow data frame, duration.

    Returns
    -------
    returns present value as a scalar if output=='price. Otherwise returns dictionary with swap price, cash flow data frame and duration

    Examples
    --------
    >>> import risktools as rt
    >>> usSwapCurves = rt.data.open_data('usSwapCurves')
    >>> rt.swap_irs(trade_date="2020-01-04", eff_date="2020-01-06", mat_date="2022-01-06", notional=1000000, pay_rec = "rec", fixed_rate=0.05, float_curve=usSwapCurves, reset_freq='Q', disc_curve=usSwapCurves, days_in_year=360, convention="act", bus_calendar="NY", output = "all")
    """
    dates = custom_date_range(eff_date, mat_date, freq=reset_freq)
    dates = pd.Index([pd.to_datetime(trade_date)]).append(dates)

    if (days_in_year in [360, 365]) == False:
        raise ValueError("days_in_year must be either 360 or 365")

    # placeholder
    if convention != "act":
        raise ValueError("function only defined for convention='act'")

    df = pd.DataFrame(
        {
            "dates": dates,
            "day2next": (dates[1:] - dates[:-1]).days.append(
                pd.Index([0])
            ),  # calc days to next period, short one element at end so add zero
            "times": (dates - dates[0]).days
            / 365,  # calc days to maturity from trade_date
        }
    )
    disc = interpolate.splrep(disc_curve["times"], disc_curve["discounts"])
    df["disc"] = interpolate.splev(df.times, disc)

    df["fixed"] = notional * fixed_rate * (df.day2next / days_in_year)
    df.loc[df.day2next <= 20, "fixed"] = 0
    df.fixed = df.fixed.shift() * df.disc

    disc_float = interpolate.splrep(float_curve["times"], float_curve["discounts"])
    df["disc_float"] = interpolate.splev(df.times, disc_float)
    df["floating"] = notional * (df.disc_float / df.disc_float.shift(-1) - 1)
    df.loc[df.day2next <= 20, "floating"] = 0
    df.floating = df.floating.shift() * df.disc
    df["net"] = df.fixed - df.floating

    df = df.fillna(0)
    pv = df.net.sum()
    df["duration"] = (
        (dates - pd.to_datetime(trade_date)).days / days_in_year * df.net / pv
    )
    duration = df.duration.sum()

    if pay_rec == "pay":
        pv *= -1

    if output == "price":
        return pv
    else:
        return {"pv": pv, "df": df, "duration": duration}


def npv(
    init_cost=-375,
    C=50,
    cf_freq=0.25,
    F=250,
    T=2,
    disc_factors=None,
    break_even=False,
    be_yield=0.01,
):
    """
    Compute Net Present Value using discount factors from ir_df_us()

    Parameters
    ----------
    init_cost : float | int
        Initial investment cost
    C : float | int
        Periodic cash flow
    cf_freq : float
        Cash flow frequency in year fraction e.g. quarterly = 0.25
    F : float | int
        Final terminal value
    T : float | int
        Final maturity in years
    disc_factors : DataFrame
        Data frame of discount factors using ir_df_us() function.
    break_even : bool
        True when using a flat discount rate assumption.
    be_yield : float | int
        Set the flat IR rate when beak_even = True.

    Returns
    -------
    A python dictionary with elements 'df' as dataframe and 'npv' value as a float

    Examples
    --------
    >>> import risktools as rt
    >>> ir = rt.ir_df_us(ir_sens=0.01)
    >>> rt.npv(init_cost=-375, C=50, cf_freq=0.5, F=250, T=2, disc_factors=ir, break_even=True, be_yield=.0399)
    """
    disc_factors = disc_factors.copy()

    if disc_factors is None:
        raise ValueError(
            "Please input a discount factor dataframe into disc_factors (use ir_df_us to get dataframe)"
        )

    if break_even == True:
        print("test")
        disc_factors["yield"] = be_yield
        disc_factors["discountfactor"] = np.exp(
            -disc_factors["yield"] * disc_factors.maturity
        )

    n = len(np.arange(0, T, cf_freq)) + 1

    disc_intp = interpolate.splrep(disc_factors.maturity, disc_factors.discountfactor)

    df = pd.DataFrame(
        {
            "t": np.append(
                np.arange(0, T, cf_freq), [T]
            ),  # need to append T since np.arange is of type [a,b)
            "cf": np.ones(n) * C,
        }
    )

    df.loc[df.t == 0, "cf"] = init_cost
    df.loc[df.t == T, "cf"] = F

    df["df"] = interpolate.splev(df.t, disc_intp)
    df["pv"] = df.cf * df.df

    return df


def crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call"):
    """
    European option binomial model on a stock without dividends. For academic purposes only.
    Use fOptions::CRRBinomialTreeOptions for real-life usage in R or Python equivalent.

    Parameters
    ----------
    s : float
        Stock price, by default 100
    x : float
        Strike price, by default 100
    sigma : float
        Implied volatility, by default 0.20
    Rf : float
        Risk-free rate, by default 0.1
    T : float
        Time to maturity in years, by default 1
    n :
        Number of time steps, by default 5. Internally dt = T/n.
    type : str
        "call" or "put", by default "call"

    Returns
    -------
    List of asset price tree, option value tree and option price.

    Examples
    --------
    >>> import risktools as rt
    >>> rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")
    """

    dt = T / n

    # define u, d, and risk-neutral probability
    u = np.exp(sigma * sqrt(dt))
    d = np.exp(-sigma * sqrt(dt))
    q = (np.exp(Rf * dt) - d) / (u - d)

    # define our asset tree prices
    asset = np.zeros([n + 1, n + 1])

    for i in range(0, n + 1):
        for j in range(0, i + 1):
            asset[i, j] = s * (u ** j) * (d ** (i - j))

    # create matrix of the same dims as asset price tree
    option = np.zeros([n + 1, n + 1])
    # replace last row with maturity payoffs
    if type == "call":
        option[-1, :] = asset[-1, :] - x
    elif type == "put":
        option[-1, :] = x - asset[-1, :]
    else:
        raise ValueError("type is not of ('call','put')")
    option[option < 0] = 0

    # we discount recursively starting from final payoff (last row of tree)
    # starting at the second last period based on final payoffs
    for i in range(n - 1, -1, -1):  # loop for n-1 to 0 because range(a,b) is [a,b)
        for j in range(0, i + 1):
            option[i, j] = (
                (1 - q) * option[i + 1, j] + q * option[i + 1, j + 1]
            ) / np.exp(Rf * dt)

    # indicator if model can be used sigma > rsqrt(dt)
    if sigma > sqrt(dt) * Rf:
        note = "ok"
    else:
        note = "sigma < Rf*sqrt(dt) do not use"

    return {"asset": asset, "option": option, "price": option[0, 0], "note": note}


def stl_decomposition(
    df,
    output="chart",
    period=None,
    seasonal=7,
    seasonal_deg=1,
    resample_freq="M",
    **kwargs,
):
    """
    Provides a summary of returns distribution using the statsmodels.tsa.seasonal.STL class.
    Resamples all data to  prior to STL if

    Parameters
    ----------
    df : Series
        A pandas series (univariate) with a datetime index.
    output : str
        "chart" to see output as a graph, "data" for results as a list. By default "chart"
    seasonal : int, optional
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7 (default).
    seasonal_deg : int, optional
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend). By default 1
    resample_freq : str
        Resampling frequency to use in pandas resample function. Set to None to not resample
    **kwargs :
        Other parms



    season - The length of the seasonal smoother. Must be odd.

    trend - The length of the trend smoother, usually around 150% of season. Must be odd and larger than season.

    low_pass - The length of the low-pass estimation window, usually the smallest odd number larger than the periodicity of the data.


    Returns
    -------
    a chart or statsmodels.tsa.seasonal.DecomposeResult object of results

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.open_data('dflong')
    >>> df = df['CL01']
    >>> rt.stl_decomposition(df, output="chart", seasonal=13, seasonal_deg=1, resample_freq='M')
    >>> rt.stl_decomposition(df, output="data", seasonal=13, seasonal_deg=1, resample_freq='M')
    """

    register_matplotlib_converters()
    sns.set_style("darkgrid")

    if resample_freq is not None:
        df = df.resample(resample_freq).mean()

    stl = STL(
        df,
        period=period,
        seasonal=seasonal,
        seasonal_deg=seasonal_deg,
        robust=True,
        **kwargs,
    )
    res = stl.fit()

    if output == "chart":
        fig = res.plot()
        fig.set_size_inches(15, 15)
        return fig  # res.plot()
    else:
        return res


def get_eia_df(tables, key):
    """
    Function for download data from the US Government EIA and return it as a pandas dataframe/series

    To get an api key, go to https://www.eia.gov/opendata/register.php
    To get table names, search via API explorer at https://www.eia.gov/opendata/qb.php

    Parameters
    ----------
    tables : List[Tuple[str]]
        EIA series to return. Can be a list or tuple of tables as well.
    key : str
        EIA key.

    Returns
    -------
    pandas dataframe or series depending on the number of tables requested

    Examples
    --------
    >>> import risktools as rt
    >>> rt.get_eia_df('PET.WDIIM_R20-Z00_2.W', key=eia_key)
    """
    import requests
    import json

    if isinstance(tables, list) == False:
        tables = [tables]

    eia = pd.DataFrame()

    for tbl in tables:
        url = r"http://api.eia.gov/series/?api_key={}&series_id={}&out=json".format(
            key, tbl
        )

        tmp = json.loads(requests.get(url).text)

        tf = pd.DataFrame(tmp["series"][0]["data"], columns=["date", "value"])
        tf["table_name"] = tmp["series"][0]["name"]
        tf["series_id"] = tmp["series"][0]["series_id"]
        eia = eia.append(tf)

    eia.loc[eia.date.str.len() < 7, "date"] += "01"

    eia.date = pd.to_datetime(eia.date)
    return eia


def chart_perf_summary(df, geometric=True, title=None):
    """
    Function to plot the cumulative performance and drawdowns of
    multiple assets

    Parameters
    ----------
    df : DataFrame | Series
        wide dataframe with a datetime index and assets as columns or a
        series with a datetime index for a univariate chart
    geometric : bool
        True to plot geometric returns, False to plot absolute
    title : str
        Title for plot


    >>> import risktools as rt
    >>> df = rt.data.open_data('dfwide')
    >>> df = df[['CL01', 'CL12', 'CL36']]
    >>> df = rt.returns(df, period_return=1)
    >>> rt.chart_perf_summary(df, title="Cummulative Returns and Drawdowns")
    """
    df = df.copy()
    if geometric == True:
        ret = df.add(1).cumprod()
    else:
        ret = df.cumsum()

    dd = drawdowns(df, geometric=geometric)

    cols = DEFAULT_PLOTLY_COLORS

    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Cummulative Returns", "Drawdowns")
    )
    for i, c in enumerate(ret.columns):
        fig.add_trace(
            go.Scatter(
                x=ret.index,
                y=ret[c],
                name=c,
                line=dict(width=2, color=cols[i]),
                legendgroup="a",
            ),
            row=1,
            col=1,
        )

    for i, c in enumerate(dd.columns):
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd[c],
                name=c,
                line=dict(width=2, color=cols[i]),
                legendgroup="b",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title_text=title)
    return fig


def chart_forward_curves(
    df,
    code=None,
    cmdty=None,
    resample_freq=None,
    skip=5,
    curve_len=365 * 2,
    yaxis_title="$/BBL",
    **kwargs,
):
    """
    Chart forward curves of a futures contract strip

    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with a datetime index and columns representing the futures contract, ordered by most recent expiry
    code : str, optional
        commodity code to use to filter
    cmdty : str, optional
        expiry_table code to use to get expiry dates to plot forward curves. See data.open_data('expiry_data') for allowable
        values, using the column 'cmdty'
    resample_freq : str, optional
        pandas resample frequency type to downsample by. Assumes that new freq of lower periodicity than old freq. Uses mean to downsample
    skip : int
        number of days between forward curves, must be 1 or greater
    curve_len : int
        how long the draw the curve on the plot, in days. Required if cmdty is not given.
    uom : str
        Unit of Measure to show on the final chart y-axis
    ***kwargs
        keyword arguments to pass to fig.update_layout function for Plotly figures. Mainly
        to adjust figure layout.

    Returns
    -------
    Chart of forward curves varying with time

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.data.open_data('dfwide')
    >>> rt.chart_forward_curves(df, 'HO', yaxis_title='$/g', skip=10)
    >>> rt.chart_forward_curves_new(df, 'HO', cmdty='cmeulsd', yaxis_title='$/g', skip=2)
    """
    df = df.copy()
    fig = go.Figure()

    # if cmdty is defined, then use the expiry_table table to extract the expiry
    # schedule to pass to the _plot_crv function. Otherwise pass None to the function
    # so that it uses the defined time step
    if isinstance(cmdty, str):
        exp = data.open_data("expiry_table")
        exp = exp.loc[exp.cmdty == cmdty, "Last_Trade"]
    else:
        exp = None

    if code is not None:
        df = df[df.columns[df.columns.str.startswith(code)]]

    if resample_freq is not None:
        df = df.resample(resample_freq).mean()

    def _plot_crv(df, date, d=curve_len, exp=exp):
        """
        Convert series index to align with front contract dates with the curve spanning 400 days thereafter
        df
            dataframe with all contracts to be plotted
        d : int
            number of days that the curve should span

        Returns
        -------
        pandas series to plot with modified datetime index
        """
        df = df.copy()
        df.columns = df.columns.str.replace("[^0-9]", "").astype(int)
        se = df.loc[date, :]

        n = len(se)
        days_per_step = ceil(d / n)

        if exp is None:
            idx = []
            for i, r in se.iteritems():
                idx.append(
                    pd.to_datetime(date) + pd.DateOffset((i - 1) * days_per_step)
                )
        else:
            idx = exp[exp > date]
            idx = idx[: len(se)]

        se.index = pd.Index(idx)

        return se

    cnt = 0
    for i, r in df.iterrows():
        if cnt % skip == 0:
            crv = _plot_crv(df, i)
            fig.add_trace(
                go.Scattergl(
                    x=crv.index,
                    y=crv,
                    line=dict(width=1, dash="dash"),
                    legendgroup="b",
                    showlegend=False,
                    mode="lines",
                )
            )
        cnt += 1

    # add front month trace
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df.iloc[:, 0],
            name="Front Contract",
            legendgroup="a",
            line_color="royalblue",
        )
    )

    # leave this here before **kwargs so that users can adjust
    fig.update_layout(width=800, height=800)

    fig.update_layout(
        dict(
            title=f"{code} Forward Curves",
            yaxis_title=yaxis_title,
            **kwargs,
        )
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True, autorange=True, thickness=0.05),
            type="date",
        )
    )

    if kwargs is not None:
        fig.update_layout(**kwargs)

    return fig


def chart_pairs(df, title="Time Series Pairs Plot", **kwargs):
    """
    Pairwise scatter matrix plot for timeseries

    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with a datetime index and columns representing the futures contract, ordered by most recent expiry
    title : str, optional
        Chart title, by default "Time Series Pairs Plot"
    **kwargs
        keyword arguments to pass to plotly.graph_objects.Figure.update_layout function
    """
    dt_idx = df.index.name
    df = df.reset_index().copy()

    dims = []
    for c, i in df.iteritems():
        dims.append(
            dict(
                label=c,
                values=df[c],
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Splom(
            dimensions=dims,
            showupperhalf=False,
            marker=dict(color=df[dt_idx].astype(int), colorscale="Portland"),
            diagonal_visible=False,
        )
    )
    fig.update_layout(width=1000, height=1000, title=title)
    if kwargs is not None:
        fig.update_layout(**kwargs)

    return fig


def chart_spreads(
    username,
    password,
    pairs,
    days_from_expiry=200,
    feed=None,
    output="chart",
    start_dt="2012-01-01",
    **kwargs,
):
    """
    Chart commodity price spreads given as a list of tuple with two commodity codes and a name

    Parameters
    ----------
    username : str
        username for Morningstar API
    password : str
        password for Morningstar API
    pairs : List of Tuples with String
        A list of tuples of the format [(code_1, code_2, legend_label), (..,..,..)]. The spread calculated is then
        (code_1 - code_2). For example to plot the March/April time spread for NYMEX ULSD for 2014 & 2019, it would
        be [('@HO4H', '@HO4J', '2014'), ('@HO9H', '@HO9J', '2019')]
    days_from_expiry : int
        Days from expiry of the earliest contract to start the plot from (i.e. 200 days). By default 200
    feed : str
        feed to use to get data from Morningstar
    start_dt : str
        Start date for the data retrival as a string, in the format "yyyy-mm-dd"
    output : ['data', 'chart']
        whether to return data or a chart, by default 'chart'
    **kwargs
        Arguments to pass to the plotly update_layout function to control the layout elements of the final figure. You
        can also modify the returned figure object if required.

    Returns
    -------
    a dataframe or a plotly figure object

    Examples
    --------
    >>> import risktools as rt
    >>> df = rt.chart_spreads(
            username=up['m*']['user'],
            password=up['m*']['pass'],
            pairs=[('@HO4H', '@HO4J', '2014'), ('@HO9H', '@HO9J', '2019')],
            feed="CME_NymexFutures_EOD")
    """

    codes = []

    for c in pairs:
        codes.append(c[0])
        codes.append(c[1])

    df = get_prices(
        username=username, password=password, feed=feed, codes=codes, start_dt=start_dt
    )

    df = df.unstack(level=0).settlement_price

    res = pd.DataFrame()
    fig = go.Figure()

    for c in pairs:
        tmp = (df[c[0]] - df[c[1]]).dropna().reset_index(drop=True)[-days_from_expiry:]

        # reset index twice to bring in index number
        tmp = tmp.reset_index().reset_index()
        tmp.columns = ["days_from_exp", "date", "spread"]
        tmp.days_from_exp -= days_from_expiry
        tmp["year"] = c[2]
        res = res.append(tmp)

        if output == "chart":
            fig.add_trace(
                go.Scatter(
                    x=tmp.dropna().days_from_exp, y=tmp.dropna().spread, name=c[2]
                )
            )

    if kwargs is not None:
        fig.update_layout(**kwargs)

    if output == "chart":
        return fig
    else:
        return res


#     codes = []

#     for c in pairs:
#         codes.append(c[0])
#         codes.append(c[1])

#     df = get_prices(
#         username=username, password=password, feed=feed, codes=codes, start_dt=start_dt
#     )

#     df = df.unstack(level=0).settlement_price

#     res = pd.DataFrame()
#     fig = go.Figure()

#     for c in pairs:
#         tmp = (df[c[0]] - df[c[1]]).dropna()[-days_from_expiry:].values
#         res[c[2]] = pd.Series(tmp)

#         fig.add_trace(
#             go.Scatter(x=res[c[2]].dropna().index, y=res[c[2]].dropna(), name=c[2])
#         )

#     if kwargs is not None:
#         fig.update_layout(**kwargs)

#     if output == "chart":
#         return fig
#     else:
#         return res


def chart_zscore(df, freq=None, output="zscore", chart="seasons", **kwargs):
    """
    Function for calculating and plotting the seasonal decomposition and zscore of seasonal data
    Z-score is computed on the residuals conditional on their seasonal period. Beware that most
    seasonal charts in industry (i.e. NG Storage) is not detrended so results once you apply
    STL decomposition will vary from the unadjusted seasonal plot

    Parameters
    ----------
    df : Series
        Series with date and a values column. Datetime index must have a frequency defined (such as 'D', 'B', 'W', 'M' etc...)
    freq : str, optional
        Resampling frequency, by default None. Resamples using mean if downsampling. Not designed for
        upsampling.
    output : ['stl','zscore','seasonal']
        'stl' : for seasonal decomposition object. Run the object method .plot() to plot the STL decomposition.
            Also use the object attributes "observed", "resid", "seasonal", "trend" and "weights" to see components
            and calculate stats.
        'zscore' : return residuals of szore
        'seasonal' : for standard seasonal chart
        By default 'zscore'
    chart : str, optional
        [description], by default 'seasons'

    Returns
    -------
    'stl' : Statsmodels STL object
    'seasonal' : Plotly figure object
    'zscore' : Plotly figure object

    Examples
    --------
    >>> df = rt.data.open_data('eiaStocks')
    >>> df = df.loc[df.series=='NGLower48',['date','value']].set_index('date')['value']
    >>> df = df.resample('W-FRI').mean()
    >>> stl = rt.chart_zscore(df, freq='M')
    >>> stl.plot()
    >>> stl = rt.chart_zscore(df, output='seasonal')
    >>> stl.show()
    >>> stl = rt.chart_zscore(df, output='zscore')
    >>> stl.show()
    """

    # either resample or figure out frequency of the series
    if freq is not None:
        df = df.resample(freq).mean()
    elif df.index.freq is None:
        df.index.freq = pd.infer_freq(df.index[0:10])

    if str(df.index.freqstr)[0] == "M":
        seasonal = 13
    elif str(df.index.freqstr)[0] == "W":
        seasonal = 53
    else:
        seasonal = 7

    stl = STL(df, seasonal=seasonal, seasonal_deg=0, robust=False).fit()

    if output == "stl":
        return stl
    elif output == "seasonal":
        df = pd.concat([df], keys=["value"], axis=1)
        df["year"] = df.index.year
        df.index = df.index + pd.DateOffset(year=df.year.max())
        df = df.set_index("year", append=True).sort_index()
        df = df.unstack(level=-1).droplevel(0, 1)

        fig = go.Figure()
        for c in df.columns:
            if c == df.index.year.max():
                mode = "lines+markers"
            else:
                mode = "lines"
            fig.add_trace(
                go.Scattergl(
                    x=df[c].dropna().index, y=df[c].dropna(), name=c, mode=mode
                )
            )
        fig.update_layout(xaxis_title="freq", yaxis_title="value")
        fig.update_layout(
            xaxis={
                "tickformat": "%b",
                # tickmode': 'auto',
                "nticks": 13,  # [where value is the max # of ticks]
                # 'tick0': '01', # [where value is the first tick]
                # 'dtick': 1, # [where value is the step between ticks]
            }
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)

        return fig
    elif output == "zscore":
        zs = pd.DataFrame()

        # first calculate mean and std of the residuals by frequency (i.e. by week, month etc...) for all years
        zs["resid"] = stl.resid
        zs["year"] = zs.index.year
        tt = stl.resid.index.to_series()
        zs["per"] = (
            tt.groupby(tt.dt.year).cumcount() + 1
        )  # elegant way to get weeknum, monthnum, quarter etc... regardless of freq
        ag = zs.groupby("per").resid.agg(["mean", "std"]).reset_index()

        zs = (
            zs.reset_index().merge(ag, left_on="per", right_on="per").set_index("date")
        )  # use reset index or it drops the datetime index of zs
        zs["z_score"] = (zs.resid - zs["mean"]) / zs["std"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=zs.index,
                    y=zs.z_score,
                    marker_color=zs.z_score,
                    marker_colorscale=["red", "orange", "green", "orange", "red"],
                    marker_colorbar=dict(title="Z-Score"),
                )
            ]
        )

        fig.update_layout(
            title="Z-Score of Seasonally Adjusted Residuals",
            xaxis_title="date",
            yaxis_title="Z Score",
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)

        return fig


def chart_eia_sd(market, key, start_dt="2010-01-01", output="chart", **kwargs):
    """
    Function for plotting and returning data from the EIA Supply & Demand Balances for
    refined oil products such as mogas, diesel, jet and resid.

    Parameters
    ----------
    market : ['mogas', 'diesel', 'jet', 'resid']
        Refined product type to build the S&D balance for
    key : str
        EIA API key
    start_dt : str | datetime
        Starting date for S&D balance data
    output : ['chart','data']
        Output as a chart or return data as a dataframe, by default 'chart'

    Returns
    -------
    pandas dataframe or a plotly figure object

    Examples
    --------
    >>> import risktools as rt
    >>> fig = rt.chart_eia_sd('mogas', up['eia'])
    >>> fig.show()
    """
    eia = data.open_data("tickers_eia")
    eia = eia[eia.sd_category == market]

    df = get_eia_df(eia.tick_eia.to_list(), key=key)
    df = df.merge(
        eia[["tick_eia", "category"]], left_on=["series_id"], right_on=["tick_eia"]
    ).drop("tick_eia", axis=1)

    df.date = pd.to_datetime(df.date)
    df = df.set_index("date").sort_index()

    # create list of plotly figure objects using repeating calls to the
    # five_year_plot function to loop through later to create subplot
    figs = []
    for c in df.category.unique():
        tf = df.loc[df.category == c, "value"]
        figs.append(five_year_plot(tf, title=c))

    # calc shape of final subplot
    n = len(df.category.unique())
    m = 2
    n = ceil(n / m)
    fig = make_subplots(
        n,
        m,
        subplot_titles=(" ", " ", " ", " ", " ", " "),
    )

    # Copy returns figures from five_year_plot to a single subplot figure
    a = 1
    b = 1
    for i, _ in enumerate(figs):
        for j, _ in enumerate(figs[i]["data"]):
            fig.add_trace(figs[i]["data"][j], row=a, col=b)
            fig["layout"]["annotations"][(a - 1) * 2 + b - 1]["text"] = figs[i][
                "layout"
            ]["title"]["text"]

        # copy xaxis nticks and tickformat to subplots so that it keeps that formatting.
        # if they don't exist, pass
        try:
            fig["layout"][f"xaxis{i+1}"]["nticks"] = figs[i]["layout"]["xaxis"][
                "nticks"
            ]
            fig["layout"][f"xaxis{i+1}"]["tickformat"] = figs[i]["layout"]["xaxis"][
                "tickformat"
            ]
        except:
            pass

        if b == m:
            b = 1
            a += 1
        else:
            b += 1

    fig.update_layout(showlegend=False)
    # return figs
    if output == "chart":
        return fig
    else:
        return df


def five_year_plot(df, **kwargs):
    """
    Function to output a 5 yr range plot for commodity prices

    Parameters
    ----------
    df : Series
        Series or single column dataframe with a datetime index
    """
    df = df.copy()
    if isinstance(df, pd.Series):
        df = pd.DataFrame({"value": df})

    freq = pd.infer_freq(df.index[-10:])

    df["year"] = df.index.year
    start_dt = str(df.index.year.max() - 5) + "-01-01"

    cy = df.index.year.max()
    py = cy - 1

    df = df.sort_index().loc[start_dt:, :]
    fy = df.index.year.min()

    df.index = df.index + pd.DateOffset(year=cy)
    df = df.set_index("year", append=True).sort_index().unstack().droplevel(0, 1)
    df = df.resample(freq).mean()

    # shift back columns where the first row is NaN for offset weeks
    for c in df.columns:
        if np.isnan(df[c].iloc[0]):
            df[c] = df[c].shift(-1)
    df = df.dropna(thresh=2, axis=0)
    mf = df[range(fy, cy)].resample(freq).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mf.min(axis=1).index, y=mf.min(axis=1), line_color="lightgrey", name="min"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mf.max(axis=1).index,
            y=mf.max(axis=1),
            line_color="lightgrey",
            name="max",
            fill="tonexty",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[cy].dropna().index,
            y=df[cy].dropna(),
            name=str(cy),
            line_color="#EF553B",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[py].dropna().index,
            y=df[py].dropna(),
            name=str(py),
            line_color="royalblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mf.mean(axis=1).index,
            y=mf.mean(axis=1),
            name="5 yr mean",
            line_dash="dash",
            line_color="black",
        )
    )

    fig.update_layout(xaxis={"tickformat": "%b", "nticks": 13})
    fig.update_layout(**kwargs)

    return fig


def chart_eia_steo(key, start_dt=None, market="globalOil", output="chart", **kwargs):
    """
    Function to plot the global oil supply and demand balance using the US EIA data

    Parameters
    ----------
    key : str
        Your EIA API key
    start_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    market : ['globalOil']
        What market to build the S&D balance for. For now, only globalOil working.
    output : ['chart', 'data'], optional
        What kind of output to return, by default 'chart'
    **kwargs
        keyword arguments to pass to plotly.graph_objects.figure.update_layout function to modify figure

    Returns
    -------
    output = 'chart'
        Plotly figure object
    output = 'data'
        dataframe

    Examples
    --------
    >>> import risktool as rt
    >>> fig = rt.chart_eia_steo(up['eia'])
    >>> fig.show()
    """
    if market == "globalOil":
        tickers = {
            "STEO.PAPR_NONOPEC.M": "SupplyNOPEC",
            "STEO.PAPR_OPEC.M": "SupplyOPEC",
            "STEO.PATC_WORLD.M": "Demand",
            "STEO.T3_STCHANGE_WORLD.M": "Inv_Change",
        }

    df = get_eia_df(list(tickers.keys()), key=key)
    df["name"] = df["series_id"].map(tickers)
    df = (
        df[["date", "value", "name"]]
        .set_index(["date", "name"])
        .sort_index()
        .unstack()
        .droplevel(0, 1)
        .dropna()
    )
    df["Supply"] = df.SupplyNOPEC + df.SupplyOPEC
    df.Inv_Change *= -1
    df = df.drop(["SupplyNOPEC", "SupplyOPEC"], axis=1)

    if start_dt is not None:
        df = df[start_dt:]

    if output == "data":
        return df
    else:
        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
        fig.update_layout(
            title=f"US EIA {market} Supply and Demand Balance",
            xaxis_title="date",
            yaxis_title="millions of barrels per day",
        )
        if kwargs is not None:
            fig.update_layout(**kwargs)
        return fig


def swap_com(df, futures_names, start_dt, end_dt, cmdty, exchange):
    """
    Function to calculate commodity swap pricing from futures contract exchange settlements of two contracts.
    Mostly used when the contract expires mid-month to work out the calendar month average

    Parameters
    ----------
    df : DataFrame
        Wide dataframe of futures prices
    futures_names : List[Tuple[str]]
        2 element List or tuple of futures contract names in order of expiry (i.e. [first expirying contract, second expirying contract])
    start_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    end_dt : str | datetime
        A string in the format of 'YYYY-mm-dd' or a datetime object
    cmdty : str
        name of expiry contract type as specified in the expiry_table of this package. See risktools.data.open_data('expiry_table')['cmdty']
    exchange : ['nymex','ice']
        Name of exchange. Only 'nymex' and 'ice' supported. Used to filter the dataframe risktools.data.open_data('holidaysOil')

    Returns
    -------
    a dataframe of swap prices

    Examples
    --------
    >>> df = rt.get_prices(up['m*']['user'], up['m*']['pass'], codes=['CL0M','CL0N','CL0Q'], start_dt='2019-08-26')
    >>> df = df.settlement_price.unstack(level=0)
    >>> rt.swap_com(df=df, futures_names=["CL0M","CL0N"], start_dt="2020-05-01", end_dt="2020-05-30", cmdty="cmewti", exchange="nymex")
    """
    # get contract expiry dates
    exp = data.open_data("expiry_table")
    exp = exp.query(
        f"Last_Trade >= '{start_dt}' & Last_Trade <= '{end_dt}' & cmdty == '{cmdty}'"
    )["Last_Trade"]

    hol = data.open_data("holidaysOil").query(f"key == '{exchange}'")

    biz_dates = set(pd.date_range(start_dt, end_dt, freq="B").to_list())
    biz_dates = biz_dates - set(hol.value)

    res = pd.DataFrame(index=biz_dates).sort_index()
    res["up2expiry"] = 1

    next_contract_dt = pd.to_datetime(exp.values[0]) + pd.DateOffset(days=1)

    res.loc[next_contract_dt:, "up2expiry"] = 0

    first_fut_weight = res.up2expiry.sum() / res.up2expiry.count()

    df = df[futures_names]
    df["swap"] = df.iloc[:, 0] * first_fut_weight + df.iloc[:, 1] * (
        1 - first_fut_weight
    )

    return df.dropna()


def _check_df(df):
    # if isinstance(df.index, pd.DatetimeIndex):
    #     # reset index if the df index is a datetime object
    #     df = df.reset_index().copy()

    return df.copy()


def chart_ols_diag(fit, **kwargs):
    """
    Function used to chart OLS regression diagnostic plots for statsmodels linear regression
    models. Based on R's regression diagnostic plots using the base function plot() or autoplot()
    from the ggfortify package.

    The diagnostic plots show residuals in four different ways [1]:

    1. Residuals vs Fitted. Used to check the linear relationship assumptions. A horizontal line,
    without distinct patterns is an indication for a linear relationship, what is good.

    2. Normal Q-Q. Used to examine whether the residuals are normally distributed. It’s good if
    residuals points follow the straight dashed line.

    3. Scale-Location (or Spread-Location). Used to check the homogeneity of variance of the
    residuals (homoscedasticity). Horizontal line with equally spread points is a good indication of
    homoscedasticity. Any degree of slope indicates a heteroscedasticity problem. A possible
    solution to reduce the heteroscedasticity problem is to use a log or square root transformation
    of the outcome variable (y).

    4. Residuals vs Leverage. Used to identify influential cases, that is extreme values that
    might influence the regression results when included or excluded from the analysis.

        Outliers:
        An outlier is a point that has an extreme outcome variable value. The presence of outliers
        may affect the interpretation of the model, because it increases the RSE.

        Outliers can be identified by examining the standardized residual (or studentized residual),
        which is the residual divided by its estimated standard error. Standardized residuals can
        be interpreted as the number of standard errors away from the regression line.

        Observations whose standardized residuals are greater than 3 in absolute value are possible
        outliers (James et al. 2014).

        High leverage points:

        A data point has high leverage, if it has extreme predictor x values. This can be detected
        by examining the leverage statistic or the hat-value. A value of this statistic above 2(p + 1)/n
        indicates an observation with high leverage (P. Bruce and Bruce 2017); where, p is the number
        of predictors and n is the number of observations. Outliers and high leverage points can be
        identified by inspecting the Residuals vs Leverage plot

        Influential values

        An influential value is a value, which inclusion or exclusion can alter the results of the
        regression analysis. Such a value is associated with a large residual.

        Not all outliers (or extreme data points) are influential in linear regression analysis.

        Statisticians have developed a metric called Cook’s distance to determine the influence of a
        value. This metric defines influence as a combination of leverage and residual size.

        A rule of thumb is that an observation has high influence if Cook’s distance exceeds 4/(n - p - 1)
        (P. Bruce and Bruce 2017), where n is the number of observations and p the number of predictor variables.

        The Residuals vs Leverage plot can help us to find influential observations if any. On this plot,
        outlying values are generally located at the upper right corner or at the lower right corner.
        Those spots are the places where data points can be influential against a regression line.

    Parameters
    ----------
    fit : statsmodels RegressionResults
        Resultant fit object from statsmodels linear regression fit
    **kwargs
        parameters to pass to plotting functions (matplotlib). Not working yet.

    Returns
    -------
    matplotlib figure object

    Reference
    ---------
    [1] http://www.sthda.com/english/articles/39-regression-model-diagnostics/161-linear-regression-assumptions-and-diagnostics-in-r-essentials/

    Examples
    --------
    >>> import risktools as rt
    >>> import pandas_datareader.data as web
    >>> import statsmodels.formula.api as smf
    >>> df = web.DataReader(['XLE','XOM'], data_source='yahoo', start='2007-01-01')
    >>> df = df.stack(level=1).swaplevel().sort_index().reset_index()
    >>> df['ret_log'] = df.groupby('Symbols')['Adj Close'].apply(lambda x: np.log(x/x.shift()))
    >>> df = df.dropna()
    >>> df_ret = df.set_index(['Symbols','Date']).unstack(level=0)['ret_log']
    >>> fit = smf.ols('XLE ~ XOM', data=df_ret.reset_index()).fit()
    >>> rt.chart_ols_diag(fit)
    """
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(10, 7)
    fig.suptitle("OLS Regression Diagnostic Plots")

    # Residuals vs Fitted
    sns.regplot(
        x=fit.fittedvalues,
        y=fit.resid,
        ax=ax[0, 0],
        lowess=True,
        line_kws={"color": "red"},
    )
    ax[0, 0].set_title("Residuals vs Fitted")
    ax[0, 0].axhline(0, color="k", linestyle="--", lw="0.5")
    ax[0, 0].set_xlabel("Fitted Values")
    ax[0, 0].set_ylabel("Residuals")

    # Normal Q-Q Plot
    sm.qqplot(
        fit.resid,
        fit=True,
        line="s",
        ax=ax[0, 1],
        markerfacecolor=sns.color_palette()[0],
        markeredgecolor=sns.color_palette()[0],
        alpha=0.8,
    )
    ax[0, 1].set_title("Normal Q-Q")

    # Std Residuals vs Fitted
    sns.regplot(
        x=fit.fittedvalues,
        y=np.sqrt(np.abs(fit.resid / fit.resid.std())),
        ax=ax[1, 0],
        lowess=True,
        line_kws={"color": "red"},
    )
    ax[1, 0].set_title("Scale-Location")
    ax[1, 0].set_xlabel("Fitted Values")
    ax[1, 0].set_ylabel(r"$\sqrt{|\enspace standardized \enspace residuals \enspace|}$")

    # Residuals vs Leverage
    infl = fit.get_influence()
    sf = infl.summary_frame()
    sns.regplot(
        x=sf.hat_diag,
        y=sf.standard_resid,
        ax=ax[1, 1],
        lowess=True,
        line_kws={"color": "red"},
    )
    ax[1, 1].axhline(0, color="k", linestyle="--", lw="0.5")
    ax[1, 1].set_xlabel("Leverage (hat_diag)")
    ax[1, 1].set_ylabel("Standardized Residuals")
    ax[1, 1].set_title("Residuals vs Leverage")
    fig.tight_layout()

    return fig


# if __name__ == "__main__":
#     np.random.seed(42)
#     import json

#     with open("../../user.json") as jfile:
#         userfile = jfile.read()

#     up = json.loads(userfile)

#     username = up["m*"]["user"
#     password = up["m*"]["pass"]

#     print(
#         chart_spreads(
#             [("@HO4H", "@HO4J", "2014"), ("@HO9H", "@HO9J", "2019")],
#             200,
#             username,
#             password,
#             feed="CME_NymexFutures_EOD",
#             output="chart",
#             start_dt="2012-01-01",
#         )
#     )

# sp = simOU()
# print(fitOU(sp))
# print(sp)

# print(bond(output='price'))
# print(bond(output='duration'))
# print(bond(output='df'))

# dflong = data.open_data('dflong')
# ret = returns(df=dflong, ret_type="abs", period_return=1, spread=True).iloc[:,0:2]
# roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade").head(50)

# from pandas_datareader import data, wb
# from datetime import datetime
# df = data.DataReader(["SPY","AAPL"],  "yahoo", datetime(2000,1,1), datetime(2012,1,1))
# df = df.pct_change()
# df = df.asfreq('B')

# R = df[('Adj Close','SPY')]

# rt.returns(df = rt.data.open_data('dflong'), ret_type = "rel", period_return = 1, spread = False)
# rt.returns(df = rt.data.open_data('dflong'), ret_type = "log", period_return = 1, spread = True)

# print(y)

# print(_check_ts(R.dropna(), scale=252))

# print(trade_stats(df[('Adj Close','SPY')]))
# print(trade_stats(df['Adj Close']))

# tt = _sr(df['Adj Close'], Rf=0, scale=252)
# print(tt)

# print(drawdowns(df['Adj Close']))
# print(drawdowns(df[('Adj Close','SPY')]))

# rs = find_drawdowns(df['Adj Close'])
# print(rs['SPY']['peaktotrough'])
# print(find_drawdowns(df[('Adj Close','SPY')]))

# print(sharpe_ratio_annualized(df['Adj Close']))
# print(sharpe_ratio_annualized(df[('Adj Close','SPY')]))

# print(omega_sharpe_ratio(df['Adj Close'],MAR=0))
# print(upside_risk(df['Adj Close'], MAR=0))
# print(upside_risk(df[('Adj Close','SPY')], MAR=0))
# print(upside_risk(df['Adj Close'], MAR=0, stat='variance'))
# print(upside_risk(df['Adj Close'], MAR=0, stat='potential'))

# print(downside_deviation(df[('Adj Close','SPY')], MAR=0))
# print(downside_deviation(df['Adj Close'], MAR=0))

# print(return_cumulative(r=df['Adj Close'], geometric=True))
# print(return_cumulative(r=df['Adj Close'], geometric=False))

# print(return_annualized(r=df[('Adj Close','SPY')], geometric=True))
# print(return_annualized(r=df[('Adj Close','SPY')], geometric=False))

# print(return_annualized(r=df['Adj Close'], geometric=True))
# print(return_annualized(r=df['Adj Close'], geometric=False))

# print(sd_annualized(x=df[('Adj Close','SPY')]))
# print(sd_annualized(x=df['Adj Close']))

####################
# R code for testing
# library(timetk)
# library(tidyverse)
# library(PerformanceAnalytics)
# df <- tq_get('SPY', from='2000-01-01', to='2012-01-01')
# df <- df %>% dplyr::mutate(adjusted = adjusted/lag(adjusted)-1)
# R = tk_xts(df, select=adjusted, date_var=date)
# DownsideDeviation(R, MAR=0)
# UpsideRisk(R, 0)
# OmegaSharpeRatio(R, 0)
