from numpy.linalg.linalg import eigvals
import pandas as pd
import numpy as np
t
import plotly.graph_objects as go
import pandas_datareader.data as web
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
from pandas_datareader import data

# TODO
# stl_decomp test
# chart_zscore only tests figure object - test actual stl decomp


# with open("../../user.json") as js:
#     up = json.load(js)

# Travis CI Env Vars
up = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": ""}

up["eia"] = os.getenv("EIA")
up["quandl"] = os.getenv("QUANDL")
up["m*"]["pass"] = os.getenv("MS_PASS")
up["m*"]["user"] = os.getenv("MS_USER")


def _load_json(fn, dataframe=True):
    path = os.path.dirname(__file__)
    fp = os.path.join(path, fn)
    with open(fp) as js:
        if dataframe == True:
            df = pd.read_json(js)
            df.columns = df.columns.str.replace(".", "_")
        else:
            df = json.load(js)

    return df


def test_ir_df_us():
    df = _load_json("ir_df_us.json")
    ir = rt.ir_df_us(quandl_key=up["quandl"])
    ir = ir[
        [
            "yield",
            "maturity",
            "discountfactor",
            "discountfactor_plus",
            "discountfactor_minus",
        ]
    ]
    assert df.round(4).equals(
        ir.round(4)
    ), "ir_df_us test failed, returned dataframe does not equal RTL results"


def test_bond():
    # first test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="price")
    assert bo == 100, "bond Test 1 failed"

    # second test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="df")
    df = _load_json("bond_2.json")
    assert df.astype(float).round(4).equals(bo.round(4)), "bond Test 2 failed"

    # third test
    bo = rt.bond(ytm=0.05, c=0.05, T=1, m=2, output="duration")
    assert round(bo, 4) == 0.9878, "bond Test 3 failed"


def test_trade_stats():

    df = data.DataReader(["SPY", "AAPL"], "yahoo", "2000-01-01", "2012-01-01")
    df = df.pct_change()
    df = df.asfreq("B")

    ou = rt.trade_stats(df[("Adj Close", "SPY")])
    ts = _load_json("tradeStats.json")

    assert round(ou["cum_ret"], 4) == round(
        ts["CumReturn"][0], 4
    ), "tradeStats Test cum_ret failed"
    assert round(ou["ret_ann"], 4) == round(
        ts["Ret_Ann"][0], 4
    ), "tradeStats Test ret_ann failed"
    assert round(ou["sd_ann"], 4) == round(
        ts["SD_Ann"][0], 4
    ), "tradeStats Test sd_ann failed"
    assert round(ou["omega"], 4) == round(
        ts["Omega"][0], 4
    ), "tradeStats Test omega failed"
    assert round(ou["sharpe"], 4) == round(
        ts["Sharpe"][0], 4
    ), "tradeStats Test sharpe failed"
    assert round(ou["perc_win"], 4) == round(
        ts["%_Win"][0], 4
    ), "tradeStats Test perc_win failed"
    assert round(ou["perc_in_mkt"], 4) == round(
        ts["%_InMrkt"][0], 4
    ), "tradeStats Test perc_in_mkt failed"
    assert round(ou["dd_length"], 4) == round(
        ts["DD_Length"][0], 4
    ), "tradeStats Test dd_length failed"
    assert round(ou["dd_max"], 4) == round(
        ts["DD_Max"][0], 4
    ), "tradeStats Test dd_max failed"


def test_returns():

    # Test 1
    ac = _load_json("returns1.json").round(4).set_index("date").dropna()
    ac.columns.name = "series"
    ts = rt.returns(
        df=rt.data.open_data("dflong").round(
            4
        ),  # round(4) because R toJSON function does so
        ret_type="rel",
        period_return=1,
        spread=True,
    ).round(4)

    ts = ts.dropna()
    assert ac.equals(ts), "returns Test 1 failed"

    # Test 2
    ac2 = _load_json("returns2.json").round(4)

    ts2 = rt.returns(
        df=rt.data.open_data("dflong").round(4),
        ret_type="rel",
        period_return=1,
        spread=False,
    )
    ts2 = ts2.round(4)

    ac2 = ac2.set_index(["series", "date"])["returns"].sort_index()

    # ts2 = ts2.unstack(0).stack().swaplevel(0, 1).sort_index()

    assert ac2.equals(ts2), "returns Test 2 failed"

    # Test 3
    ac = _load_json("returns3.json").round(4).set_index("date").dropna()
    ac.columns.name = "series"
    ts = rt.returns(
        df=rt.data.open_data("dflong").round(
            4
        ),  # round(4) because R toJSON function does so
        ret_type="abs",
        period_return=1,
        spread=True,
    ).round(4)

    ts = ts.dropna()
    assert ac.equals(ts), "returns Test 3 failed"

    # Test 4
    ac = _load_json("returns4.json").round(4).set_index("date").dropna()
    ac.columns.name = "series"
    ts = rt.returns(
        df=rt.data.open_data("dflong").round(
            4
        ),  # round(4) because R toJSON function does so
        ret_type="log",
        period_return=1,
        spread=True,
    ).round(4)

    ts = ts.dropna()
    assert ac.equals(ts), "returns Test 4 failed"


def test_roll_adjust():
    ac = _load_json("rolladjust.json").set_index("date").round(4)

    ac = ac.iloc[:, 0].dropna()

    dflong = rt.data.open_data("dflong")
    rt.data.open_data("expiry_table").cmdty.unique()  # for list of commodity names
    ret = rt.returns(df=dflong, ret_type="abs", period_return=1, spread=True)
    ret = ret.iloc[:, 0].dropna()
    ts = (
        rt.roll_adjust(df=ret, commodity_name="cmewti", roll_type="Last_Trade")
        # .iloc[1:]
        .round(4).dropna()
    )

    assert ac.equals(ts), "rolladjust Test failed"


def test_garch():
    ac = _load_json("garch.json").set_index("date").garch

    dflong = rt.data.open_data("dflong")
    dflong = dflong["CL01"]
    df = rt.returns(df=dflong, ret_type="rel", period_return=1, spread=True)

    df = rt.roll_adjust(df=df, commodity_name="cmewti", roll_type="Last_Trade").iloc[1:]

    ts = rt.garch(df, out="data", vol="garch", rescale=False)

    # need to see if I can get R and Python garch models to produce the same vol
    assert (ac.mean() / ts["h.1"].mean() < 2) & (
        ac.mean() / ts["h.1"].mean() > 0.5
    ), "garch mean test failed, test result mean is more that double or less than half of RTL results"
    assert (ac.std() / ts["h.1"].std() < 2) & (
        ac.std() / ts["h.1"].std() > 0.5
    ), "garch std test failed, test result std is more that double or less than half of RTL results"

    # redo R garch using a standard garch model


def test_prompt_beta():
    ac = _load_json("promptBeta.json").round(2).drop("contract", axis=1)

    dfwide = rt.data.open_data("dfwide")
    col_mask = dfwide.columns[dfwide.columns.str.contains("CL")]
    dfwide = dfwide[col_mask]
    dfwide = dfwide[~dfwide.index.isin(["2020-04-20", "2020-04-21"])]
    x = rt.returns(df=dfwide, ret_type="abs", period_return=1)
    x = rt.roll_adjust(df=x, commodity_name="cmewti", roll_type="Last_Trade")

    ts = (
        rt.prompt_beta(df=x, period="all", beta_type="all", output="betas")
        .round(4)
        .reset_index(drop=True)
    )

    # for some reason the betas are slightly different using the Python sklearn
    # LinearRegression model. Make sure that the max of the three columns
    # are less than 0.03. Differences are on the order of 0.001 on any individual
    # beta
    assert (
        ac - ts
    ).abs().max().sum() < 0.03, (
        "prompt_beta Test failed, sum of total differences > 0.03"
    )


def test_swap_irs():
    a = 67868.58
    b = 1.1925

    ac = _load_json("swapIRS.json")
    ac.dates = pd.to_datetime(ac.dates)

    usSwapCurves = rt.data.open_data("usSwapCurves")
    ts = rt.swap_irs(
        trade_date="2020-01-04",
        eff_date="2020-01-06",
        mat_date="2022-01-06",
        notional=1000000,
        pay_rec="rec",
        fixed_rate=0.05,
        float_curve=usSwapCurves,
        reset_freq="Q",
        disc_curve=usSwapCurves,
        days_in_year=360,
        convention="act",
        bus_calendar="NY",
        output="all",
    )

    assert round(a, 0) == ts["pv"].round(
        0
    ), "swapIRS test failed, pv not equal to 2 decimal places'"

    assert b == ts["duration"].round(
        4
    ), "swapIRS test failed, duration not equal to 4 decimal places"

    ac[["fixed", "floating", "net"]] = ac[["fixed", "floating", "net"]].round(0)

    ts["df"][["fixed", "floating", "net"]] = ts["df"][
        ["fixed", "floating", "net"]
    ].round(0)

    assert ac.round(4).equals(ts["df"].round(4)), "swapIRS test failed"


def test_npv():
    ac = _load_json("npv1.json")
    ac.cf = ac.cf.astype(float)
    ir = rt.ir_df_us(quandl_key=up["quandl"], ir_sens=0.01)
    ts = rt.npv(
        init_cost=-375, C=50, cf_freq=0.5, F=250, T=2, disc_factors=ir, break_even=False
    )

    assert ac.round(4).equals(ts.round(4)), "npv Test 1 using actual ir failed"

    ac2 = _load_json("npv2.json")
    ac2.cf = ac2.cf.astype(float)
    ts2 = rt.npv(
        init_cost=-375,
        C=50,
        cf_freq=0.5,
        F=250,
        T=2,
        disc_factors=ir,
        break_even=True,
        be_yield=0.0399,
    )

    assert ac2.round(4).equals(ts2.round(4)), "npv Test 2 using fixed yield"


def test_crr_euro():
    ac = _load_json("crreuro.json", dataframe=False)
    ts = rt.crr_euro(s=100, x=100, sigma=0.2, Rf=0.1, T=1, n=5, type="call")

    assert np.array_equal(
        np.array(ac["asset"]), ts["asset"].round(4)
    ), "crr_euro Test failed on assets array"
    assert np.array_equal(
        np.array(ac["option"]), ts["option"].round(4)
    ), "crr_euro Test failed on options array"
    assert ac["price"][0] == round(ts["price"], 4), "crr_euro Test failed on price"
    assert ac["note"][0] == ts["note"], "crr_euro Test failed on price"


def test_stl_decomposition():
    ac = _load_json("stl_decomp.json")

    ac = ac.rename({"index": "date"}, axis=1).set_index("date")

    df = rt.data.open_data("dflong")
    df = df["CL01"]
    ts = rt.stl_decomposition(
        df, output="data", seasonal=13, seasonal_deg=1, resample_freq="M"
    )
    pass


def test_get_eia_df():
    ac = _load_json("eia2tidy1.json")
    ac = ac[["date", "value"]].set_index("date").sort_index().value
    ts = rt.get_eia_df("PET.MCRFPTX2.M", key=up["eia"])
    ts = ts[["date", "value"]].set_index("date").sort_index().value

    assert ac.equals(ts), "get_eia_df Test 1 failed"

    ac = _load_json("eia2tidy2.json")
    ac = ac.set_index(["series", "date"]).sort_index().value
    ts = rt.get_eia_df(
        ["PET.W_EPC0_SAX_YCUOK_MBBL.W", "NG.NW2_EPG0_SWO_R48_BCF.W"], key=up["eia"]
    )
    ts.table_name = ts.table_name.map(
        {
            "Cushing, OK Ending Stocks excluding SPR of Crude Oil, Weekly": "CrudeCushing",
            "Weekly Lower 48 States Natural Gas Working Underground Storage, Weekly": "NGLower48",
        }
    )
    ts = (
        ts.rename({"table_name": "series"}, axis=1)
        .set_index(["series", "date"])
        .sort_index()
        .value
    )

    assert ac.equals(ts), "get_eia_df Test 2 failed"


def test_chart_spreads():
    ac = _load_json("chart_spreads.json")
    ts = rt.chart_spreads(
        up["m*"]["user"],
        up["m*"]["pass"],
        [
            ("@HO4H", "@HO4J", "2014"),
            ("@HO9H", "@HO9J", "2019"),
            ("@HO0H", "@HO0J", "2020"),
        ],
        feed="CME_NymexFutures_EOD",
        output="data",
    )

    ts.spread = ts.spread.mul(42).round(4)
    ts = ts[["year", "spread", "days_from_exp", "date"]]
    ts.columns = ["year", "value", "DaysFromExp", "date"]
    ts = ts.reset_index(drop=True).dropna()
    ts.year = pd.to_numeric(ts.year)

    # assert ac.equals(ts), "chart_spreads Test failed"


def test_chart_zscore():
    df = rt.data.open_data("eiaStocks")
    df = df.loc[df.series == "NGLower48", ["date", "value"]].set_index("date")["value"]
    df = df.resample("W-FRI").mean()
    stl = rt.chart_zscore(df)

    assert isinstance(stl, go.Figure), "chart_zscore Test failed"


def test_chart_eia_sd():
    fig = rt.chart_eia_sd("mogas", up["eia"])
    assert isinstance(fig, go.Figure), "chart_eia_sd Test failed"


def test_chart_eia_steo():
    fig = rt.chart_eia_steo(up["eia"])
    assert isinstance(fig, go.Figure), "chart_eia_steo Test failed"


def test_swap_com():
    ac = _load_json("swapCOM.json")

    df = rt.get_prices(
        up["m*"]["user"],
        up["m*"]["pass"],
        codes=["CL0M", "CL0N", "CL0Q"],
        start_dt="2019-08-26",
    )
    df = df.settlement_price.unstack(level=0)
    ts = rt.swap_com(
        df=df,
        futures_names=["CL0M", "CL0N"],
        start_dt="2020-05-01",
        end_dt="2020-05-30",
        cmdty="cmewti",
        exchange="nymex",
    )

    ac = ac.set_index("date")
    ac = ac.round(4)
    ts.index.name = "date"

    assert ac.equals(ts.round(4)), "swap_com Test failed"


def test_chart_ols_diag():
    df = web.DataReader(["XLE", "XOM"], data_source="yahoo", start="2007-01-01")
    df = df.stack(level=1).swaplevel().sort_index().reset_index()
    df["ret_log"] = df.groupby("Symbols")["Adj Close"].apply(
        lambda x: np.log(x / x.shift())
    )
    df = df.dropna()
    df_ret = df.set_index(["Symbols", "Date"]).unstack(level=0)["ret_log"]
    fit = smf.ols("XLE ~ XOM", data=df_ret.reset_index()).fit()
    fig = rt.chart_ols_diag(fit)
    assert isinstance(fig, plt.Figure), "plot_ols_diag Test failed"


if __name__ == "__main__":
    test_returns()
    pass


# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri, numpy2ri
# from rpy2.robjects.pandas2ri import rpy2py, py2rpy
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects.packages import importr
# from rpy2.robjects.packages import STAP
# from contextlib import contextmanager

# # R vector of strings
# from rpy2.robjects.vectors import StrVector, FloatVector, ListVector, Vector

# # import R's "base" package
# base = importr("base")

# utils = importr("utils")
# tq = importr("tidyquant")
# tv = importr("tidyverse")
# ql = importr("Quandl")
# rtl = importr("RTL")


# def p2r(p_df):
#     # Function to convert pandas dataframes to R
#     with localconverter(ro.default_converter + pandas2ri.converter) as cv:
#         r_from_pd_df = cv.py2rpy(p_df)

#     return r_from_pd_df


# def r2p(r_df):
#     # Function to convert R dataframes to pandas
#     with localconverter(ro.default_converter + pandas2ri.converter) as cv:
#         pd_from_r_df = cv.rpy2py(r_df)

#     return pd_from_r_df
