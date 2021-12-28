from numpy.linalg.linalg import eigvals
import pandas as pd
import numpy as np
import os
import json
import sys
import plotly.graph_objects as go
from test_risktools import _load_json

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import risktools as rt
from pandas_datareader import data

# TODO

test_date = "2021-12-24"

# Github Actions CI Env Vars
up = {"m*": {"user": "", "pass": ""}, "eia": "", "quandl": ""}

up["eia"] = os.getenv("EIA")
up["quandl"] = os.getenv("QUANDL")
up["m*"]["pass"] = os.getenv("MS_PASS")
up["m*"]["user"] = os.getenv("MS_USER")

ms = dict(username=os.getenv("MS_USER"), password=os.getenv("MS_PASS"))


def test_get_curves():
    cl = _load_json("getCurveCL.json")
    cl.expirationDate = pd.to_datetime(cl.expirationDate)

    bg = _load_json("getCurveBG.json")
    bg.expirationDate = pd.to_datetime(bg.expirationDate)

    df_cl = rt.get_curves(
        up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["CL"]
    )

    df_bg = rt.get_curves(
        up["m*"]["user"], up["m*"]["pass"], date="2021-12-20", contract_roots=["BG"]
    )

    pd.testing.assert_frame_equal(
        cl, df_cl, check_like=True
    ), "get_curves test failed on CL"

    pd.testing.assert_frame_equal(
        bg, df_bg, check_like=True
    ), "get_curves test failed on BG"

    cl["root"] = "CL"
    bg["root"] = "BG"
    combo = cl.append(bg)

    df = rt.get_curves(
        up["m*"]["user"],
        up["m*"]["pass"],
        date="2021-12-20",
        contract_roots=["CL", "BG"],
    )

    pd.testing.assert_frame_equal(
        combo, df, check_like=True
    ), "get_curves test failed on combined ['CL','BG']"


def test_get_ir_swap_curve():
    ac = _load_json("getIRSwapCurve.json")
    ac.date = pd.to_datetime(ac.date)
    ac = ac.set_index("date")
    ac.index.name = "Date"

    ts = rt.get_ir_swap_curve(up["m*"]["user"], up["m*"]["pass"], end_dt="2021-12-27")

    pd.testing.assert_frame_equal(ac, ts, check_like=True)


def test_swap_info():
    ac = _load_json("swapInfo.json").dropna()
    ac.bizDays = pd.to_datetime(ac.bizDays)
    ac = ac.set_index("bizDays")
    ac.index.name = "date"
    ac = ac.rename({"fut_contract": "futures_contract"}, axis=1)
    ac = ac.replace("1stLineSettled", "first_line_settled")

    ts = rt.swap_info(**ms, date="2020-05-06", output="dataframe")
    print(ts)

    pd.testing.assert_frame_equal(ac, ts, check_like=True)

