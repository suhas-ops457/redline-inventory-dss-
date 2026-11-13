"""Test Cases: 4 in-code scenarios with expected behavior, actual output and computed pass/fail."""
import copy

import numpy as np
import pandas as pd
import streamlit as st

from pages._engine import example_inputs, run_pipeline, validate_inputs
from pages._ui import page_header, pct


def case_inputs(name: str) -> dict:
    """Build raw inputs for a test case from the illustrative example."""
    raw = copy.deepcopy(example_inputs())
    if name == "extreme":
        raw["stock"]["Elasticity α"] = 0.0          # invalid on purpose -> clipped to −0.1
    elif name == "zero":
        raw["stock"].loc[raw["stock"].MFC == "MFC-C", "Red-line stock"] = 0
    elif name == "highcost":
        raw["cost"] = raw["cost"] * 0 + 1000.0
    return raw


def non_increasing(res) -> bool:
    """Every optimal expected price path is non-increasing (MDP action constraint)."""
    return all(np.all(np.diff(tr["IR + CP"]["price"]) <= 1e-6) for tr in res["trajectories"].values())


def finite(res) -> bool:
    return bool(np.isfinite(res["detail"].select_dtypes("number").to_numpy()).all())


def check(name, res, msgs):
    """Return (actual-output dict, list of (check, passed))."""
    K, reb = res["kpis"], res["rebalancing"]
    st_ = reb["stock"].groupby("MFC")[["Out", "In"]].sum()
    units = int(reb["transfers"]["Units"].sum()) if len(reb["transfers"]) else 0
    actual = {"Items moved": units,
              "Store A change": int(st_.loc["MFC-A", "In"] - st_.loc["MFC-A", "Out"]),
              "Store C change": int(st_.loc["MFC-C", "In"] - st_.loc["MFC-C", "Out"]),
              "Money: rule / smart plan": f"₹{K['Rule-based']['Revenue']:,.0f} / ₹{K['IR + CP']['Revenue']:,.0f}",
              "Waste: rule / smart plan": f"{pct(K['Rule-based']['ESR'])} / {pct(K['IR + CP']['ESR'])}",
              "Avg discount (smart plan)": pct(K["IR + CP"]["CDR"]),
              "Data warnings shown": len(msgs)}
    checks = [("App runs without errors", finite(res)),
              ("Prices only go down during the day", non_increasing(res))]
    if name == "normal":
        checks += [("Crowded stores (A/B) send stock", st_.loc[["MFC-A", "MFC-B"], "Out"].sum() > 0),
                   ("Store C receives stock", actual["Store C change"] > 0),
                   ("Smart plan earns at least as much as the current rule", K["IR + CP"]["Revenue"] >= K["Rule-based"]["Revenue"]),
                   ("Smart plan wastes no more than the current rule", K["IR + CP"]["ESR"] <= K["Rule-based"]["ESR"]),
                   ("Moving stock doesn't lower profit", K["IR + CP"]["Net revenue"] >= K["CP only"]["Net revenue"] - 1e-6)]
    elif name == "extreme":
        checks += [("Invalid value caught with a friendly message", any("clipped" in m for _, m in msgs)),
                   ("Prices stay near full price (avg discount ≤ 5%)", K["IR + CP"]["CDR"] <= 0.05)]
    elif name == "zero":
        checks += [("Empty Store C receives stock", actual["Store C change"] > 0),
                   ("Current rule sells nothing at empty Store C", res["detail"].query("Scenario=='Rule-based' and MFC=='MFC-C'")["Units sold"].sum() == 0)]
    elif name == "highcost":
        checks += [("No stock is moved", units == 0),
                   ("Result equals smart pricing only", abs(K["IR + CP"]["Revenue"] - K["CP only"]["Revenue"]) < 1e-6)]
    return actual, checks


CASES = [
    ("normal", "Check 1 · Normal day", "The example data, unchanged.",
     "Stock moves from crowded stores to Store C. The smart plan earns more and wastes less than the current rule. "
     "Prices never go up."),
    ("extreme", "Check 2 · Customers ignore discounts", "Price sensitivity set to 0 for everything (an invalid value).",
     "The app shows a friendly warning instead of crashing. Since discounts don't help, prices stay at full price."),
    ("zero", "Check 3 · A store with no stock", "Store C has 0 expiring items of every product.",
     "No crash. Store C gets stock moved to it at night. Without moves, it sells nothing."),
    ("highcost", "Check 4 · Moving stock is too expensive", "Moving any item costs ₹1,000.",
     "Nothing is moved, because no move pays for itself. The result is the same as smart pricing alone."),
]

page_header("🧪 Checks", "Four built-in situations to show the tool behaves sensibly, including tricky ones. "
            "Each check runs the real calculation and is marked PASS or FAIL automatically.", tag="TESTING")


@st.cache_data(show_spinner="Running the checks …")
def run_case(name: str):
    inp, msgs = validate_inputs(case_inputs(name))
    return run_pipeline(inp), msgs, inp


results = []
for name, title, inp_desc, expected in CASES:
    try:
        res, msgs, inp = run_case(name)
        actual, checks = check(name, res, msgs)
    except Exception as e:  # noqa: BLE001
        actual, checks, msgs, inp = {"Error": str(e)}, [("Pipeline ran", False)], [], None
    passed = all(ok for _, ok in checks)
    results.append((title, passed))
    with st.container(border=True):
        h = st.columns([4, 1])
        h[0].subheader(title)
        h[1].markdown(f"<div style='text-align:right;font-size:1.3rem;font-weight:700;color:{'#0f7a4f' if passed else '#b42318'}'>"
                      f"{'✅ PASS' if passed else '❌ FAIL'}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.1, 1.2])
        with c1:
            st.markdown("**Situation**")
            st.write(inp_desc)
            if inp is not None:
                with st.expander("Show stock used"):
                    st.dataframe(inp["stock"][["MFC", "SKU", "Red-line stock"]].rename(columns={"MFC": "Store", "SKU": "Product", "Red-line stock": "Expiring stock"}), hide_index=True)
            for lvl, m in msgs:
                st.caption(f"⚠️ {m}")
        with c2:
            st.markdown("**What should happen**")
            st.write(expected)
            st.markdown("**Results**")
            for lbl, ok in checks:
                st.markdown(f"{'✅' if ok else '❌'} {lbl}")
        with c3:
            st.markdown("**What actually happened**")
            st.dataframe(pd.DataFrame({"Value": {k: str(v) for k, v in actual.items()}}))

n = sum(p for _, p in results)
(st.success if n == len(results) else st.error)(f"{n} of {len(results)} checks passed.")
