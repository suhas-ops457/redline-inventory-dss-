"""Step 2 · Set Prices (Day): hour-by-hour price plan vs the '20% off after 6 pm' rule."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._engine import HOURS, evaluate_schedule, rule_prices, solve_pricing_dp
from pages._ui import BLUE, GRAY, get_inputs, kpi_card, money, page_header, pct, safe_pipeline

page_header("☀️ Step 2 · Set Prices During the Day",
            "For each store and product, the tool picks a price for every hour. It starts at full price and "
            "<b>only lowers it when needed</b> to sell everything before closing. It is compared with today's simple rule: "
            "<b>full price, then 20% off after 6 pm</b>.", tag="STEP 2 OF 2")
inp, msgs = get_inputs()
res = safe_pipeline(inp)
if res is None:
    st.stop()


@st.cache_data(show_spinner="Working out the best prices …")
def dp_cached(q0, p0, floor, pbar, qbar, alpha, ch, shares):
    """Cached price plan + rule result for one store-product."""
    sh = np.array(shares)
    opt = solve_pricing_dp(q0, p0=p0, floor=floor, pbar=pbar, qbar=qbar, alpha=alpha, ch=ch, shares=sh)
    rule = evaluate_schedule(q0, rule_prices(p0, floor), p0=p0, pbar=pbar, qbar=qbar, alpha=alpha, ch=ch, shares=sh)
    return opt, rule


st_df = inp["stock"]
c1, c2, c3 = st.columns([1.3, 1, 1.4])
opts = {f"Store {r.MFC[-1]} · {r.SKU}": (r.MFC, r.SKU) for r in st_df.itertuples()}
label = c1.selectbox("Store and product", list(opts), key="day_item")
mfc, sku = opts[label]
row = st_df[(st_df.MFC == mfc) & (st_df.SKU == sku)].iloc[0]
srow = inp["skus"].set_index("SKU").loc[sku]
after = res["rebalancing"]["stock"].set_index(["MFC", "SKU"]).loc[(mfc, sku)]
src = c2.radio("Starting stock", ["After night moves", "Without night moves"], key="day_src")
q0 = int(after["Stock after"] if src == "After night moves" else after["Stock before"])
alpha = c3.slider("How much do customers react to discounts?", -4.0, -0.1,
                  float(np.clip(row["Elasticity α"], -4.0, -0.1)), 0.1, key=f"day_alpha_{label}",
                  help="Further left = customers react strongly to discounts. Near the right end = they barely react.")
c3.caption("◀ react a lot · barely react ▶")

shares = inp["hourly"][sku].to_numpy(float)
shares = tuple(shares / shares.sum())
p0, floor, ch = float(srow["Retail price p0"]), float(srow["Price floor"]), float(srow["Holding penalty c_h"])
if q0 == 0:
    st.info(f"Store {mfc[-1]} has **no expiring {sku}** today, so there is nothing to price. Pick another store or product.", icon="ℹ️")
    st.stop()
if alpha >= -0.15:
    st.warning("Customers barely react to discounts here, so lowering the price doesn't help much. "
               "The tool keeps prices high, and some waste can't be avoided.")
opt, rule = dp_cached(q0, p0, floor, float(row["Baseline price p̄"]), float(row["Baseline sales q̄"]), alpha, ch, shares)


def tot(tr):
    rev, lv = tr["revenue"].sum(), tr["list_revenue"].sum()
    return {"Revenue": rev, "Spoiled": tr["stock_end"][-1], "Waste": tr["stock_end"][-1] / q0,
            "Disc": 1 - rev / lv if lv > 0 else 0.0}


O, R = tot(opt), tot(rule)
k = st.columns(4)
kpi_card("Money earned (smart prices)", money(O["Revenue"]), f"{O['Revenue'] - R['Revenue']:+,.0f} ₹ vs current rule",
         O["Revenue"] >= R["Revenue"], k[0])
kpi_card("Items thrown away", f"{O['Spoiled']:.0f}", f"current rule: {R['Spoiled']:.0f} items", O["Spoiled"] <= R["Spoiled"], k[1])
kpi_card("Wasted stock", pct(O["Waste"], 0), f"current rule: {pct(R['Waste'], 0)}", O["Waste"] <= R["Waste"], k[2])
kpi_card("Average discount", pct(O["Disc"], 0), f"current rule: {pct(R['Disc'], 0)}", None, k[3])
st.caption(f"Starting with {q0} items · full price ₹{p0:.0f} · never below ₹{floor:.0f}. "
           "Numbers are averages: real sales vary a little from day to day.")

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    fig.add_scatter(x=HOURS, y=opt["price"], name="Smart price", mode="lines+markers",
                    line=dict(color=BLUE, width=2, shape="hv"), marker=dict(size=8),
                    hovertemplate="%{x}: ₹%{y:.0f}<extra>Smart price</extra>")
    fig.add_scatter(x=HOURS, y=rule["price"], name="Current rule", mode="lines+markers",
                    line=dict(color=GRAY, width=2, dash="dash", shape="hv"), marker=dict(size=8),
                    hovertemplate="%{x}: ₹%{y:.0f}<extra>Current rule</extra>")
    fig.add_hline(y=p0, line_dash="dot", line_color="#b0b0aa", annotation_text="full price", annotation_position="top left")
    fig.add_hline(y=floor, line_dash="dot", line_color="#b0b0aa", annotation_text="lowest allowed", annotation_position="bottom left")
    fig.update_layout(title="Price for each hour", height=380, yaxis_title="Price (₹)", xaxis_title="Hour",
                      yaxis_range=[floor * 0.85, p0 * 1.08])
    st.plotly_chart(fig, theme=None)
with g2:
    x = ["Open"] + HOURS
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.r_[q0, opt["stock_end"]], name="Smart price", line=dict(color=BLUE, width=2),
                    mode="lines+markers", marker=dict(size=8), hovertemplate="%{x}: %{y:.0f} items left<extra>Smart price</extra>")
    fig.add_scatter(x=x, y=np.r_[q0, rule["stock_end"]], name="Current rule", line=dict(color=GRAY, width=2, dash="dash"),
                    mode="lines+markers", marker=dict(size=8), hovertemplate="%{x}: %{y:.0f} items left<extra>Current rule</extra>")
    fig.update_layout(title="Items left on the shelf", height=380, yaxis_title="Items left",
                      xaxis_title="Hour", yaxis_rangemode="tozero")
    st.plotly_chart(fig, theme=None)
st.markdown(f"<div class='explain'>📌 <b>In short:</b> with smart prices this store earns <b>{money(O['Revenue'])}</b> "
            f"and throws away about <b>{O['Spoiled']:.0f}</b> items. With the current rule it would earn "
            f"<b>{money(R['Revenue'])}</b> and throw away about <b>{R['Spoiled']:.0f}</b>.</div>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_bar(name="Current rule", x=["Money earned (₹)"], y=[R["Revenue"]], marker_color=GRAY,
            text=[money(R["Revenue"])], textposition="outside")
fig.add_bar(name="Smart price", x=["Money earned (₹)"], y=[O["Revenue"]], marker_color=BLUE,
            text=[money(O["Revenue"])], textposition="outside")
fig.update_layout(barmode="group", title="Money earned today", height=320, bargap=0.45, bargroupgap=0.1)
g3, g4 = st.columns(2)
g3.plotly_chart(fig, theme=None)
fig = go.Figure()
fig.add_bar(name="Current rule", x=["Items thrown away"], y=[R["Spoiled"]], marker_color=GRAY,
            text=[f"{R['Spoiled']:.0f}"], textposition="outside")
fig.add_bar(name="Smart price", x=["Items thrown away"], y=[O["Spoiled"]], marker_color=BLUE,
            text=[f"{O['Spoiled']:.0f}"], textposition="outside")
fig.update_layout(barmode="group", title="Food thrown away at closing", height=320, bargap=0.45, bargroupgap=0.1)
g4.plotly_chart(fig, theme=None)

with st.expander("See hour-by-hour numbers"):
    tab = pd.DataFrame({"Hour": HOURS, "Smart price (₹)": opt["price"], "Items sold": opt["sales"],
                        "Items left": opt["stock_end"], "Rule price (₹)": rule["price"],
                        "Items sold (rule)": rule["sales"], "Items left (rule)": rule["stock_end"]})
    st.dataframe(tab.round(0), hide_index=True)
