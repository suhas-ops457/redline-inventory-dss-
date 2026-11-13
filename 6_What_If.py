"""Try Scenarios: friendly sliders with live-updating results."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._engine import scale_inputs
from pages._ui import BLUE, GRAY, SCEN_COLORS, SCEN_SHORT, cached_pipeline, get_inputs, kpi_card, money, page_header, pct

page_header("🎛️ Try Scenarios",
            "Ask <b>“what if…?”</b>. Move a slider and every number and chart updates. "
            "1.0× means the same as your data, 2.0× means double.", tag="WHAT-IF")
inp, msgs = get_inputs()

c = st.columns(4)
am = c[0].slider("Customers react to discounts", 0.25, 2.0, 1.0, 0.05, key="wi_a", format="%.2f×",
                 help="Higher = customers buy a lot more when the price drops.")
sm = c[1].slider("Amount of expiring stock", 0.0, 2.5, 1.0, 0.1, key="wi_s", format="%.1f×")
cm = c[2].slider("Cost of moving stock", 0.0, 20.0, 1.0, 0.5, key="wi_c", format="%.1f×")
hm = c[3].slider("Cost of unsold stock on shelf", 0.0, 5.0, 1.0, 0.25, key="wi_h", format="%.2f×")

try:
    base_res = cached_pipeline(inp)
    res = cached_pipeline(scale_inputs(inp, am, sm, cm, hm))
except Exception:  # noqa: BLE001
    st.error("This combination couldn't be calculated. Try less extreme slider values.")
    st.stop()

K, K0 = res["kpis"], base_res["kpis"]
o, o0 = K["IR + CP"], K0["IR + CP"]
t, t0 = res["rebalancing"]["transfers"], base_res["rebalancing"]["transfers"]
units = int(t["Units"].sum()) if len(t) else 0
units0 = int(t0["Units"].sum()) if len(t0) else 0
k = st.columns(4)
kpi_card("Items moved at night", f"{units}", f"{units - units0:+d} vs your data", None, k[0])
kpi_card("Money earned", money(o["Revenue"]), f"{o['Revenue'] - o0['Revenue']:+,.0f} ₹ vs your data", None, k[1])
kpi_card("Wasted stock", pct(o["ESR"], 0), f"your data: {pct(o0['ESR'], 0)}", o["ESR"] <= o0["ESR"] + 1e-9, k[2])
kpi_card("Average discount", pct(o["CDR"], 0), f"your data: {pct(o0['CDR'], 0)}", o["CDR"] <= o0["CDR"] + 1e-9, k[3])
if res["total_stock"] == 0:
    st.info("With 0× stock there is nothing to sell, so everything is zero.", icon="ℹ️")
if units == 0 and units0 > 0:
    st.info("Moving stock is now **too expensive to be worth it**, so no stock is moved.", icon="🚚")

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    for sc in ["Rule-based", "CP only", "IR + CP"]:
        fig.add_bar(name=SCEN_SHORT[sc], x=[SCEN_SHORT[sc]], y=[K[sc]["Revenue"]], marker_color=SCEN_COLORS[sc],
                    text=[money(K[sc]["Revenue"])], textposition="outside", showlegend=False)
    fig.update_layout(title="Money earned in this scenario (₹)", height=360, bargap=0.45)
    st.plotly_chart(fig, theme=None)
with g2:
    fig = go.Figure()
    for sc in ["Rule-based", "CP only", "IR + CP"]:
        fig.add_bar(name=SCEN_SHORT[sc], x=[SCEN_SHORT[sc]], y=[K[sc]["ESR"] * 100], marker_color=SCEN_COLORS[sc],
                    text=[f"{K[sc]['ESR'] * 100:.0f}%"], textposition="outside", showlegend=False)
    fig.update_layout(title="Share of stock thrown away (%)", height=360, bargap=0.45, yaxis_rangemode="tozero")
    st.plotly_chart(fig, theme=None)

s = res["rebalancing"]["stock"].groupby("MFC", as_index=False)[["Stock before", "Stock after"]].sum()
fig = go.Figure()
fig.add_bar(name="Before moving", x=[m.replace("MFC-", "Store ") for m in s.MFC], y=s["Stock before"], marker_color=GRAY)
fig.add_bar(name="After moving", x=[m.replace("MFC-", "Store ") for m in s.MFC], y=s["Stock after"], marker_color=BLUE)
fig.update_layout(barmode="group", title="Stock at each store (all products)", height=340, bargap=0.3, bargroupgap=0.08)
st.plotly_chart(fig, theme=None)

st.subheader("When does moving stock stop being worth it?")
rows = []
for mult in [0, 1, 2, 4, 6, 8, 10, 15, 20]:
    r = cached_pipeline(scale_inputs(inp, am, sm, mult, hm))
    tt = r["rebalancing"]["transfers"]
    rows.append({"Cost ×": mult, "Items moved": int(tt["Units"].sum()) if len(tt) else 0})
sw = pd.DataFrame(rows)
fig = go.Figure(go.Scatter(x=sw["Cost ×"], y=sw["Items moved"], mode="lines+markers", line=dict(color=BLUE, width=2),
                           marker=dict(size=8), hovertemplate="moving cost %{x}×: %{y} items moved<extra></extra>"))
fig.add_vline(x=cm, line_dash="dot", line_color=GRAY, annotation_text="your slider")
fig.update_layout(title="Items moved as moving gets more expensive", height=320,
                  xaxis_title="Cost of moving stock (× your data)", yaxis_title="Items moved")
st.plotly_chart(fig, theme=None)
st.caption("As transport gets more expensive, fewer moves pay off, until none are recommended.")
