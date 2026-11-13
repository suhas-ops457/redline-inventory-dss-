"""Results: today's rule vs the smart plan, in plain words, + paper reference + CSV download."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._engine import PAPER_BENCHMARK
from pages._ui import (KPI_LABEL, SCEN_COLORS, SCEN_SHORT, get_inputs, kpi_card, kpi_frame, money,
                       page_header, pct, safe_pipeline)

page_header("📊 Results",
            "The big picture across all stores and products: <b>today's rule</b> vs the <b>smart plan</b> "
            "(move stock at night + smart prices during the day). All numbers update when you change the data.",
            tag="SUMMARY")
inp, msgs = get_inputs()
res = safe_pipeline(inp)
if res is None:
    st.stop()
K = res["kpis"]
if res["total_stock"] == 0:
    st.info("There is no expiring stock in the data, so every number is zero.", icon="ℹ️")
base, opt, cp = K["Rule-based"], K["IR + CP"], K["CP only"]
order = ["Rule-based", "CP only", "IR + CP"]

st.markdown(f"<div class='explain'>📌 <b>In short:</b> the smart plan earns <b>{money(opt['Revenue'])}</b> instead of "
            f"<b>{money(base['Revenue'])}</b> and throws away <b>{opt['Spoiled units']:.0f}</b> items instead of "
            f"<b>{base['Spoiled units']:.0f}</b>.</div>", unsafe_allow_html=True)

c = st.columns(4)
kpi_card("Money earned", money(opt["Revenue"]), f"{opt['Revenue'] - base['Revenue']:+,.0f} ₹ vs current rule",
         opt["Revenue"] >= base["Revenue"], c[0])
kpi_card("Wasted stock", pct(opt["ESR"], 0), f"current rule: {pct(base['ESR'], 0)}", opt["ESR"] <= base["ESR"], c[1])
kpi_card("Value recovered", pct(opt["RIRCR"], 0), f"current rule: {pct(base['RIRCR'], 0)}", opt["RIRCR"] >= base["RIRCR"], c[2])
kpi_card("Average discount", pct(opt["CDR"], 0), f"current rule: {pct(base['CDR'], 0)}", opt["CDR"] <= base["CDR"], c[3])

with st.expander("What do these measures mean?"):
    st.markdown("""
- **Wasted stock (ESR):** share of expiring items that don't sell and get thrown away. *Lower is better.*
- **Value recovered (RIRCR):** money earned (minus small shelf and transport costs) as a share of what all the stock would earn at full price. *Higher is better.*
- **Average discount (CDR):** how much below full price items sold for, on average. *Lower is better.*
- **Smart pricing only** = smart prices but no stock moved at night. It shows how much the night moves add.
""")

left, right = st.columns([1.55, 1])
with left:
    fig = go.Figure()
    for sc in order:
        vals = [K[sc]["ESR"] * 100, K[sc]["RIRCR"] * 100, K[sc]["CDR"] * 100]
        fig.add_bar(name=SCEN_SHORT[sc], x=[KPI_LABEL["ESR"], KPI_LABEL["RIRCR"], KPI_LABEL["CDR"]], y=vals,
                    marker_color=SCEN_COLORS[sc], text=[f"{v:.0f}%" for v in vals], textposition="outside",
                    hovertemplate=f"{SCEN_SHORT[sc]} · %{{x}}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(barmode="group", title="Comparing the three approaches (%)", height=400, yaxis_title="%",
                      bargap=0.25, bargroupgap=0.06)
    st.plotly_chart(fig, theme=None)
    fig = go.Figure()
    for sc in order:
        fig.add_bar(name=SCEN_SHORT[sc], x=[SCEN_SHORT[sc]], y=[K[sc]["Revenue"]], marker_color=SCEN_COLORS[sc],
                    text=[money(K[sc]["Revenue"])], textposition="outside", showlegend=False,
                    hovertemplate=f"{SCEN_SHORT[sc]}: ₹%{{y:,.0f}}<extra></extra>")
    fig.update_layout(title="Money earned (₹)", height=340, yaxis_title="₹", bargap=0.45)
    st.plotly_chart(fig, theme=None)
with right:
    st.markdown("<div class='ref'><b>📄 Real-world results from the paper (reference only)</b><br>"
                "<span class='small'>Meituan's real stores, not calculated by this app.</span></div>", unsafe_allow_html=True)
    b = PAPER_BENCHMARK
    st.markdown(f"""
- Smart pricing → **{abs(b['cp_city_a']['ESR']):.0f}% less waste**
- Smart pricing → **{b['cp_city_a']['RIRCR']:.0f}% more money recovered**
- Smart pricing → **{abs(b['cp_city_a']['CDR']):.0f}% smaller discounts**
- Moving stock at night → discounts **{min(b['ir_cdr_cities'].values()):.0f}–{max(b['ir_cdr_cities'].values()):.0f}% smaller** in 5 cities
- Nationwide → **20% less waste**, **14% more money** from expiring stock
""")
    st.markdown("**This example's numbers**")
    st.markdown(f"""
- Waste: **{pct(base['ESR'], 0)} → {pct(opt['ESR'], 0)}**
- Money earned: **{(opt['Revenue'] / base['Revenue'] - 1) * 100 if base['Revenue'] else 0:+.0f}%**
- Moving stock made discounts **{abs((opt['CDR'] / cp['CDR'] - 1) * 100) if cp['CDR'] else 0:.0f}% smaller** than smart pricing alone
""")
    st.caption("Why can discounts look bigger than the current rule? The rule hardly discounts, but it throws away "
               "a lot of food. Selling at a discount still earns far more than throwing it away.")

st.subheader("Money earned by store")
d = res["detail"].copy()
by_mfc = d.groupby(["Scenario", "MFC"], as_index=False)["Revenue"].sum()
fig = go.Figure()
for sc in order:
    s = by_mfc[by_mfc.Scenario == sc]
    fig.add_bar(name=SCEN_SHORT[sc], x=[m.replace("MFC-", "Store ") for m in s.MFC], y=s.Revenue, marker_color=SCEN_COLORS[sc],
                hovertemplate=f"{SCEN_SHORT[sc]} · %{{x}}: ₹%{{y:,.0f}}<extra></extra>")
fig.update_layout(barmode="group", height=340, yaxis_title="₹", bargap=0.3, bargroupgap=0.06, title="")
st.plotly_chart(fig, theme=None)

with st.expander("See the detailed table"):
    tab = d.rename(columns={"MFC": "Store", "SKU": "Product", "Stock": "Starting stock",
                            "Unsold (spoiled)": "Thrown away", "Revenue": "Money earned (₹)"})
    tab["Scenario"] = tab["Scenario"].map(SCEN_SHORT)
    st.dataframe(tab[["Scenario", "Store", "Product", "Starting stock", "Units sold", "Thrown away", "Money earned (₹)"]].round(0),
                 hide_index=True)

kf = kpi_frame(K).reset_index().rename(columns={"index": "Scenario"})
kf["Scenario"] = kf["Scenario"].map(SCEN_SHORT)
tr = res["rebalancing"]["transfers"]
csv = ("# Summary\n" + kf.to_csv(index=False) + "\n# By store and product\n" + d.to_csv(index=False) +
       "\n# Night moves\n" + tr.to_csv(index=False))
st.download_button("⬇️ Download results (CSV, opens in Excel)", csv, file_name="near_expiry_results.csv",
                   mime="text/csv", type="primary")
