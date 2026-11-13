"""Step 1 · Move Stock (Night): which store sends how much to which, in plain language."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._ui import (BLUE, GRAY, INK, MFC_COLORS, ORANGE, get_inputs, kpi_card, money,
                       page_header, safe_pipeline)

page_header("🌙 Step 1 · Move Stock at Night",
            "Before stores open, the tool checks whether moving expiring food from a <b>crowded</b> store to a store that "
            "<b>needs it</b> earns more than the cost of moving it. It only recommends moves that pay off.",
            tag="STEP 1 OF 2")
inp, msgs = get_inputs()
res = safe_pipeline(inp)
if res is None:
    st.stop()
reb = res["rebalancing"]
tr, stock = reb["transfers"], reb["stock"]
mfcs = sorted(inp["stock"]["MFC"].unique())

sku_opts = ["All products"] + list(inp["skus"]["SKU"])
sku = st.segmented_control("Show", sku_opts, default="All products", key="night_sku") or "All products"
t = tr if sku == "All products" else tr[tr.SKU == sku]
s = stock if sku == "All products" else stock[stock.SKU == sku]
sg = s.groupby("MFC", as_index=False)[["Stock before", "Out", "In", "Stock after", "Breakpoint q'",
                                        "π_lin before", "π_lin after"]].sum()

units = int(t["Units"].sum()) if len(t) else 0
cost = float(t["Cost"].sum()) if len(t) else 0.0
# Real gain = (smart plan − smart pricing only) money earned, minus shelf costs and transport (from the day model)
dd = res["detail"] if sku == "All products" else res["detail"][res["detail"].SKU == sku]
net = lambda sc: float((dd[dd.Scenario == sc]["Revenue"] - dd[dd.Scenario == sc]["Holding penalty"]).sum())
gain = net("IR + CP") - net("CP only") - cost
status = list(reb["status"].values()) if sku == "All products" else [reb["status"].get(sku, "")]
ok = all("Optimal" in x or "No stock" in x for x in status)
c = st.columns(4)
kpi_card("Items to move", f"{units}", f"{len(t)} trip(s)", None, c[0])
kpi_card("Cost of moving", money(cost), "transport cost", None, c[1])
kpi_card("Extra money from moving", money(gain), "vs not moving, after transport cost", gain >= 0, c[2])
kpi_card("Plan quality", "Best plan ✓" if ok else "Approximate", "the best possible moves were found" if ok else "quick estimate used", ok, c[3])
if any("Greedy" in x for x in status):
    st.warning("The exact calculation wasn't available, so a quick estimate was used. Results are approximate.")

st.subheader("✅ Recommended moves")
if units == 0:
    st.info("**No moves needed.** Moving stock would cost more than it earns, or the stores are already balanced.", icon="ℹ️")
else:
    cols = st.columns(min(3, len(t)))
    for k, (_, r) in enumerate(t.iterrows()):
        cols[k % len(cols)].markdown(
            f"<div class='reco'>🚚 Move <b>{int(r.Units)} × {r.SKU}</b><br>"
            f"<span class='small'>from <b>{r.From.replace('MFC-', 'Store ')}</b> to <b>{r.To.replace('MFC-', 'Store ')}</b> · "
            f"cost {money(r.Cost)}</span></div>", unsafe_allow_html=True)

left, right = st.columns([1, 1.1])
with left:
    mat = pd.DataFrame(0, index=mfcs, columns=mfcs, dtype=float)
    for _, r in t.iterrows():
        mat.loc[r.From, r.To] += r.Units
    pos = {m: (np.cos(np.pi / 2 + 2 * np.pi * k / len(mfcs)), np.sin(np.pi / 2 + 2 * np.pi * k / len(mfcs)))
           for k, m in enumerate(mfcs)}
    fig = go.Figure()
    mx = max(mat.values.max(), 1)
    for a in mfcs:
        for b in mfcs:
            u = mat.loc[a, b]
            if u <= 0:
                continue
            (x0, y0), (x1, y1) = pos[a], pos[b]
            dx, dy = x1 - x0, y1 - y0
            off = np.array([-dy, dx]) * 0.06
            fig.add_annotation(x=x1 - 0.18 * dx + off[0], y=y1 - 0.18 * dy + off[1],
                               ax=x0 + 0.18 * dx + off[0], ay=y0 + 0.18 * dy + off[1],
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                               arrowsize=1, arrowwidth=1.5 + 6 * u / mx, arrowcolor=ORANGE)
            fig.add_annotation(x=(x0 + x1) / 2 + off[0] * 2.2, y=(y0 + y1) / 2 + off[1] * 2.2,
                               text=f"<b>{int(u)} items</b>", showarrow=False, font=dict(color=INK, size=12),
                               bgcolor="rgba(255,255,255,.85)")
    for m in mfcs:
        row = sg[sg.MFC == m].iloc[0]
        fig.add_trace(go.Scatter(x=[pos[m][0]], y=[pos[m][1]], mode="markers+text",
                                 marker=dict(size=46, color=MFC_COLORS.get(m, GRAY), line=dict(color="white", width=2)),
                                 text=[m.replace("MFC-", "")], textfont=dict(color="white", size=16), showlegend=False,
                                 hovertemplate=f"Store {m[-1]}: {row['Stock before']:.0f} → {row['Stock after']:.0f} items<extra></extra>"))
        fig.add_annotation(x=pos[m][0], y=pos[m][1] - 0.36, showarrow=False,
                           text=f"Store {m[-1]}<br>{row['Stock before']:.0f} → {row['Stock after']:.0f}", font=dict(size=12))
    fig.update_layout(title="Where the stock goes (thicker arrow = more items)", height=380,
                      xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.0, 1.4], scaleanchor="x"))
    st.plotly_chart(fig, theme=None)
with right:
    fig = go.Figure()
    names = [m.replace("MFC-", "Store ") for m in sg.MFC]
    fig.add_bar(x=names, y=sg["Stock before"], name="Before moving", marker_color=GRAY,
                hovertemplate="%{x}: %{y:.0f} items before<extra></extra>")
    fig.add_bar(x=names, y=sg["Stock after"], name="After moving", marker_color=BLUE,
                hovertemplate="%{x}: %{y:.0f} items after<extra></extra>")
    fig.add_scatter(x=names, y=sg["Breakpoint q'"], mode="markers", name="How many can sell at full price",
                    marker=dict(symbol="line-ew-open", size=40, line=dict(width=3, color=ORANGE)),
                    hovertemplate="%{x}: about %{y:.0f} can sell at full price<extra></extra>")
    fig.update_layout(barmode="group", title="Stock at each store, before and after", height=380,
                      yaxis_title="Items", bargap=0.3, bargroupgap=0.08)
    st.plotly_chart(fig, theme=None)
    st.caption("The orange line shows roughly how many items each store could sell **without any discount**. "
               "Stock above the line will need discounts; stock below it is a missed chance to sell at full price.")

with st.expander("See the full table"):
    tab = s[["MFC", "SKU", "Stock before", "Out", "In", "Stock after"]].rename(columns={
        "MFC": "Store", "SKU": "Product", "Out": "Sent out", "In": "Received"})
    st.dataframe(tab, hide_index=True)
