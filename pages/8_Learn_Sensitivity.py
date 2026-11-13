"""Learn Price Sensitivity: estimate it from past sales (stand-in for the paper's ML model)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pages._engine import estimate_sensitivity, example_history
from pages._ui import BLUE, GRAY, ORANGE, editor, init_state, page_header

page_header("🧠 Learn Price Sensitivity from Past Sales",
            "Instead of typing in how strongly customers react to discounts, the tool can <b>learn it from past sales</b>. "
            "It looks at days when the price was lower or higher and sees how sales changed. "
            "Then you can use the learned values in the plan with one click.", tag="LEARNING FROM DATA")
init_state()
raw = st.session_state.raw
if "history" not in raw:
    raw["history"] = example_history()

st.info("**Example sales history, made up for illustration**: the last 14 days of prices and items sold for each "
        "store and product. With real data, you'd load the company's actual sales records here.", icon="ℹ️")

est = estimate_sensitivity(raw["history"])
typed = raw["stock"][["MFC", "SKU", "Elasticity α", "Baseline sales q̄", "Baseline price p̄"]]
m = est.merge(typed, on=["MFC", "SKU"], how="inner")


def reliability(r2):
    if pd.isna(r2):
        return "⚪ Not enough data"
    return "🟢 Good" if r2 >= 0.7 else ("🟡 OK" if r2 >= 0.4 else "🔴 Weak")


st.subheader("1 · See what was learned")
opts = {f"Store {r.MFC[-1]} · {r.SKU}": (r.MFC, r.SKU) for r in m.itertuples()}
label = st.selectbox("Store and product", list(opts), key="learn_item")
mfc, sku = opts[label]
h = raw["history"]
g = h[(h.MFC == mfc) & (h.SKU == sku)].copy()
g["Price"] = pd.to_numeric(g["Price"], errors="coerce")
g["Units sold"] = pd.to_numeric(g["Units sold"], errors="coerce")
r = m[(m.MFC == mfc) & (m.SKU == sku)].iloc[0]

left, right = st.columns([1.6, 1])
with left:
    fig = go.Figure()
    fig.add_scatter(x=g["Price"], y=g["Units sold"], mode="markers", name="One past day",
                    marker=dict(size=10, color=GRAY, line=dict(color="white", width=1)),
                    hovertemplate="Day %{customdata}: ₹%{x:.0f} → %{y} sold<extra></extra>", customdata=g["Day"])
    if not pd.isna(r["Learned α"]) and g["Price"].notna().any():
        xs = np.linspace(g["Price"].min() * 0.95, g["Price"].max() * 1.03, 80)
        fig.add_scatter(x=xs, y=np.exp(r["Intercept"]) * xs ** r["Learned α"], mode="lines", name="What the tool learned",
                        line=dict(color=BLUE, width=3), hoverinfo="skip")
    fig.update_layout(title="Lower price → more items sold?", height=380,
                      xaxis_title="Price that day (₹)", yaxis_title="Items sold that day", yaxis_rangemode="tozero")
    st.plotly_chart(fig, theme=None)
with right:
    lv = r["Learned α"]
    st.markdown("<div class='kpi'><div class='lbl'>Learned price sensitivity</div>"
                f"<div class='val'>{'—' if pd.isna(lv) else f'{lv:.1f}'}</div>"
                f"<div class='dlt neu'>typed-in value: {r['Elasticity α']:.1f}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='kpi'><div class='lbl'>How reliable?</div>"
                f"<div class='val' style='font-size:1.2rem'>{reliability(r['R2'])}</div>"
                f"<div class='dlt neu'>based on {int(r['Days of data'])} days of sales</div></div>", unsafe_allow_html=True)
    st.write("")
    if not pd.isna(lv):
        st.markdown(f"<div class='explain'>📌 <b>In plain words:</b> a <b>10% discount</b> on {sku} at Store {mfc[-1]} "
                    f"sells about <b>{(0.9 ** lv - 1) * 100:.0f}% more</b>.</div>", unsafe_allow_html=True)
    elif r["Note"]:
        st.warning(r["Note"])

st.subheader("2 · All stores and products")
tab = pd.DataFrame({
    "Store": m.MFC.str.replace("MFC-", "Store "), "Product": m.SKU,
    "Typed-in value": m["Elasticity α"].round(1), "Learned value": m["Learned α"].round(1),
    "How reliable": m["R2"].map(reliability),
    "Avg daily sales (14 days)": m["Avg sales"].round(0), "Avg price (14 days)": m["Avg price"].round(0)})
st.dataframe(tab, hide_index=True)
st.caption("🔴 Weak means the past prices didn't move enough, or sales were too irregular, to learn a clear pattern. "
           "Treat those values with care.")

c1, c2 = st.columns([1, 1.6])
also = c2.checkbox("Also update normal daily sales and normal price to the 14-day averages (this is what the paper does)",
                   value=False, key="learn_also")
if c1.button("✅ Use learned values in the plan", type="primary", key="learn_apply"):
    stock = raw["stock"].copy()
    n = 0
    for _, row in m.iterrows():
        if pd.isna(row["Learned α"]):
            continue
        i = stock.index[(stock.MFC == row.MFC) & (stock.SKU == row.SKU)]
        stock.loc[i, "Elasticity α"] = round(float(row["Learned α"]), 2)
        if also:
            stock.loc[i, "Baseline sales q̄"] = round(float(row["Avg sales"]), 1)
            stock.loc[i, "Baseline price p̄"] = round(float(row["Avg price"]), 0)
        n += 1
    raw["stock"] = stock
    st.session_state.ver += 1
    st.success(f"Done. {n} learned value(s) are now used everywhere in the app. Open **Results** to see the effect. "
               "Use **Reset to example** to go back.")

now = est.merge(raw["stock"][["MFC", "SKU", "Elasticity α"]], on=["MFC", "SKU"])
using = np.allclose(now["Elasticity α"].to_numpy(float), now["Learned α"].fillna(now["Elasticity α"]).round(2).to_numpy(float), atol=0.006)
st.caption("Currently the plan uses: **" + ("learned values" if using else "typed-in values") + "**.")

with st.expander("✏️ Edit the sales history"):
    editor("history", disabled=["MFC", "SKU", "Day"], hide_index=True, height=320, column_config={
        "MFC": st.column_config.TextColumn("Store"), "SKU": st.column_config.TextColumn("Product"),
        "Price": st.column_config.NumberColumn("Price (₹)", min_value=0.0, format="₹%.0f"),
        "Units sold": st.column_config.NumberColumn("Items sold", min_value=0, step=1)})

with st.expander("How does this work, and how is it different from the paper?"):
    st.markdown("""
- The tool compares **cheaper days with more expensive days** and draws the best-fitting curve through them. How steep the curve is gives the price sensitivity.
- This uses **the same demand model as the paper**, fitted with simple regression.
- The paper uses a more advanced machine-learning method (a **causal forest**). It also accounts for things like weekday, weather and store type, and is retrained every day on the latest 3 weeks of data. That needs far more data than a prototype has.
- **In production** this page would be replaced by that model. Everything after it (moving stock, setting prices) stays the same.
""")
