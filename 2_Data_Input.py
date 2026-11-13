"""Enter Data: editable example tables with friendly column names, reset button, checks."""
import streamlit as st

from pages._ui import TABLE_LABELS, editor, get_inputs, page_header, reset_example

page_header("Enter Data",
            "These are the numbers the plan is built from. <b>Click any white cell to change it</b>. "
            "Every other page updates automatically. Grey columns are names and can't be edited.",
            tag="ILLUSTRATIVE EXAMPLE DATA")

top = st.columns([3, 1])
top[0].info("**Example data, made up for illustration.** 3 dark stores (A, B, C), 3 products, "
            "12 selling hours (10 am – 9 pm). Store A has **too much** stock, Store C has **too little**. "
            "Prices are in ₹.", icon="ℹ️")
if top[1].button("↺ Reset to example", type="primary", key="reset_main"):
    reset_example()
    st.rerun()

nc = st.column_config.NumberColumn
st.subheader("1 · " + TABLE_LABELS["skus"])
editor("skus", disabled=["SKU", "Category"], hide_index=True, column_config={
    "SKU": st.column_config.TextColumn("Product"),
    "Retail price p0": nc("Full price (₹)", help="Normal price, no discount", min_value=0.0, format="₹%.0f"),
    "Price floor": nc("Lowest allowed price (₹)", help="We never sell below this", min_value=0.0, format="₹%.0f"),
    "Holding penalty c_h": nc("Cost of unsold stock (₹ per item per hour)",
                              help="Small charge for every item still sitting on the shelf each hour", min_value=0.0, format="₹%.2f"),
})

st.subheader("2 · " + TABLE_LABELS["stock"])
editor("stock", disabled=["MFC", "SKU"], hide_index=True, column_config={
    "MFC": st.column_config.TextColumn("Store"),
    "SKU": st.column_config.TextColumn("Product"),
    "Red-line stock": nc("Expiring stock (items)", help="Items with less than 24 hours left", min_value=0, step=1),
    "Baseline sales q̄": nc("Normal daily sales (items)", help="How many usually sell in a day", min_value=0.0, format="%.0f"),
    "Baseline price p̄": nc("Normal selling price (₹)", help="Usual average price paid", min_value=0.0, format="₹%.0f"),
    "Elasticity α": nc("Price sensitivity", max_value=0.0, format="%.1f",
                       help="How strongly customers react to a discount. −2 means: 10% cheaper → about 20% more sold. "
                            "Closer to 0 = customers barely react."),
})

c1, c2 = st.columns([1, 1.4])
with c1:
    st.subheader("3 · " + TABLE_LABELS["cost"])
    editor("cost", column_config={c: nc(c.replace("MFC-", "To store "), min_value=0.0, format="₹%.0f")
                                  for c in ["MFC-A", "MFC-B", "MFC-C"]})
    st.caption("Each row is the store sending the stock.")
with c2:
    st.subheader("4 · " + TABLE_LABELS["hourly"])
    editor("hourly", disabled=["Hour"], hide_index=True, height=280)
    st.caption("Only the pattern matters: an hour with 12 is twice as busy as an hour with 6.")

st.subheader("Data check")
inp, msgs = get_inputs(show=True)
with st.expander("Which stores have too much stock?"):
    s = inp["stock"].merge(inp["skus"][["SKU", "Retail price p0"]], on="SKU")
    full = s["Baseline sales q̄"] * (s["Retail price p0"] / s["Baseline price p̄"]) ** s["Elasticity α"]
    out = s[["MFC", "SKU", "Red-line stock"]].rename(columns={"MFC": "Store", "SKU": "Product", "Red-line stock": "Expiring stock"})
    out["Can sell at full price"] = full.round(0).astype(int)
    out["Status"] = [("🔴 Too much" if a > 1.2 * b else "🔵 Needs more" if a < 0.8 * b else "🟡 About right")
                     for a, b in zip(out["Expiring stock"], full)]
    st.dataframe(out, hide_index=True)
