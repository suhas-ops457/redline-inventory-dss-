"""Home: plain-language problem, 3-step picture, how to use, paper results (reference only)."""
import plotly.graph_objects as go
import streamlit as st

from pages._engine import PAPER_BENCHMARK
from pages._ui import BLUE, ORANGE, kpi_card, page_header

page_header("Selling Food Before It Expires",
            "Online grocery companies (think Blinkit, Zepto, Instamart) keep stock in small city warehouses called "
            "<b>dark stores</b>. Every night some food has <b>less than 24 hours</b> left before it expires. "
            "This tool decides <b>where that food should go</b> and <b>what price to sell it at</b>, so less is "
            "thrown away and more money is recovered.",
            tag="ADVANCED OPERATIONS RESEARCH · PROTOTYPE")

c1, c2 = st.columns(2)
with c1:
    st.subheader("😟 The problem")
    st.markdown("""
- One store has **too much** expiring food → big discounts, and a lot still gets thrown away.
- Another store has **too little** → customers there would happily pay full price, but it runs out.
- A fixed rule like **"20% off after 6 pm"** is the same everywhere and ignores how customers actually behave.
""")
with c2:
    st.subheader("💡 The idea")
    st.markdown("""
- **At night:** move some food from crowded stores to stores that need it (only when it's worth the trip).
- **During the day:** lower prices step by step, only as much as needed to sell out by closing.
- **Prices only go down**, never back up, so customers aren't confused.
""")

st.subheader("How it works")
st.markdown("""
<div class="flow">
 <div class="step"><b>📥 1. Your data</b>How much expiring stock each store has, normal prices and normal sales.</div>
 <div class="arrow">➜</div>
 <div class="step night"><b>🌙 2. Move stock at night</b>Finds the cheapest moves that earn the most money.</div>
 <div class="arrow">➜</div>
 <div class="step day"><b>☀️ 3. Set prices by the hour</b>Picks a price for every hour, from 10 am to 9 pm.</div>
 <div class="arrow">➜</div>
 <div class="step"><b>📊 4. Compare</b>More money, less waste, smaller discounts vs. today's rule.</div>
</div>""", unsafe_allow_html=True)

st.subheader("How to use this app")
st.markdown("""
1. **Enter Data**: look at (or change) the example numbers.
   - Optional: **Learn from Past Sales** works out how strongly customers react to discounts from 14 days of sales.
2. **Step 1 · Move Stock**: see which store sends food to which.
3. **Step 2 · Set Prices**: see the suggested price for each hour.
4. **Results**: compare the smart plan with today's rule and download the numbers.
5. **Try Scenarios**: move the sliders and see what changes.
""")

st.subheader("What happened in the real world")
st.markdown("<div class='ref'><b>From the research paper (for reference only).</b> These are real results from Meituan, "
            "a large Chinese grocery company. They were <b>not</b> calculated by this app.</div>", unsafe_allow_html=True)
st.write("")
b = PAPER_BENCHMARK
cols = st.columns(4)
kpi_card("Discounts", f"{abs(b['cp_city_a']['CDR']):.1f}% smaller", "with smart pricing", True, cols[0])
kpi_card("Wasted food", f"{abs(b['cp_city_a']['ESR']):.1f}% less", "with smart pricing", True, cols[1])
kpi_card("Money recovered", f"{b['cp_city_a']['RIRCR']:.1f}% more", "with smart pricing", True, cols[2])
kpi_card("Across China", "≈50M yuan", "extra profit per year", None, cols[3])

g1, g2 = st.columns(2)
with g1:
    d = b["ir_cdr_cities"]
    fig = go.Figure(go.Bar(x=[f"City {k}" for k in d], y=list(d.values()), marker_color=BLUE,
                           text=[f"{v:.1f}%" for v in d.values()], textposition="outside",
                           hovertemplate="%{x}: discounts %{y:.1f}% smaller<extra></extra>"))
    fig.update_layout(title="Moving stock at night made discounts smaller in every city", height=320,
                      yaxis_title="Discounts reduced by (%)", yaxis_range=[0, 14])
    st.plotly_chart(fig, theme=None)
with g2:
    d = b["category_esr"]
    fig = go.Figure(go.Bar(y=list(d), x=list(d.values()), orientation="h", marker_color=ORANGE,
                           text=[f"{v:.1f}%" for v in d.values()], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}% less waste<extra></extra>"))
    fig.update_layout(title="Less food thrown away, by product type", height=320,
                      xaxis_title="Waste reduced by (%)", xaxis_range=[0, 66], yaxis_autorange="reversed")
    st.plotly_chart(fig, theme=None)
st.caption("Nationwide the company reported 20% less waste and 14% more money from expiring stock. "
           "The paper reports money in Chinese yuan; this app's example uses Indian rupees (₹).")
