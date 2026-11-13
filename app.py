"""Red-Line Inventory DSS — entry point (sidebar navigation + shared theme).

Run:  streamlit run app.py
"""
import streamlit as st

from pages._ui import CSS, init_state, reset_example

st.set_page_config(page_title="Near-Expiry Stock Planner", page_icon="🥬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_state()

PAGES = {
    "Start here": [st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True),
                   st.Page("pages/2_Data_Input.py", title="Enter Data", icon="📝"),
                   st.Page("pages/8_Learn_Sensitivity.py", title="Learn from Past Sales", icon="🧠")],
    "The plan": [st.Page("pages/3_Night_Rebalancing.py", title="Step 1 · Move Stock (Night)", icon="🌙"),
                 st.Page("pages/4_Day_Clearance_Pricing.py", title="Step 2 · Set Prices (Day)", icon="☀️")],
    "Results": [st.Page("pages/5_Results_Dashboard.py", title="Results", icon="📊"),
                st.Page("pages/6_What_If.py", title="Try Scenarios", icon="🎛️"),
                st.Page("pages/7_Test_Cases.py", title="Checks", icon="🧪")],
}
nav = st.navigation(PAGES)
st.session_state["_via_app"] = True  # pages check this (see page_header)

with st.sidebar:
    st.markdown("### 🥬 Near-Expiry Stock Planner")
    st.caption("Helps a grocery delivery company sell food that expires today: "
               "move it to the right store at night, then price it smartly during the day.")
    if st.button("↺ Reset to example", key="reset_sidebar"):
        reset_example()
        st.rerun()
    st.caption("Using *illustrative example data* — you can edit it on the Enter Data page.")

nav.run()
