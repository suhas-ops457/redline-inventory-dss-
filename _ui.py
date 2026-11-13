"""Shared Streamlit helpers: theme, session state, editors, KPI cards, caching."""
from __future__ import annotations

import copy

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from pages._engine import example_inputs, run_pipeline, validate_inputs

# Palette (validated categorical order) + neutral for the baseline
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, MUTED = "#8a8984", "#1f2933", "#5f6b7a"
SCEN_COLORS = {"Rule-based": GRAY, "CP only": AQUA, "IR + CP": BLUE}
# Friendly display names for the three scenarios
SCEN_LABEL = {"Rule-based": "Current rule (20% off after 6 pm)", "CP only": "Smart pricing only",
              "IR + CP": "Smart plan (move stock + smart pricing)"}
SCEN_SHORT = {"Rule-based": "Current rule", "CP only": "Smart pricing only", "IR + CP": "Smart plan"}
# Friendly KPI names
KPI_LABEL = {"CDR": "Average discount", "ESR": "Wasted stock", "RIRCR": "Value recovered"}
MFC_COLORS = {"MFC-A": BLUE, "MFC-B": ORANGE, "MFC-C": AQUA}
TABLE_LABELS = {"skus": "Products", "stock": "Stock and sales at each store",
                "cost": "Cost to move one item between stores (₹)", "hourly": "When customers shop (busier hour = bigger number)"}

pio.templates["dss"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#e8e8e4", zerolinecolor="#d0d0cb", linecolor="#d0d0cb"),
    yaxis=dict(gridcolor="#e8e8e4", zerolinecolor="#d0d0cb", linecolor="#d0d0cb"),
    title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12)),
    margin=dict(l=10, r=10, t=78, b=10), hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=12),
    colorway=[BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", VIOLET, "#e34948"],
))
pio.templates.default = "plotly+dss"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.kpi {border: 1px solid #e3e5e8; border-radius: 12px; padding: 14px 16px; background: #ffffff;
      box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%;}
.kpi .lbl {font-size: .78rem; color: #5f6b7a; text-transform: uppercase; letter-spacing: .04em;}
.kpi .val {font-size: 1.55rem; font-weight: 650; color: #1f2933; margin-top: 2px;}
.kpi .dlt {font-size: .85rem; margin-top: 2px;}
.kpi .good {color: #0f7a4f;} .kpi .bad {color: #b42318;} .kpi .neu {color: #5f6b7a;}
.tag {display:inline-block; font-size:.72rem; padding:2px 8px; border-radius:999px;
      background:#eef4fc; color:#1d5fae; font-weight:600; letter-spacing:.03em;}
.ref {background:#fbf7ef; border:1px dashed #d9b46a; border-radius:12px; padding:12px 16px;}
.flow {display:flex; gap:10px; align-items:stretch; flex-wrap:wrap; margin:6px 0 4px 0;}
.flow .step {flex:1; min-width:150px; border-radius:12px; padding:12px 14px; border:1px solid #e3e5e8; background:#fff;}
.flow .step b {display:block; margin-bottom:4px;}
.flow .night {background:#eef1fb; border-color:#c9d1f2;}
.flow .day {background:#fff6e8; border-color:#f1d6a8;}
.flow .arrow {align-self:center; font-size:1.4rem; color:#8a8984;}
.small {font-size:.85rem; color:#5f6b7a;}
.explain {background:#f3f7fd; border-left:4px solid #2a78d6; border-radius:8px; padding:12px 16px; margin:4px 0 14px 0;
          font-size:.97rem; color:#1f2933;}
.explain b {color:#1d5fae;}
.reco {border:1px solid #e3e5e8; border-radius:10px; padding:10px 14px; margin-bottom:8px; background:#fff;}
</style>
"""


def init_state() -> None:
    """Create session-state inputs from the hard-coded example (once)."""
    if "raw" not in st.session_state:
        st.session_state.raw = copy.deepcopy(example_inputs())
        st.session_state.ver = 0


def reset_example() -> None:
    """Restore the illustrative example inputs."""
    st.session_state.raw = copy.deepcopy(example_inputs())
    st.session_state.ver = st.session_state.get("ver", 0) + 1


def _apply_edits(name: str, key: str) -> None:
    """on_change callback: fold data_editor deltas into session inputs."""
    df = st.session_state.raw[name].copy()
    for r, changes in st.session_state[key].get("edited_rows", {}).items():
        for col, val in changes.items():
            df.iloc[int(r), df.columns.get_loc(col)] = val
    st.session_state.raw[name] = df
    st.session_state.ver += 1


def editor(name: str, **kw) -> None:
    """Editable table bound to st.session_state.raw[name] (fixed rows)."""
    key = f"ed_{name}_{st.session_state.ver}"
    st.data_editor(st.session_state.raw[name], key=key, num_rows="fixed",
                   on_change=_apply_edits, args=(name, key), **kw)


def get_inputs(show: bool = False):
    """Validated current inputs (+ optionally render validation messages)."""
    init_state()
    inp, msgs = validate_inputs(st.session_state.raw)
    if show:
        show_messages(msgs)
    elif msgs:
        st.caption(f"⚠️ {len(msgs)} note(s) about your data — see the **Enter Data** page.")
    return inp, msgs


def show_messages(msgs) -> None:
    """Render validation messages as friendly alerts."""
    if not msgs:
        st.success("Your data looks good.", icon="✅")
    for lvl, txt in msgs:
        {"error": st.error, "warning": st.warning}.get(lvl, st.info)(txt)


@st.cache_data(show_spinner="Working out the best plan …")
def cached_pipeline(inp: dict) -> dict:
    """Cached full pipeline (recomputes only when inputs change)."""
    return run_pipeline(inp)


def safe_pipeline(inp: dict):
    """Run the pipeline; on any failure show a friendly message instead of crashing."""
    try:
        return cached_pipeline(inp)
    except Exception as e:  # noqa: BLE001
        st.error("Something in the data doesn't add up, so the plan couldn't be calculated. "
                 "Check the Enter Data page or press 'Reset to example'.")
        return None


def page_header(title: str, subtitle: str, tag: str | None = None) -> None:
    """Consistent page title block.

    Guard: if a page is opened directly before app.py has run (e.g. a deep link
    right after the server starts), send the visitor through app.py so the
    proper sidebar menu (st.navigation) is used.
    """
    if not st.session_state.get("_via_app"):
        st.switch_page("app.py")
    st.markdown(CSS, unsafe_allow_html=True)
    if tag:
        st.markdown(f"<span class='tag'>{tag}</span>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='explain'>{subtitle}</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", good: bool | None = None, col=None) -> None:
    """HTML KPI card; good=True green delta, False red, None neutral."""
    cls = "neu" if good is None else ("good" if good else "bad")
    html = f"<div class='kpi'><div class='lbl'>{label}</div><div class='val'>{value}</div>" \
           f"<div class='dlt {cls}'>{delta}&nbsp;</div></div>"
    (col or st).markdown(html, unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def money(x: float) -> str:
    return f"₹{x:,.0f}"


def kpi_frame(k: dict) -> pd.DataFrame:
    """KPI dict-of-dicts -> tidy frame (scenarios as rows)."""
    return pd.DataFrame(k).T[["Revenue", "Net revenue", "CDR", "ESR", "RIRCR", "Units sold", "Spoiled units", "Transfer cost"]]
