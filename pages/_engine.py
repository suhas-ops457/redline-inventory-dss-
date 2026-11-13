"""Core OR engine: example inputs, validation, night MILP, day DP, KPIs.

Pure Python (no Streamlit calls) so it can be tested and cached.
Logic follows the paper "Rebalancing and Clearance Pricing of Near-Expiry
Inventory in Online Grocery Retail" (Meituan, NeurIPS 2025 MLxOR).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import poisson

HOURS = [f"{h:02d}:00" for h in range(10, 22)]  # 12 hourly periods, 10:00-21:00
RULE_DISCOUNT = 0.20       # baseline rule: 20% off ...
RULE_START_HOUR = 18       # ... from 18:00 onwards
N_PRICE_LEVELS = 11        # DP price grid size (floor ... retail)
MAX_DP_STOCK = 400         # cap on DP inventory state per MFC-SKU
ALPHA_MAX = -0.1           # elasticities above this are clipped

# --------------------------------------------------------------------------
# Illustrative example inputs (hard-coded, editable in the app)
# --------------------------------------------------------------------------

def example_inputs() -> dict:
    """Return the illustrative example: 3 MFCs x 3 SKUs x 12 hours (inputs only)."""
    skus = pd.DataFrame({
        "SKU": ["Alphonso Mango 1kg", "Tiger Prawns 250g", "Pav 6-pack"],
        "Category": ["Fruit", "Seafood", "Bakery"],
        "Retail price p0": [399.0, 349.0, 40.0],
        "Price floor": [160.0, 150.0, 15.0],
        "Holding penalty c_h": [2.0, 2.5, 0.25],
    })
    stock = pd.DataFrame({
        "MFC": ["MFC-A"] * 3 + ["MFC-B"] * 3 + ["MFC-C"] * 3,
        "SKU": list(skus["SKU"]) * 3,
        "Red-line stock": [60, 40, 70, 30, 18, 35, 8, 4, 10],
        "Baseline sales q̄": [28.0, 16.0, 32.0, 28.0, 15.0, 30.0, 30.0, 18.0, 34.0],
        "Baseline price p̄": [339.0, 299.0, 34.0, 339.0, 299.0, 34.0, 349.0, 309.0, 35.0],
        "Elasticity α": [-2.2, -1.8, -2.6, -2.2, -1.8, -2.6, -2.0, -1.6, -2.4],
    })
    mfcs = ["MFC-A", "MFC-B", "MFC-C"]
    cost = pd.DataFrame([[0.0, 8.0, 12.0], [8.0, 0.0, 7.0], [12.0, 7.0, 0.0]],
                        index=mfcs, columns=mfcs)
    cost.index.name = "From \\ To"
    hourly = pd.DataFrame({
        "Hour": HOURS,
        "Alphonso Mango 1kg": [5, 6, 8, 9, 7, 6, 6, 8, 12, 13, 11, 9],
        "Tiger Prawns 250g": [6, 8, 10, 9, 5, 4, 4, 6, 10, 12, 10, 6],
        "Pav 6-pack": [10, 8, 6, 5, 5, 6, 8, 10, 13, 13, 10, 6],
    })
    return {"skus": skus, "stock": stock, "cost": cost, "hourly": hourly, "history": example_history()}



# Illustrative example sales history: last 14 days, (daily price ₹, units sold) per store-product.
# Made up for illustration only — used to show how price sensitivity can be learned from data.
EXAMPLE_HISTORY = {
    ("MFC-A", "Alphonso Mango 1kg"): ([279, 339, 319, 279, 339, 319, 299, 399, 399, 399, 399, 299, 279, 399], [53, 26, 29, 46, 30, 33, 32, 19, 22, 16, 18, 28, 35, 15]),
    ("MFC-A", "Tiger Prawns 250g"): ([258, 343, 258, 292, 343, 343, 309, 343, 343, 309, 240, 309, 258, 240], [21, 13, 17, 16, 11, 11, 18, 11, 12, 17, 22, 15, 21, 24]),
    ("MFC-A", "Pav 6-pack"): ([28, 30, 32, 34, 40, 32, 36, 40, 40, 36, 32, 36, 34, 30], [72, 50, 31, 32, 23, 36, 31, 21, 23, 34, 34, 28, 30, 45]),
    ("MFC-B", "Alphonso Mango 1kg"): ([320, 358, 358, 377, 377, 358, 282, 377, 358, 263, 263, 377, 377, 301], [35, 18, 23, 22, 27, 28, 40, 21, 24, 62, 46, 21, 23, 36]),
    ("MFC-B", "Tiger Prawns 250g"): ([242, 294, 294, 294, 346, 329, 242, 329, 346, 346, 276, 346, 242, 260], [24, 15, 18, 15, 13, 10, 23, 10, 8, 11, 15, 12, 31, 17]),
    ("MFC-B", "Pav 6-pack"): ([40, 34, 40, 32, 30, 38, 28, 36, 36, 32, 30, 28, 28, 40], [17, 30, 20, 30, 43, 20, 58, 27, 26, 32, 41, 37, 42, 21]),
    ("MFC-C", "Alphonso Mango 1kg"): ([273, 371, 391, 391, 371, 371, 391, 273, 293, 332, 273, 371, 391, 391], [39, 32, 30, 24, 25, 26, 21, 58, 39, 33, 44, 24, 20, 29]),
    ("MFC-C", "Tiger Prawns 250g"): ([349, 323, 323, 251, 287, 349, 270, 270, 287, 287, 349, 287, 349, 306], [18, 16, 16, 20, 18, 19, 20, 19, 21, 25, 12, 20, 13, 14]),
    ("MFC-C", "Pav 6-pack"): ([31, 35, 39, 33, 37, 33, 39, 39, 31, 39, 31, 37, 39, 31], [39, 28, 32, 36, 31, 39, 25, 24, 50, 25, 44, 30, 31, 50]),
}


def example_history() -> pd.DataFrame:
    """Illustrative 14-day sales history as a tidy table (one row per store-product-day)."""
    rows = [{"MFC": m, "SKU": k, "Day": d + 1, "Price": float(pr[d]), "Units sold": int(u[d])}
            for (m, k), (pr, u) in EXAMPLE_HISTORY.items() for d in range(len(pr))]
    return pd.DataFrame(rows)


def estimate_sensitivity(history: pd.DataFrame) -> pd.DataFrame:
    """Learn price sensitivity per store-product from past (price, units) pairs.

    Fits the paper's demand model ln(units) = const + α·ln(price) by least squares
    (a simple stand-in for the paper's causal forest). Also returns the 14-day
    averages the paper uses as 'normal' sales and price. Never raises.
    """
    out = []
    h = history.copy()
    h["Price"] = pd.to_numeric(h["Price"], errors="coerce")
    h["Units sold"] = pd.to_numeric(h["Units sold"], errors="coerce")
    for (m, k), g in h.groupby(["MFC", "SKU"], sort=True):
        g = g.dropna(subset=["Price", "Units sold"])
        g = g[(g["Price"] > 0) & (g["Units sold"] >= 0)]
        use = g[g["Units sold"] > 0]
        row = {"MFC": m, "SKU": k, "Days of data": len(g),
               "Avg sales": g["Units sold"].mean() if len(g) else np.nan,
               "Avg price": g["Price"].mean() if len(g) else np.nan,
               "Learned α": np.nan, "R2": np.nan, "Intercept": np.nan, "Note": ""}
        if len(use) < 5 or use["Price"].nunique() < 3:
            row["Note"] = "Not enough different prices to learn from"
        else:
            x, y = np.log(use["Price"].to_numpy()), np.log(use["Units sold"].to_numpy())
            slope, icpt = np.polyfit(x, y, 1)
            ss = ((y - y.mean()) ** 2).sum()
            row["R2"] = 1 - ((y - (icpt + slope * x)) ** 2).sum() / ss if ss > 0 else 0.0
            row["Intercept"] = icpt
            if slope > ALPHA_MAX:
                row["Note"] = "Sales didn't rise when prices fell; result not usable"
            else:
                row["Learned α"] = float(max(slope, -6.0))
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def validate_inputs(raw: dict) -> tuple[dict, list[tuple[str, str]]]:
    """Clean user-edited inputs. Never raises; returns (clean, messages).

    messages: list of (level, text) with level in {"error","warning","info"}.
    """
    msgs: list[tuple[str, str]] = []
    ex = example_inputs()
    try:
        skus = raw["skus"].copy().dropna(subset=["SKU"]).reset_index(drop=True)
        stock = raw["stock"].copy().dropna(subset=["MFC", "SKU"]).reset_index(drop=True)
        cost = raw["cost"].copy()
        hourly = raw["hourly"].copy()
    except Exception:  # noqa: BLE001
        return ex, [("error", "The data couldn't be read, so the example data was put back.")]

    if skus.empty or stock.empty:
        msgs.append(("error", "A table is empty, so the example data was put back."))
        return ex, msgs

    skus["SKU"] = skus["SKU"].astype(str).str.strip()
    if skus["SKU"].duplicated().any():
        msgs.append(("warning", "Some products appear twice; only the first one is used."))
        skus = skus.drop_duplicates("SKU").reset_index(drop=True)
    skus["Category"] = skus.get("Category", "").fillna("").astype(str)

    p0 = _num(skus["Retail price p0"])
    bad = p0 <= 0
    if bad.any():
        msgs.append(("warning", f"Full price must be above ₹0 for {', '.join(skus.SKU[bad])}; it was set to ₹1."))
        p0[bad] = 1.0
    skus["Retail price p0"] = p0

    fl = _num(skus["Price floor"])
    bad = fl <= 0
    if bad.any():
        msgs.append(("warning", f"Lowest allowed price must be above ₹0 for {', '.join(skus.SKU[bad])}; it was set to 30% of full price."))
        fl[bad] = 0.3 * p0[bad]
    bad = fl > p0
    if bad.any():
        msgs.append(("warning", f"Lowest allowed price is higher than full price for {', '.join(skus.SKU[bad])}; it was set equal to full price (so no discounts)."))
        fl[bad] = p0[bad]
    skus["Price floor"] = fl

    ch = _num(skus["Holding penalty c_h"])
    if (ch < 0).any():
        msgs.append(("warning", "A negative shelf cost was changed to ₹0."))
    skus["Holding penalty c_h"] = ch.clip(lower=0)

    # stock table
    stock["MFC"] = stock["MFC"].astype(str).str.strip()
    stock["SKU"] = stock["SKU"].astype(str).str.strip()
    unknown = ~stock["SKU"].isin(skus["SKU"])
    if unknown.any():
        msgs.append(("warning", f"Stock rows for unknown products were ignored: {', '.join(sorted(set(stock.SKU[unknown])))}."))
        stock = stock[~unknown].reset_index(drop=True)
    if stock.duplicated(["MFC", "SKU"]).any():
        msgs.append(("warning", "A store-product row appears twice; only the first one is used."))
        stock = stock.drop_duplicates(["MFC", "SKU"]).reset_index(drop=True)

    q = _num(stock["Red-line stock"])
    if (q < 0).any():
        msgs.append(("warning", "Negative stock was changed to 0."))
    q = q.clip(lower=0).round()
    if (q > MAX_DP_STOCK).any():
        msgs.append(("warning", f"Stock above {MAX_DP_STOCK} items per store-product is capped at {MAX_DP_STOCK} (prototype limit)."))
        q = q.clip(upper=MAX_DP_STOCK)
    stock["Red-line stock"] = q.astype(int)

    qb = _num(stock["Baseline sales q̄"])
    bad = qb <= 0
    if bad.any():
        msgs.append(("warning", "Normal daily sales must be above 0; empty or zero values were set to 0.1."))
        qb[bad] = 0.1
    stock["Baseline sales q̄"] = qb

    pb = _num(stock["Baseline price p̄"])
    p0_map = skus.set_index("SKU")["Retail price p0"]
    bad = pb <= 0
    if bad.any():
        msgs.append(("warning", "Normal selling price must be above ₹0; it was set to the full price."))
        pb[bad] = stock.loc[bad, "SKU"].map(p0_map)
    stock["Baseline price p̄"] = pb

    a = _num(stock["Elasticity α"], default=-1.5)
    bad = a > ALPHA_MAX
    if bad.any():
        msgs.append(("warning", f"Price sensitivity must be below 0. {int(bad.sum())} value(s) were clipped to {ALPHA_MAX} "
                                "(customers barely react to discounts)."))
        a[bad] = ALPHA_MAX
    stock["Elasticity α"] = a

    # transfer cost matrix
    mfcs = sorted(stock["MFC"].unique())
    try:
        cost.index = cost.index.astype(str)
        cost.columns = cost.columns.astype(str)
        c = cost.reindex(index=mfcs, columns=mfcs)
        if c.isna().any().any():
            msgs.append(("warning", "Some moving costs were empty; they were set to ₹999 (so those moves won't happen)."))
        c = c.apply(pd.to_numeric, errors="coerce").fillna(999.0)
        if (c.values < 0).any():
            msgs.append(("warning", "Negative moving costs were changed to ₹0."))
        arr = c.clip(lower=0).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(arr, 0.0)
        c = pd.DataFrame(arr, index=mfcs, columns=mfcs)
    except Exception:  # noqa: BLE001
        msgs.append(("warning", "The moving-cost table couldn't be read; example costs were used."))
        c = ex["cost"].reindex(index=mfcs, columns=mfcs).fillna(999.0)
    c.index.name = "From \\ To"

    # hourly demand profile (shares)
    if len(hourly) != 12:
        msgs.append(("warning", "The hourly table needs 12 hours; the example pattern was used."))
        hourly = ex["hourly"].copy()
    for s in skus["SKU"]:
        if s not in hourly.columns:
            msgs.append(("info", f"No hourly pattern for '{s}'; every hour is treated as equally busy."))
            hourly[s] = 1.0
        col = _num(hourly[s]).clip(lower=0)
        if col.sum() <= 0:
            msgs.append(("warning", f"The hourly pattern for '{s}' is all zeros; every hour is treated as equally busy."))
            col = pd.Series(1.0, index=hourly.index)
        hourly[s] = col
    hourly["Hour"] = HOURS

    if stock["Red-line stock"].sum() == 0:
        msgs.append(("info", "There is no expiring stock at all, so there is nothing to move or sell. All results will be zero."))

    return {"skus": skus, "stock": stock, "cost": c, "hourly": hourly}, msgs


# --------------------------------------------------------------------------
# Night stage: linearized revenue + MILP rebalancing
# --------------------------------------------------------------------------

def breakpoint_q(qbar, pbar, p0, alpha):
    """q' = q̄·(p0/p̄)^α: stock that sells out at retail price."""
    return qbar * (p0 / pbar) ** alpha


def revenue_linear(q, qbar, pbar, p0, alpha):
    """Paper's piecewise-linear clearance revenue π(q)."""
    qp = breakpoint_q(qbar, pbar, p0, alpha)
    lin = pbar * qbar + pbar * (1 + 1 / alpha) * (q - qbar)
    return np.where(q <= qp, p0 * q, lin)


def revenue_exact(q, qbar, pbar, p0, alpha):
    """Non-linear π(q) = q·p*(q), price capped at retail."""
    q = np.asarray(q, float)
    p = np.minimum(pbar * (np.maximum(q, 1e-9) / qbar) ** (1 / alpha), p0)
    return q * p


def solve_rebalancing_sku(stock_df: pd.DataFrame, cost: pd.DataFrame, p0: float):
    """Solve the night MILP for one SKU across one MFC cluster.

    max Σ_i r_i − Σ_ij c_ij x_ij
    s.t. q_i ≤ q_i^pre − Σ_j x_ij + Σ_j x_ji ;  Σ_j x_ij ≤ q_i^pre ; x ≥ 0 integer
         r_i = π(q_i) (piecewise-linear, regime binary z_i, big-M).
    Returns dict(x=matrix, q=array, status=str, method=str).
    """
    mfcs = list(stock_df["MFC"])
    n = len(mfcs)
    qpre = stock_df["Red-line stock"].to_numpy(float)
    qbar = stock_df["Baseline sales q̄"].to_numpy(float)
    pbar = stock_df["Baseline price p̄"].to_numpy(float)
    alpha = stock_df["Elasticity α"].to_numpy(float)
    C = cost.reindex(index=mfcs, columns=mfcs).to_numpy(float)
    total = qpre.sum()
    if total <= 0 or n < 2:
        return {"x": np.zeros((n, n)), "q": qpre.copy(), "status": "No stock to move", "method": "MILP"}

    qp = breakpoint_q(qbar, pbar, p0, alpha)
    A = pbar * qbar - pbar * (1 + 1 / alpha) * qbar      # intercept of linear branch
    B = pbar * (1 + 1 / alpha)                            # slope of linear branch
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    nx = len(pairs)
    # variable order: x (nx), q (n), r (n), z (n)
    iq, ir, iz = nx, nx + n, nx + 2 * n
    nv = nx + 3 * n
    Mq = total + 1
    Mr = (p0 * total + np.abs(A) + np.abs(B) * total + 1).max() * 2

    rows, lb, ub = [], [], []

    def add(coefs: dict, lo, hi):
        r = np.zeros(nv)
        for k, v in coefs.items():
            r[k] += v
        rows.append(r); lb.append(lo); ub.append(hi)

    for i in range(n):
        out_k = [k for k, (a, b) in enumerate(pairs) if a == i]
        in_k = [k for k, (a, b) in enumerate(pairs) if b == i]
        # q_i + Σ out − Σ in ≤ q_pre
        co = {iq + i: 1.0}
        for k in out_k: co[k] = co.get(k, 0) + 1.0
        for k in in_k: co[k] = co.get(k, 0) - 1.0
        add(co, -np.inf, qpre[i])
        # Σ out ≤ q_pre
        add({k: 1.0 for k in out_k}, -np.inf, qpre[i])
        # r_i ≤ p0 q_i + Mr z_i
        add({ir + i: 1.0, iq + i: -p0, iz + i: -Mr}, -np.inf, 0.0)
        # r_i ≤ A + B q_i + Mr (1 − z_i)
        add({ir + i: 1.0, iq + i: -B[i], iz + i: Mr}, -np.inf, A[i] + Mr)
        # z_i = 0  ->  q_i ≤ q'_i ;  z_i = 1 -> q_i ≥ q'_i
        add({iq + i: 1.0, iz + i: -Mq}, -np.inf, qp[i])
        add({iq + i: 1.0, iz + i: -qp[i]}, 0.0, np.inf)

    c = np.zeros(nv)
    for k, (i, j) in enumerate(pairs):
        c[k] = C[i, j]
    c[ir:ir + n] = -1.0
    lo = np.zeros(nv); hi = np.full(nv, np.inf)
    hi[iq:iq + n] = Mq
    lo[ir:ir + n] = -Mr; hi[ir:ir + n] = Mr
    hi[iz:iz + n] = 1
    integ = np.zeros(nv); integ[:nx] = 1; integ[iz:iz + n] = 1
    try:
        res = milp(c, constraints=LinearConstraint(np.array(rows), lb, ub),
                   bounds=Bounds(lo, hi), integrality=integ,
                   options={"time_limit": 10, "disp": False})
        if res.x is None:
            raise RuntimeError(res.message)
        x = np.zeros((n, n))
        for k, (i, j) in enumerate(pairs):
            x[i, j] = round(res.x[k])
        return {"x": x, "q": res.x[iq:iq + n], "status": "Optimal" if res.status == 0 else res.message,
                "method": "MILP (HiGHS)"}
    except Exception as e:  # noqa: BLE001
        x = _greedy_rebalance(qpre, qbar, pbar, alpha, p0, C)
        return {"x": x, "q": qpre - x.sum(1) + x.sum(0), "status": f"Solver fallback ({e})",
                "method": "Greedy demo"}


def _greedy_rebalance(qpre, qbar, pbar, alpha, p0, C):
    """Greedy fallback: move one unit at a time along the best marginal-gain arc."""
    n = len(qpre)
    q = qpre.copy(); x = np.zeros((n, n))
    for _ in range(int(qpre.sum())):
        best, arc = 0.0, None
        for i in range(n):
            if q[i] < 1 or x[i].sum() >= qpre[i]:
                continue
            for j in range(n):
                if i == j: continue
                g = (revenue_linear(q[j] + 1, qbar[j], pbar[j], p0, alpha[j]) - revenue_linear(q[j], qbar[j], pbar[j], p0, alpha[j])
                     - revenue_linear(q[i], qbar[i], pbar[i], p0, alpha[i]) + revenue_linear(q[i] - 1, qbar[i], pbar[i], p0, alpha[i])
                     - C[i, j])
                if g > best + 1e-9:
                    best, arc = g, (i, j)
        if arc is None:
            break
        i, j = arc; q[i] -= 1; q[j] += 1; x[i, j] += 1
    return x


def run_rebalancing(inp: dict) -> dict:
    """Run the night MILP for every SKU. Returns transfers + stock before/after."""
    skus, stock, cost = inp["skus"], inp["stock"], inp["cost"]
    tr_rows, after_rows, status = [], [], {}
    for _, s in skus.iterrows():
        df = stock[stock.SKU == s.SKU].sort_values("MFC").reset_index(drop=True)
        if df.empty:
            continue
        p0 = s["Retail price p0"]
        sol = solve_rebalancing_sku(df, cost, p0)
        status[s.SKU] = f"{sol['method']}: {sol['status']}"
        mfcs = list(df.MFC)
        x = sol["x"]
        phys = df["Red-line stock"].to_numpy(float) - x.sum(1) + x.sum(0)
        for i, a in enumerate(mfcs):
            for j, b in enumerate(mfcs):
                if x[i, j] > 0:
                    tr_rows.append({"SKU": s.SKU, "From": a, "To": b, "Units": int(x[i, j]),
                                    "Unit cost": cost.loc[a, b], "Cost": x[i, j] * cost.loc[a, b]})
        for i, r in df.iterrows():
            args = (r["Baseline sales q̄"], r["Baseline price p̄"], p0, r["Elasticity α"])
            after_rows.append({
                "MFC": r.MFC, "SKU": s.SKU, "Stock before": int(r["Red-line stock"]),
                "Out": int(x[i].sum()), "In": int(x[:, i].sum()), "Stock after": int(round(phys[i])),
                "Breakpoint q'": float(breakpoint_q(*args)),
                "π_lin before": float(revenue_linear(r["Red-line stock"], *args)),
                "π_lin after": float(revenue_linear(phys[i], *args)),
            })
    transfers = pd.DataFrame(tr_rows, columns=["SKU", "From", "To", "Units", "Unit cost", "Cost"])
    after = pd.DataFrame(after_rows)
    return {"transfers": transfers, "stock": after, "status": status,
            "transfer_cost": float(transfers["Cost"].sum()) if len(transfers) else 0.0}


# --------------------------------------------------------------------------
# Day stage: clearance-pricing MDP solved by backward DP
# --------------------------------------------------------------------------

def _sales_matrix(lam: float, Q: int):
    """Transition matrix M[q, q'] for q' = q − min(q, D), D ~ Poisson(lam)."""
    qs = np.arange(Q + 1)
    pmf = poisson.pmf(qs, lam)
    k = qs[:, None] - qs[None, :]                    # units sold to go q -> j
    M = np.where((k >= 0) & (qs[None, :] >= 1), pmf[np.clip(k, 0, Q)], 0.0)
    tail = np.ones(Q + 1)
    tail[1:] = poisson.sf(qs[1:] - 1, lam)          # P(D ≥ q): sell out
    M[:, 0] = tail
    return M


def hourly_means(row_qbar, row_pbar, alpha, shares, price):
    """Poisson mean λ_t(p) = q̄·share_t·(p/p̄)^α."""
    return row_qbar * shares * (price / row_pbar) ** alpha


def solve_pricing_dp(q0: int, p0: float, floor: float, pbar: float, qbar: float,
                     alpha: float, ch: float, shares: np.ndarray, n_levels: int = N_PRICE_LEVELS) -> dict:
    """Backward induction for the clearance MDP of one MFC-SKU.

    State (prev price index, stock); action price ≤ prev price; reward
    p·E[min(q,D)] − c_h·E[q_{t+1}]; V_T = 0; start price = retail p0.
    Returns expected hourly trajectory under the optimal policy + policy table.
    """
    T = len(shares)
    Q = int(max(q0, 0))
    grid = np.unique(np.round(np.linspace(floor, p0, n_levels), 2)) if p0 > floor else np.array([p0])
    P = len(grid)
    qs = np.arange(Q + 1)
    Ms = [[_sales_matrix(hourly_means(qbar, pbar, alpha, shares[t], grid[a]), Q) for a in range(P)] for t in range(T)]
    V = np.zeros((P, Q + 1))
    policy = np.zeros((T, P, Q + 1), dtype=int)
    for t in range(T - 1, -1, -1):
        Cval = np.empty((P, Q + 1))
        for a in range(P):
            M = Ms[t][a]
            left = M @ qs
            sold = qs - left
            Cval[a] = grid[a] * sold - ch * left + M @ V[a]
        Vn = np.empty_like(V)
        best = np.full(Q + 1, -np.inf); arg = np.zeros(Q + 1, dtype=int)
        for prev in range(P):                       # actions allowed: a ≤ prev
            better = Cval[prev] >= best - 1e-9      # ties -> keep the higher price
            best = np.where(better, Cval[prev], best); arg = np.where(better, prev, arg)
            Vn[prev] = best; policy[t, prev] = arg
        V = Vn
    traj = _forward(q0, grid, Ms, policy, ch, T)
    traj["value"] = float(V[P - 1, Q])
    traj["grid"] = grid
    traj["policy_from_retail"] = policy[:, P - 1, :]
    return traj


def _forward(q0, grid, Ms, policy, ch, T):
    """Propagate the exact state distribution forward under the DP policy."""
    P = len(grid); Q = int(q0)
    qs = np.arange(Q + 1)
    dist = np.zeros((P, Q + 1)); dist[P - 1, Q] = 1.0
    out = {k: np.zeros(T) for k in ["price", "stock_start", "sales", "revenue", "holding", "stock_end", "list_revenue"]}
    for t in range(T):
        new = np.zeros_like(dist)
        out["stock_start"][t] = (dist.sum(0) * qs).sum()
        instock = dist[:, 1:].sum()
        for prev in range(P):
            d = dist[prev]
            if d.sum() <= 0: continue
            acts = policy[t, prev]
            for a in np.unique(acts):
                m = acts == a
                dm = np.where(m, d, 0.0)
                if dm.sum() <= 0: continue
                M = Ms[t][a]
                nd = dm @ M
                sold = dm @ (qs - M @ qs)
                out["price"][t] += dm[1:].sum() * grid[a]   # average price while stock remains
                out["sales"][t] += sold
                out["revenue"][t] += sold * grid[a]
                out["list_revenue"][t] += sold * grid[-1]
                out["holding"][t] += ch * (nd * qs).sum()
                new[a] += nd
        out["price"][t] = out["price"][t] / instock if instock > 1e-12 else (out["price"][t - 1] if t else grid[-1])
        dist = new
        out["stock_end"][t] = (dist.sum(0) * qs).sum()
    return out


def evaluate_schedule(q0: int, prices: np.ndarray, p0: float, pbar: float, qbar: float,
                      alpha: float, ch: float, shares: np.ndarray) -> dict:
    """Expected outcome of a fixed hourly price schedule (rule-based baseline)."""
    T = len(prices); Q = int(max(q0, 0)); qs = np.arange(Q + 1)
    d = np.zeros(Q + 1); d[Q] = 1.0
    out = {k: np.zeros(T) for k in ["price", "stock_start", "sales", "revenue", "holding", "stock_end", "list_revenue"]}
    for t in range(T):
        M = _sales_matrix(hourly_means(qbar, pbar, alpha, shares[t], prices[t]), Q)
        out["stock_start"][t] = (d * qs).sum()
        sold = d @ (qs - M @ qs)
        d = d @ M
        out["price"][t] = prices[t]; out["sales"][t] = sold
        out["revenue"][t] = sold * prices[t]; out["list_revenue"][t] = sold * p0
        out["holding"][t] = ch * (d * qs).sum(); out["stock_end"][t] = (d * qs).sum()
    return out


def rule_prices(p0: float, floor: float) -> np.ndarray:
    """Baseline: retail price, then 20% off from 18:00 (clipped to floor)."""
    hrs = np.array([int(h[:2]) for h in HOURS])
    return np.where(hrs >= RULE_START_HOUR, max(p0 * (1 - RULE_DISCOUNT), floor), p0)


# --------------------------------------------------------------------------
# Full pipeline + KPIs
# --------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _dp_cached(q0, p0, floor, pbar, qbar, alpha, ch, shares):
    """Memoized DP (shared across scenarios / what-if sweeps)."""
    return solve_pricing_dp(q0, p0=p0, floor=floor, pbar=pbar, qbar=qbar, alpha=alpha, ch=ch, shares=np.array(shares))


@lru_cache(maxsize=4096)
def _rule_cached(q0, p0, floor, pbar, qbar, alpha, ch, shares):
    """Memoized rule-based baseline evaluation."""
    return evaluate_schedule(q0, rule_prices(p0, floor), p0=p0, pbar=pbar, qbar=qbar, alpha=alpha, ch=ch,
                             shares=np.array(shares))


def summarize(tr: dict, q0: int) -> dict:
    """Totals for one MFC-SKU trajectory."""
    return {"Stock": q0, "Units sold": tr["sales"].sum(), "Revenue": tr["revenue"].sum(),
            "List value of sales": tr["list_revenue"].sum(), "Holding penalty": tr["holding"].sum(),
            "Unsold (spoiled)": tr["stock_end"][-1]}


def kpis(df: pd.DataFrame, total_stock: float, stock_value: float, transfer_cost: float = 0.0) -> dict:
    """CDR, ESR, RIRCR and revenue from aggregated scenario rows.

    CDR   = 1 − revenue / (units sold × retail)  (sales-weighted average discount)
    ESR   = unsold units / red-line units
    RIRCR = net revenue / red-line stock value at retail
            (net = revenue − holding penalty − transfer cost)
    """
    rev = df["Revenue"].sum(); lv = df["List value of sales"].sum()
    net = rev - df["Holding penalty"].sum() - transfer_cost
    return {"Revenue": rev, "Net revenue": net,
            "CDR": (1 - rev / lv) if lv > 0 else 0.0,
            "ESR": df["Unsold (spoiled)"].sum() / total_stock if total_stock > 0 else 0.0,
            "RIRCR": net / stock_value if stock_value > 0 else 0.0,
            "Units sold": df["Units sold"].sum(), "Spoiled units": df["Unsold (spoiled)"].sum(),
            "Transfer cost": transfer_cost}


def run_pipeline(inp: dict) -> dict:
    """Night MILP + day DP + rule baseline for all MFC-SKUs; returns tables & KPIs.

    Scenarios: 'Rule-based' (no transfers, 20%-off-after-6pm),
               'CP only'    (no transfers, DP pricing),
               'IR + CP'    (MILP transfers, DP pricing).
    """
    skus, stock, hourly = inp["skus"], inp["stock"], inp["hourly"]
    reb = run_rebalancing(inp)
    after = reb["stock"].set_index(["MFC", "SKU"])
    rows, trajs = [], {}
    for _, r in stock.iterrows():
        s = skus.set_index("SKU").loc[r.SKU]
        shares = hourly[r.SKU].to_numpy(float); shares = shares / shares.sum()
        common = dict(p0=s["Retail price p0"], pbar=r["Baseline price p̄"], qbar=r["Baseline sales q̄"],
                      alpha=r["Elasticity α"], ch=s["Holding penalty c_h"], shares=shares)
        q_pre = int(r["Red-line stock"]); q_post = int(after.loc[(r.MFC, r.SKU), "Stock after"])
        key = (float(s["Retail price p0"]), float(s["Price floor"]), float(r["Baseline price p̄"]),
               float(r["Baseline sales q̄"]), float(r["Elasticity α"]), float(s["Holding penalty c_h"]),
               tuple(np.round(shares, 10)))
        rule = _rule_cached(q_pre, *key)
        cp = _dp_cached(q_pre, *key)
        ircp = _dp_cached(q_post, *key)
        trajs[(r.MFC, r.SKU)] = {"Rule-based": rule, "CP only": cp, "IR + CP": ircp}
        for name, tr, q0 in [("Rule-based", rule, q_pre), ("CP only", cp, q_pre), ("IR + CP", ircp, q_post)]:
            rows.append({"Scenario": name, "MFC": r.MFC, "SKU": r.SKU, **summarize(tr, q0)})
    detail = pd.DataFrame(rows)
    total_stock = float(stock["Red-line stock"].sum())
    value = float((stock["Red-line stock"] * stock.SKU.map(skus.set_index("SKU")["Retail price p0"])).sum())
    k = {sc: kpis(detail[detail.Scenario == sc], total_stock, value, reb["transfer_cost"] if sc == "IR + CP" else 0.0)
         for sc in ["Rule-based", "CP only", "IR + CP"]}
    return {"rebalancing": reb, "detail": detail, "trajectories": trajs, "kpis": k,
            "total_stock": total_stock, "stock_value": value}


def scale_inputs(inp: dict, alpha_mult=1.0, stock_mult=1.0, cost_mult=1.0, ch_mult=1.0) -> dict:
    """What-if transform of validated inputs (returns a new dict)."""
    out = {k: v.copy() for k, v in inp.items()}
    out["stock"]["Elasticity α"] = np.minimum(out["stock"]["Elasticity α"] * alpha_mult, ALPHA_MAX)
    out["stock"]["Red-line stock"] = (out["stock"]["Red-line stock"] * stock_mult).round().clip(0, MAX_DP_STOCK).astype(int)
    out["cost"] = out["cost"] * cost_mult
    out["skus"]["Holding penalty c_h"] = out["skus"]["Holding penalty c_h"] * ch_mult
    return out


PAPER_BENCHMARK = {
    "cp_city_a": {"CDR": -9.46, "ESR": -22.85, "RIRCR": +38.89},
    "ir_cdr_cities": {"A": 8.04, "B": 11.83, "C": 10.89, "D": 11.36, "E": 6.63},
    "category_esr": {"Fruits": 18.80, "Vegetables": 10.77, "Seafood": 57.00, "Meat & Poultry": 11.52, "Bakery": 22.12},
    "nationwide": {"Spoilage": "−20%", "Red-line revenue": "+14%", "Profit": "≈50M CNY", "Food-waste saving": "≈18M CNY"},
}
