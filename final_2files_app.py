from __future__ import annotations
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from io import StringIO
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================================================
# CONSTANTS & DEFAULTS (Team ROBO-A | ROB-3 | AP Calculus)
# =========================================================
DEFAULT_TEAM_ID = "ROBO-A"
PATHWAY_ID = "ROB-3"
COURSE = "AP Calculus"
LEVEL = "Advanced"

DEFAULT_VLINES = {
    "components_COGS": "30.00",
    "assembly_labor": "20.00",
    "packaging": "10.00",
    "warranty_consumables": "5.00",
}

DEFAULT_FLINES = {
    "workspace": "250.00",
    "software": "100.00",
    "equipment_maintenance": "150.00",
    "admin": "100.00",
}

BUILT_IN_FIXTURE_CSV = """month,requested,capacity_override,ad_cost
2027-01,40,30,0
2027-02,44,30,100
2027-03,36,22,0
"""


def d(val: Any) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def create_initial_state(
    team_id: str = DEFAULT_TEAM_ID,
    selected_level: str = LEVEL,
    random_seed: int = 2027,
) -> Dict[str, Any]:
    return {
        "team_id": team_id,
        "course": COURSE,
        "pathway_id": PATHWAY_ID,
        "snapshot_id": "SNAP-000",
        "state_version": 1,
        "simulation_date": "2026-12-31",
        "base_period": "monthly",
        "working_days": 20,
        "currency": "USD",
        "level_by_skill": {
            "Constrained Profit Optimization (Derivatives)": selected_level,
            "Accumulated Net Change (Integrals)": selected_level,
        },
        "assumptions": [
            "Fictional baseline teaching assumptions from Career Venture Lab Handbook (ROB-3).",
            "Components are expensed when consumed on attempted builds; cash outflow occurs upon restock.",
            "Non-component variable costs ($35.00/attempted unit) are paid in cash in the period incurred.",
            "Baseline borrowing, interest, refunds, depreciation, and sales tax are disabled ($0.00).",
        ],
        "sources": [
            "Career Venture Lab Student Handbook (November 2026), ROB-3 Pathway Card & ROB Package."
        ],
        "offerings": [
            {
                "offering_id": "OFF-01",
                "name": "Introductory robot kit",
                "unit": "kit",
                "price": 120.00,
                "demand_rule": "q(p) = max(0, 80 - (40/120)*p)",
                "capacity_rule": "floor(staff_hours / hours_per_unit)",
            }
        ],
        "funding": [
            {
                "date": "2026-12-31",
                "type": "owner_contribution",
                "amount": 12000.00,
                "notes": "Initial owner equity contribution",
            }
        ],
        "cost_lines": {
            "variable_per_attempted_unit": {k: float(d(v)) for k, v in DEFAULT_VLINES.items()},
            "fixed_monthly": {k: float(d(v)) for k, v in DEFAULT_FLINES.items()},
        },
        "parameters": {
            "base_price": 120.00,
            "D0": 40,
            "staff_hours": 90.0,
            "hours_per_unit": 3.0,
            "component_unit_cost": 30.00,
            "defect_rate": 0.0,
            "random_seed": random_seed,
        },
        "balances": {
            "owner_equity": 12000.00,
            "equipment_assets": 4000.00,
            "pre_opening_expense": 0.00,
            "opening_cash": 5000.00,
            "cash": 5000.00,
            "opening_stock_units": 100,
            "stock_units": 100,
            "inventory_value": 3000.00,
            "retained_earnings": 0.00,
            "debt_principal": 0.00,
        },
        "capacity": {
            "staff_hours": 90.0,
            "hours_per_unit": 3.0,
            "normal_monthly_capacity": 30,
        },
        "ledger": [],
        "executed_run_ids": [],
        "campaigns": [],
        "decisions": [],
    }


def compute_normal_capacity(staff_hours: float, hours_per_unit: float) -> int:
    if hours_per_unit <= 0:
        raise ValueError("hours_per_unit must be > 0")
    if staff_hours < 0:
        raise ValueError("staff_hours must be >= 0")
    return int(math.floor(staff_hours / hours_per_unit))


def step_simulation(
    state: Dict[str, Any],
    *,
    run_id: str,
    sim_date: str,
    requested: int,
    price: float | None = None,
    restock_units: int = 0,
    ad_cost: float = 0.0,
    defect_rate: float | None = None,
    staff_hours: float | None = None,
    hours_per_unit: float | None = None,
    component_unit_cost: float | None = None,
    refund_amount: float = 0.0,
    loan_cash: float = 0.0,
    loan_principal_repayment: float = 0.0,
    interest: float = 0.0,
    additional_capex: float = 0.0,
    new_owner_cash: float = 0.0,
    fixture_mode: bool = False,
    capacity_override: int | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not run_id or not run_id.strip():
        raise ValueError("A unique non-empty run_id is required.")
    if run_id in state["executed_run_ids"]:
        raise ValueError(f"Duplicate run_id '{run_id}' rejected: transaction already recorded.")
    if requested < 0 or restock_units < 0:
        raise ValueError("requested and restock_units must be >= 0.")

    new_state = deepcopy(state)
    params = new_state["parameters"]
    bal = new_state["balances"]

    unit_price = d(price if price is not None else params["base_price"])
    comp_cost = d(component_unit_cost if component_unit_cost is not None else params["component_unit_cost"])
    d_rate = float(defect_rate if defect_rate is not None else params["defect_rate"])
    if not (0.0 <= d_rate <= 1.0):
        raise ValueError("defect_rate must be in [0, 1].")

    s_hours = float(staff_hours if staff_hours is not None else params["staff_hours"])
    h_unit = float(hours_per_unit if hours_per_unit is not None else params["hours_per_unit"])

    if fixture_mode:
        if capacity_override is None or capacity_override < 0:
            raise ValueError("Fixture mode requires a non-negative capacity_override.")
        capacity = int(capacity_override)
    else:
        capacity = compute_normal_capacity(s_hours, h_unit)

    vlines = new_state["cost_lines"]["variable_per_attempted_unit"]
    flines = new_state["cost_lines"]["fixed_monthly"]

    vlines_dec = {k: d(v) for k, v in vlines.items()}
    vlines_dec["components_COGS"] = comp_cost
    noncomp_var_per_unit = sum(
        v for k, v in vlines_dec.items() if k != "components_COGS"
    )
    fixed_expense = sum(d(v) for v in flines.values())

    opening_stock = int(bal["stock_units"])
    available_stock = opening_stock + int(restock_units)
    attempted = min(int(requested), capacity, available_stock)
    failed = int(math.floor(attempted * d_rate))
    sold = attempted - failed
    unserved_demand = max(0, int(requested) - attempted)

    gross_revenue = unit_price * Decimal(sold)
    refund_dec = d(refund_amount)
    net_revenue = gross_revenue - refund_dec

    component_consumed_expense = comp_cost * Decimal(attempted)
    noncomp_variable_expense = noncomp_var_per_unit * Decimal(attempted)
    variable_expense = component_consumed_expense + noncomp_variable_expense

    marketing_dec = d(ad_cost)
    interest_dec = d(interest)
    principal_dec = d(loan_principal_repayment)
    capex_dec = d(additional_capex)
    new_owner_dec = d(new_owner_cash)
    loan_cash_dec = d(loan_cash)

    total_operating_expense = variable_expense + fixed_expense + marketing_dec
    operating_profit = net_revenue - total_operating_expense
    net_profit = operating_profit - interest_dec

    restock_cash_outflow = comp_cost * Decimal(restock_units)
    opening_cash = d(bal["cash"])
    ending_cash = (
        opening_cash
        + net_revenue
        - noncomp_variable_expense
        - restock_cash_outflow
        - fixed_expense
        - marketing_dec
        - interest_dec
        - principal_dec
        - capex_dec
        + new_owner_dec
        + loan_cash_dec
    )

    ending_stock = available_stock - attempted
    ending_inventory_value = comp_cost * Decimal(ending_stock)

    new_version = int(new_state["state_version"]) + 1
    new_state["state_version"] = new_version
    new_state["snapshot_id"] = f"SNAP-{new_version - 1:03d}"
    new_state["simulation_date"] = sim_date
    new_state["capacity"]["staff_hours"] = s_hours
    new_state["capacity"]["hours_per_unit"] = h_unit
    new_state["capacity"]["normal_monthly_capacity"] = compute_normal_capacity(s_hours, h_unit)

    bal["cash"] = float(ending_cash)
    bal["stock_units"] = ending_stock
    bal["inventory_value"] = float(ending_inventory_value)
    bal["retained_earnings"] = float(d(bal["retained_earnings"]) + net_profit)
    bal["equipment_assets"] = float(d(bal["equipment_assets"]) + capex_dec)
    bal["owner_equity"] = float(d(bal["owner_equity"]) + new_owner_dec)
    bal["debt_principal"] = float(d(bal["debt_principal"]) + loan_cash_dec - principal_dec)

    record = {
        "transaction_id": f"TXN-{run_id}",
        "run_id": run_id,
        "date": sim_date,
        "mode": "fixture_override" if fixture_mode else "normal",
        "state_version": new_version,
        "requested": int(requested),
        "capacity_used": capacity,
        "attempted": attempted,
        "failed": failed,
        "sold": sold,
        "unserved_demand": unserved_demand,
        "unit_price": float(unit_price),
        "revenue": float(net_revenue),
        "variable_expense": float(variable_expense),
        "fixed_expense": float(fixed_expense),
        "marketing_expense": float(marketing_dec),
        "total_operating_expense": float(total_operating_expense),
        "operating_profit": float(operating_profit),
        "net_profit": float(net_profit),
        "restock_units": int(restock_units),
        "restock_cash_outflow": float(restock_cash_outflow),
        "closing_cash": float(ending_cash),
        "opening_stock": opening_stock,
        "closing_stock": ending_stock,
        "closing_inventory_value": float(ending_inventory_value),
    }

    new_state["ledger"].append(record)
    new_state["executed_run_ids"].append(run_id)
    return new_state, record


def replay_fixture_rows(
    base_state: Dict[str, Any], rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    state = deepcopy(base_state)
    results = []
    for idx, row in enumerate(rows, start=1):
        month_str = str(row["month"]).strip()
        sim_date = f"{month_str}-28" if len(month_str) == 7 else month_str
        state, rec = step_simulation(
            state,
            run_id=f"FIXTURE-{month_str}-{idx}",
            sim_date=sim_date,
            requested=int(row["requested"]),
            ad_cost=float(row.get("ad_cost", 0.0)),
            fixture_mode=True,
            capacity_override=int(row["capacity_override"]),
        )
        results.append(rec)
    return state, results


def export_snapshot_json(state: Dict[str, Any]) -> str:
    return json.dumps(state, indent=2)


def import_snapshot_json(
    raw_json: str, current_state: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        loaded = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {exc}") from exc

    required_keys = [
        "team_id",
        "course",
        "pathway_id",
        "snapshot_id",
        "state_version",
        "balances",
        "parameters",
        "cost_lines",
        "ledger",
    ]
    for key in required_keys:
        if key not in loaded:
            raise ValueError(f"Missing required snapshot field: '{key}'.")

    if loaded["team_id"] != current_state["team_id"]:
        raise ValueError(
            f"Team ID mismatch: expected '{current_state['team_id']}', got '{loaded['team_id']}'."
        )

    if loaded["pathway_id"] != current_state["pathway_id"]:
        raise ValueError(
            f"Pathway ID mismatch: expected '{current_state['pathway_id']}', got '{loaded['pathway_id']}'."
        )

    if int(loaded["state_version"]) < int(current_state["state_version"]):
        raise ValueError(
            f"Stale state_version rejected: imported v{loaded['state_version']} is older than active v{current_state['state_version']}."
        )

    if float(loaded["balances"]["cash"]) < 0 or int(loaded["balances"]["stock_units"]) < 0:
        raise ValueError("Negative cash or stock_units detected in imported snapshot.")

    return loaded


def calc_adapter_metrics(
    D0: float = 40.0,
    p0: float = 120.0,
    v: float = 65.0,
    F: float = 600.0,
    capacity: float = 30.0,
) -> Dict[str, Any]:
    a = 2.0 * D0
    b = D0 / p0
    p_max = 2.0 * p0
    p_corner = (a - capacity) / b
    p_unconstrained_opt = (a + b * v) / (2.0 * b)

    def q_dem(p: float) -> float:
        return max(0.0, a - b * p)

    def s_constrained(p: float) -> float:
        return min(capacity, q_dem(p))

    def profit_unconstrained(p: float) -> float:
        return (p - v) * q_dem(p) - F

    def profit_constrained(p: float) -> float:
        return (p - v) * s_constrained(p) - F

    candidates = [0.0, p_max]
    if 0.0 <= p_corner <= p_max:
        candidates.append(p_corner)
    if p_corner <= p_unconstrained_opt <= p_max:
        candidates.append(p_unconstrained_opt)

    best_p = max(candidates, key=profit_constrained)
    best_profit = profit_constrained(best_p)
    best_units_cont = s_constrained(best_p)
    best_units_int = int(math.floor(best_units_cont + 1e-9))
    best_profit_int = (best_p - v) * best_units_int - F

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_corner": p_corner,
        "p_unconstrained_opt": p_unconstrained_opt,
        "profit_at_unconstrained_opt_unconstrained": profit_unconstrained(p_unconstrained_opt),
        "profit_at_unconstrained_opt_constrained": profit_constrained(p_unconstrained_opt),
        "constrained_opt_price": best_p,
        "constrained_opt_profit_continuous": best_profit,
        "constrained_opt_units_continuous": best_units_cont,
        "constrained_opt_units_integer": best_units_int,
        "constrained_opt_profit_integer": best_profit_int,
    }


def integrate_rates(
    r0: float = 100.0,
    r_slope: float = 20.0,
    c0: float = 60.0,
    c_slope: float = 5.0,
    t0: float = 0.0,
    t1: float = 10.0,
) -> Dict[str, float]:
    rev = r0 * (t1 - t0) + 0.5 * r_slope * (t1**2 - t0**2)
    cost = c0 * (t1 - t0) + 0.5 * c_slope * (t1**2 - t0**2)
    net = rev - cost
    return {
        "accumulated_revenue": round(rev, 2),
        "accumulated_cost": round(cost, 2),
        "accumulated_net_profit": round(net, 2),
    }


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================
st.set_page_config(
    page_title="Career Venture Lab | ROB-3 (ROBO-A)",
    layout="wide",
)

if "biz_state" not in st.session_state:
    st.session_state.biz_state = create_initial_state(team_id=DEFAULT_TEAM_ID, selected_level=LEVEL)

state = st.session_state.biz_state
bal = state["balances"]
params = state["parameters"]

with st.sidebar:
    st.header("Team Configuration")
    new_tid = st.text_input("Team ID", value=state["team_id"])
    if new_tid.strip() and new_tid.strip() != state["team_id"]:
        state["team_id"] = new_tid.strip()
        st.rerun()

st.title("Robotics Kit & Repair Studio — Career Venture Lab")
col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
col_b1.metric("Team ID", state["team_id"])
col_b2.metric("Pathway", state["pathway_id"])
col_b3.metric("Course & Level", f"{state['course']} ({state['level_by_skill']['Constrained Profit Optimization (Derivatives)']})")
col_b4.metric("Snapshot / Version", f"{state['snapshot_id']} (v{state['state_version']})")
col_b5.metric("Sim Date", state["simulation_date"])

st.caption("Stage 1 Active: Deterministic Ledger, Inputs & Funding, AP Calculus Math Panel, Built-In Fixture Tests, and Save/Load JSON. Stage 2/3 modules are marked as placeholders.")

tabs = st.tabs([
    "1. Overview",
    "2. Inputs & Funding",
    "3. Math Skills & Levels (AP Calculus)",
    "4. Operations & Ledger",
    "5. Save / Load & Fixture Tests",
    "6. Stage 2 & 3 Placeholders",
])

with tabs[0]:
    st.subheader("Business Operating Snapshot")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spendable Cash", f"${bal['cash']:,.2f}")
    m2.metric("Inventory Units", f"{bal['stock_units']} kits", f"${bal['inventory_value']:,.2f} value")
    m3.metric("Normal Capacity", f"{state['capacity']['normal_monthly_capacity']} kits/mo")
    m4.metric("Equipment Assets", f"${bal['equipment_assets']:,.2f}")
    m5.metric("Retained Earnings", f"${bal['retained_earnings']:,.2f}")

    st.markdown("### Persistent Math Panel Summary")
    st.info(
        "**Course:** AP Calculus | **Level:** Advanced | **Skills:** Constrained Profit Optimization (Derivatives) & Accumulated Net Change (Integrals)\n\n"
        "**Business Question:** What kit price maximizes monthly operating profit without exceeding our 30-kit/month assembly capacity, and how does net profit accumulate over a multi-day campaign?\n\n"
        "**Core Equations:** $q(p) = \max(0, 80 - \frac{1}{3}p)$, $s(p) = \min(\text{capacity}, q(p))$, $P(p) = (p - v)s(p) - F$ on $0 \le p \le 240$."
    )

    if state["ledger"]:
        st.markdown("### Recorded Ledger History")
        st.dataframe(pd.DataFrame(state["ledger"]), use_container_width=True)
    else:
        st.warning("No simulation rounds recorded yet. Use Tab 4 (Operations & Ledger) to advance a period or Tab 5 to run the fixture acceptance test.")

with tabs[1]:
    st.subheader("Baseline Parameters, Cost Lines & Startup Balances")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Editable Baseline Parameters")
        new_p0 = st.number_input("Base Price p0 ($/kit)", min_value=1.0, value=float(params["base_price"]), step=5.0)
        new_d0 = st.number_input("Baseline Demand D0 (kits/mo)", min_value=0, value=int(params["D0"]), step=1)
        new_hours = st.number_input("Staff Hours / Month", min_value=0.0, value=float(params["staff_hours"]), step=3.0)
        new_hpu = st.number_input("Hours per Kit", min_value=0.1, value=float(params["hours_per_unit"]), step=0.5)
        new_comp_cost = st.number_input("Component Unit Cost ($/set)", min_value=0.0, value=float(params["component_unit_cost"]), step=1.0)
        new_defect = st.slider("Defect Rate (failed build fraction)", min_value=0.0, max_value=1.0, value=float(params["defect_rate"]), step=0.05)

        if st.button("Save Baseline Parameter Updates"):
            params["base_price"] = float(new_p0)
            params["D0"] = int(new_d0)
            params["staff_hours"] = float(new_hours)
            params["hours_per_unit"] = float(new_hpu)
            params["component_unit_cost"] = float(new_comp_cost)
            params["defect_rate"] = float(new_defect)
            state["cost_lines"]["variable_per_attempted_unit"]["components_COGS"] = float(new_comp_cost)
            state["capacity"]["staff_hours"] = float(new_hours)
            state["capacity"]["hours_per_unit"] = float(new_hpu)
            state["capacity"]["normal_monthly_capacity"] = int(np.floor(new_hours / new_hpu))
            st.success("Baseline parameters updated (ledger history unchanged until a period is executed).")

    with c2:
        st.markdown("#### Startup Opening Balances (Fictional Teaching Baseline)")
        st.table(pd.DataFrame([
            {"Account": "Owner Contribution (Equity)", "Opening Amount": "$12,000.00", "Accounting Rule": "Equity, not sales revenue"},
            {"Account": "Equipment Assets", "Opening Amount": "$4,000.00", "Accounting Rule": "Capital asset, not expensed again"},
            {"Account": "Opening Component Inventory (100 sets)", "Opening Amount": "$3,000.00", "Accounting Rule": "Expensed when consumed; cash paid on restock"},
            {"Account": "Post-Startup Spendable Cash", "Opening Amount": "$5,000.00", "Accounting Rule": "Opening cash balance"},
            {"Account": "Pre-Opening Expense", "Opening Amount": "$0.00", "Accounting Rule": "Retained earnings start at $0.00"},
        ]))

        st.markdown("#### Unit Variable & Monthly Fixed Cost Lines")
        vlines = state["cost_lines"]["variable_per_attempted_unit"]
        flines = state["cost_lines"]["fixed_monthly"]
        st.write(f"**Variable Cost per Attempted Kit:** ${sum(vlines.values()):.2f} ({vlines})")
        st.write(f"**Monthly Fixed Costs:** ${sum(flines.values()):.2f} ({flines})")

with tabs[2]:
    st.subheader("AP Calculus Math Panel — Constrained Optimization & Accumulation")
    active_level = st.selectbox(
        "Challenge Level (Teacher-Approved)",
        ["Foundation", "Applied", "Advanced"],
        index=["Foundation", "Applied", "Advanced"].index(
            state["level_by_skill"]["Constrained Profit Optimization (Derivatives)"]
        ),
    )
    if active_level != state["level_by_skill"]["Constrained Profit Optimization (Derivatives)"]:
        state["level_by_skill"]["Constrained Profit Optimization (Derivatives)"] = active_level
        state["level_by_skill"]["Accumulated Net Change (Integrals)"] = active_level

    st.markdown("#### Interactive Sensitivity Controls (Projection Only — Does Not Alter Ledger)")
    sc1, sc2, sc3 = st.columns(3)
    comp_shock_pct = sc1.slider("Component Cost Shock (%)", min_value=-20, max_value=50, value=0, step=5)
    sim_cap = sc2.slider("Capacity Cap (kits/mo)", min_value=5, max_value=60, value=int(state["capacity"]["normal_monthly_capacity"]), step=1)
    test_price = sc3.slider("Inspect Price Point p ($)", min_value=0, max_value=int(2 * params["base_price"]), value=150, step=5)

    base_v = sum(state["cost_lines"]["variable_per_attempted_unit"].values())
    comp_part = state["cost_lines"]["variable_per_attempted_unit"]["components_COGS"]
    adj_v = base_v + comp_part * (comp_shock_pct / 100.0)
    adj_F = sum(state["cost_lines"]["fixed_monthly"].values())

    metrics = calc_adapter_metrics(
        D0=params["D0"],
        p0=params["base_price"],
        v=adj_v,
        F=adj_F,
        capacity=sim_cap,
    )

    p_grid = np.linspace(0, metrics["p_max"], 241)
    a, b = metrics["a"], metrics["b"]
    q_vals = np.maximum(0.0, a - b * p_grid)
    s_vals = np.minimum(sim_cap, q_vals) if active_level != "Foundation" else q_vals
    prof_vals = (p_grid - adj_v) * s_vals - adj_F

    if active_level == "Foundation":
        dprof_vals = a + b * adj_v - 2.0 * b * p_grid
    else:
        dprof_vals = np.where(
            p_grid < metrics["p_corner"],
            float(sim_cap),
            a + b * adj_v - 2.0 * b * p_grid,
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Monthly Profit P(p) [$] — {active_level} Mode",
            "Marginal Profit P'(p) [$/dollar of price] (Undefined at Capacity Corner)",
        ),
    )
    fig.add_trace(go.Scatter(x=p_grid, y=prof_vals, name="Profit P(p)", line=dict(color="#1f77b4", width=3)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    if active_level != "Foundation":
        fig.add_vline(x=metrics["p_corner"], line_dash="dot", line_color="#d95f02", annotation_text=f"Capacity Corner p=${metrics['p_corner']:.2f}", row=1, col=1)

    fig.add_trace(go.Scatter(x=p_grid, y=dprof_vals, name="Derivative P'(p)", line=dict(color="#2ca02c", width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    if active_level != "Foundation":
        fig.add_trace(
            go.Scatter(
                x=[metrics["p_corner"]],
                y=[sim_cap],
                mode="markers",
                marker=dict(size=10, color="white", line=dict(color="red", width=2)),
                name="Corner (P' undefined)",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=550, xaxis2_title="Kit Price p (USD)", yaxis_title="Profit ($)", yaxis2_title="P'(p) (kits)")
    st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unconstrained Critical Price", f"${metrics['p_unconstrained_opt']:.2f}")
    k2.metric("Capacity Corner Price", f"${metrics['p_corner']:.2f}")
    k3.metric("Constrained Optimal Price", f"${metrics['constrained_opt_price']:.2f}")
    k4.metric("Max Constrained Profit (Cont. / Int.)", f"${metrics['constrained_opt_profit_continuous']:.2f} / ${metrics['constrained_opt_profit_integer']:.2f}")

    q_at_p = max(0.0, a - b * test_price)
    s_at_p = min(sim_cap, q_at_p) if active_level != "Foundation" else q_at_p
    int_s = int(np.floor(s_at_p))
    st.markdown(f"**Selected Price Inspection ($p = \${test_price}$):** Continuous demand $q({test_price}) = {q_at_p:.2f}$ kits; Constrained sales $s({test_price}) = {s_at_p:.2f}$ kits (Integer rounded: **{int_s} kits**); Continuous profit = **${(test_price - adj_v)*s_at_p - adj_F:,.2f}**; Integer profit = **${(test_price - adj_v)*int_s - adj_F:,.2f}**.")

    if active_level == "Advanced":
        st.markdown("---")
        st.markdown("### Advanced Integration Module: Accumulated Revenue, Cost & Profit")
        st.write("Models continuous daily revenue rate $r(t) = r_0 + r_1 t$ and cost rate $c(t) = c_0 + c_1 t$ ($/day) over campaign interval $[t_0, t_1]$ without differentiating `floor()`.")
        ic1, ic2, ic3, ic4, ic5 = st.columns(5)
        r0 = ic1.number_input("r0 ($/day)", value=100.0)
        r1 = ic2.number_input("r_slope ($/day²)", value=20.0)
        c0 = ic3.number_input("c0 ($/day)", value=60.0)
        c1 = ic4.number_input("c_slope ($/day²)", value=5.0)
        t1 = ic5.number_input("Horizon t1 (days)", min_value=1.0, value=10.0)

        integ = integrate_rates(r0=r0, r_slope=r1, c0=c0, c_slope=c1, t0=0.0, t1=t1)
        im1, im2, im3 = st.columns(3)
        im1.metric("∫ r(t) dt (Accumulated Revenue)", f"${integ['accumulated_revenue']:,.2f}")
        im2.metric("∫ c(t) dt (Accumulated Cost)", f"${integ['accumulated_cost']:,.2f}")
        im3.metric("∫ [r(t) - c(t)] dt (Net Profit)", f"${integ['accumulated_net_profit']:,.2f}")

    st.info("**Interpretation Prompts:** What changed when you adjusted component cost or capacity? Why can an unconstrained critical point be infeasible when capacity is 30 kits/month? Note: Continuous curves are mathematical models; the ledger records integer units only.")

with tabs[3]:
    st.subheader("Execute Approved Simulation Period (Normal Mode)")
    st.write("Each approved run increments `state_version` and records one immutable transaction in the dated ledger.")

    with st.form("run_period_form"):
        fc1, fc2, fc3 = st.columns(3)
        run_id_in = fc1.text_input("Unique Run ID", value=f"RUN-{state['state_version']:03d}")
        sim_date_in = fc2.text_input("Simulation Date (YYYY-MM-DD)", value="2027-01-31")
        req_in = fc3.number_input("Requested Kits (Demand)", min_value=0, value=int(params["D0"]), step=1)

        fc4, fc5, fc6 = st.columns(3)
        price_in = fc4.number_input("Selling Price ($/kit)", min_value=0.0, value=float(params["base_price"]), step=5.0)
        restock_in = fc5.number_input("Restock Component Sets (units)", min_value=0, value=0, step=10)
        ad_in = fc6.number_input("Approved Marketing Expense ($)", min_value=0.0, value=0.0, step=25.0)

        submitted = st.form_submit_button("Approve & Advance Simulation One Period")
        if submitted:
            try:
                updated, rec = step_simulation(
                    state,
                    run_id=run_id_in.strip(),
                    sim_date=sim_date_in.strip(),
                    requested=int(req_in),
                    price=float(price_in),
                    restock_units=int(restock_in),
                    ad_cost=float(ad_in),
                    fixture_mode=False,
                )
                st.session_state.biz_state = updated
                st.success(
                    f"Recorded {rec['transaction_id']}: Sold {rec['sold']} kits | Revenue ${rec['revenue']:,.2f} | "
                    f"Operating Profit ${rec['operating_profit']:,.2f} | Closing Cash ${rec['closing_cash']:,.2f} | Closing Stock {rec['closing_stock']}"
                )
                st.rerun()
            except ValueError as err:
                st.error(str(err))

with tabs[4]:
    st.subheader("Durable Save/Load JSON & Stage 1 Acceptance Fixtures")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Export & Import Snapshot JSON")
        snap_str = export_snapshot_json(state)
        safe_tid = state["team_id"].replace("[", "").replace("]", "").replace(" ", "_")
        st.download_button(
            label=f"Download Snapshot JSON ({state['snapshot_id']}_v{state['state_version']}.json)",
            data=snap_str,
            file_name=f"{safe_tid}_{state['snapshot_id']}_v{state['state_version']}.json",
            mime="application/json",
        )

        uploaded_json = st.file_uploader("Restore / Import Snapshot JSON", type=["json"])
        if uploaded_json is not None:
            raw_text = uploaded_json.read().decode("utf-8")
            if st.button("Validate & Import JSON"):
                try:
                    imported = import_snapshot_json(raw_text, state)
                    st.session_state.biz_state = imported
                    st.success(f"Snapshot {imported['snapshot_id']} (v{imported['state_version']}) restored successfully.")
                    st.rerun()
                except ValueError as err:
                    st.error(f"Import Rejected: {err}")

    with col_s2:
        st.markdown("#### Replay Handbook 3-Month Acceptance Fixture")
        st.warning("Fixture-test mode resets to opening balances and applies explicit `capacity_override` values strictly to verify ledger math.")
        fixture_df = pd.read_csv(StringIO(BUILT_IN_FIXTURE_CSV))
        st.dataframe(fixture_df, use_container_width=True)

        if st.button("Reset to Opening State & Replay 3-Month Fixture"):
            fresh = create_initial_state(team_id=state["team_id"], selected_level=LEVEL)
            rows = fixture_df.to_dict(orient="records")
            replayed_state, recs = replay_fixture_rows(fresh, rows)
            st.session_state.biz_state = replayed_state
            st.success("Fixture replay complete! Verify: 2027-01 Cash $6,950 | 2027-02 Cash $8,800 | 2027-03 Cash $10,070.")
            st.rerun()

with tabs[5]:
    st.subheader("Upcoming Modules (Not Implemented in Stage 1)")
    st.info(
        "**Stage 2 Placeholders:** AI Departmental Assistants (Finance, Operations, Market/Marketing, Independent Reviewer), "
        "Full Report Center (Income Statement, Cash-Flow, Budget Variance, Balance Sheet, PDF/CSV exports), Costed Marketing Studio, and Two-Report Decision Memos.\n\n"
        "**Stage 3 Placeholders:** Cross-Course Connections Panel (2 non-math subjects), Frozen-Snapshot Practice Assessments, and Optional Fictional Storefront."
    )
