"""OptiBuy analyst dashboard.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optibuy.config import CONTRACTS, PolicyConfig  # noqa: E402
from optibuy.data import sample_future  # noqa: E402
from optibuy.forecast import backtest_forecaster  # noqa: E402
from optibuy.planner import Engine, run_policies, simulate  # noqa: E402

st.set_page_config(page_title="OptiBuy", layout="wide")
CR = 1e7  # INR crore


@st.cache_resource
def base_engine() -> Engine:
    return Engine.build(seed=7)


def engine_for(cfg: PolicyConfig) -> Engine:
    b = base_engine()
    return Engine(world=b.world, cfg=cfg, forecaster=b.forecaster, cancel=b.cancel)


@st.cache_data(show_spinner="Running seed, rebalance and markdown for one season...")
def one_season(cfg_items: tuple, season_seed: int):
    cfg = replace(PolicyConfig(), **dict(cfg_items))
    rng = np.random.default_rng(season_seed)
    future = sample_future(rng)
    res, stt = run_policies(engine_for(cfg), future, rng)
    u = stt.units.copy()
    u["true_demand"] = future["true_demand"].to_numpy() * u["price_mult"] ** -u["elasticity"]
    u["disrupted"] = future["true_disruption"].to_numpy() < 1
    return res, u, stt.log, stt.seed_result.cvar_loss, stt.seed_result.expected_profit


@st.cache_data(show_spinner="Simulating seasons (about 10 s each)...")
def many_seasons(cfg_items: tuple, n: int):
    cfg = replace(PolicyConfig(), **dict(cfg_items))
    return simulate(engine_for(cfg), n_sims=n, seed=11)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("OptiBuy")
st.sidebar.caption("Pre-buy optimisation for hotel and flight inventory")
budget = st.sidebar.slider("Pre-buy budget (INR crore)", 2.0, 40.0, 20.0, 1.0)
risk = st.sidebar.slider("Risk cap: worst-5% loss allowed (% of budget)", 0.0, 20.0, 4.0, 0.5)
dest = st.sidebar.slider("Max share of budget per destination", 0.15, 1.0, 0.35, 0.05)
cover = st.sidebar.slider("Peak-week cover target (share of P50)", 0.0, 1.0, 0.45, 0.05)
season = st.sidebar.number_input("Season sample (seed)", 1, 999, 2)
cfg_items = (("budget", budget * CR), ("risk_cap_pct", risk / 100), ("max_dest_share", dest),
             ("peak_cover_ratio", cover), ("n_scenarios", 80))

res, u, log, cvar, exp_profit = one_season(cfg_items, int(season))

tab_plan, tab_fc, tab_roll, tab_kra, tab_how = st.tabs(
    ["Buy plan", "Forecast and risk", "Rolling plan", "KRA scorecard", "How it works"])

# ------------------------------------------------------------------ buy plan
with tab_plan:
    spend = u["spend"].sum()
    c = st.columns(5)
    c[0].metric("Seed spend", f"₹{spend / CR:.1f} cr", f"of ₹{budget:.0f} cr")
    c[1].metric("Expected profit (plan)", f"₹{exp_profit / CR:.1f} cr")
    c[2].metric("Expected ROI on capital", f"{exp_profit / max(spend, 1):.0%}")
    c[3].metric("Worst-5% profit", f"₹{-cvar / CR:.1f} cr")
    c[4].metric("Units pre-bought", f"{u['buy_qty'].sum():,.0f}")
    left, right = st.columns(2)
    by_dest = u.groupby(["destination", "contract"], as_index=False)["spend"].sum()
    left.plotly_chart(px.bar(by_dest, x="destination", y="spend", color="contract", title="Spend by destination and contract"),
                      use_container_width=True)
    heat = u.pivot_table(index="product_id", columns=u["week_start"].dt.strftime("%d %b"), values="buy_qty", sort=False)
    right.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues", title="Units to pre-buy (product x arrival week)"),
                       use_container_width=True)
    show = u[["product_id", "week_start", "contract", "p50_adj", "price_mult", "sell_price", "unit_cost", "buy_qty",
              "spend", "exp_sell_through", "p_stockout"]].rename(columns={"p50_adj": "demand_p50"})
    show["week_start"] = show["week_start"].dt.date
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ forecast
with tab_fc:
    pid = st.selectbox("Product", u["product_id"].unique())
    hist = base_engine().forecaster.history
    h = hist[hist["product_id"] == pid].tail(52)
    f = u[u["product_id"] == pid]
    fig = go.Figure()
    fig.add_scatter(x=h["week_start"], y=h["bookings"], name="Recorded bookings", mode="lines")
    fig.add_scatter(x=h["week_start"], y=h["demand_est"], name="Unconstrained demand", mode="lines", line=dict(dash="dot"))
    fig.add_scatter(x=f["week_start"], y=f["p90"], mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=f["week_start"], y=f["p10"], mode="lines", fill="tonexty", line=dict(width=0), name="P10 to P90")
    fig.add_scatter(x=f["week_start"], y=f["p50"], mode="lines", name="P50 forecast")
    fig.add_bar(x=f["week_start"], y=f["buy_qty"], name="Seed buy", opacity=0.5)
    fig.update_layout(title=f"{pid}: last 52 weeks, then the planning horizon (forecast and seed buy)", height=420)
    st.plotly_chart(fig, use_container_width=True)
    bt = backtest_forecaster(base_engine().world.history)
    c = st.columns(3)
    c[0].metric("Forecast WAPE (26-week holdout)", f"{bt['wape_model']:.1%}")
    c[1].metric("Seasonal-naive WAPE", f"{bt['wape_seasonal_naive']:.1%}")
    c[2].metric("P10 to P90 coverage", f"{bt['p10_p90_coverage']:.0%}")
    curves = base_engine().cancel.curves()
    st.plotly_chart(px.line(curves, x="days_before", y=["p_cancel_nonref", "p_cancel_ref"], facet_col="ptype",
                            title="Cancellation probability by booking lead time"), use_container_width=True)

# ------------------------------------------------------------------ rolling
with tab_roll:
    st.markdown("One sampled season played through the three stages. Forecast error shrinks as booking pace arrives.")
    c = st.columns(3)
    c[0].metric("WAPE at seed (T-90)", f"{log['wape_t90']:.1%}")
    c[1].metric("WAPE at rebalance (T-35)", f"{log['wape_t35']:.1%}")
    c[2].metric("WAPE at markdown (T-10)", f"{log['wape_t10']:.1%}")
    c = st.columns(4)
    c[0].metric("Units topped up", f"{u['topup_qty'].sum():,.0f}")
    c[1].metric("Units released to suppliers", f"{u['release_qty'].sum():,.0f}")
    c[2].metric("Cohorts marked down", f"{(u['markdown_mult'] < 1).sum()}")
    c[3].metric("Disrupted cohorts this season", f"{u['disrupted'].sum()}")
    pid2 = st.selectbox("Product ", u["product_id"].unique(), key="roll")
    f = u[u["product_id"] == pid2]
    fig = go.Figure()
    fig.add_bar(x=f["week_start"], y=f["buy_qty"], name="Seed buy (T-90)")
    fig.add_bar(x=f["week_start"], y=f["final_qty"], name="After rebalance (T-35)")
    fig.add_scatter(x=f["week_start"], y=f["true_demand"], name="Actual demand", mode="lines+markers")
    fig.add_scatter(x=f["week_start"], y=f["p50_adj"], name="Seed forecast P50", mode="lines", line=dict(dash="dot"))
    fig.update_layout(barmode="group", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(res).T.round(3), use_container_width=True)

# ------------------------------------------------------------------ KRA
with tab_kra:
    n = st.slider("Seasons to simulate", 3, 30, 8)
    if st.button("Run simulation"):
        df, summ = many_seasons(cfg_items, n)
        s = summ.copy()
        view = pd.DataFrame({
            "Profit (₹ cr)": s["profit"] / CR,
            "Profit, worst 5% (₹ cr)": s["profit_p5"] / CR,
            "ROI on pre-buy capital": s["roi"],
            "Sell-through": s["sell_through"],
            "Demand served (fill rate)": s["fill_rate"],
            "Sold-out cohorts": s["soldout_share"],
            "Spoilage (₹ cr)": s["spoilage_value"] / CR,
            "Price realisation": s["price_realisation"],
        })
        st.dataframe(view.style.format("{:.2f}"), use_container_width=True)
        st.plotly_chart(px.box(df, x="policy", y="profit", title="Season profit across simulations (INR)"),
                        use_container_width=True)

with tab_how:
    st.markdown(Path(__file__).resolve().parents[1].joinpath("docs", "approach.md").read_text())
    st.json({k: v.__dict__ for k, v in CONTRACTS.items()})
