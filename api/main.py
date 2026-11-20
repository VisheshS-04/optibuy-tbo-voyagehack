"""OptiBuy REST API.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from optibuy.config import CONTRACTS, PolicyConfig
from optibuy.data import sample_future
from optibuy.forecast import backtest_forecaster
from optibuy.planner import Engine, simulate

app = FastAPI(title="OptiBuy API", version="0.1.0",
              description="Inventory pre-buy optimisation for hotels and flight blocks (TBO VoyageHack 4.0 prototype).")


@lru_cache(maxsize=1)
def engine() -> Engine:
    return Engine.build(seed=7)


class PlanRequest(BaseModel):
    budget: float = Field(2.0e8, gt=0, description="Capital available for pre-buys, INR")
    risk_cap_pct: float = Field(0.04, ge=0, le=1, description="Allowed worst-5% loss, share of budget")
    max_dest_share: float = Field(0.35, gt=0, le=1)
    peak_cover_ratio: float = Field(0.45, ge=0, le=1)
    n_scenarios: int = Field(80, ge=20, le=300)
    seed: int = 1


class SimRequest(PlanRequest):
    n_sims: int = Field(5, ge=1, le=50)


def _cfg(req: PlanRequest) -> PolicyConfig:
    return replace(PolicyConfig(), budget=req.budget, risk_cap_pct=req.risk_cap_pct,
                   max_dest_share=req.max_dest_share, peak_cover_ratio=req.peak_cover_ratio,
                   n_scenarios=req.n_scenarios)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/catalog")
def catalog() -> dict:
    e = engine()
    return {"products": e.world.catalog.to_dict(orient="records"),
            "contracts": {k: v.__dict__ for k, v in CONTRACTS.items()}}


@app.get("/forecast/backtest")
def forecast_backtest(holdout_weeks: int = 26) -> dict:
    return backtest_forecaster(engine().world.history, holdout_weeks)


@app.get("/forecast")
def forecast(seed: int = 1) -> list[dict]:
    """P10/P50/P90 demand for every product x arrival week in the planning horizon."""
    fut = sample_future(np.random.default_rng(seed))
    fc = engine().forecaster.predict(fut)
    fc["week_start"] = fc["week_start"].dt.date.astype(str)
    return fc.round(1).to_dict(orient="records")


@app.get("/cancellation/curves")
def cancellation_curves() -> dict:
    e = engine()
    return {"coefficients": e.cancel.coefficients(), "curves": e.cancel.curves().round(4).to_dict(orient="records")}


@app.post("/plan")
def plan(req: PlanRequest) -> dict:
    """Stage 1 seed buy plan: what to pre-buy, how much, at what cost and sell price."""
    base = engine()
    e = Engine(world=base.world, cfg=_cfg(req), forecaster=base.forecaster, cancel=base.cancel)
    rng = np.random.default_rng(req.seed)
    try:
        st = e.seed(sample_future(rng), rng)
    except RuntimeError as exc:  # pragma: no cover
        raise HTTPException(500, str(exc)) from exc
    u = st.units
    cols = ["product_id", "name", "ptype", "destination", "week_start", "contract", "p10_adj", "p50_adj", "p90_adj",
            "price_mult", "sell_price", "unit_cost", "buy_qty", "spend", "exp_sell_through", "p_stockout", "cover_floor"]
    rows = u[cols].rename(columns={"p10_adj": "demand_p10", "p50_adj": "demand_p50", "p90_adj": "demand_p90"}).copy()
    rows["week_start"] = rows["week_start"].dt.date.astype(str)
    r = st.seed_result
    return {
        "summary": {
            "budget": req.budget, "spend": float(u["spend"].sum()),
            "expected_profit": r.expected_profit, "cvar95_loss": r.cvar_loss,
            "expected_roi": r.expected_profit / max(float(u["spend"].sum()), 1.0),
            "risk_cap_breach": r.slack_risk, "destination_cap_breach": float(r.slack_dest.sum()),
            "peak_cover_shortfall_units": float(r.slack_cover.sum()), "solve_seconds": r.solve_seconds,
        },
        "plan": rows.round(3).fillna(0).to_dict(orient="records"),
    }


@app.post("/simulate")
def simulate_endpoint(req: SimRequest) -> dict:
    """Play the full three-stage policy and baselines over sampled seasons; return KRAs."""
    base = engine()
    e = Engine(world=base.world, cfg=_cfg(req), forecaster=base.forecaster, cancel=base.cancel)
    df, summary = simulate(e, n_sims=req.n_sims, seed=req.seed)
    acc = df.groupby("sim")[["wape_t90", "wape_t35", "wape_t10"]].first().mean().round(4).to_dict()
    return {"kpis": summary.round(4).reset_index().to_dict(orient="records"), "forecast_wape": acc}
