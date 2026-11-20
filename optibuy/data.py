"""Synthetic data generator.

TBO's real booking, search and supplier data is not public, so the prototype
generates a world with the same structure and the same difficulties:

* weekly demand per product (hotel room-nights / flight seats) with trend,
  annual seasonality, destination peaks and Indian festival spikes;
* correlated shocks: a destination-level shock hits its hotels and flights together;
* **censored history**: past capacity was limited, so peak weeks sold out and the
  recorded bookings understate true demand (the forecaster must unconstrain them);
* a **leading indicator**: search volume observed ~90 days before arrival
  (look-to-book), noisy but informative;
* **booking-level data** with lead times and cancellations whose probability
  rises with lead time and refundability;
* **spot availability** that dries up in peak weeks (the reason to pre-buy at all).

Columns that start with ``true_`` are ground truth used only by the simulator
to score decisions. The planner never reads them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CATALOG, EVENT_WEEKS, Product

PLAN_DATE = pd.Timestamp("2026-09-21")  # Monday; the "today" of the demo
HISTORY_WEEKS = 156
HORIZON_OFFSET_WEEKS = 13  # first arrival week is ~90 days after plan date
HORIZON_WEEKS = 12

DEST_PHASE = {  # month of natural high season, drives a smooth annual cycle
    "Goa": 12, "Dubai": 1, "Bangkok": 12, "Bali": 7, "Manali": 5,
    "Jaipur": 11, "Singapore": 6, "Maldives": 2,
}
ANNUAL_GROWTH = 0.08
SIGMA_DEST = 0.12
SIGMA_IDIO = 0.20
# Disruptions (travel advisory, cyclone, airline grounding): per destination they
# start on average DISRUPTION_RATE times a year, last DISRUPTION_WEEKS weeks and cut
# demand to DISRUPTION_MULT. Rare, correlated and invisible 90 days out: the tail
# risk that makes non-refundable blocks dangerous.
DISRUPTION_RATE = 0.5
DISRUPTION_WEEKS = 3
DISRUPTION_MULT = 0.45
SEARCH_NOISE = 0.18
LEAD_SCALE = {"hotel": 22.0, "flight": 30.0}
LEAD_SHAPE = 1.6
REFUNDABLE_SHARE = {"hotel": 0.60, "flight": 0.35}
# True cancellation model: logit p = b0 + b1*log1p(lead) + b2*refundable + b3*is_flight
TRUE_CANCEL_COEF = (-3.6, 0.50, 0.80, -0.30)


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([p.__dict__ for p in CATALOG])


def _week_distance(a: np.ndarray, b: int) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 52 - d)


def season_index(p: Product, weeks: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Multiplicative seasonality for product ``p`` (1.0 = average week)."""
    phase = DEST_PHASE[p.destination]
    annual = 1.0 + 0.15 * np.cos(2 * np.pi * (months - phase) / 12.0)
    peak = np.zeros_like(weeks, dtype=float)
    for pw in p.peak_weeks:
        peak = np.maximum(peak, 0.55 * np.exp(-0.5 * _week_distance(weeks, pw) ** 2))
    event = np.array([EVENT_WEEKS.get(int(w), 0.0) for w in weeks])
    return annual * (1.0 + peak) * (1.0 + event)


def expected_demand(p: Product, week_starts: pd.DatetimeIndex) -> np.ndarray:
    weeks = week_starts.isocalendar().week.to_numpy().astype(int)
    months = week_starts.month.to_numpy()
    years = (week_starts - PLAN_DATE).days.to_numpy() / 365.0
    trend = (1.0 + ANNUAL_GROWTH) ** years
    return p.base_weekly_demand * trend * season_index(p, weeks, months)


def spot_capacity(p: Product, mean_demand: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Units the supplier will still sell on request; scarce in peak weeks."""
    pressure = np.maximum(0.0, mean_demand / p.base_weekly_demand - 1.0)
    level = 1.0 * p.base_weekly_demand * np.exp(-1.5 * pressure)
    return level * rng.uniform(0.6, 1.2, size=np.shape(mean_demand))


def _draw_demand(mean: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Lognormal demand with a shared destination shock per week."""
    dest_shock = {}
    out = np.empty(len(mean))
    for i, (dest, wk) in enumerate(zip(mean["destination"], mean["week_start"])):
        key = (dest, wk)
        if key not in dest_shock:
            dest_shock[key] = rng.normal(0, SIGMA_DEST)
        eps = dest_shock[key] + rng.normal(0, SIGMA_IDIO)
        out[i] = mean["mean_demand"].iat[i] * np.exp(eps - 0.5 * (SIGMA_DEST**2 + SIGMA_IDIO**2))
    return np.maximum(0.0, out)


def disruption_multiplier(destinations: np.ndarray, week_idx: np.ndarray, rng: np.random.Generator,
                          n: int = 1) -> np.ndarray:
    """Demand multiplier from disruptions, shape (units, n). ``week_idx`` = 0.. horizon."""
    n_weeks = int(week_idx.max()) + 1
    out = np.ones((len(destinations), n))
    for d in np.unique(destinations):
        rows = np.where(destinations == d)[0]
        span = n_weeks + DISRUPTION_WEEKS - 1
        starts = rng.poisson(DISRUPTION_RATE * span / 52.0, size=n)
        for k in np.where(starts > 0)[0]:
            for _ in range(starts[k]):
                s0 = rng.integers(-(DISRUPTION_WEEKS - 1), n_weeks)
                hit = rows[(week_idx[rows] >= s0) & (week_idx[rows] < s0 + DISRUPTION_WEEKS)]
                out[hit, k] = DISRUPTION_MULT
    return out


def _week_index(week_start: pd.Series) -> np.ndarray:
    return ((week_start - week_start.min()).dt.days // 7).to_numpy()


def _unit_frame(week_starts: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for p in CATALOG:
        mu = expected_demand(p, week_starts)
        for ws, m in zip(week_starts, mu):
            rows.append(
                dict(product_id=p.product_id, ptype=p.ptype, destination=p.destination,
                     week_start=ws, mean_demand=m)
            )
    df = pd.DataFrame(rows)
    iso = df["week_start"].dt.isocalendar()
    df["iso_week"] = iso.week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["event_uplift"] = df["iso_week"].map(lambda w: EVENT_WEEKS.get(int(w), 0.0))
    return df


def generate_history(rng: np.random.Generator) -> pd.DataFrame:
    """Three years of weekly history, as TBO would have recorded it (censored)."""
    weeks = pd.date_range(end=PLAN_DATE - pd.Timedelta(days=7), periods=HISTORY_WEEKS, freq="7D")
    df = _unit_frame(weeks)
    true_d = _draw_demand(df, rng)
    df["early_searches"] = np.round(true_d * 4.0 * np.exp(rng.normal(0, SEARCH_NOISE, len(df))))
    true_d = true_d * disruption_multiplier(df["destination"].to_numpy(), _week_index(df["week_start"]), rng)[:, 0]
    prod = {p.product_id: p for p in CATALOG}
    base = df["product_id"].map(lambda k: prod[k].base_weekly_demand).to_numpy()
    capacity = base * rng.uniform(1.10, 1.50, size=len(df))  # what TBO could sell back then
    df["capacity"] = np.round(capacity)
    df["bookings"] = np.round(np.minimum(true_d, capacity))
    df["sold_out"] = true_d > capacity
    df["searches"] = np.round(true_d * 6.0 * np.exp(rng.normal(0, 0.10, len(df))))
    df["true_demand"] = true_d
    return df.drop(columns=["mean_demand"])


def generate_bookings(rng: np.random.Generator, n: int = 30000) -> pd.DataFrame:
    """Booking-level sample (lead time, refundability, cancellation outcome)."""
    ptype = rng.choice(["hotel", "flight"], size=n, p=[0.62, 0.38])
    scale = np.where(ptype == "hotel", LEAD_SCALE["hotel"], LEAD_SCALE["flight"])
    lead = np.round(rng.gamma(LEAD_SHAPE, scale)) + 1
    refundable = rng.random(n) < np.where(ptype == "hotel", REFUNDABLE_SHARE["hotel"], REFUNDABLE_SHARE["flight"])
    b0, b1, b2, b3 = TRUE_CANCEL_COEF
    logit = b0 + b1 * np.log1p(lead) + b2 * refundable + b3 * (ptype == "flight")
    cancelled = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(dict(ptype=ptype, lead_days=lead.astype(int), refundable=refundable, cancelled=cancelled))


def horizon_weeks() -> pd.DatetimeIndex:
    start = PLAN_DATE + pd.Timedelta(weeks=HORIZON_OFFSET_WEEKS)
    return pd.date_range(start=start, periods=HORIZON_WEEKS, freq="7D")


def true_booked_fraction(ptype: str, days_before: float) -> float:
    """Share of final bookings already made ``days_before`` arrival (true curve)."""
    from scipy.stats import gamma

    return float(gamma.sf(days_before - 1, LEAD_SHAPE, scale=LEAD_SCALE[ptype]))


def sample_future(rng: np.random.Generator) -> pd.DataFrame:
    """One possible future for the planning horizon.

    Returns the features the planner can see at the plan date (early searches)
    plus ``true_*`` columns the simulator uses to play the season out.
    """
    df = _unit_frame(horizon_weeks())
    true_d = _draw_demand(df, rng)
    df["early_searches"] = np.round(true_d * 4.0 * np.exp(rng.normal(0, SEARCH_NOISE, len(df))))
    df["true_disruption"] = disruption_multiplier(df["destination"].to_numpy(), _week_index(df["week_start"]), rng)[:, 0]
    true_d = true_d * df["true_disruption"].to_numpy()
    prod = {p.product_id: p for p in CATALOG}
    caps = np.empty(len(df))
    for pid, idx in df.groupby("product_id").groups.items():
        caps[idx] = spot_capacity(prod[pid], df.loc[idx, "mean_demand"].to_numpy(), rng)
    df["true_demand"] = true_d
    df["true_spot_cap"] = caps
    df["true_pace_noise"] = np.exp(rng.normal(0, 0.10, len(df)))
    return df.drop(columns=["mean_demand"])


@dataclass
class World:
    catalog: pd.DataFrame
    history: pd.DataFrame
    bookings: pd.DataFrame
    plan_date: pd.Timestamp = PLAN_DATE


def make_world(seed: int = 7) -> World:
    rng = np.random.default_rng(seed)
    return World(catalog=catalog_frame(), history=generate_history(rng), bookings=generate_bookings(rng))
