"""Probabilistic demand forecasting.

Pipeline
1. **Unconstrain** history: weeks that sold out recorded fewer bookings than
   true demand. We estimate lost demand from searches x the product's normal
   look-to-book conversion.
2. **Features** known ~90 days before arrival: seasonality (week of year),
   festival uplift, destination peak flag, same week last year (lag-52) and the
   early search signal (look-to-book leading indicator).
3. **Quantile gradient boosting** (P10 / P50 / P90) on log(demand / product level),
   pooled across all products so thin products borrow strength.
4. **Scenario sampler**: quantiles -> lognormal per unit, with a shared
   destination factor so Goa hotels and Goa flights move together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import CATALOG

QUANTILES = (0.1, 0.5, 0.9)
Z90 = 1.2815515655446004
FEATURES = [
    "product_code", "is_flight", "dest_code", "woy_sin", "woy_cos", "event_uplift",
    "is_peak", "lag52_ratio", "search_signal",
]
_PRODUCTS = {p.product_id: p for p in CATALOG}
_PRODUCT_CODES = {p.product_id: i for i, p in enumerate(CATALOG)}
_DEST_CODES = {d: i for i, d in enumerate(sorted({p.destination for p in CATALOG}))}


def unconstrain(history: pd.DataFrame) -> pd.DataFrame:
    """Add ``demand_est``: bookings, lifted for sold-out weeks using searches."""
    h = history.copy()
    open_weeks = h[~h["sold_out"]]
    conv = (open_weeks["bookings"] / open_weeks["searches"]).groupby(open_weeks["product_id"]).median()
    implied = h["searches"] * h["product_id"].map(conv)
    h["demand_est"] = np.where(h["sold_out"], np.maximum(h["bookings"], implied), h["bookings"])
    return h


def _is_peak(pid: str, week: int) -> int:
    for pw in _PRODUCTS[pid].peak_weeks:
        d = abs(week - pw)
        if min(d, 52 - d) <= 1:
            return 1
    return 0


@dataclass
class DemandForecaster:
    max_iter: int = 300
    learning_rate: float = 0.05
    models: dict = field(default_factory=dict)
    level: pd.Series | None = None  # product demand level (last 52 weeks)
    search_ratio: pd.Series | None = None  # early searches per unit of demand
    history: pd.DataFrame | None = None

    # ---------------------------------------------------------------- features
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["product_code"] = df["product_id"].map(_PRODUCT_CODES)
        X["is_flight"] = (df["ptype"] == "flight").astype(int)
        X["dest_code"] = df["destination"].map(_DEST_CODES)
        angle = 2 * np.pi * df["iso_week"] / 52.0
        X["woy_sin"], X["woy_cos"] = np.sin(angle), np.cos(angle)
        X["event_uplift"] = df["event_uplift"]
        X["is_peak"] = [_is_peak(p, w) for p, w in zip(df["product_id"], df["iso_week"])]
        lvl = df["product_id"].map(self.level)
        lag = self._lag52(df)
        X["lag52_ratio"] = np.log(np.maximum(lag, 1.0) / lvl)
        expected_searches = df["product_id"].map(self.search_ratio) * lvl
        X["search_signal"] = np.log(np.maximum(df["early_searches"], 1.0) / expected_searches)
        return X[FEATURES]

    def _lag52(self, df: pd.DataFrame) -> np.ndarray:
        h = self.history.set_index(["product_id", "week_start"])["demand_est"]
        keys = list(zip(df["product_id"], df["week_start"] - pd.Timedelta(weeks=52)))
        vals = h.reindex(keys).to_numpy()
        fallback = df["product_id"].map(self.level).to_numpy()
        return np.where(np.isnan(vals), fallback, vals)

    # -------------------------------------------------------------------- fit
    def fit(self, history: pd.DataFrame, until: pd.Timestamp | None = None) -> "DemandForecaster":
        h = unconstrain(history)
        self.history = h
        train = h if until is None else h[h["week_start"] < until]
        recent = train[train["week_start"] >= train["week_start"].max() - pd.Timedelta(weeks=52)]
        self.level = recent.groupby("product_id")["demand_est"].mean()
        self.search_ratio = (train["early_searches"] / train["demand_est"]).groupby(train["product_id"]).median()
        # rows with a full year of history behind them
        train = train[train["week_start"] >= h["week_start"].min() + pd.Timedelta(weeks=52)]
        X = self._features(train)
        y = np.log(np.maximum(train["demand_est"], 1.0) / train["product_id"].map(self.level))
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=self.max_iter, learning_rate=self.learning_rate,
                max_leaf_nodes=15, min_samples_leaf=15, categorical_features=[0, 2], random_state=0,
            )
            self.models[q] = m.fit(X, y)
        return self

    # --------------------------------------------------------------- predict
    def predict(self, units: pd.DataFrame) -> pd.DataFrame:
        """Return P10/P50/P90 demand per unit (product x arrival week)."""
        X = self._features(units)
        lvl = units["product_id"].map(self.level).to_numpy()
        out = units[["product_id", "ptype", "destination", "week_start", "iso_week"]].copy()
        preds = np.column_stack([self.models[q].predict(X) for q in QUANTILES])
        preds.sort(axis=1)  # guard against quantile crossing
        for j, q in enumerate(QUANTILES):
            out[f"p{int(q * 100)}"] = lvl * np.exp(preds[:, j])
        out["level"] = lvl
        return out.reset_index(drop=True)

    def seasonal_naive(self, units: pd.DataFrame) -> np.ndarray:
        """Baseline: same week last year, scaled by recent growth."""
        return self._lag52(units)


def sample_scenarios(fc: pd.DataFrame, n: int, rng: np.random.Generator, rho: float = 0.4, inflate: float = 1.2,
                     disruptions: bool = True,
                     price_mult: np.ndarray | None = None, elasticity: np.ndarray | None = None) -> np.ndarray:
    """Draw ``n`` correlated demand scenarios, shape (units, n).

    Each unit is lognormal, matched to its P50 and the P10-P90 spread. A shared
    standard-normal factor per destination gives correlation ``rho``. Optionally
    overlays rare destination disruptions (same process the history shows), which
    the point forecast cannot see but the risk model must price.
    """
    mu = np.log(np.maximum(fc["p50"].to_numpy(), 0.5))
    sigma = (np.log(np.maximum(fc["p90"].to_numpy(), 1.0)) - np.log(np.maximum(fc["p10"].to_numpy(), 0.5))) / (2 * Z90)
    sigma = np.clip(sigma * inflate, 0.05, 1.5)  # inflate: holdout P10-P90 coverage runs ~70%, not 80%
    dests = fc["destination"].to_numpy()
    uniq = {d: i for i, d in enumerate(np.unique(dests))}
    common = rng.standard_normal((len(uniq), n))
    z = np.sqrt(rho) * common[[uniq[d] for d in dests]] + np.sqrt(1 - rho) * rng.standard_normal((len(fc), n))
    d = np.exp(mu[:, None] + sigma[:, None] * z)
    if disruptions:
        from .data import disruption_multiplier

        wk = ((fc["week_start"] - fc["week_start"].min()).dt.days // 7).to_numpy()
        d = d * disruption_multiplier(dests, wk, rng, n)
    if price_mult is not None:
        d = d * (np.asarray(price_mult)[:, None] ** (-np.asarray(elasticity)[:, None]))
    return d


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(actual - forecast).sum() / np.abs(actual).sum())


def backtest_forecaster(history: pd.DataFrame, holdout_weeks: int = 26) -> dict:
    """Fit on all but the last ``holdout_weeks`` and score on them against true demand."""
    cutoff = history["week_start"].max() - pd.Timedelta(weeks=holdout_weeks - 1)
    model = DemandForecaster().fit(history, until=cutoff)
    test = history[history["week_start"] >= cutoff].reset_index(drop=True)
    fc = model.predict(test)
    naive = model.seasonal_naive(test)
    actual = test["true_demand"].to_numpy()
    recorded = test["bookings"].to_numpy()
    inside = (actual >= fc["p10"]) & (actual <= fc["p90"])
    return {
        "holdout_weeks": holdout_weeks,
        "wape_model": wape(actual, fc["p50"]),
        "wape_seasonal_naive": wape(actual, naive),
        "censoring_gap_recorded_vs_true": wape(actual, recorded),
        "p10_p90_coverage": float(inside.mean()),
    }
