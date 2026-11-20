"""Three-stage rolling planner and season simulator.

Stage 1  SEED      ~90 days out. Price each unit, then buy a risk-aware block at the
                   deepest discount (scenario LP with CVaR, budget, soft limits).
Stage 2  REBALANCE ~35 days out (before release deadlines). Read booking pace net of
                   expected cancellations, update the forecast, then top up hot units
                   and release cold ones (same LP, with holdings fixed).
Stage 3  MARKDOWN  ~10 days out. For stock that pace says will not sell, pick a
                   price cut that maximises expected recovery; leftovers are salvaged.

``simulate`` plays many possible seasons against ground truth and compares the
policy with two baselines: buying everything on request (spot only) and a rule
of thumb (pre-buy 80% of last year's volume as hard blocks).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cancellation import CancellationModel
from .config import CATALOG, CONTRACTS, SPOT_COST_RATIO, PolicyConfig
from .data import World, make_world, sample_future, spot_capacity, true_booked_fraction
from .forecast import Z90, DemandForecaster, sample_scenarios, wape
from .optimizer import LPInput, LPResult, cvar_loss, solve_plan

_PROD = {p.product_id: p for p in CATALOG}


# --------------------------------------------------------------------------- units
def unit_table(fc: pd.DataFrame) -> pd.DataFrame:
    """Attach commercial and contract attributes to each forecast unit."""
    u = fc.copy()
    u["name"] = u["product_id"].map(lambda k: _PROD[k].name)
    u["contract"] = u["product_id"].map(lambda k: _PROD[k].contract)
    u["market_rate"] = u["product_id"].map(lambda k: _PROD[k].market_rate).astype(float)
    u["elasticity"] = u["product_id"].map(lambda k: _PROD[k].elasticity).astype(float)
    u["base_demand"] = u["product_id"].map(lambda k: _PROD[k].base_weekly_demand).astype(float)
    for attr in ("cost_ratio", "release_fraction", "release_value", "salvage_value", "topup_cost_ratio", "release_days"):
        u[attr] = u["contract"].map(lambda k, a=attr: getattr(CONTRACTS[k], a)).astype(float)
    dests = sorted(u["destination"].unique())
    u["dest_idx"] = u["destination"].map({d: i for i, d in enumerate(dests)})
    return u


def spot_scenarios(u: pd.DataFrame, K: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((len(u), K))
    for i, row in enumerate(u.itertuples()):
        out[i] = spot_capacity(_PROD[row.product_id], np.full(K, row.p50), rng)
    return out


def _unit_profit(A, D, spot, p, c0, m, a, r1, r2):
    """Vectorised scenario profit of holding A units (no purchase cost)."""
    s = np.minimum(A, D)
    y = np.minimum(spot, D - s)
    u = A - s
    rec = np.minimum(c0 * r1 * u, c0 * ((r1 * a + r2 * (1 - a)) * A - r2 * s))
    return p * s + m * y + rec


# --------------------------------------------------------------------------- pricing
def recommend_prices(u: pd.DataFrame, D: np.ndarray, spot: np.ndarray, grid) -> tuple[np.ndarray, pd.DataFrame]:
    """Pick a sell-price multiplier per unit by evaluating each grid price with its
    best newsvendor quantity (demand responds with the product's elasticity)."""
    mk, e = u["market_rate"].to_numpy()[:, None], u["elasticity"].to_numpy()[:, None]
    c = (u["cost_ratio"].to_numpy() * u["market_rate"].to_numpy())[:, None]
    a, r1, r2 = (u[k].to_numpy()[:, None] for k in ("release_fraction", "release_value", "salvage_value"))
    best, table = None, {}
    for mult in grid:
        Dm = D * mult ** (-e)
        p = mk * mult
        m = np.maximum(p - SPOT_COST_RATIO * mk, 0)
        qs = np.quantile(Dm, np.linspace(0, 0.95, 20), axis=1).T  # (n, 20) candidate quantities
        vals = np.stack([(_unit_profit(qs[:, [j]], Dm, spot, p, c, m, a, r1, r2) - c * qs[:, [j]]).mean(1)
                         for j in range(qs.shape[1])], axis=1)
        table[mult] = vals.max(1)
    t = pd.DataFrame(table)
    mults = np.array(grid)[t.to_numpy().argmax(1)]
    return mults, t


# --------------------------------------------------------------------------- engine
@dataclass
class Engine:
    world: World
    cfg: PolicyConfig
    forecaster: DemandForecaster
    cancel: CancellationModel

    @classmethod
    def build(cls, seed: int = 7, cfg: PolicyConfig | None = None) -> "Engine":
        w = make_world(seed)
        return cls(world=w, cfg=cfg or PolicyConfig(),
                   forecaster=DemandForecaster().fit(w.history),
                   cancel=CancellationModel().fit(w.bookings))

    # ---------------------------------------------------------------- stage 1
    def seed(self, future: pd.DataFrame, rng: np.random.Generator, budget: float | None = None) -> "SeasonState":
        cfg = self.cfg
        budget = cfg.budget if budget is None else budget
        fc = self.forecaster.predict(future)
        u = unit_table(fc)
        K = cfg.n_scenarios
        D0 = sample_scenarios(u, K, rng)
        spot = spot_scenarios(u, K, rng)
        mult, price_table = recommend_prices(u, D0, spot, cfg.price_grid)
        e = u["elasticity"].to_numpy()
        D = D0 * (mult ** -e)[:, None]
        for q in ("p10", "p50", "p90"):
            u[f"{q}_adj"] = u[q] * mult ** -e
        u["price_mult"] = mult
        u["sell_price"] = u["market_rate"] * mult
        u["unit_cost"] = u["market_rate"] * u["cost_ratio"]
        peak = (u["p50"] / u["level"]) >= 1.25
        u["cover_floor"] = np.where(peak, cfg.peak_cover_ratio * u["p50_adj"], 0.0)
        u["supplier_cap"] = cfg.supplier_cap_mult * u["base_demand"]
        p = u["sell_price"].to_numpy()
        inp = LPInput(
            price=p, spot_margin=np.maximum(p - SPOT_COST_RATIO * u["market_rate"].to_numpy(), 0),
            held=np.zeros(len(u)), add_cost=u["unit_cost"].to_numpy(), add_cap=u["supplier_cap"].to_numpy(),
            rel_cap=np.zeros(len(u)), rel_value=np.zeros(len(u)), basis=u["unit_cost"].to_numpy(),
            rel_frac=u["release_fraction"].to_numpy(), r1=u["release_value"].to_numpy(),
            r2=u["salvage_value"].to_numpy(), cover_floor=u["cover_floor"].to_numpy(),
            cover_pen=cfg.penalties["cover"] * p, dest=u["dest_idx"].to_numpy(), demand=D, spot=spot,
        )
        res = solve_plan(inp, budget=budget, lam=cfg.risk_lambda(cfg.seed_days), alpha=cfg.alpha,
                         risk_cap=cfg.risk_cap_pct * budget, dest_cap=cfg.max_dest_share * budget,
                         penalties=cfg.penalties)
        u["buy_qty"] = np.floor(res.add + 1e-6)
        u["spend"] = u["buy_qty"] * u["unit_cost"]
        A = u["buy_qty"].to_numpy()[:, None]
        u["exp_sold"] = np.minimum(A, D).mean(1)
        u["exp_sell_through"] = np.where(A[:, 0] > 0, u["exp_sold"] / np.maximum(A[:, 0], 1), np.nan)
        u["p_stockout"] = (D > A + spot).mean(1)
        u["cover_shortfall"] = res.slack_cover
        return SeasonState(units=u, budget=budget, seed_result=res, D_seed=D, spot=spot, price_table=price_table)

    # ---------------------------------------------------------------- pace
    def _posterior(self, u: pd.DataFrame, onbooks_gross: np.ndarray, days: int) -> pd.DataFrame:
        """Blend the seed forecast with booking pace (log-normal precision weighting)."""
        F = np.array([self.cancel.booked_fraction(t, days) for t in u["ptype"]])
        fcr = np.array([self.cancel.future_cancel_rate(t, days) for t in u["ptype"]])
        net = onbooks_gross * (1 - fcr)
        pace = np.maximum(net, 0.5) / np.maximum(F, 1e-3)
        mu0 = np.log(u["p50_adj"].to_numpy())
        s0 = 1.2 * (np.log(u["p90_adj"]) - np.log(u["p10_adj"])).to_numpy() / (2 * Z90)
        so = 0.10 + 0.30 * (1 - F)
        w = (1 / so**2) / (1 / so**2 + 1 / s0**2)
        mu = w * np.log(pace) + (1 - w) * mu0
        s = np.sqrt(1 / (1 / so**2 + 1 / s0**2))
        out = u[["product_id", "ptype", "destination", "week_start"]].copy()
        out["p10"], out["p50"], out["p90"] = np.exp(mu - Z90 * s), np.exp(mu), np.exp(mu + Z90 * s)
        out["onbooks_net"], out["pace_weight"] = net, w
        return out

    # ---------------------------------------------------------------- stage 2
    def rebalance(self, st: "SeasonState", onbooks_gross: np.ndarray, rng: np.random.Generator) -> None:
        cfg, u = self.cfg, st.units
        post = self._posterior(u, onbooks_gross, cfg.rebalance_days)
        D = sample_scenarios(post, cfg.n_scenarios, rng, inflate=1.0, disruptions=False)
        q = u["buy_qty"].to_numpy()
        mk = u["market_rate"].to_numpy()
        p = u["sell_price"].to_numpy()
        can_release = u["release_days"].to_numpy() < cfg.rebalance_days
        peak = u["cover_floor"].to_numpy() > 0
        inp = LPInput(
            price=p, spot_margin=np.maximum(p - SPOT_COST_RATIO * mk, 0), held=q,
            add_cost=u["topup_cost_ratio"].to_numpy() * mk,
            add_cap=0.5 * np.maximum(u["supplier_cap"].to_numpy() - q, 0),
            rel_cap=np.where(can_release, u["release_fraction"].to_numpy() * q, 0.0),
            rel_value=u["release_value"].to_numpy() * u["unit_cost"].to_numpy(),
            basis=u["unit_cost"].to_numpy(), rel_frac=np.zeros(len(u)),
            r1=u["salvage_value"].to_numpy(), r2=u["salvage_value"].to_numpy(),
            cover_floor=np.where(peak, cfg.peak_cover_ratio * post["p50"].to_numpy(), 0.0),
            cover_pen=cfg.penalties["cover"] * p, dest=u["dest_idx"].to_numpy(), demand=D, spot=st.spot,
        )
        res = solve_plan(inp, budget=st.budget - st.units["spend"].sum(), lam=cfg.risk_lambda(cfg.rebalance_days),
                         alpha=cfg.alpha, penalties=cfg.penalties)
        u["p50_t35"] = post["p50"].to_numpy()
        u["onbooks_net_t35"] = post["onbooks_net"].to_numpy()
        u["topup_qty"] = np.floor(res.add + 1e-6)
        u["release_qty"] = np.floor(res.release + 1e-6)
        u["final_qty"] = q + u["topup_qty"] - u["release_qty"]
        st.rebalance_result = res

    # ---------------------------------------------------------------- stage 3
    def markdown(self, st: "SeasonState", onbooks_gross: np.ndarray, rng: np.random.Generator) -> None:
        cfg, u = self.cfg, st.units
        post = self._posterior(u, onbooks_gross, cfg.markdown_days)
        D = sample_scenarios(post, cfg.n_scenarios, rng, inflate=1.0, disruptions=False)
        n10 = post["onbooks_net"].to_numpy()[:, None]
        resid = np.maximum(D - n10, 0)
        A = u.get("final_qty", u["buy_qty"]).to_numpy()[:, None]
        avail = np.maximum(A - np.minimum(n10, A), 0)
        p = u["sell_price"].to_numpy()[:, None]
        sc = (SPOT_COST_RATIO * u["market_rate"].to_numpy())[:, None]
        e = u["elasticity"].to_numpy()[:, None]
        c0r2 = (u["unit_cost"] * u["salvage_value"]).to_numpy()[:, None]
        vals = []
        for mm in cfg.markdown_grid:
            late = resid * mm ** (-e)
            s = np.minimum(avail, late)
            y = np.minimum(st.spot, late - s)
            vals.append((p * mm * s + np.maximum(p * mm - sc, 0) * y + c0r2 * (avail - s)).mean(1))
        vals = np.stack(vals, 1)
        u["p50_t10"] = post["p50"].to_numpy()
        u["markdown_mult"] = np.array(cfg.markdown_grid)[vals.argmax(1)]
        u["exp_leftover_t10"] = np.maximum(avail - resid, 0).mean(1)


@dataclass
class SeasonState:
    units: pd.DataFrame
    budget: float
    seed_result: LPResult
    D_seed: np.ndarray
    spot: np.ndarray
    price_table: pd.DataFrame
    rebalance_result: LPResult | None = None
    log: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- truth
def observed_onbooks(u: pd.DataFrame, future: pd.DataFrame, cancel: CancellationModel, days: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Gross bookings on the books ``days`` before arrival, as the system would see them."""
    mult = u["price_mult"].to_numpy() if "price_mult" in u else 1.0
    D = future["true_demand"].to_numpy() * mult ** -u["elasticity"].to_numpy()
    F = np.array([true_booked_fraction(t, days) for t in u["ptype"]])
    fcr = np.array([cancel.future_cancel_rate(t, days) for t in u["ptype"]]) * rng.uniform(0.85, 1.15, len(u))
    net = D * F * future["true_pace_noise"].to_numpy()
    return net / (1 - fcr)


def play(u: pd.DataFrame, future: pd.DataFrame, cfg: PolicyConfig, holdings: np.ndarray,
         price_mult: np.ndarray, markdown: np.ndarray, seed_cost: np.ndarray, topup_cost: np.ndarray,
         refunds: np.ndarray) -> dict:
    """Score a set of decisions against the true season."""
    mk = u["market_rate"].to_numpy()
    e = u["elasticity"].to_numpy()
    p = mk * price_mult
    sc = SPOT_COST_RATIO * mk
    D = future["true_demand"].to_numpy() * price_mult ** -e
    F10 = np.array([true_booked_fraction(t, cfg.markdown_days) for t in u["ptype"]])
    early, late = D * F10, D * (1 - F10) * markdown ** -e
    A = holdings
    s_e = np.minimum(A, early)
    s_l = np.minimum(A - s_e, late)
    cap = future["true_spot_cap"].to_numpy()
    y_e = np.where(p > sc, np.minimum(early - s_e, cap), 0)
    y_l = np.where(p * markdown > sc, np.minimum(late - s_l, cap - y_e), 0)
    left = A - s_e - s_l
    salvage = left * u["unit_cost"].to_numpy() * u["salvage_value"].to_numpy()
    own_rev = p * s_e + p * markdown * s_l
    spot_profit = (p - sc) * y_e + (p * markdown - sc) * y_l
    purchase = seed_cost.sum() + topup_cost.sum()
    prebuy_profit = own_rev.sum() + salvage.sum() + refunds.sum() - purchase
    served = s_e + s_l + y_e + y_l
    fill = np.where(D > 0, served / np.maximum(D, 1e-9), 1.0)
    sold = (s_e + s_l + y_e + y_l).sum()
    revenue = own_rev.sum() + (p * y_e + p * markdown * y_l).sum()
    list_rev = (p * (s_e + s_l + y_e + y_l)).sum()  # vs. the recommended sell price
    return {
        "profit": float(prebuy_profit + spot_profit.sum()),
        "prebuy_profit": float(prebuy_profit),
        "capital": float(purchase),
        "roi": float(prebuy_profit / purchase) if purchase > 0 else np.nan,
        "sell_through": float((s_e + s_l).sum() / A.sum()) if A.sum() > 0 else np.nan,
        "fill_rate": float(served.sum() / D.sum()),
        "soldout_share": float((fill < 0.97).mean()),
        "spoilage_value": float((left * u["unit_cost"].to_numpy()).sum() - salvage.sum()),
        "price_realisation": float(revenue / list_rev) if list_rev > 0 else np.nan,
        "units_sold": float(sold),
    }


# --------------------------------------------------------------------------- policies
def run_policies(engine: Engine, future: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, SeasonState]:
    cfg = engine.cfg
    st = engine.seed(future, rng)
    u = st.units
    n = len(u)
    ones, zeros = np.ones(n), np.zeros(n)
    results = {}

    # Baseline A: everything on request
    results["spot_only"] = play(u, future, cfg, zeros, ones, ones, zeros, zeros, zeros)

    # Baseline B: rule of thumb, 80% of last year's same week, as contracted, scaled to budget
    naive = 0.8 * engine.forecaster.seasonal_naive(future)
    naive = np.minimum(naive, u["supplier_cap"].to_numpy())
    cost = u["unit_cost"].to_numpy()
    scale = min(1.0, st.budget / (naive * cost).sum())
    hq = np.floor(naive * scale)
    results["rule_of_thumb"] = play(u, future, cfg, hq, ones, ones, hq * cost, zeros, zeros)

    # OptiBuy, stage 1 only
    q = u["buy_qty"].to_numpy()
    mult = u["price_mult"].to_numpy()
    results["optibuy_seed_only"] = play(u, future, cfg, q, mult, ones, q * cost, zeros, zeros)

    # OptiBuy, full rolling plan
    engine.rebalance(st, observed_onbooks(u, future, engine.cancel, cfg.rebalance_days, rng), rng)
    engine.markdown(st, observed_onbooks(u, future, engine.cancel, cfg.markdown_days, rng), rng)
    topup = u["topup_qty"].to_numpy() * u["topup_cost_ratio"].to_numpy() * u["market_rate"].to_numpy()
    refunds = u["release_qty"].to_numpy() * u["release_value"].to_numpy() * cost
    results["optibuy"] = play(u, future, cfg, u["final_qty"].to_numpy(), mult, u["markdown_mult"].to_numpy(),
                              q * cost, topup, refunds)

    true_d = future["true_demand"].to_numpy() * mult ** -u["elasticity"].to_numpy()
    st.log = {
        "wape_t90": wape(true_d, u["p50_adj"]),
        "wape_t35": wape(true_d, u["p50_t35"]),
        "wape_t10": wape(true_d, u["p50_t10"]),
        "seed_cvar_loss": st.seed_result.cvar_loss,
        "seed_expected_profit": st.seed_result.expected_profit,
        "seed_spend": float(u["spend"].sum()),
        "risk_slack": st.seed_result.slack_risk,
    }
    return results, st


def simulate(engine: Engine, n_sims: int = 20, seed: int = 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo over possible seasons. Returns (per-sim rows, KPI summary per policy)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sim in range(n_sims):
        future = sample_future(rng)
        res, st = run_policies(engine, future, rng)
        for pol, kpi in res.items():
            rows.append({"sim": sim, "policy": pol, **kpi, **{k: v for k, v in st.log.items() if k.startswith("wape")}})
    df = pd.DataFrame(rows)
    return df, summarise(df, engine.cfg.alpha)


def summarise(df: pd.DataFrame, alpha: float = 0.95) -> pd.DataFrame:
    g = df.groupby("policy", sort=False)
    out = g[["profit", "prebuy_profit", "capital", "roi", "sell_through", "fill_rate", "soldout_share",
             "spoilage_value", "price_realisation"]].mean()
    out["profit_p5"] = g["profit"].quantile(1 - alpha)
    out["profit_std"] = g["profit"].std()
    base = out.loc["spot_only", "profit"]
    out["uplift_vs_spot"] = out["profit"] - base
    return out
