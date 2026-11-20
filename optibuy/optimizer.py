"""Scenario-based buy-plan optimiser (linear program, solved with HiGHS).

For every decision unit i (product x arrival week) and demand scenario k:

    maximise   E_k[ profit_k ]  -  lambda(t) * CVaR_alpha(loss)  -  penalties * slacks

    profit_k = sum_i  p_i s_ik + m_i y_ik + rec_ik  -  c_i x_i  +  v_i rel_i

    s_ik  <= H_i + x_i - rel_i          own inventory sold (H = already held)
    s_ik + y_ik <= D_ik                 can't sell more than demand
    y_ik  <= Spot_ik                    on-request (spot) sales, thin margin m_i
    rec_ik <= recovery(unsold)          release / salvage value, concave piecewise
    sum c_i x_i - v_i rel_i <= budget                         (hard)
    CVaR_alpha(loss) <= risk_cap + slack_risk                 (soft)
    spend per destination <= share * budget + slack_dest      (soft)
    H_i + x_i - rel_i >= cover_floor_i - slack_cover_i         (soft)

The same LP serves stage 1 (seed buy: H = 0) and stage 2 (rebalance: H = seed
holdings, x = top-ups, rel = releases). Hard limits that can never conflict
(budget, supplier caps) stay hard; everything that might conflict is softened
with a priced slack, so the solver always returns a plan and the slacks tell
the analyst which policy had to give way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

SCALE = 1e-3  # work in thousands of INR for numerical comfort


@dataclass
class LPInput:
    price: np.ndarray        # p_i, sell price per unit
    spot_margin: np.ndarray  # m_i, margin per spot unit (>= 0)
    held: np.ndarray         # H_i
    add_cost: np.ndarray     # c_i
    add_cap: np.ndarray      # U_i
    rel_cap: np.ndarray      # R_i
    rel_value: np.ndarray    # v_i
    basis: np.ndarray        # c0_i, cost basis for recovery of unsold units
    rel_frac: np.ndarray     # a_i, share of final holding still releasable at arrival
    r1: np.ndarray           # recovery rate on releasable units
    r2: np.ndarray           # recovery rate on the rest (salvage)
    cover_floor: np.ndarray  # f_i
    cover_pen: np.ndarray    # penalty per unit of cover shortfall
    dest: np.ndarray         # destination index per unit
    demand: np.ndarray       # D, shape (n, K)
    spot: np.ndarray         # Spot capacity, shape (n, K)


@dataclass
class LPResult:
    add: np.ndarray
    release: np.ndarray
    holding: np.ndarray
    scenario_profit: np.ndarray
    expected_profit: float
    cvar_loss: float
    spend: float
    slack_risk: float
    slack_dest: np.ndarray
    slack_cover: np.ndarray
    status: str
    solve_seconds: float


def solve_plan(inp: LPInput, budget: float, lam: float, alpha: float = 0.95,
               risk_cap: float | None = None, dest_cap: float | None = None,
               penalties: dict | None = None) -> LPResult:
    import time

    pen = {"risk": 4.0, "dest": 0.5} | (penalties or {})
    n, K = inp.demand.shape
    nd = int(inp.dest.max()) + 1
    nK = n * K
    # variable offsets
    o_x, o_rel, o_cs = 0, n, 2 * n
    o_s = 3 * n
    o_y = o_s + nK
    o_rec = o_y + nK
    o_z = o_rec + nK
    o_eta = o_z + K
    o_sr = o_eta + 1
    o_sd = o_sr + 1
    nvar = o_sd + nd

    S = SCALE
    p, m, c, v = inp.price * S, inp.spot_margin * S, inp.add_cost * S, inp.rel_value * S
    c0 = inp.basis * S
    g = inp.r1 * inp.rel_frac + inp.r2 * (1 - inp.rel_frac)
    tail = 1.0 / ((1 - alpha) * K)

    # ----------------------------------------------------------- objective
    obj = np.zeros(nvar)
    obj[o_x:o_x + n] = c
    obj[o_rel:o_rel + n] = -v
    obj[o_cs:o_cs + n] = inp.cover_pen * S
    obj[o_s:o_s + nK] = -np.repeat(p, K) / K
    obj[o_y:o_y + nK] = -np.repeat(m, K) / K
    obj[o_rec:o_rec + nK] = -1.0 / K
    obj[o_z:o_z + K] = lam * tail
    obj[o_eta] = lam
    obj[o_sr] = pen["risk"]
    obj[o_sd:o_sd + nd] = pen["dest"]

    ik = np.arange(nK)
    unit = np.repeat(np.arange(n), K)
    scen = np.tile(np.arange(K), n)
    rows, cols, vals, rhs = [], [], [], []
    r0 = 0

    def add(rr, cc, vv):
        rows.append(np.asarray(rr)); cols.append(np.asarray(cc)); vals.append(np.asarray(vv, float))

    H = inp.held
    # (1) s_ik - x_i + rel_i <= H_i
    add(r0 + ik, o_s + ik, np.ones(nK)); add(r0 + ik, o_x + unit, -np.ones(nK)); add(r0 + ik, o_rel + unit, np.ones(nK))
    rhs.append(H[unit]); r0 += nK
    # (2) s_ik + y_ik <= D_ik
    add(r0 + ik, o_s + ik, np.ones(nK)); add(r0 + ik, o_y + ik, np.ones(nK))
    rhs.append(inp.demand.ravel()); r0 += nK
    # (3) rec + c0 r1 s - c0 r1 (x - rel) <= c0 r1 H
    k1 = (c0 * inp.r1)[unit]
    add(r0 + ik, o_rec + ik, np.ones(nK)); add(r0 + ik, o_s + ik, k1)
    add(r0 + ik, o_x + unit, -k1); add(r0 + ik, o_rel + unit, k1)
    rhs.append(k1 * H[unit]); r0 += nK
    # (4) rec + c0 r2 s - c0 g (x - rel) <= c0 g H
    k2, kg = (c0 * inp.r2)[unit], (c0 * g)[unit]
    add(r0 + ik, o_rec + ik, np.ones(nK)); add(r0 + ik, o_s + ik, k2)
    add(r0 + ik, o_x + unit, -kg); add(r0 + ik, o_rel + unit, kg)
    rhs.append(kg * H[unit]); r0 += nK
    # (5) CVaR: -profit_k - eta - z_k <= 0
    add(r0 + scen, o_s + ik, -p[unit]); add(r0 + scen, o_y + ik, -m[unit]); add(r0 + scen, o_rec + ik, -np.ones(nK))
    add(np.repeat(r0 + np.arange(K), n), np.tile(o_x + np.arange(n), K), np.tile(c, K))
    add(np.repeat(r0 + np.arange(K), n), np.tile(o_rel + np.arange(n), K), -np.tile(v, K))
    add(r0 + np.arange(K), np.full(K, o_eta), -np.ones(K)); add(r0 + np.arange(K), o_z + np.arange(K), -np.ones(K))
    rhs.append(np.zeros(K)); r0 += K
    # (6) risk cap (soft)
    if risk_cap is not None:
        add([r0], [o_eta], [1.0]); add(np.full(K, r0), o_z + np.arange(K), np.full(K, tail)); add([r0], [o_sr], [-1.0])
        rhs.append([risk_cap * S]); r0 += 1
    # (7) budget (hard)
    add(np.full(n, r0), o_x + np.arange(n), c); add(np.full(n, r0), o_rel + np.arange(n), -v)
    rhs.append([budget * S]); r0 += 1
    # (8) destination concentration (soft)
    if dest_cap is not None:
        add(r0 + inp.dest, o_x + np.arange(n), c); add(r0 + np.arange(nd), o_sd + np.arange(nd), -np.ones(nd))
        rhs.append(np.full(nd, dest_cap * S)); r0 += nd
    # (9) cover floor (soft): -x + rel - cs <= H - f
    add(r0 + np.arange(n), o_x + np.arange(n), -np.ones(n)); add(r0 + np.arange(n), o_rel + np.arange(n), np.ones(n))
    add(r0 + np.arange(n), o_cs + np.arange(n), -np.ones(n))
    rhs.append(H - inp.cover_floor); r0 += n

    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, nvar))
    b = np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in rhs])

    lb = np.zeros(nvar); ub = np.full(nvar, np.inf)
    ub[o_x:o_x + n] = inp.add_cap
    ub[o_rel:o_rel + n] = inp.rel_cap
    ub[o_s:o_s + nK] = inp.demand.ravel()
    ub[o_y:o_y + nK] = inp.spot.ravel()
    lb[o_eta] = -np.inf

    t0 = time.perf_counter()
    res = linprog(obj, A_ub=A, b_ub=b, bounds=np.column_stack([lb, ub]), method="highs")
    secs = time.perf_counter() - t0
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    xv = res.x
    add_q, rel_q = xv[o_x:o_x + n], xv[o_rel:o_rel + n]
    s = xv[o_s:o_s + nK].reshape(n, K); y = xv[o_y:o_y + nK].reshape(n, K); rec = xv[o_rec:o_rec + nK].reshape(n, K)
    prof = (p[:, None] * s + m[:, None] * y + rec).sum(0) - c @ add_q + v @ rel_q
    eta = xv[o_eta]; z = xv[o_z:o_z + K]
    return LPResult(
        add=add_q, release=rel_q, holding=H + add_q - rel_q,
        scenario_profit=prof / S, expected_profit=float(prof.mean() / S),
        cvar_loss=float((eta + tail * z.sum()) / S),
        spend=float((c @ add_q - v @ rel_q) / S),
        slack_risk=float(xv[o_sr] / S), slack_dest=xv[o_sd:o_sd + nd] / S,
        slack_cover=xv[o_cs:o_cs + n], status=res.message, solve_seconds=secs,
    )


def cvar_loss(profits: np.ndarray, alpha: float = 0.95) -> float:
    """Average loss in the worst (1-alpha) share of outcomes (negative = still a profit)."""
    losses = np.sort(-np.asarray(profits))[::-1]
    k = max(1, int(np.ceil(round((1 - alpha) * len(losses), 9))))
    return float(losses[:k].mean())
