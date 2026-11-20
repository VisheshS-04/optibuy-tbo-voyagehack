import numpy as np
import pytest

from optibuy.cancellation import CancellationModel
from optibuy.config import PolicyConfig
from optibuy.data import make_world, sample_future
from optibuy.forecast import backtest_forecaster, unconstrain
from optibuy.optimizer import cvar_loss
from optibuy.planner import Engine, run_policies


@pytest.fixture(scope="module")
def world():
    return make_world(7)


@pytest.fixture(scope="module")
def engine():
    return Engine.build(7, PolicyConfig(n_scenarios=30))


def test_history_shape_and_censoring(world):
    h = world.history
    assert len(h) == 16 * 156
    assert h["sold_out"].any()
    assert (h.loc[h["sold_out"], "bookings"] <= h.loc[h["sold_out"], "true_demand"] + 0.5).all()
    # unconstraining lifts sold-out weeks closer to truth
    u = unconstrain(h)
    so = u[u["sold_out"]]
    assert (abs(so["demand_est"] - so["true_demand"]).sum() < abs(so["bookings"] - so["true_demand"]).sum())


def test_forecaster_beats_seasonal_naive(world):
    m = backtest_forecaster(world.history)
    assert m["wape_model"] < m["wape_seasonal_naive"]
    assert m["wape_model"] < 0.25


def test_cancellation_rises_with_lead_time(world):
    cm = CancellationModel().fit(world.bookings)
    p = cm.prob(np.array([1, 30, 90]), False, "hotel")
    assert p[0] < p[1] < p[2]
    assert cm.booked_fraction("hotel", 90) < cm.booked_fraction("hotel", 10)


def test_cvar_loss():
    profits = np.array([-10.0] + [5.0] * 19)
    assert cvar_loss(profits, 0.95) == 10.0


def test_plan_respects_budget_and_always_returns(engine):
    rng = np.random.default_rng(0)
    engine.cfg.risk_cap_pct = -10.0  # impossible risk cap: soft constraint must absorb it
    st = engine.seed(sample_future(rng), rng, budget=3e7)
    engine.cfg.risk_cap_pct = 0.04
    assert st.units["spend"].sum() <= 3e7 + 1
    assert st.seed_result.slack_risk > 0


def test_rolling_policy_beats_baselines(engine):
    rng = np.random.default_rng(3)
    res, st = run_policies(engine, sample_future(rng), rng)
    assert res["optibuy"]["profit"] > res["rule_of_thumb"]["profit"] > res["spot_only"]["profit"]
    assert res["optibuy"]["fill_rate"] > res["spot_only"]["fill_rate"]
    assert st.log["wape_t35"] < st.log["wape_t90"]
