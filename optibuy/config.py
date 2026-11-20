"""Static configuration: catalogue, supplier contract types and policy knobs.

Everything here is synthetic but shaped like a B2B travel wholesaler's world:
hotel room-nights and flight seat blocks, sold to travel agents, bought from
suppliers under different contract types.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContractType:
    """How a block of pre-bought inventory is paid for and what can be handed back.

    cost_ratio        purchase cost as a fraction of the market (sell) rate
    release_days      last day (days before arrival) unsold units may be released
    release_fraction  share of the block that may be released back to the supplier
    release_value     refund per released unit, as a fraction of the purchase cost
    salvage_value     value recovered per unit still unsold at arrival
                      (dumped to a wholesale/opaque channel), fraction of cost
    topup_cost_ratio  cost of buying extra units later, at the rebalance checkpoint
    """

    name: str
    cost_ratio: float
    release_days: int
    release_fraction: float
    release_value: float
    salvage_value: float
    topup_cost_ratio: float
    description: str = ""


CONTRACTS: dict[str, ContractType] = {
    "HARD_BLOCK": ContractType(
        name="HARD_BLOCK",
        cost_ratio=0.78,
        release_days=0,
        release_fraction=0.0,
        release_value=0.0,
        salvage_value=0.15,
        topup_cost_ratio=0.87,
        description="Prepaid, non-refundable hotel block. Deepest discount, full risk.",
    ),
    "SOFT_ALLOTMENT": ContractType(
        name="SOFT_ALLOTMENT",
        cost_ratio=0.86,
        release_days=28,
        release_fraction=1.0,
        release_value=0.95,
        salvage_value=0.15,
        topup_cost_ratio=0.91,
        description="Hotel allotment, releasable until 28 days before arrival (5% fee).",
    ),
    "FLIGHT_GROUP_BLOCK": ContractType(
        name="FLIGHT_GROUP_BLOCK",
        cost_ratio=0.83,
        release_days=30,
        release_fraction=0.20,
        release_value=1.0,
        salvage_value=0.10,
        topup_cost_ratio=0.90,
        description="Airline group seats; up to 20% attrition free until 30 days out.",
    ),
}

#: Cost of buying on request ("spot") from the supplier at booking time.
SPOT_COST_RATIO = 0.94


@dataclass(frozen=True)
class Product:
    product_id: str
    ptype: str  # "hotel" | "flight"
    destination: str
    name: str
    market_rate: float  # INR per room-night or seat
    base_weekly_demand: float
    contract: str
    peak_weeks: tuple[int, ...]  # ISO weeks with seasonal peaks
    elasticity: float  # price elasticity of B2B demand (agents compare wholesalers)


CATALOG: tuple[Product, ...] = (
    Product("H-GOA-5", "hotel", "Goa", "Goa beach resort (5*)", 14000, 140, "HARD_BLOCK", (51, 52, 1), 4.5),
    Product("H-GOA-3", "hotel", "Goa", "Goa city hotel (3*)", 5200, 220, "SOFT_ALLOTMENT", (51, 52, 1), 4.8),
    Product("H-DXB-5", "hotel", "Dubai", "Dubai downtown hotel (5*)", 18000, 120, "SOFT_ALLOTMENT", (50, 51, 52, 1, 2), 4.2),
    Product("H-DXB-4", "hotel", "Dubai", "Dubai marina hotel (4*)", 11000, 160, "HARD_BLOCK", (50, 51, 52, 1, 2), 4.5),
    Product("H-BKK-4", "hotel", "Bangkok", "Bangkok Sukhumvit hotel (4*)", 7000, 190, "SOFT_ALLOTMENT", (52, 1, 15), 4.65),
    Product("H-DPS-5", "hotel", "Bali", "Bali villa resort (5*)", 21000, 70, "HARD_BLOCK", (26, 27, 52, 1), 4.05),
    Product("H-MNL-3", "hotel", "Manali", "Manali hill hotel (3*)", 4500, 150, "SOFT_ALLOTMENT", (21, 22, 23, 52), 4.95),
    Product("H-JAI-4", "hotel", "Jaipur", "Jaipur heritage hotel (4*)", 8000, 110, "HARD_BLOCK", (44, 45, 52), 4.5),
    Product("H-SIN-4", "hotel", "Singapore", "Singapore Orchard hotel (4*)", 16000, 130, "SOFT_ALLOTMENT", (22, 23, 51, 52), 4.35),
    Product("H-MLE-5", "hotel", "Maldives", "Maldives water villa (5*)", 45000, 35, "HARD_BLOCK", (52, 1, 2, 7), 3.75),
    Product("F-DEL-GOI", "flight", "Goa", "Delhi to Goa seats", 6500, 230, "FLIGHT_GROUP_BLOCK", (51, 52, 1), 3.9),
    Product("F-BOM-DXB", "flight", "Dubai", "Mumbai to Dubai seats", 14000, 200, "FLIGHT_GROUP_BLOCK", (50, 51, 52, 1), 3.6),
    Product("F-DEL-BKK", "flight", "Bangkok", "Delhi to Bangkok seats", 15000, 170, "FLIGHT_GROUP_BLOCK", (52, 1, 15), 3.75),
    Product("F-BLR-SIN", "flight", "Singapore", "Bengaluru to Singapore seats", 17000, 140, "FLIGHT_GROUP_BLOCK", (22, 23, 51, 52), 3.6),
    Product("F-BOM-DPS", "flight", "Bali", "Mumbai to Bali seats", 24000, 90, "FLIGHT_GROUP_BLOCK", (26, 27, 52, 1), 3.45),
    Product("F-BOM-MLE", "flight", "Maldives", "Mumbai to Male seats", 19000, 60, "FLIGHT_GROUP_BLOCK", (52, 1, 2, 7), 3.45),
)

#: Indian festival / holiday weeks that lift outbound leisure demand everywhere.
EVENT_WEEKS: dict[int, float] = {
    44: 0.35,  # Diwali window
    45: 0.25,
    52: 0.30,  # Christmas / New Year
    1: 0.20,
    12: 0.10,  # Holi long weekend
    21: 0.15,  # summer school holidays start
    22: 0.20,
    23: 0.15,
}


@dataclass
class PolicyConfig:
    """Knobs for the decision engine. Defaults are used by the API and dashboard."""

    budget: float = 2.0e8  # INR of capital available for pre-buys over the horizon
    alpha: float = 0.95  # CVaR confidence level (worst 5% of scenarios)
    risk_cap_pct: float = 0.04  # worst-5% average program loss allowed, as % of budget
    lambda0: float = 0.10  # base risk aversion
    lambda1: float = 0.60  # extra risk aversion close to arrival
    lambda_tau: float = 20.0  # days; decay of the time-varying term
    n_scenarios: int = 80
    max_dest_share: float = 0.35  # concentration limit per destination (soft)
    peak_cover_ratio: float = 0.45  # guaranteed cover of P50 demand on peak cohorts (soft)
    seed_days: int = 90  # stage 1 checkpoint (days before arrival)
    rebalance_days: int = 35  # stage 2 checkpoint
    markdown_days: int = 10  # stage 3 checkpoint
    price_grid: tuple[float, ...] = (0.95, 1.0, 1.05, 1.10)
    markdown_grid: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7)
    supplier_cap_mult: float = 1.6  # max block size vs. product base weekly demand
    penalties: dict = field(
        default_factory=lambda: {"risk": 4.0, "dest": 0.5, "cover": 0.30}
    )

    def risk_lambda(self, days_to_arrival: float) -> float:
        """Time-decaying risk weight lambda(t): grows as arrival gets close.

        Close to arrival there is little time left to sell, so the same unsold
        unit is more likely to become a loss; the planner becomes more cautious.
        """
        import math

        return self.lambda0 + self.lambda1 * math.exp(-days_to_arrival / self.lambda_tau)
