"""Lead-time-dependent cancellation risk and booking pace curves.

A booking made 90 days out is far more likely to cancel than one made 5 days
out, and refundable rates cancel more. We fit a logistic model on booking-level
history and derive two curves used by the rolling planner:

* ``booked_fraction(ptype, t)``: share of final bookings already on the books
  ``t`` days before arrival (the pace curve);
* ``future_cancel_rate(ptype, t)``: share of the bookings on the books at ``t``
  that will still cancel before arrival. Pace is measured on bookings net of
  this, so a flurry of refundable early bookings does not trigger over-buying.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _design(lead, refundable, is_flight) -> np.ndarray:
    return np.column_stack([np.log1p(lead), refundable, is_flight]).astype(float)


@dataclass
class CancellationModel:
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(C=10.0))
    bookings: pd.DataFrame | None = None

    def fit(self, bookings: pd.DataFrame) -> "CancellationModel":
        self.bookings = bookings.copy()
        X = _design(bookings["lead_days"], bookings["refundable"], bookings["ptype"] == "flight")
        self.model.fit(X, bookings["cancelled"])
        self.bookings["p_cancel"] = self.model.predict_proba(X)[:, 1]
        return self

    def prob(self, lead_days, refundable, ptype) -> np.ndarray:
        lead = np.atleast_1d(lead_days)
        X = _design(lead, np.broadcast_to(refundable, lead.shape), np.broadcast_to(ptype == "flight", lead.shape))
        return self.model.predict_proba(X)[:, 1]

    def booked_fraction(self, ptype: str, days_before: float) -> float:
        b = self.bookings[self.bookings["ptype"] == ptype]
        return float((b["lead_days"] >= days_before).mean())

    def future_cancel_rate(self, ptype: str, days_before: float) -> float:
        b = self.bookings[(self.bookings["ptype"] == ptype) & (self.bookings["lead_days"] >= days_before)]
        if b.empty:
            return 0.0
        # cancellation time ~ uniform between booking and arrival
        return float((b["p_cancel"] * days_before / b["lead_days"]).mean())

    def curves(self, max_days: int = 120) -> pd.DataFrame:
        rows = []
        for ptype in ("hotel", "flight"):
            for t in range(0, max_days + 1, 5):
                rows.append(dict(
                    ptype=ptype, days_before=t,
                    booked_fraction=self.booked_fraction(ptype, t),
                    future_cancel_rate=self.future_cancel_rate(ptype, t),
                    p_cancel_nonref=float(self.prob(t, False, ptype)[0]),
                    p_cancel_ref=float(self.prob(t, True, ptype)[0]),
                ))
        return pd.DataFrame(rows)

    def coefficients(self) -> dict:
        return dict(zip(["log1p_lead", "refundable", "is_flight"], self.model.coef_[0].round(3))) | {
            "intercept": round(float(self.model.intercept_[0]), 3)
        }
