# Architecture

```mermaid
flowchart LR
    subgraph Data["Data layer (synthetic stand-in for TBO data)"]
        H[Weekly bookings, searches, sold-outs]
        B[Booking-level lead times and cancellations]
        S[Supplier contracts and spot availability]
    end
    subgraph Models
        U[Unconstrain sold-out weeks]
        F[Quantile GBM forecaster P10/P50/P90]
        C[Cancellation model by lead time]
        SC[Scenario sampler with destination correlation and disruptions]
        PR[Price recommender with elasticity]
    end
    subgraph Decide["Decision engine (HiGHS LP)"]
        S1[Stage 1 seed buy T-90]
        S2[Stage 2 rebalance T-35: top-up and release]
        S3[Stage 3 markdown T-10]
    end
    subgraph Serve
        API[FastAPI /plan /forecast /simulate]
        DB[Streamlit analyst dashboard]
        BT[Backtest and KRA scorecard]
    end
    H --> U --> F --> SC --> PR --> S1
    B --> C --> S2
    S --> S1
    S1 --> S2 --> S3
    S1 & S2 & S3 --> API & DB & BT
```

## Modules

| Path | Responsibility |
|---|---|
| `optibuy/config.py` | Product catalogue, contract types, policy knobs, λ(t) |
| `optibuy/data.py` | Synthetic world: demand process, censoring, searches, bookings, disruptions, spot capacity |
| `optibuy/forecast.py` | Unconstraining, features, quantile GBM, scenario sampler, WAPE backtest |
| `optibuy/cancellation.py` | Logistic cancellation model, pace curve, future-cancel rate |
| `optibuy/optimizer.py` | Scenario LP with CVaR, budget, soft constraints (scipy + HiGHS) |
| `optibuy/planner.py` | Pricing, three-stage rolling engine, baselines, season simulator |
| `api/main.py` | REST API |
| `dashboard/app.py` | Analyst UI |
| `scripts/run_backtest.py` | Monte Carlo KRA scorecard to `reports/` |

## Size and speed

16 products x 12 arrival weeks = 192 decision units. With 80 scenarios the seed LP has about
47,000 variables and 62,000 constraints and solves in 2 to 7 seconds on a laptop CPU. It scales
linearly in scenarios and units; at TBO scale the same model would run per region or per
destination cluster nightly, or switch to a commercial solver.

## Moving from synthetic to real data

Replace `optibuy/data.py` with loaders that return the same columns:

* weekly history: `product_id, week_start, bookings, searches, early_searches, sold_out`
* bookings: `ptype, lead_days, refundable, cancelled`
* supplier offers: market rate, contract type, block cap, release terms

Everything downstream keeps working unchanged.
