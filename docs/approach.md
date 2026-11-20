# How OptiBuy decides

**The question TBO asks:** which hotel rooms and flight seats should we buy in advance, how many,
when, and at what sell price, so that a fixed budget earns the most risk-adjusted profit and
agents can still book in peak weeks?

## 1. Forecast demand (P10 / P50 / P90)

* **Unconstrain history.** Weeks that sold out recorded fewer bookings than true demand. Lost
  demand is estimated as searches x the product's normal look-to-book conversion.
* **Features known ~90 days out:** week of year, festival uplift (Diwali, Christmas / New Year,
  Holi, summer holidays), destination peak flag, same week last year, and the early search
  signal (look-to-book as a leading indicator).
* **Quantile gradient boosting** pooled across products, predicting log(demand / product level).
* **Scenarios:** each unit becomes a lognormal matched to its quantiles; a shared factor per
  destination makes Goa hotels and Goa flights move together; rare **disruptions** (advisory,
  cyclone, grounding) are overlaid because no 90-day forecast can see them.

## 2. Price each unit

For each candidate price (0.95x to 1.10x market) demand is re-scaled with the product's price
elasticity, the best newsvendor quantity is found and the expected profit compared. Scarce
peak weeks price up; softer weeks stay near market.

## 3. Seed buy (T-90): a scenario linear program

Maximise expected profit minus **λ(t) x CVaR95** of loss, across all products and weeks at once:

* revenue from own stock, thin margin from on-request (spot) sales when own stock runs out;
* unsold units recover value by **contract type**: hard blocks only salvage, soft allotments can
  be released (minus a fee) until 28 days out, flight group blocks allow 20% free attrition until
  30 days out;
* **hard:** budget, supplier block caps;
* **soft (priced slack):** CVaR risk cap, destination concentration (35% of budget), minimum
  cover on peak weeks. The solver always returns a plan; non-zero slack tells the analyst which
  policy had to give way and by how much.

λ(t) = λ0 + λ1·e^(−t/τ), with t = days to arrival at the re-plan. Close to arrival there is
little time left to sell, so the same exposure is weighted as riskier.

## 4. Rebalance (T-35): read the booking pace

Bookings on the books are discounted by the **lead-time-dependent cancellation model**
(logistic on log lead time, refundability, product type), divided by the share of bookings
normally made by now, and blended with the seed forecast by precision. The same LP then runs
with current holdings fixed: **top up** units that are running hot (at a smaller discount) and
**release** cold units before their contract deadlines.

## 5. Markdown and release (T-10)

For stock that pace says will not sell, choose the price cut (0 to 30%) with the best expected
recovery against salvage. Whatever is left at arrival is salvaged at the contract's salvage value.

## 6. Judge it on many seasons

`scripts/run_backtest.py` plays sampled seasons against ground truth and compares OptiBuy with
buying nothing in advance and with a rule of thumb (80% of last year's volume as hard blocks).
