Your plan is strong. I’d tweak the order so you’re not “measuring with a rubber ruler.” In short: **make the live decision rule correct first**, then validate it.

## What to do first (and why)

1. **Fix production to true probabilistic EV (with no-vig odds)** – *same day impact*
   Right now the live system still uses the heuristic `0.5 + edge/100`, so all results/learning from live bets are noisy. Switch to:

* **Player sigma:** estimate per-player σ from residuals (last 20–40 games), with shrinkage:
  [
  \hat\sigma_i^2=\frac{n}{n+\lambda}s_i^2+\frac{\lambda}{n+\lambda}s_{\text{global}}^2
  ]
  (e.g., (n=20), (\lambda=10)). Use robust (s_i) from MAD: (1.4826\cdot \text{MAD}).
* **Over probability:** (p_{\text{over}}=1-\Phi!\left(\frac{\text{line}-\mu_i}{\hat\sigma_i}\right))
* **No-vig price:** for American odds (A), convert to implied (q); for both sides (q_{+},q_{-}),
  [
  q_{\text{novig}}=\frac{q}{q_{+}+q_{-}}
  ]
* **EV:** (\text{EV}=p_{\text{model}}\cdot \text{payout}-(1-p_{\text{model}})) (use decimal odds minus 1 for payout). Bet only if EV > threshold.

2. **Add player-specific variance now** – *the lever that makes #1 work*
   Giannis ≠ Poole. Without σ_player, a “3-pt edge” treats high-variance guys like low-variance guys and breaks calibration.

3. **Bootstrap CIs on your backtests** – *1 day to de-risk*
   Block-bootstrap by date, report 95% CI for ROI & WR. If the ROI CI straddles 0, assume fragile edge and tighten filters.

4. **Start lightweight CLV tracking in parallel** – *keystone validity check*
   Even a simple “entry line vs line 30 mins pre-tip” logger is enough to see if you beat the close. If CLV trends positive, you’re on the right track—even before P&L converges.

After those four, move to: edge-bucket monotonicity, feature de-duplication (cut near-duplicate recency features with RFECV), embargo window, fractional Kelly.

---

## A 7-day mini-sprint (tight, realistic)

**Day 1–2: Probabilistic EV + σ_player**

* Compute per-player residuals → robust σ with shrinkage.
* Replace heuristic with (p_{\text{over}}) and **no-vig** EV gating in production.
* Add fractional Kelly (0.25×):
  (f=\max(0, \frac{bp - (1-p)}{b})) with (b=\text{decimalOdds}-1); stake = (0.25f\cdot)bankroll (cap at 1–2% bankroll).

**Day 3: Bootstrap CI**

* Block-bootstrap (by date) 2–5k reps; log WR/ROI CIs.
* Surface these CIs in your report so you stop eyeballing point estimates.

**Day 4–5: CLV**

* Log: game_id, market, entry_line, entry_time, close_line, (\Delta=) (close–entry)×direction.
* KPI: % bets with positive CLV and mean CLV. Target **>55%** positive and mean Δ > 0.

**Day 6–7: Edge-bucket check + quick calibration**

* Bin by model edge (0–2, 2–4, 4–6, 6–8, 8+). Expect monotone lift.
* If U-shaped: fit **isotonic regression** mapping raw (p_{\text{model}}\to p_{\text{cal}}); re-compute EV with (p_{\text{cal}}).

---

## Acceptance criteria (green lights)

* **Probability plumbing:** live bets use (p_{\text{over}}(\mu,\sigma_i)) + no-vig.
* **σ_player sanity:** σ shrinks toward global when n is small; distribution looks plausible (stars low, gunners high).
* **Backtest robustness:** ROI 95% CI **doesn’t** include 0; WR CI lower bound **> 52.38%** (your breakeven on -110).
* **CLV:** mean CLV > 0; **>55%** of bets beat the close.
* **Monotonicity:** higher edge buckets → higher realized WR/EV after isotonic calibration.

---

## Tiny code sketch (drop-in logic)

```python
# sigma per player (robust with shrinkage)
def player_sigma(residuals, s_global, lam=10):
    import numpy as np
    n = len(residuals)
    s_i = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))  # MAD
    w = n / (n + lam)
    return np.sqrt(w * s_i**2 + (1 - w) * s_global**2)

# prob over using player sigma
from math import erf, sqrt
def norm_cdf(x):  # Φ(x)
    return 0.5 * (1 + erf(x / sqrt(2)))

def prob_over(mu, sigma, line):
    z = (line - mu) / sigma
    return 1 - norm_cdf(z)

# no-vig for two-way market given implied probs q_over, q_under
def novig(q_over, q_under):
    total = q_over + q_under
    return q_over / total, q_under / total
```

---

### Bottom line

Start with **(1) probabilistic EV + (2) σ_player**, add **(3) bootstrap CIs**, and begin **(4) CLV logging** immediately. Those four give you **clean signal**; everything else (feature pruning, embargo, etc.) then compounds your edge instead of masking it.
