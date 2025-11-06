# Methodology & Trade-offs

This project is intentionally simple and neutral in its main view: a schedule-adjusted table based on **Pot × Venue** averages from 2024–25. Below is a brief explanation of key choices and the trade-offs involved.

## Why Pot × Venue (and not per-opponent Elo/coefficients)?
- **Alignment with competition design**: Pots are the scheduling primitive. Reading difficulty through pots matches how fixtures are constructed.
- **Clarity & comparability**: One global baseline for all teams. Per-opponent Elo/coefficients would vary expected points by team, which complicates cross-team comparisons.
- **Maintainability**: Pot × Venue needs minimal upkeep and is transparent to non-technical readers.

**Trade-off**: Pots are coarse. There is real spread within each pot. Future alternative views may scale by coefficients/Elo for readers who value that granularity.

## Baseline size and stability
We currently use one full season (2024–25) under the new format. That’s limited, so bucket means are noisy. We prefer to avoid subjective “fixes” (e.g., adjusting Pot 1 home down because of one season’s quirk). As more seasons accrue, we can smooth the baseline (e.g., multi-year rolling averages).

## Projections as a bonus, not a forecast
The projections show “what if current over/under versus schedule persists.” They:
- cap per game at 3 points,
- optionally winsorize & shrink the form factor (**F_raw**).

They are **not** probabilistic forecasts; a real forecast would add team strength models, injuries, rest, etc.

## Effort vs precision
Richer models (per-opponent ratings, margins, Bayesian priors, etc.) increase effort and complexity while yielding diminishing returns for a mid-phase reading tool. The current design favors:
- neutral inputs,
- explainability,
- reproducibility.

## Neutrality and guardrails
- We avoid hand-tuning bucket means. Adjustments invite bias unless methodologically justified and repeatable.
- The pipeline caches inputs and records source URLs. Helper scripts are deterministic given the scraped pages.

## Known limitations
- Pots are imperfect proxies for actual team strength.
- One-season baseline is noisy; expanding the baseline is on the roadmap.
- Draws are 1 point (no margin or xG). That’s deliberate for readability; variants could be added separately.

## Alternatives we might add (as separate views)
- Per-opponent coefficient/Elo scaling.
- Goal-margin adjusted baselines.
- Multi-season baselines with decay.
- Europa League / Conference League versions.
