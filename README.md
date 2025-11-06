# UCL League-Phase: Schedule-Adjusted Table (2025–26)

A small, focused project that computes a **schedule-adjusted table** for the UEFA Champions League league phase. The idea: teams have played **different opponent strengths** and **different home/away splits** after matchdays 3–6. This table adjusts for that, so we can read current performance **relative to strength of schedule**. Projections are included as a **bonus**, not the main output.

## What this shows
- **So far**: Actual points vs **expected** points based on opponent **pot** and **venue** (home/away).
- **Over/under**: How far a team is ahead or behind what an average team typically gets vs the same pot/venue mix.
- **Bonus projections**: Simple carry-forward estimates to 8 matches (with and without a small tempering step).

> Scope: this is most relevant **mid-phase** (after MD3–MD6). Once 8/8 games are finished, the official table is sufficient.

## Method (short)
1. Use **2024–25** (first season with the new format) to build **8 baseline averages**:  
   `avg points` for (Pot 1–4) × (Home/Away).  
2. For **2025–26**, compute each team’s **expected points so far** by mapping each played match into one of the 8 buckets and summing those baseline averages.  
3. Compare **actual vs expected** to get over/under and **F_raw** = actual / expected.  
4. Project to 8 matches by adding baseline points for remaining buckets, optionally scaled by **F_raw** (capped at 3 per game). A tempered version winsorizes F_raw and shrinks it slightly toward 1 as games are played.

**Why pots?** Pots are the competition’s native schedule design and are easy to explain and maintain. Using coefficients/Elo per opponent is possible, but trades clarity and comparability for complexity (see “Design choices & alternatives”).

## How to run

### 0) Environment
- Python 3.11+
- `pip install -r requirements.txt` (or `uv pip sync`)

### 1) Scrape Wikipedia
python src/01_scrape.py

Outputs to `data/raw/` and caches raw HTML in `data/raw/_cache/`.

**Cache controls in `01_scrape.py`:**
- `FORCE_REFRESH` – ignore cache for all pages.
- `CACHE_MAX_AGE_SECONDS` – TTL for cached HTML (default 6h).
- `ALWAYS_REFRESH_RESULTS` – always refetch results (pots remain cached).

### 2) Build helper CSVs
python src/02_build_helpers.py

Outputs to `data/out/`:
- `baselines_ucl_2024-25.csv`  
- `teamgames_ucl_2024-25.csv`, `teamgames_ucl_2025-26.csv`  
- `buckets_played_ucl_2025-26.csv`  
- `future_slots_ucl_2025-26.csv` (internal helper; not published as HTML)

> Step 2 also **cleans pots** to drop any stray non-team rows and aligns pots with teams that actually appear in the results.

### 3) Make tables (CSV + HTML)
python src/03_make_tables.py

- Writes **CSV** projections to `data/out/`.
- Writes **HTML** pages to `data/html/` (sortable tables, small legends, and a cross-link list):
  - `index.html` (entry page)
  - `projections_ucl_2025-26.html` (**main adjusted table**; `F_raw` highlighted)
  - `baselines_ucl_2024-25.html`
  - `buckets_ucl_2025-26.html`
  - `teamgames_ucl_2024-25.html`
  - `teamgames_ucl_2025-26.html`

## Interpreting the numbers

- **expected_pts_so_far**: baseline points for the *exact* Pot×Venue buckets a team has already played.
- **over_under**: actual minus expected (positive = overperformed vs schedule).
- **F_raw**: form factor = actual / expected (1.0 = on baseline).
- **Projections**: simple add-ons to explore “what if current form persisted.” They’re capped at **3 per game**, and the tempered version winsorizes and shrinks F_raw slightly.

### Example question (PSG after MD3)
“Why is PSG’s **expected_pts_so_far** low?”  
Because expected points measure **schedule difficulty**, not team quality. If a team opened with the toughest buckets (e.g., P1 away), their expected baseline is low. If they scored many points from that run, they’ll sit **well above** expectation.

## Design choices & alternatives

- **Use pots, not per-opponent Elo/coefficients** for the main table.  
  Rationale: aligned with competition design, neutral, and comparable. Elo/coefficients add nuance but also complexity and subjectivity.  
- **Single baseline season (2024–25)**.  
  It’s the only complete data under the new format (so far). Over time, the baseline can be expanded or smoothed.  
- **No per-opponent adjustment** in the main view.  
  Alternative views can scale by coefficients/Elo or include goal margins. Those are valid but serve different questions.

## Limitations

- Baseline season is one year; bucket means can be noisy.  
- Pots are imperfect proxies for actual strength.  
- Projections are illustrative, not forecasts; they reflect current over/under vs schedule, not expected team strength.

## Roadmap
- More seasons in the baseline once available.
- Optional opponent-strength variants (coefficients/Elo).
- Europa League / Conference League support (structure is already in place).
- Small visual improvements.

## Repo structure
data/
raw/ # scraped CSV + HTML cache
out/ # helper and result CSVs
html/ # published HTML tables
src/
01_scrape.py
02_build_helpers.py
03_make_tables.py


## License and data sources
- Match and pot info parsed from Wikipedia league-phase pages (see `source_url` columns).
- Use at your own risk; check Wikipedia/UEFA pages for authoritative data and terms.
