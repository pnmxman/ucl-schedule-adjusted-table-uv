#!/usr/bin/env python3
# src/02_build_helpers.py
"""
Step 2 — Build helper CSVs from raw scraped data.
- Validates input CSVs (schema + quick sanity checks)
- Cleans pots tables (remove non-team rows, align to teams present in results)
- Explodes match rows to team-games
- Computes baseline (venue × opponent_pot) from 2024-25
- Builds 2025-26 buckets played and remaining slots

Run:
  python src/02_build_helpers.py

Inputs (from data/raw/):
  pots_ucl_2024-25.csv
  results_ucl_2024-25.csv
  pots_ucl_2025-26.csv
  results_ucl_2025-26.csv

Outputs (to data/out/):
  baselines_ucl_2024-25.csv
  teamgames_ucl_2024-25.csv
  teamgames_ucl_2025-26.csv
  buckets_played_ucl_2025-26.csv
  future_slots_ucl_2025-26.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# ===============================
# EDIT THESE IF YOU WANT
# ===============================
COMPETITION = "ucl"
BASELINE_SEASON = "2024-25"
CURRENT_SEASON = "2025-26"

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "out")

# ===============================
# Utilities (inline; no external imports)
# ===============================
# NAME_MAP for normalization might be incomplete. mechanism to identify missing aliases only identifies which team in results is not reperesented in other data. So it does not hand out the missing alias name. This should be modified for next season.
# ===============================
NAME_MAP = {
    "AFC Ajax": "Ajax",
    "Bayern München": "Bayern Munich",
    "Athletic Club": "Athletic Bilbao",
    "Crvena Zvezda": "Red Star Belgrade",
    "GNK Dinamo": "Dinamo Zagreb",
    "Leverkusen": "Bayer Leverkusen",
    "Leipzig": "RB Leipzig",
    "Inter": "Inter Milan",
    "PSV": "PSV Eindhoven",
    "Tottenham": "Tottenham Hotspur",
    "København": "Copenhagen",
    "Kobenhavn": "Copenhagen",
    "Union Saint": "Union Saint-Gilloise",
    "Bodo/Glimt": "Bodø/Glimt",
    "Bodø / Glimt": "Bodø/Glimt",
    "Bodø- Glimt": "Bodø/Glimt",
    "Qarabağ": "Qarabag",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "FC Bayern Munich": "Bayern Munich",
    "RB Salzburg": "Red Bull Salzburg",
    "AC Sparta Prague": "Sparta Prague",
    "Sparta Praha": "Sparta Prague",
    "Paris SG": "Paris Saint-Germain",
    "Paris Saint- Germain": "Paris Saint-Germain",
    "AC Milan": "Milan",
    "A.C. Milan": "Milan",
    "Atletico Madrid": "Atlético Madrid",
    "Atlético de Madrid": "Atlético Madrid",
    "B. Dortmund": "Borussia Dortmund",
    "Man City": "Manchester City",
    "Manchester City FC": "Manchester City",
    "Lille": "LOSC Lille",
    "LOSC": "LOSC Lille",
    "Kairat Almaty": "Kairat",
    "Frankfurt": "Eintracht Frankfurt",
    "Slavia Praha": "Slavia Prague",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip().replace("\xa0", " ")
    for k, v in NAME_MAP.items():
        if k in s:
            if k == "Union Saint":  # ensure full official name
                return "Union Saint-Gilloise"
            return v
    return s


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ===============================
# Validation helpers
# ===============================
EXPECTED_RESULTS_COLS = [
    "competition",
    "season",
    "matchday",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "source_url",
]
EXPECTED_POTS_COLS = [
    "competition",
    "season",
    "team",
    "pot",
    "uefa_coefficient",
    "source_url",
]


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        fail(f"Missing required file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        fail(f"Failed to read {path}: {e}")


def validate_schema(df: pd.DataFrame, expected_cols: list, label: str):
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing:
        fail(f"{label}: missing expected columns: {missing}")
    if extra:
        print(f"[WARN] {label}: extra columns present (ignored by downstream): {extra}")


def validate_results_counts(df_prev: pd.DataFrame, df_curr: pd.DataFrame):
    # baseline year should be complete (144 matches) for UCL league phase
    prev_n = len(df_prev)
    if prev_n < 140:
        print(
            f"[WARN] Baseline season has {prev_n} matches (<144). Baselines may be slightly off if page is incomplete."
        )
    # current season should be non-zero
    curr_n = len(df_curr)
    if curr_n == 0:
        fail(
            "Current season results CSV has 0 rows. Step 1 likely failed to parse results."
        )


def coerce_results_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["matchday"] = pd.to_numeric(out["matchday"], errors="coerce").astype("Int64")
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce").astype(
        "Int64"
    )
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce").astype(
        "Int64"
    )
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    return out


def coerce_pots_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team"] = out["team"].astype(str)
    out["pot"] = pd.to_numeric(out["pot"], errors="coerce").astype("Int64")
    out["uefa_coefficient"] = pd.to_numeric(out["uefa_coefficient"], errors="coerce")
    return out


# === NEW: Clean pots (remove non-team rows and align to teams in results) ===
def clean_pots_df(df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-team rows from a pots table and align to teams that actually appear in results."""
    out = df.copy()
    out["team"] = out["team"].astype(str).str.strip()

    # Rule 1: obvious non-team rows (captions/headers/notes that slipped in)
    bad = out["team"].str.len() < 2
    bad |= out["team"].str.match(r"^\d+$", na=False)  # pure numbers
    bad |= out["team"].str.contains(r"\bpot\s*\d", case=False, na=False)
    bad |= out["team"].str.contains(
        r"title|holder|winner|coefficient|associat|seed|note|path|qualif|teams?\s+and|seeding",
        case=False,
        na=False,
    )
    # very long strings (likely notes)
    bad |= out["team"].str.len() > 60

    out = out[~bad].copy()

    # Rule 2: keep only clubs that appear in results (normalized)
    teams_in_results = set()
    if not results_df.empty:
        teams_in_results |= set(results_df["home_team"].astype(str).map(normalize_name))
        teams_in_results |= set(results_df["away_team"].astype(str).map(normalize_name))

    if teams_in_results:
        out["team_norm"] = out["team"].map(normalize_name)
        out = out[out["team_norm"].isin(teams_in_results)].drop(
            columns=["team_norm"], errors="ignore"
        )

    # Final sanity: pot must be 1..4
    out = out[out["pot"].isin([1, 2, 3, 4])]
    # Drop exact duplicate teams (keep first)
    out = out.drop_duplicates(subset=["team"], keep="first")

    return out.reset_index(drop=True)


def ensure_names_align(
    pots_df: pd.DataFrame, results_df: pd.DataFrame, season_label: str
):
    """
    Fail fast when a team name in results cannot be matched to any team in pots
    after normalization. This catches newly-appeared aliases that are not yet
    added to NAME_MAP and would otherwise lead to missing opponent_pot values.
    """
    pots_norm = set(pots_df["team"].astype(str).map(normalize_name))
    res_names = pd.concat([results_df["home_team"], results_df["away_team"]]).astype(
        str
    )
    res_norm_series = res_names.map(normalize_name)
    res_norm = set(res_norm_series)

    missing_in_pots = sorted(res_norm - pots_norm)
    if missing_in_pots:
        raw_examples = (
            res_names[res_norm_series.isin(missing_in_pots)].drop_duplicates().tolist()
        )
        fail(
            f"{season_label}: found team names in results that are not present in pots "
            f"after normalization: {missing_in_pots}. Likely a new alias; please add it "
            f"to NAME_MAP. Raw examples: {raw_examples[:10]}"
        )


# ===============================
# Core transforms
# ===============================
def explode_team_games(
    results_df: pd.DataFrame, pots_df: pd.DataFrame, season_label: str
) -> pd.DataFrame:
    pot_map = dict(zip(pots_df["team"].map(normalize_name), pots_df["pot"]))
    rows = []
    for _, r in results_df.iterrows():
        ht = normalize_name(str(r["home_team"]))
        at = normalize_name(str(r["away_team"]))
        # home perspective
        rows.append(
            {
                "season": season_label,
                "team": ht,
                "venue": "home",
                "opponent": at,
                "opponent_pot": pot_map.get(at, pd.NA),
                "points": int(
                    3
                    if r["home_goals"] > r["away_goals"]
                    else 1
                    if r["home_goals"] == r["away_goals"]
                    else 0
                ),
                "matchday": r["matchday"],
            }
        )
        # away perspective
        rows.append(
            {
                "season": season_label,
                "team": at,
                "venue": "away",
                "opponent": ht,
                "opponent_pot": pot_map.get(ht, pd.NA),
                "points": int(
                    3
                    if r["home_goals"] < r["away_goals"]
                    else 1
                    if r["home_goals"] == r["away_goals"]
                    else 0
                ),
                "matchday": r["matchday"],
            }
        )
    df = pd.DataFrame(rows)
    # type tidy
    df["opponent_pot"] = pd.to_numeric(df["opponent_pot"], errors="coerce").astype(
        "Int64"
    )
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    return df


def build_baselines(prev_team_games: pd.DataFrame) -> pd.DataFrame:
    g = prev_team_games.groupby(["venue", "opponent_pot"])["points"]
    out = g.agg(avg_points="mean", matches="count", std="std").reset_index()
    out["se"] = out["std"] / np.sqrt(out["matches"])
    # for neatness: sort Venue (home, away) then pot 1..4
    out["venue"] = out["venue"].astype(str)
    out = out.sort_values(["venue", "opponent_pot"]).reset_index(drop=True)
    return out


def compute_buckets_played(
    curr_team_games: pd.DataFrame, all_teams_norm: pd.Series
) -> pd.DataFrame:
    played = (
        curr_team_games.groupby(["team", "venue", "opponent_pot"])
        .size()
        .rename("count")
        .reset_index()
    )
    teams = sorted(all_teams_norm.unique().tolist())
    grid = pd.MultiIndex.from_product(
        [teams, ["home", "away"], [1, 2, 3, 4]], names=["team", "venue", "opponent_pot"]
    )
    played = (
        played.set_index(["team", "venue", "opponent_pot"])
        .reindex(grid)
        .fillna({"count": 0})
        .reset_index()
    )
    played["remaining"] = (1 - played["count"]).clip(0, 1)
    return played


def make_future_slots(played: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    bmap = baselines.set_index(["venue", "opponent_pot"])["avg_points"]
    played["baseline_ep_per_game"] = played.apply(
        lambda r: bmap.get((r["venue"], int(r["opponent_pot"])))
        if pd.notna(r["opponent_pot"])
        else np.nan,
        axis=1,
    )
    rem = played[played["remaining"] > 0].copy()
    rem["Venue"] = rem["venue"].str.title()
    rem["Slot"] = rem.groupby("team").cumcount() + 1
    out = (
        rem.rename(
            columns={
                "team": "Team",
                "opponent_pot": "OpponentPot",
                "baseline_ep_per_game": "BaselineEP_per_game",
            }
        )[["Team", "Slot", "Venue", "OpponentPot", "BaselineEP_per_game"]]
        .sort_values(["Team", "Slot"])
        .reset_index(drop=True)
    )
    return out


# ===============================
# Main
# ===============================
def main():
    ensure_dirs([OUT_DIR])

    # ---- Load & validate raw inputs
    pots_prev_path = os.path.join(RAW_DIR, f"pots_{COMPETITION}_{BASELINE_SEASON}.csv")
    res_prev_path = os.path.join(
        RAW_DIR, f"results_{COMPETITION}_{BASELINE_SEASON}.csv"
    )
    pots_curr_path = os.path.join(RAW_DIR, f"pots_{COMPETITION}_{CURRENT_SEASON}.csv")
    res_curr_path = os.path.join(RAW_DIR, f"results_{COMPETITION}_{CURRENT_SEASON}.csv")

    pots_prev = load_csv(pots_prev_path)
    res_prev = load_csv(res_prev_path)
    pots_curr = load_csv(pots_curr_path)
    res_curr = load_csv(res_curr_path)

    # Schema checks
    validate_schema(pots_prev, EXPECTED_POTS_COLS, f"pots {BASELINE_SEASON}")
    validate_schema(res_prev, EXPECTED_RESULTS_COLS, f"results {BASELINE_SEASON}")
    validate_schema(pots_curr, EXPECTED_POTS_COLS, f"pots {CURRENT_SEASON}")
    validate_schema(res_curr, EXPECTED_RESULTS_COLS, f"results {CURRENT_SEASON}")

    # Coerce dtypes
    pots_prev = coerce_pots_types(pots_prev)
    pots_curr = coerce_pots_types(pots_curr)
    res_prev = coerce_results_types(res_prev)
    res_curr = coerce_results_types(res_curr)

    # === NEW: Clean pots to drop non-team rows and align to teams in results ===
    pots_prev = clean_pots_df(pots_prev, res_prev)
    pots_curr = clean_pots_df(pots_curr, res_curr)

    # Ensure no unseen aliases slipped into results (must match pots after normalization)
    ensure_names_align(pots_prev, res_prev, BASELINE_SEASON)
    ensure_names_align(pots_curr, res_curr, CURRENT_SEASON)

    # Sanity checks (counts)
    validate_results_counts(res_prev, res_curr)

    # ---- Explode to team-games
    team_prev = explode_team_games(res_prev, pots_prev, BASELINE_SEASON)
    team_curr = explode_team_games(res_curr, pots_curr, CURRENT_SEASON)

    # ---- Baselines from previous season
    baselines = build_baselines(team_prev)

    # ---- Buckets played & future slots for current season
    all_teams_norm = pots_curr["team"].map(normalize_name)
    played = compute_buckets_played(team_curr, all_teams_norm)
    future_slots = make_future_slots(played, baselines)

    # ---- Write outputs
    baselines.to_csv(
        os.path.join(OUT_DIR, f"baselines_{COMPETITION}_{BASELINE_SEASON}.csv"),
        index=False,
    )
    team_prev.to_csv(
        os.path.join(OUT_DIR, f"teamgames_{COMPETITION}_{BASELINE_SEASON}.csv"),
        index=False,
    )
    team_curr.to_csv(
        os.path.join(OUT_DIR, f"teamgames_{COMPETITION}_{CURRENT_SEASON}.csv"),
        index=False,
    )
    played.to_csv(
        os.path.join(OUT_DIR, f"buckets_played_{COMPETITION}_{CURRENT_SEASON}.csv"),
        index=False,
    )
    future_slots.to_csv(
        os.path.join(OUT_DIR, f"future_slots_{COMPETITION}_{CURRENT_SEASON}.csv"),
        index=False,
    )

    # ---- Summary
    print("OK — helpers written to data/out/:")
    print(f"  baselines_{COMPETITION}_{BASELINE_SEASON}.csv   ({len(baselines)} rows)")
    print(f"  teamgames_{COMPETITION}_{BASELINE_SEASON}.csv   ({len(team_prev)} rows)")
    print(f"  teamgames_{COMPETITION}_{CURRENT_SEASON}.csv    ({len(team_curr)} rows)")
    print(f"  buckets_played_{COMPETITION}_{CURRENT_SEASON}.csv ({len(played)} rows)")
    print(
        f"  future_slots_{COMPETITION}_{CURRENT_SEASON}.csv ({len(future_slots)} rows)"
    )


if __name__ == "__main__":
    main()
