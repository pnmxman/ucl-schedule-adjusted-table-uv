#!/usr/bin/env python3
# src/01_scrape.py
"""
Step 1 – Scrape Wikipedia league-phase inputs (pots + results) and save CSVs.

No CLI args. Edit the constants below.

Run:
  python src/01_scrape.py

Outputs:
  data/raw/pots_{comp}_{season}.csv
  data/raw/results_{comp}_{season}.csv
Also writes raw HTML cache to data/raw/_cache/ for debugging.
"""

import os, re, time, random
from io import StringIO
from typing import Tuple
from urllib.parse import quote

import requests
import pandas as pd
from bs4 import BeautifulSoup

# =========================
# EDIT THESE CONSTANTS
# =========================
COMPETITIONS = ["ucl"]  # ucl, uel, uecl
SEASONS = ["2024-25", "2025-26"]  # add/remove as needed

# --- Cache behavior ---
# If True, always refetch (ignore cache) for ALL pages (pots + results).
FORCE_REFRESH = False
# If not forcing, refetch when the cached HTML is older than this many seconds.
# Example: 6 hours = 6*3600; 1 day = 24*3600
CACHE_MAX_AGE_SECONDS = 6 * 3600

# You can also refresh only results on every run (pots are relatively stable).
ALWAYS_REFRESH_RESULTS = True  # set True if you want to ignore cache only for results

USE_UEFA_FALLBACK = False  # set True only if you implement the parser below
VERBOSE = True

DATA_DIR_RAW = os.path.join("data", "raw")
CACHE_DIR = os.path.join(DATA_DIR_RAW, "_cache")

COMPETITION_META = {
    "ucl": {"name": "UEFA Champions League", "slug": "UEFA_Champions_League"},
    "uel": {"name": "UEFA Europa League", "slug": "UEFA_Europa_League"},
    "uecl": {
        "name": "UEFA Europa Conference League",
        "slug": "UEFA_Europa_Conference_League",
    },
}

EN_DASH = "\u2013"  # Wikipedia season dash


# --- HTTP session with proper headers & polite delays ---
def make_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "ucl-schedule-adjusted/1.0 (GitHub; contact: youremail@example.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return s


def backoff_sleep(i):
    time.sleep(min(8, (2**i)) + random.random())


def log(*args):
    if VERBOSE:
        print(*args)


# --- Wiki URL helpers ---
def page_title(season: str, comp_slug: str) -> str:
    season_title = season.replace("-", EN_DASH)
    return f"{season_title}_{comp_slug}_league_phase"


def rest_html_url(title: str) -> str:
    return f"https://en.wikipedia.org/api/rest_v1/page/html/{quote(title)}"


def web_html_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title)}"


# --- Cache helpers ---
def cache_is_fresh(path: str, max_age_seconds: int) -> bool:
    """Return True if cache file exists and is newer than TTL."""
    if not os.path.exists(path):
        return False
    if max_age_seconds is None or max_age_seconds <= 0:
        # TTL disabled -> treat cache as always fresh unless FORCE_REFRESH
        return True
    mtime = os.path.getmtime(path)
    age = time.time() - mtime
    return age <= max_age_seconds


# --- Fetch with fallback: REST → web ---
def fetch_html(
    session: requests.Session, title: str, cache_key: str, *, ignore_cache: bool = False
) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.html")

    # Use cache if allowed and fresh
    if (
        not FORCE_REFRESH
        and not ignore_cache
        and cache_is_fresh(cache_path, CACHE_MAX_AGE_SECONDS)
    ):
        with open(cache_path, "r", encoding="utf-8") as f:
            html = f.read()
        log(f"  [cache] {cache_key}.html (fresh)")
        return html

    # Otherwise refetch
    urls = [(rest_html_url(title), "REST"), (web_html_url(title), "WEB")]
    last_err = None
    for url, kind in urls:
        for attempt in range(3):
            try:
                log(f"  [{kind}] GET {url}")
                r = session.get(url, timeout=30)
                if r.status_code == 429:
                    log("   429 Too Many Requests; backing off…")
                    backoff_sleep(attempt)
                    continue
                r.raise_for_status()
                html = r.text
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(html)
                time.sleep(0.7 + random.random() * 0.5)
                return html
            except Exception as e:
                last_err = e
                log(f"   ERROR: {e}")
                backoff_sleep(attempt)
    raise RuntimeError(f"Failed to fetch {title}: {last_err}")


# --- Common helpers ---
def clean_text(x: str) -> str:
    x = re.sub(r"\[\d+\]", "", str(x))
    x = x.replace("\xa0", " ").strip()
    return x


def to_float(x: str):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def parse_score(s: str) -> Tuple[int, int]:
    s = s.replace("–", "-").replace("—", "-")
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# --- POTS parsing (tables near “Teams and seeding”) ---
def parse_pots_html(html: str, comp: str, season: str, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    # Locate the Teams / Teams and seeding section
    header = None
    for tag in soup.find_all(["h2", "h3"]):
        txt = (tag.get_text(" ", strip=True) or "").lower()
        if "teams" in txt and (
            "seeding" in txt or "teams and seeding" in txt or "teams" in txt
        ):
            header = tag
            break

    if header is None:
        tables = soup.select("table.wikitable")
    else:
        tables = []
        for sib in header.find_all_next():
            if sib.name and sib.name in ("h2", "h3") and sib is not header:
                break
            if sib.name == "table" and "wikitable" in (sib.get("class") or []):
                tables.append(sib)

    rows = []
    for t in tables:
        try:
            df = pd.read_html(StringIO(str(t)))[0]
        except Exception:
            continue
        df.columns = [clean_text(c).lower() for c in df.columns]
        df = df.map(clean_text)

        team_col = "team" if "team" in df.columns else df.columns[0]
        coeff_col = next((c for c in df.columns if "coeff" in c), None)

        pot_hint = None
        cap = t.find("caption")
        if cap:
            m = re.search(
                r"pot\s*(\d+)", clean_text(cap.get_text(" ", strip=True)).lower()
            )
            if m:
                pot_hint = int(m.group(1))

        for _, row in df.iterrows():
            team = clean_text(row.get(team_col, ""))
            if not team or team.lower() in ("team", "teams"):
                continue
            if re.search(r"pot\s*\d+", team.lower()):
                continue

            pot = None
            if "pot" in df.columns and str(row.get("pot", "")).strip().isdigit():
                pot = int(row["pot"])
            elif pot_hint:
                pot = pot_hint
            else:
                for v in row.values:
                    m = re.search(r"pot\s*(\d+)", str(v).lower())
                    if m:
                        pot = int(m.group(1))
                        break

            coeff = to_float(row.get(coeff_col, "")) if coeff_col else None
            rows.append(
                {
                    "competition": comp,
                    "season": season,
                    "team": team,
                    "pot": pot if pot in (1, 2, 3, 4) else None,
                    "uefa_coefficient": coeff,
                    "source_url": source_url,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        log("  [pots] WARNING: no rows parsed from HTML")
        return pd.DataFrame(
            columns=[
                "competition",
                "season",
                "team",
                "pot",
                "uefa_coefficient",
                "source_url",
            ]
        )
    out = out.dropna(subset=["team"]).drop_duplicates(
        subset=["competition", "season", "team"], keep="first"
    )
    log(f"  [pots] parsed {len(out)} rows")
    return out


# --- RESULTS parsing (footballbox strict) ---
def parse_results_html(
    html: str, comp: str, season: str, source_url: str
) -> pd.DataFrame:
    """
    Parse results from Wikipedia REST HTML using the footballbox template.

    It collects ONLY matches that have a played score (e.g. '2–1') and
    pulls home/away names from the exact fhome/faway cells.
    """
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    current_md = None  # Matchday N tracker

    for el in soup.find_all(True, recursive=True):
        if el.name in ("h2", "h3") and el.has_attr("id"):
            m = re.match(r"Matchday_(\d+)", el["id"], flags=re.I)
            if m:
                try:
                    current_md = int(m.group(1))
                except Exception:
                    current_md = None

        if el.name == "div" and "footballbox" in el.get("class", []):
            name_row = el.select_one('table.fevent tr[itemprop="name"]')
            if not name_row:
                continue

            home_cell = name_row.select_one("th.fhome [itemprop=name]")
            away_cell = name_row.select_one("th.faway [itemprop=name]")
            score_cell = name_row.select_one("th.fscore")

            home = (
                clean_text(home_cell.get_text(" ", strip=True)) if home_cell else None
            )
            away = (
                clean_text(away_cell.get_text(" ", strip=True)) if away_cell else None
            )
            score = (
                clean_text(score_cell.get_text(" ", strip=True)) if score_cell else ""
            )

            hg, ag = parse_score(score)
            if home and away and hg is not None and ag is not None:
                rows.append(
                    {
                        "competition": comp,
                        "season": season,
                        "matchday": current_md,  # may be None if the page omitted the heading; OK
                        "home_team": home,
                        "away_team": away,
                        "home_goals": hg,
                        "away_goals": ag,
                        "source_url": source_url,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        log(
            "  [results] WARNING: no rows parsed from footballbox (no completed matches found)."
        )
        return pd.DataFrame(
            columns=[
                "competition",
                "season",
                "matchday",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "source_url",
            ]
        )

    out = out.drop_duplicates(
        subset=[
            "competition",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    )
    log(f"  [results] parsed {len(out)} rows (footballbox strict)")
    return out


# --- UEFA fallback stub (optional) ---
def scrape_results_from_uefa_stub(comp: str, season: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "competition",
            "season",
            "matchday",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "source_url",
        ]
    )


# --- MAIN ---
def run():
    os.makedirs(DATA_DIR_RAW, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    session = make_session()

    for comp in COMPETITIONS:
        comp_name = COMPETITION_META.get(comp, {}).get("name", comp)
        comp_slug = COMPETITION_META[comp]["slug"]
        for season in SEASONS:
            print(f"\n=== {comp.upper()} · {comp_name} · {season} ===")
            title = page_title(season, comp_slug)

            # --- POTS ---
            try:
                html = fetch_html(
                    session,
                    title,
                    cache_key=f"{comp}_{season}_pots",
                    ignore_cache=False,  # we rarely need to refetch pots mid-season
                )
                pots = parse_pots_html(
                    html, comp, season, source_url=web_html_url(title)
                )
            except Exception as e:
                print(f"  [pots] ERROR: {e}")
                pots = pd.DataFrame(
                    columns=[
                        "competition",
                        "season",
                        "team",
                        "pot",
                        "uefa_coefficient",
                        "source_url",
                    ]
                )
            pots_path = os.path.join(DATA_DIR_RAW, f"pots_{comp}_{season}.csv")
            pots.to_csv(pots_path, index=False)
            print(f"  -> wrote {pots_path} ({len(pots)} rows)")

            # --- RESULTS ---
            try:
                html = fetch_html(
                    session,
                    title,
                    cache_key=f"{comp}_{season}_results",
                    ignore_cache=(ALWAYS_REFRESH_RESULTS or False),
                )
                results = parse_results_html(
                    html, comp, season, source_url=web_html_url(title)
                )
            except Exception as e:
                print(f"  [results] ERROR: {e}")
                results = pd.DataFrame(
                    columns=[
                        "competition",
                        "season",
                        "matchday",
                        "home_team",
                        "away_team",
                        "home_goals",
                        "away_goals",
                        "source_url",
                    ]
                )
            if results.empty and USE_UEFA_FALLBACK:
                print("  Wikipedia results empty; trying UEFA fallback stub.")
                results = scrape_results_from_uefa_stub(comp, season)

            res_path = os.path.join(DATA_DIR_RAW, f"results_{comp}_{season}.csv")
            results.to_csv(res_path, index=False)
            print(f"  -> wrote {res_path} ({len(results)} rows)")

    print("\nDone. Raw CSVs and HTML cache are in data/raw/")
    print(
        "Cache policy: FORCE_REFRESH=%s, CACHE_MAX_AGE_SECONDS=%s, ALWAYS_REFRESH_RESULTS=%s"
        % (FORCE_REFRESH, CACHE_MAX_AGE_SECONDS, ALWAYS_REFRESH_RESULTS)
    )
    print(
        "If parsing returns 0 rows, open the cached HTML files in data/raw/_cache/ and adjust selectors."
    )


if __name__ == "__main__":
    run()
