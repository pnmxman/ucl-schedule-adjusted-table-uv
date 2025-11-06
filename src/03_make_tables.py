#!/usr/bin/env python3
# src/03_make_tables.py
"""
Step 3 — Build schedule-adjusted table (main output) + projections (bonus), export CSV + rich HTML.

- Validates inputs from Step 2 (data/out).
- Computes the schedule-adjusted table SO FAR (Actual vs Expected by Pot×Venue).
- Adds two projection variants as a bonus (carry-forward F and winsor+shrink).
- Exports CSV to data/out/.
- Exports ALL HTML to data/html/ with:
  • sortable tables
  • concise legends (only where needed)
  • cross-page link list (projections page marked "(main adjusted table)")
  • F_raw column styled red in the projections page

Run:
  python src/03_make_tables.py
"""

import os
import sys
import numpy as np
import pandas as pd
import html
from datetime import datetime, timezone

# -----------------------------
# Config
# -----------------------------
COMPETITION = "ucl"
BASELINE_SEASON = "2024-25"
CURRENT_SEASON = "2025-26"

OUT_DIR = os.path.join("data", "out")  # CSV inputs/outputs
HTML_DIR = os.path.join("data", "html")  # ALL HTML here


def fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ------------- HTML helpers -------------
def df_to_sortable_html(
    df: pd.DataFrame,
    title: str,
    legend_items=None,
    links=None,
    active_slug=None,
    highlight_col_name: str | None = None,
) -> str:
    """Render a sortable table + optional legend + cross-link list.
    If highlight_col_name is given, color that column's cells red.
    """
    cols = list(df.columns)

    # Build nav links
    nav_html = ""
    if links:
        nav_html = ["<nav><h2>Pages</h2><ul>"]
        for slug, label, fname, is_main in links:
            annot = " <strong>(main adjusted table)</strong>" if is_main else ""
            active = " style='font-weight:bold;'" if slug == active_slug else ""
            nav_html.append(
                f"<li{active}><a href='{html.escape(fname)}'>{html.escape(label)}</a>{annot}</li>"
            )
        nav_html.append("</ul></nav>")
        nav_html = "\n".join(nav_html)

    # Legends (only for non-obvious tables)
    leg_html = ""
    if legend_items:
        leg_html = "<section class='legend'><h2>Legend</h2><ul>"
        for item in legend_items:
            leg_html += f"<li>{html.escape(item)}</li>"
        leg_html += "</ul></section>"

    # Column index to highlight (if any)
    highlight_col_idx = (
        cols.index(highlight_col_name) if (highlight_col_name in cols) else None
    )
    highlight_idx_js = "null" if highlight_col_idx is None else str(highlight_col_idx)

    head = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial, Segoe UI, Helvetica, sans-serif; margin:20px}}
table{{border-collapse:collapse; width:100%}}
th,td{{border:1px solid #ccc; padding:6px 8px; text-align:right}}
th{{cursor:pointer; background:#f7f7f7; text-align:center}}
td:first-child, th:first-child{{text-align:left}}
tr:nth-child(even){{background:#fafafa}}
.asc::after{{content:" ▲"}} .desc::after{{content:" ▼"}}
legend, .legend h2{{margin-top:24px}}
nav h2{{margin-top:24px}}
footer{{margin-top:24px; font-size:0.9em; color:#555}}
/* red emphasis for F_raw cells (applied by JS) */
td.emph{{color:#b00020; font-weight:600}}
</style>
<script>
function sortTable(n){{
 var t=document.getElementById('tbl'); var rows=Array.from(t.tBodies[0].rows);
 var ths=t.tHead.rows[0].cells; var dir=t.getAttribute('data-dir')||'asc';
 var last=parseInt(t.getAttribute('data-col')||'-1');
 dir=(last===n && dir==='asc')?'desc':'asc';
 for(var i=0;i<ths.length;i++){{ths[i].classList.remove('asc');ths[i].classList.remove('desc');}}
 ths[n].classList.add(dir);
 rows.sort(function(a,b){{
   var A=a.cells[n].innerText.trim(); var B=b.cells[n].innerText.trim();
   var x=parseFloat(A.replace(',','.')); var y=parseFloat(B.replace(',','.'));
   var na=isNaN(x), nb=isNaN(y);
   return (na||nb? (A>B?1:-1) : (x>y?1:(x<y?-1:0))) * (dir==='asc'?1:-1);
 }});
 rows.forEach(r=>t.tBodies[0].appendChild(r));
 t.setAttribute('data-col',n); t.setAttribute('data-dir',dir);
}}
function highlightColumnByIndex(idx){{
  if(idx==null) return;
  var t=document.getElementById('tbl'); var rows=t.tBodies[0].rows;
  for(var i=0;i<rows.length;i++){{ if(rows[i].cells[idx]) rows[i].cells[idx].classList.add('emph'); }}
}}
document.addEventListener('DOMContentLoaded', function(){{
  highlightColumnByIndex({highlight_idx_js});
}});
</script></head><body>
<h1>{html.escape(title)}</h1>
<table id="tbl"><thead><tr>"""
    for i, c in enumerate(cols):
        head += f'<th onclick="sortTable({i})">{html.escape(str(c))}</th>'
    head += "</tr></thead><tbody>"
    body = ""
    for _, row in df.iterrows():
        body += (
            "<tr>"
            + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols)
            + "</tr>"
        )
    tail = f"""</tbody></table>
{leg_html}
{nav_html}
<footer>Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</footer>
</body></html>"""
    return head + body + tail


# ------------- Load inputs -------------
def load_inputs():
    paths = {
        "baselines": os.path.join(
            OUT_DIR, f"baselines_{COMPETITION}_{BASELINE_SEASON}.csv"
        ),
        "team_prev": os.path.join(
            OUT_DIR, f"teamgames_{COMPETITION}_{BASELINE_SEASON}.csv"
        ),
        "team_curr": os.path.join(
            OUT_DIR, f"teamgames_{COMPETITION}_{CURRENT_SEASON}.csv"
        ),
        "buckets": os.path.join(
            OUT_DIR, f"buckets_played_{COMPETITION}_{CURRENT_SEASON}.csv"
        ),
        "future": os.path.join(
            OUT_DIR, f"future_slots_{COMPETITION}_{CURRENT_SEASON}.csv"
        ),  # read only; no HTML emitted
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            fail(f"Missing required input: {p}")
    bl = pd.read_csv(paths["baselines"])
    tp = pd.read_csv(paths["team_prev"])
    tc = pd.read_csv(paths["team_curr"])
    bk = pd.read_csv(paths["buckets"])
    fs = pd.read_csv(
        paths["future"]
    )  # kept for completeness, but we don't render an HTML page now

    # quick schema checks
    need_bl = {"venue", "opponent_pot", "avg_points", "matches"}
    need_tp = {
        "season",
        "team",
        "venue",
        "opponent",
        "opponent_pot",
        "points",
        "matchday",
    }
    need_tc = {
        "season",
        "team",
        "venue",
        "opponent",
        "opponent_pot",
        "points",
        "matchday",
    }
    need_bk = {"team", "venue", "opponent_pot", "count", "remaining"}
    need_fs = {"Team", "Slot", "Venue", "OpponentPot", "BaselineEP_per_game"}
    if not need_bl.issubset(bl.columns):
        fail("Baselines schema mismatch.")
    if not need_tp.issubset(tp.columns):
        fail("Teamgames (prev) schema mismatch.")
    if not need_tc.issubset(tc.columns):
        fail("Teamgames (current) schema mismatch.")
    if not need_bk.issubset(bk.columns):
        fail("Buckets schema mismatch.")
    if not need_fs.issubset(fs.columns):
        fail("Remaining slots (future_slots) schema mismatch.")

    return bl, tp, tc, bk, fs


# ------------- Projections -------------
def projection_tables(
    bl: pd.DataFrame, tc: pd.DataFrame, bk: pd.DataFrame
) -> pd.DataFrame:
    bmap = bl.set_index(["venue", "opponent_pot"])["avg_points"]

    # Actual so far
    actual = tc.groupby("team")["points"].sum().to_frame("actual_pts")

    # Expected so far (by buckets played)
    counts = (
        tc.groupby(["team", "venue", "opponent_pot"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["exp_per_game"] = counts.apply(
        lambda r: bmap.get((r["venue"], int(r["opponent_pot"]))), axis=1
    )
    counts["exp_pts"] = counts["count"] * counts["exp_per_game"]
    expected = counts.groupby("team")["exp_pts"].sum().to_frame("expected_pts_so_far")

    tbl = actual.join(expected, how="outer").fillna(0.0)
    tbl["over_under"] = tbl["actual_pts"] - tbl["expected_pts_so_far"]
    tbl["F_raw"] = np.where(
        tbl["expected_pts_so_far"] > 0,
        tbl["actual_pts"] / tbl["expected_pts_so_far"],
        1.0,
    )

    # Remaining baseline
    bk2 = bk.copy()
    bk2["baseline_per_game"] = bk2.apply(
        lambda r: bmap.get((r["venue"], int(r["opponent_pot"]))), axis=1
    )
    rem_base = (
        bk2.assign(rem_pts=lambda d: d["remaining"] * d["baseline_per_game"])
        .groupby("team")["rem_pts"]
        .sum()
    )
    tbl["expected_pts_remaining_baseline"] = tbl.index.map(rem_base.to_dict()).fillna(
        0.0
    )
    tbl["Projection_baseline"] = (
        tbl["actual_pts"] + tbl["expected_pts_remaining_baseline"]
    )

    # Remaining with F (carry-forward), cap per-game at 3
    def rem_with_F(F_series, winsor=None, shrink=None, cap=3.0):
        out = {}
        for team, sub in bk2.groupby("team"):
            F = F_series.get(team, 1.0)
            if winsor is not None:
                F = max(winsor[0], min(winsor[1], F))
            if shrink is not None:
                F = 1.0 + shrink * (F - 1.0)
            total = 0.0
            for _, r in sub.iterrows():
                if r["remaining"] <= 0:
                    continue
                base = r["baseline_per_game"]
                if pd.isna(base):
                    continue
                per_game = min(cap, base * F)
                total += per_game * r["remaining"]
            out[team] = total
        return pd.Series(out)

    tbl["Projection_F"] = tbl["actual_pts"] + rem_with_F(tbl["F_raw"])

    # Winsor + Shrink (shrink weight = games played / 8)
    games_played = tc.groupby("team").size()
    shrink_weight = (games_played / 8.0).reindex(tbl.index).fillna(0.0)
    proj_ws = []
    for team in tbl.index:
        F = tbl.at[team, "F_raw"]
        F = max(2 / 3, min(1.5, F))  # winsorize
        F = 1.0 + shrink_weight.get(team, 0.0) * (F - 1.0)  # shrink
        total = 0.0
        for _, r in bk2[bk2["team"] == team].iterrows():
            if r["remaining"] <= 0:
                continue
            base = r["baseline_per_game"]
            if pd.isna(base):
                continue
            per_game = min(3.0, base * F)
            total += per_game * r["remaining"]
        proj_ws.append(total)
    tbl["Projection_F_winsor_shrink"] = tbl["actual_pts"] + pd.Series(
        proj_ws, index=tbl.index
    )

    # Order + sort by F_raw (desc)
    tbl_out = tbl.reset_index().rename(columns={"index": "team"})
    tbl_out = (
        tbl_out[
            [
                "team",
                "actual_pts",
                "expected_pts_so_far",
                "over_under",
                "F_raw",
                "expected_pts_remaining_baseline",
                "Projection_baseline",
                "Projection_F",
                "Projection_F_winsor_shrink",
            ]
        ]
        .copy()
        .round(3)
    )
    tbl_out = tbl_out.sort_values("F_raw", ascending=False).reset_index(drop=True)
    return tbl_out


# ------------- Legends & links -------------
def legend_for_projections():
    return [
        "You can re-sort the table by clicking the column headers.",
        "[expected_pts_so_far] — reflects schedule difficulty faced so far: lower value → tougher opponents; higher value → easier opponents.",
        "[over_under] — actual_pts minus expected_pts_so_far (how many more/less points a team has earned compared to an average team vs the same schedule).",
        "[F_raw] — actual_pts relative to expected_pts_so_far (table is sorted by this column).",
        "[expected_pts_remaining_baseline] — reflects schedule difficulty yet to be played.",
        "[Projection_baseline] — actual_pts + expected_pts_remaining_baseline (remaining games at baseline, not adjusted for form).",
        "[Projection_F] — remaining games scaled by F_raw (per-game cap 3).",
        "[Projection_F_winsor_shrink] — like Projection_F but tempered: F_raw winsorized to [0.67, 1.50] and shrunk toward 1 with season progress; per-game cap 3.",
    ]


def legend_for_baselines():
    return [
        "[avg_points] — average points per game in the baseline season for this (venue, opponent_pot) bucket.",
        "[matches] — number of team-games contributing to this average.",
        "[std] — standard deviation of points across those matches.",
        "[se] — standard error of the mean (std / sqrt(matches)).",
    ]


def legend_for_buckets():
    return [
        "[count] — how many times this team has already played this (venue, opponent_pot) bucket (0 or 1).",
        "[remaining] — 1 if this bucket is still to be played; otherwise 0.",
    ]


def make_links():
    # All links relative to data/html/
    return [
        (
            "projections",
            f"Projections {COMPETITION.upper()} {CURRENT_SEASON}",
            f"projections_{COMPETITION}_{CURRENT_SEASON}.html",
            True,
        ),
        (
            "baselines",
            f"Baselines {COMPETITION.upper()} {BASELINE_SEASON}",
            f"baselines_{COMPETITION}_{BASELINE_SEASON}.html",
            False,
        ),
        (
            "buckets",
            f"Buckets {COMPETITION.upper()} {CURRENT_SEASON}",
            f"buckets_{COMPETITION}_{CURRENT_SEASON}.html",
            False,
        ),
        (
            "team_prev",
            f"Teamgames {COMPETITION.upper()} {BASELINE_SEASON}",
            f"teamgames_{COMPETITION}_{BASELINE_SEASON}.html",
            False,
        ),
        (
            "team_curr",
            f"Teamgames {COMPETITION.upper()} {CURRENT_SEASON}",
            f"teamgames_{COMPETITION}_{CURRENT_SEASON}.html",
            False,
        ),
        # removed: remaining slots HTML
    ]


def write_index_html(links):
    desc = (
        "This project’s main output is a schedule-adjusted table that reflects the different "
        "difficulty of opponents faced so far.<br/>"
        "We compare each team’s actual points against an expected points value for the difficulty "
        "of the matches (opponent pot and venue: home or away).<br/>"
        "Projections show end-of-phase totals based solely on results to date. "
        "(A legitimate prognosis would require additional information and interpretation.)"
    )
    items = []
    for slug, label, fname, is_main in links:
        annot = " <strong>(main adjusted table)</strong>" if is_main else ""
        items.append(
            f"<li><a href='{html.escape(fname)}'>{html.escape(label)}</a>{annot}</li>"
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Schedule-adjusted table</title>
<style>
body{{font-family:Arial, Segoe UI, Helvetica, sans-serif; margin:20px}}
li strong{{color:#333}}
</style></head><body>
<h1>Schedule-adjusted table</h1>
<p>{desc}</p>
<h2>Pages</h2>
<ul>
{"".join(items)}
</ul>
<p><small>Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</small></p>
</body></html>"""
    with open(os.path.join(HTML_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_text)


# ------------- Main -------------
def main():
    ensure_dirs([OUT_DIR, HTML_DIR])

    # Inputs from Step 2
    bl, tp, tc, bk, fs = load_inputs()

    # Compute main table + bonus projections
    proj = projection_tables(bl, tc, bk)

    # ---- CSV stays in data/out
    proj_csv = os.path.join(OUT_DIR, f"projections_{COMPETITION}_{CURRENT_SEASON}.csv")
    proj.to_csv(proj_csv, index=False)

    # ---- Build link list (for footers and index)
    links = make_links()

    # ---- HTML to data/html (no "remaining slots" page)
    pages = [
        (
            "projections",
            f"Schedule-adjusted table (with end-of-phase projections) — {COMPETITION.upper()} {CURRENT_SEASON}",
            proj,
            legend_for_projections(),
            f"projections_{COMPETITION}_{CURRENT_SEASON}.html",
            "F_raw",
        ),
        (
            "baselines",
            f"Baselines — {COMPETITION.upper()} {BASELINE_SEASON}",
            bl,
            legend_for_baselines(),
            f"baselines_{COMPETITION}_{BASELINE_SEASON}.html",
            None,
        ),
        (
            "buckets",
            f"Buckets played — {COMPETITION.upper()} {CURRENT_SEASON}",
            bk,
            legend_for_buckets(),
            f"buckets_{COMPETITION}_{CURRENT_SEASON}.html",
            None,
        ),
        (
            "team_prev",
            f"Teamgames — {COMPETITION.upper()} {BASELINE_SEASON}",
            tp,
            None,
            f"teamgames_{COMPETITION}_{BASELINE_SEASON}.html",
            None,
        ),  # legend removed
        (
            "team_curr",
            f"Teamgames — {COMPETITION.upper()} {CURRENT_SEASON}",
            tc,
            None,
            f"teamgames_{COMPETITION}_{CURRENT_SEASON}.html",
            None,
        ),  # legend removed
    ]

    for slug, title, df, legend, fname, highlight in pages:
        path = os.path.join(HTML_DIR, fname)
        html_text = df_to_sortable_html(
            df,
            title,
            legend_items=legend,
            links=links,
            active_slug=slug,
            highlight_col_name=highlight,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_text)

    # index.html
    write_index_html(links)

    print("OK — outputs written:")
    print(f"  CSV: {proj_csv}")
    print(f"  HTML directory: {HTML_DIR}")
    print("  Created: index.html and 5 page HTMLs (projections + helpers).")


if __name__ == "__main__":
    main()
