"""Descriptive EPL statistics from a normalized match DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd


def long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (team, opponent, venue, gf, ga, shots...)."""
    def side(is_home: bool) -> pd.DataFrame:
        p, o = ("h", "a") if is_home else ("a", "h")
        out = pd.DataFrame({
            "date": matches["date"],
            "team": matches["home" if is_home else "away"],
            "opponent": matches["away" if is_home else "home"],
            "venue": "H" if is_home else "A",
            "gf": matches[f"{p}g"],
            "ga": matches[f"{o}g"],
        })
        for stat in ("shots", "sot", "corners"):
            if f"{p}_{stat}" in matches.columns:
                out[f"{stat}_for"] = matches[f"{p}_{stat}"]
                out[f"{stat}_against"] = matches[f"{o}_{stat}"]
        return out

    df = pd.concat([side(True), side(False)], ignore_index=True)
    df["points"] = np.select([df.gf > df.ga, df.gf == df.ga], [3, 1], 0)
    df["result"] = np.select([df.gf > df.ga, df.gf == df.ga], ["W", "D"], "L")
    return df.sort_values(["date", "team"]).reset_index(drop=True)


def league_table(matches: pd.DataFrame, venue: str | None = None) -> pd.DataFrame:
    """Standings. ``venue`` = 'H' or 'A' for home/away-only tables."""
    lf = long_format(matches)
    if venue:
        lf = lf[lf.venue == venue]
    g = lf.groupby("team")
    table = pd.DataFrame({
        "P": g.size(),
        "W": g["result"].apply(lambda s: (s == "W").sum()),
        "D": g["result"].apply(lambda s: (s == "D").sum()),
        "L": g["result"].apply(lambda s: (s == "L").sum()),
        "GF": g["gf"].sum(),
        "GA": g["ga"].sum(),
    })
    table["GD"] = table.GF - table.GA
    table["Pts"] = 3 * table.W + table.D
    table = table.sort_values(["Pts", "GD", "GF"], ascending=False)
    table.insert(0, "Pos", range(1, len(table) + 1))
    return table.astype(int)


def form(matches: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    """Recent form: last-N results string, points, goals for/against per team."""
    lf = long_format(matches)
    rows = []
    for team, g in lf.groupby("team"):
        r = g.tail(last_n)
        rows.append({
            "team": team,
            "form": "".join(r.result),
            "pts": int(r.points.sum()),
            "ppg": round(r.points.mean(), 2),
            "gf": int(r.gf.sum()),
            "ga": int(r.ga.sum()),
        })
    return pd.DataFrame(rows).sort_values(["pts", "gf"], ascending=False).reset_index(drop=True)


def team_profiles(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-team averages and betting-relevant rates (O2.5, BTTS, clean sheets)."""
    lf = long_format(matches)
    lf["total"] = lf.gf + lf.ga
    g = lf.groupby("team")
    prof = pd.DataFrame({
        "matches": g.size(),
        "gf_pg": g.gf.mean(),
        "ga_pg": g.ga.mean(),
        "over25_pct": g.total.apply(lambda s: (s > 2.5).mean() * 100),
        "btts_pct": g.apply(lambda d: ((d.gf > 0) & (d.ga > 0)).mean() * 100, include_groups=False),
        "clean_sheet_pct": g.ga.apply(lambda s: (s == 0).mean() * 100),
        "failed_to_score_pct": g.gf.apply(lambda s: (s == 0).mean() * 100),
    })
    if "shots_for" in lf.columns:
        prof["shots_pg"] = g.shots_for.mean()
        prof["sot_pg"] = g.sot_for.mean()
        prof["sot_against_pg"] = g.sot_against.mean()
        conv = g.apply(lambda d: d.gf.sum() / max(d.sot_for.sum(), 1), include_groups=False)
        prof["conversion_per_sot"] = conv
    return prof.round(2).sort_values("gf_pg", ascending=False)


def league_summary(matches: pd.DataFrame) -> dict:
    total = matches.hg + matches.ag
    return {
        "matches": int(len(matches)),
        "goals_per_match": round(float(total.mean()), 2),
        "home_win_pct": round(float((matches.hg > matches.ag).mean() * 100), 1),
        "draw_pct": round(float((matches.hg == matches.ag).mean() * 100), 1),
        "away_win_pct": round(float((matches.hg < matches.ag).mean() * 100), 1),
        "over25_pct": round(float((total > 2.5).mean() * 100), 1),
        "btts_pct": round(float(((matches.hg > 0) & (matches.ag > 0)).mean() * 100), 1),
    }


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> pd.DataFrame:
    m = matches[((matches.home == team_a) & (matches.away == team_b)) |
                ((matches.home == team_b) & (matches.away == team_a))]
    return m[["date", "home", "away", "hg", "ag"]].sort_values("date", ascending=False)
