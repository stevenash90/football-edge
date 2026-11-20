"""Data loading.

Sources
-------
* football-data.co.uk: one CSV per EPL season with results, match stats and
  bookmaker odds. URL pattern: https://www.football-data.co.uk/mmz4281/{code}/E0.csv
  where ``code`` is e.g. ``"2526"`` for 2025/26.
* Fantasy Premier League public API (no login needed):
  ``bootstrap-static`` (players, teams, gameweeks) and ``fixtures``.

Everything downloaded is cached under ``data/`` so the rest of the code can run
offline once the files exist.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FPL_BASE = "https://fantasy.premierleague.com/api"
DEFAULT_CACHE = Path("data")

# football-data column -> our name. Only columns that exist are renamed.
_COLS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "hg",
    "FTAG": "ag",
    "FTR": "result",
    "HS": "h_shots",
    "AS": "a_shots",
    "HST": "h_sot",
    "AST": "a_sot",
    "HC": "h_corners",
    "AC": "a_corners",
    # 1X2 odds: Pinnacle (sharp), Bet365, market average, and closing versions
    "PSH": "odds_h_ps", "PSD": "odds_d_ps", "PSA": "odds_a_ps",
    "B365H": "odds_h_b365", "B365D": "odds_d_b365", "B365A": "odds_a_b365",
    "AvgH": "odds_h_avg", "AvgD": "odds_d_avg", "AvgA": "odds_a_avg",
    "MaxH": "odds_h_max", "MaxD": "odds_d_max", "MaxA": "odds_a_max",
    "PSCH": "odds_h_psc", "PSCD": "odds_d_psc", "PSCA": "odds_a_psc",
    # Over/under 2.5
    "Avg>2.5": "odds_o25_avg", "Avg<2.5": "odds_u25_avg",
    "Max>2.5": "odds_o25_max", "Max<2.5": "odds_u25_max",
}


def season_code(start_year: int) -> str:
    """2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _http_get(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "football-edge/0.1"})
    resp.raise_for_status()
    return resp


def normalize_matches(raw: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Rename football-data columns, parse dates, drop unplayed rows."""
    df = raw.rename(columns={k: v for k, v in _COLS.items() if k in raw.columns})
    keep = [c for c in df.columns if c in _COLS.values()]
    df = df[keep].copy()
    df = df.dropna(subset=["home", "away"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if season is not None:
        df["season"] = season
    played = df["hg"].notna() & df["ag"].notna()
    df = df[played].copy()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    if "result" not in df.columns:
        df["result"] = "D"
        df.loc[df.hg > df.ag, "result"] = "H"
        df.loc[df.hg < df.ag, "result"] = "A"
    return df.sort_values("date").reset_index(drop=True)


def load_season(start_year: int, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Load one EPL season from football-data.co.uk (cached as CSV)."""
    code = season_code(start_year)
    cache_dir = Path(cache_dir)
    path = cache_dir / f"E0_{code}.csv"
    if refresh or not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_http_get(FOOTBALL_DATA_URL.format(code=code)).content)
    raw = pd.read_csv(path, encoding="latin-1")
    return normalize_matches(raw, season=code)


def load_seasons(start_years: list[int], **kw) -> pd.DataFrame:
    frames = [load_season(y, **kw) for y in start_years]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def load_matches_csv(path: str | Path) -> pd.DataFrame:
    """Load a football-data style CSV from disk (any season)."""
    return normalize_matches(pd.read_csv(path, encoding="latin-1"))


# --------------------------------------------------------------------------- FPL

def _fpl_json(endpoint: str, cache_dir: Path, refresh: bool) -> dict | list:
    cache_dir = Path(cache_dir)
    path = cache_dir / f"fpl_{endpoint.strip('/').replace('/', '_')}.json"
    if refresh or not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(_http_get(f"{FPL_BASE}/{endpoint}").text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fpl(cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> dict:
    """Return {'players', 'teams', 'events', 'fixtures'} DataFrames from the FPL API."""
    boot = _fpl_json("bootstrap-static/", cache_dir, refresh)
    fixtures = _fpl_json("fixtures/", cache_dir, refresh)
    return fpl_frames(boot, fixtures)


def load_fpl_entry(entry_id: int, event: int, cache_dir: Path = DEFAULT_CACHE, refresh: bool = True) -> dict:
    """A manager's picks for a gameweek (public endpoint) plus bank/transfers."""
    picks = _fpl_json(f"entry/{entry_id}/event/{event}/picks/", cache_dir, refresh)
    return picks


def fpl_frames(boot: dict, fixtures: list) -> dict:
    players = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    events = pd.DataFrame(boot["events"])
    fx = pd.DataFrame(fixtures)
    numeric = ["form", "ep_next", "points_per_game", "selected_by_percent",
               "expected_goals", "expected_assists", "expected_goal_involvements"]
    for c in numeric:
        if c in players.columns:
            players[c] = pd.to_numeric(players[c], errors="coerce").fillna(0.0)
    players["price"] = players["now_cost"] / 10.0
    pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    players["pos"] = players["element_type"].map(pos)
    team_names = teams.set_index("id")["short_name"]
    players["team_name"] = players["team"].map(team_names)
    return {"players": players, "teams": teams, "events": events, "fixtures": fx}


def raw_from_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), encoding="latin-1")
