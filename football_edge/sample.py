"""Synthetic data with known ground truth, for tests and offline demos.

Not real EPL data — team names are placeholders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAMS = [f"Team{c}" for c in "ABCDEFGHIJKLMNOPQRST"]


def synthetic_matches(seasons: int = 2, seed: int = 0, home_adv: float = 0.25,
                      with_odds: bool = True) -> tuple[pd.DataFrame, dict]:
    """Double round-robin seasons drawn from a Poisson model with known strengths."""
    rng = np.random.default_rng(seed)
    n = len(TEAMS)
    att = np.linspace(0.45, -0.45, n) + rng.normal(0, 0.05, n)
    att -= att.mean()
    dfn = np.linspace(-0.35, 0.35, n) + rng.normal(0, 0.05, n)
    truth = {"attack": dict(zip(TEAMS, att)), "defence": dict(zip(TEAMS, dfn)), "home_adv": home_adv}
    rows = []
    day = pd.Timestamp("2024-08-10")
    for _ in range(seasons):
        pairs = [(h, a) for h in range(n) for a in range(n) if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            lam = np.exp(home_adv + att[h] + dfn[a])
            mu = np.exp(att[a] + dfn[h])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            row = {"date": day + pd.Timedelta(days=7 * (k // 10)), "home": TEAMS[h], "away": TEAMS[a],
                   "hg": int(hg), "ag": int(ag)}
            if with_odds:
                g = np.arange(11)
                from scipy.stats import poisson
                m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
                p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
                p = p / p.sum()
                noisy = p * np.exp(rng.normal(0, 0.06, 3))
                noisy /= noisy.sum()
                o = 1 / (noisy * 1.05)
                row.update({"odds_h_avg": round(o[0], 2), "odds_d_avg": round(o[1], 2), "odds_a_avg": round(o[2], 2)})
            rows.append(row)
        day = day + pd.Timedelta(days=7 * (len(pairs) // 10) + 60)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    return df, truth


def synthetic_fpl(seed: int = 0, n_gw: int = 38, next_gw: int = 12) -> dict:
    """Minimal FPL-shaped frames: players, teams, events, fixtures."""
    rng = np.random.default_rng(seed)
    teams = pd.DataFrame({"id": range(1, 21), "name": TEAMS, "short_name": [t[-1] * 3 for t in TEAMS]})
    events = pd.DataFrame({"id": range(1, n_gw + 1)})
    events["finished"] = events.id < next_gw
    events["is_next"] = events.id == next_gw
    fx = []
    for gw in range(1, n_gw + 1):
        order = rng.permutation(20) + 1
        for k in range(0, 20, 2):
            fx.append({"event": gw, "team_h": int(order[k]), "team_a": int(order[k + 1]),
                       "team_h_difficulty": int(rng.integers(2, 6)), "team_a_difficulty": int(rng.integers(2, 6))})
    fixtures = pd.DataFrame(fx)
    # make a blank and a double in next_gw+1 for teams 1 and 2
    gw_b = next_gw + 1
    fixtures = fixtures[~((fixtures.event == gw_b) & ((fixtures.team_h == 1) | (fixtures.team_a == 1)))]
    fixtures = pd.concat([fixtures, pd.DataFrame([{"event": gw_b, "team_h": 2, "team_a": 3,
                                                   "team_h_difficulty": 2, "team_a_difficulty": 3}])],
                         ignore_index=True)
    rows = []
    pid = 1
    per_team = {1: 2, 2: 5, 3: 5, 4: 3}
    pos_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    price_base = {1: 4.5, 2: 4.5, 3: 5.5, 4: 6.0}
    for t in range(1, 21):
        quality = 1.0 - (t - 1) / 25
        for et, cnt in per_team.items():
            for _ in range(cnt):
                skill = max(0.5, rng.normal(3.5, 1.2) * quality)
                price = round(price_base[et] + max(0, skill - 2.5) * 1.6 + rng.normal(0, 0.3), 1)
                rows.append({"id": pid, "web_name": f"P{pid}", "team": t, "element_type": et,
                             "now_cost": int(max(price_base[et] - 0.5, price) * 10),
                             "form": round(max(0, skill + rng.normal(0, 1)), 1),
                             "points_per_game": round(skill, 1),
                             "minutes": int(rng.uniform(0.2, 1.0) * 90 * (next_gw - 1)),
                             "status": "a", "chance_of_playing_next_round": None})
                pid += 1
    players = pd.DataFrame(rows)
    players.loc[5, ["status", "chance_of_playing_next_round"]] = ["i", 0]
    players["price"] = players.now_cost / 10.0
    players["pos"] = players.element_type.map(pos_name)
    players["team_name"] = players.team.map(teams.set_index("id").short_name)
    return {"players": players, "teams": teams, "events": events, "fixtures": fixtures}
