"""Fantasy Premier League: expected-points projection and squad/transfer optimisation.

Projection (heuristic, transparent, easy to tune):
    xP(player, gw) = base_ppm * play_prob * availability(gw) * sum(fixture_mult(d) for fixtures in gw)

* ``base_ppm``   blend of recent ``form`` and season ``points_per_game``.
* ``play_prob``  share of available minutes the player has actually played.
* ``availability`` from FPL ``status`` / ``chance_of_playing_next_round``.
* fixture difficulty (FDR 1..5) scales the points; double gameweeks add up,
  blank gameweeks give 0.

Optimisation uses an integer program (scipy.optimize.milp / HiGHS) with the real FPL rules:
15 players (2 GK, 5 DEF, 5 MID, 3 FWD), max 3 per club, budget, valid XI
(1 GK, >=3 DEF, >=2 MID, >=1 FWD), captain doubled, -4 per extra transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD_SLOTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
FDR_MULT = {1: 1.25, 2: 1.12, 3: 1.0, 4: 0.86, 5: 0.74}
UNAVAILABLE = {"i", "s", "u", "n"}   # injured, suspended, unavailable/left, not in squad


def next_gameweek(events: pd.DataFrame) -> int:
    nxt = events[events.get("is_next", False) == True]  # noqa: E712
    if len(nxt):
        return int(nxt.id.iloc[0])
    unfinished = events[events.get("finished", False) == False]  # noqa: E712
    return int(unfinished.id.min()) if len(unfinished) else int(events.id.max())


def team_fixtures(fixtures: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    """Long table: team, gw, opponent, is_home, difficulty (from that team's view)."""
    fx = fixtures[fixtures.event.isin(gws)]
    home = pd.DataFrame({"team": fx.team_h, "gw": fx.event, "opp": fx.team_a,
                         "is_home": True, "fdr": fx.team_h_difficulty})
    away = pd.DataFrame({"team": fx.team_a, "gw": fx.event, "opp": fx.team_h,
                         "is_home": False, "fdr": fx.team_a_difficulty})
    return pd.concat([home, away], ignore_index=True).astype({"gw": int, "team": int, "fdr": int})


def availability(players: pd.DataFrame) -> pd.Series:
    chance = pd.to_numeric(players.get("chance_of_playing_next_round"), errors="coerce")
    avail = chance.fillna(100.0) / 100.0
    avail[players.status.isin(UNAVAILABLE) & chance.isna()] = 0.0
    return avail.clip(0, 1)


def project_points(data: dict, horizon: int = 5, form_weight: float = 0.3,
                   start_gw: int | None = None) -> pd.DataFrame:
    """Add xp_gw{N} columns and xp_total over the next ``horizon`` gameweeks."""
    players = data["players"].copy()
    events = data["events"]
    gw0 = start_gw or next_gameweek(events)
    gws = list(range(gw0, gw0 + horizon))
    tf = team_fixtures(data["fixtures"], gws)
    tf["mult"] = tf.fdr.map(FDR_MULT).fillna(1.0)
    mult = tf.groupby(["team", "gw"]).mult.sum().unstack(fill_value=0.0).reindex(columns=gws, fill_value=0.0)

    finished = int(events.get("finished", pd.Series(dtype=bool)).fillna(False).sum()) if len(events) else 0
    finished = max(finished, 1)
    mins_share = (players.minutes / (90.0 * finished)).clip(0, 1)
    players["play_prob"] = (mins_share * 1.15).clip(0, 1)
    base = form_weight * players.form + (1 - form_weight) * players.points_per_game
    players["base_ppm"] = base
    avail = availability(players)
    players["availability"] = avail

    cols = []
    for k, gw in enumerate(gws):
        # injuries tend to heal: availability recovers 25 pts per week ahead
        a_k = (avail + 0.25 * k).clip(upper=1.0) if k else avail
        a_k = a_k.where(~players.status.isin({"u", "n"}), 0.0)
        m = players.team.map(mult[gw]).fillna(0.0)
        col = f"xp_gw{gw}"
        players[col] = (base * players.play_prob * a_k * m).round(2)
        cols.append(col)
    players["xp_next"] = players[cols[0]]
    players["xp_total"] = players[cols].sum(axis=1).round(2)
    players.attrs["gw_cols"] = cols
    players.attrs["gws"] = gws
    return players


def fixture_ticker(data: dict, horizon: int = 6, start_gw: int | None = None) -> pd.DataFrame:
    """Team x GW grid of opponents with FDR, plus average difficulty (easiest first)."""
    gw0 = start_gw or next_gameweek(data["events"])
    gws = list(range(gw0, gw0 + horizon))
    tf = team_fixtures(data["fixtures"], gws)
    names = data["teams"].set_index("id").short_name
    tf["label"] = tf.opp.map(names) + np.where(tf.is_home, "(H)", "(a)") + tf.fdr.astype(str)
    grid = tf.groupby(["team", "gw"]).label.apply(" ".join).unstack().reindex(columns=gws).fillna("BLANK")
    grid["avg_fdr"] = tf.groupby("team").fdr.mean().round(2)
    grid["fixtures"] = tf.groupby("team").size()
    grid.index = grid.index.map(names)
    return grid.sort_values(["avg_fdr", "fixtures"], ascending=[True, False])


def blank_double(data: dict, horizon: int = 8, start_gw: int | None = None) -> pd.DataFrame:
    """Gameweeks in the horizon where clubs blank (0 games) or double (2+ games)."""
    gw0 = start_gw or next_gameweek(data["events"])
    gws = list(range(gw0, gw0 + horizon))
    tf = team_fixtures(data["fixtures"], gws)
    counts = tf.groupby(["gw", "team"]).size().unstack(fill_value=0).reindex(index=gws, fill_value=0)
    counts = counts.reindex(columns=data["teams"].id, fill_value=0)
    names = data["teams"].set_index("id").short_name
    rows = []
    for gw, r in counts.iterrows():
        rows.append({"gw": gw,
                     "blank": ", ".join(names[t] for t in r.index[r == 0]),
                     "double": ", ".join(names[t] for t in r.index[r >= 2])})
    return pd.DataFrame(rows)


@dataclass
class Plan:
    squad: pd.DataFrame
    xi: pd.DataFrame
    bench: pd.DataFrame
    captain: str
    vice: str
    transfers_in: pd.DataFrame
    transfers_out: pd.DataFrame
    hits: int
    cost: float
    objective: float

    def summary(self) -> str:
        lines = [f"Squad cost: £{self.cost:.1f}m | objective xP: {self.objective:.1f} | hits: {self.hits} (-{4*self.hits})"]
        if len(self.transfers_out):
            for o, i in zip(self.transfers_out.itertuples(), self.transfers_in.itertuples()):
                lines.append(f"  OUT {o.web_name} ({o.team_name}, £{o.price}m)  ->  IN {i.web_name} ({i.team_name}, £{i.price}m)")
        lines.append(f"Captain: {self.captain} | Vice: {self.vice}")
        return "\n".join(lines)


def optimise(projected: pd.DataFrame, budget: float = 100.0, current: list[int] | None = None,
             bank: float = 0.0, free_transfers: int = 1, max_transfers: int | None = None,
             hit_cost: float = 4.0, bench_weight: float = 0.15, objective_col: str = "xp_total",
             exclude: list[int] | None = None, include: list[int] | None = None,
             selling_prices: dict[int, float] | None = None) -> Plan:
    """Best 15-man squad (draft) or best transfers from ``current`` squad (list of player ids)."""
    df = projected.copy()
    if exclude:
        df = df[~df.id.isin(exclude) | df.id.isin(current or [])]
    df = df.reset_index(drop=True)
    ids = df.id.tolist()
    price = dict(zip(ids, df.price))
    if selling_prices:
        price.update({k: v for k, v in selling_prices.items() if k in price})
    xp = dict(zip(ids, df[objective_col]))
    xp_next = dict(zip(ids, df.get("xp_next", df[objective_col])))
    pos = dict(zip(ids, df.pos))
    team = dict(zip(ids, df.team))

    n = len(ids)
    cur = set(current or [])
    if current:
        missing = cur - set(ids)
        if missing:
            raise ValueError(f"Current squad ids not found in player data: {sorted(missing)}")
        budget = bank + sum(price[i] for i in cur)

    # variable layout: [x (squad) | s (starting XI) | c (captain) | e (extra transfers)]
    X, S, C, E = 0, n, 2 * n, 3 * n
    nv = 3 * n + 1
    xp_a = np.array([xp[i] for i in ids], dtype=float)
    obj = np.zeros(nv)
    obj[X:X + n] = bench_weight * xp_a
    obj[S:S + n] = (1 - bench_weight) * xp_a
    obj[C:C + n] = [xp_next[i] for i in ids]
    obj[E] = -hit_cost
    rows, lo, hi = [], [], []

    def add(coefs: dict[int, float], lb: float, ub: float):
        r = np.zeros(nv)
        for k, v in coefs.items():
            r[k] = v
        rows.append(r); lo.append(lb); hi.append(ub)

    idx_pos = {p: [k for k, i in enumerate(ids) if pos[i] == p] for p in SQUAD_SLOTS}
    add({X + k: price[i] for k, i in enumerate(ids)}, -np.inf, budget + 1e-6)
    for p, cnt in SQUAD_SLOTS.items():
        add({X + k: 1 for k in idx_pos[p]}, cnt, cnt)
        add({S + k: 1 for k in idx_pos[p]}, XI_MIN[p], XI_MAX[p])
    add({S + k: 1 for k in range(n)}, 11, 11)
    add({C + k: 1 for k in range(n)}, 1, 1)
    for t in set(team.values()):
        add({X + k: 1 for k, i in enumerate(ids) if team[i] == t}, -np.inf, 3)
    for k in range(n):
        add({S + k: 1, X + k: -1}, -np.inf, 0)
        add({C + k: 1, S + k: -1}, -np.inf, 0)
    id_pos = {i: k for k, i in enumerate(ids)}
    for i in include or []:
        add({X + id_pos[i]: 1}, 1, 1)
    e_ub = 0.0
    if current:
        new_in = {X + k: 1 for k, i in enumerate(ids) if i not in cur}
        add({**new_in, E: -1}, -np.inf, free_transfers)          # extra >= n_in - ft
        if max_transfers is not None:
            add(new_in, -np.inf, max_transfers)
        e_ub = 15.0

    ub = np.ones(nv); ub[E] = e_ub
    res = milp(-obj, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub))
    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")
    sol = np.round(res.x).astype(int)

    df["in_squad"] = sol[X:X + n] == 1
    df["in_xi"] = sol[S:S + n] == 1
    df["sel_price"] = [price[i] for i in ids]
    squad = df[df.in_squad].copy()
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad["_o"] = squad.pos.map(order)
    squad = squad.sort_values(["_o", objective_col], ascending=[True, False]).drop(columns="_o")
    xi = squad[squad.in_xi]
    bench = squad[~squad.in_xi].sort_values(["pos", objective_col], ascending=[True, False])
    # bench order: GK last-in-list convention -> outfield by xP, then GK
    bench = pd.concat([bench[bench.pos != "GK"].sort_values(objective_col, ascending=False), bench[bench.pos == "GK"]])
    caps = xi.sort_values("xp_next" if "xp_next" in xi else objective_col, ascending=False)
    t_in = squad[~squad.id.isin(cur)] if current else squad.iloc[0:0]
    t_out = df[df.id.isin(cur) & ~df.in_squad] if current else df.iloc[0:0]
    hits = max(0, len(t_in) - free_transfers) if current else 0
    return Plan(squad=squad, xi=xi, bench=bench, captain=caps.web_name.iloc[0], vice=caps.web_name.iloc[1],
                transfers_in=t_in, transfers_out=t_out, hits=hits, cost=float(squad.sel_price.sum()),
                objective=float(-res.fun))


def compare_transfer_options(projected: pd.DataFrame, current: list[int], bank: float,
                             free_transfers: int = 1, max_options: int = 3, **kw) -> pd.DataFrame:
    """Evaluate making 0, 1, 2, ... transfers so the manager can see the marginal gain."""
    rows = []
    for n in range(0, max_options + 1):
        plan = optimise(projected, current=current, bank=bank, free_transfers=free_transfers,
                        max_transfers=n, **kw)
        rows.append({"max_transfers": n, "made": len(plan.transfers_in), "hits": plan.hits,
                     "objective": round(plan.objective, 2),
                     "out": ", ".join(plan.transfers_out.web_name),
                     "in": ", ".join(plan.transfers_in.web_name),
                     "captain": plan.captain})
    out = pd.DataFrame(rows)
    out["gain_vs_roll"] = (out.objective - out.objective.iloc[0]).round(2)
    return out
