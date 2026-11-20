"""Odds utilities: implied probability, margin removal, value detection, Kelly, backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied(odds) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> float:
    return float(implied(odds).sum() - 1.0)


def remove_margin(odds, method: str = "power") -> np.ndarray:
    """Fair probabilities from a full market (e.g. [H, D, A]).

    * ``proportional``: divide by the booksum.
    * ``power``: find k with sum(p_i ** k) = 1 — corrects favourite/longshot bias better.
    """
    p = implied(odds)
    if method == "proportional":
        return p / p.sum()
    if method != "power":
        raise ValueError(method)
    lo, hi = 1.0, 5.0
    for _ in range(100):
        k = (lo + hi) / 2
        s = np.sum(p ** k)
        if s > 1:
            lo = k
        else:
            hi = k
    q = p ** ((lo + hi) / 2)
    return q / q.sum()


def edge(model_prob: float, odds: float) -> float:
    """Expected value per 1 unit staked: p * odds - 1."""
    return model_prob * odds - 1.0


def kelly(model_prob: float, odds: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    """Fractional Kelly stake as a share of bankroll (0 if no edge), capped."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1 - model_prob)) / b
    return float(np.clip(f * fraction, 0.0, cap))


def value_bets(preds: pd.DataFrame, odds_prefix: str = "odds_{}_avg", min_edge: float = 0.03,
               min_prob: float = 0.05, blend: float = 0.0, kelly_fraction: float = 0.25) -> pd.DataFrame:
    """List 1X2 selections where the model beats the bookmaker price.

    ``preds`` must contain p_home/p_draw/p_away and odds columns like
    ``odds_h_avg``. ``blend`` mixes the model with the de-margined market
    probability (0 = pure model, 0.5 = half/half) — blending usually improves
    calibration because the market is very efficient.
    """
    rows = []
    for r in preds.itertuples(index=False):
        rd = r._asdict()
        odds = [rd.get(odds_prefix.format(s)) for s in "hda"]
        if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
            continue
        market = remove_margin(odds)
        model = np.array([rd["p_home"], rd["p_draw"], rd["p_away"]])
        p = (1 - blend) * model + blend * market
        for k, sel in enumerate(("H", "D", "A")):
            e = edge(p[k], odds[k])
            if e >= min_edge and p[k] >= min_prob:
                rows.append({
                    "date": rd.get("date"), "home": rd["home"], "away": rd["away"],
                    "selection": sel, "odds": odds[k], "p_model": round(p[k], 4),
                    "p_market": round(market[k], 4), "edge": round(e, 4),
                    "kelly_stake": round(kelly(p[k], odds[k], kelly_fraction), 4),
                    "result": rd.get("result"),
                })
    cols = ["date", "home", "away", "selection", "odds", "p_model", "p_market", "edge", "kelly_stake", "result"]
    return pd.DataFrame(rows, columns=cols).sort_values("edge", ascending=False).reset_index(drop=True)


def backtest(bets: pd.DataFrame, staking: str = "flat", bankroll: float = 100.0) -> dict:
    """Settle value bets that have a known ``result``. Staking: 'flat' (1u) or 'kelly'."""
    b = bets.dropna(subset=["result"]).sort_values("date").copy()
    if b.empty:
        return {"bets": 0}
    b["won"] = b.selection == b.result
    if staking == "flat":
        b["stake"] = 1.0
        b["pnl"] = np.where(b.won, b.odds - 1, -1.0)
        staked = b.stake.sum()
        profit = b.pnl.sum()
        return {"bets": len(b), "hit_rate": round(float(b.won.mean()), 3), "staked": float(staked),
                "profit": round(float(profit), 2), "roi": round(float(profit / staked), 4),
                "avg_odds": round(float(b.odds.mean()), 2)}
    bank = bankroll
    for r in b.itertuples():
        stake = bank * r.kelly_stake
        bank += stake * (r.odds - 1) if r.won else -stake
    return {"bets": len(b), "hit_rate": round(float(b.won.mean()), 3), "start": bankroll,
            "end": round(bank, 2), "growth": round(bank / bankroll - 1, 4)}
