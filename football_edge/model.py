"""Dixon-Coles match model with exponential time decay.

Goals are modelled as Poisson:
    home goals ~ Poisson(lambda = exp(home_adv + attack[home] + defence[away]))
    away goals ~ Poisson(mu     = exp(attack[away] + defence[home]))
with the Dixon-Coles low-score correction ``rho`` and match weights
``exp(-xi * days_ago)`` so recent games count more.

From the score matrix we derive 1X2, over/under, BTTS and Asian handicap
probabilities, plus fair (no-margin) odds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    xi: float = 0.0019          # time decay per day (half-life ~ 1 year)
    l2: float = 0.0             # optional ridge penalty on attack/defence
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    fitted_at: pd.Timestamp | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColes":
        df = matches.dropna(subset=["hg", "ag"])
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
        df = df[df["date"] <= as_of]
        teams = sorted(set(df.home) | set(df.away))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = df.home.map(idx).to_numpy()
        ai = df.away.map(idx).to_numpy()
        hg = df.hg.to_numpy()
        ag = df.ag.to_numpy()
        days = (as_of - df["date"]).dt.days.to_numpy().astype(float)
        w = np.exp(-self.xi * days)
        lgam_h = _log_factorial(hg)
        lgam_a = _log_factorial(ag)

        def unpack(p):
            att = np.append(p[: n - 1], -p[: n - 1].sum())   # sum-to-zero constraint
            dfn = p[n - 1: 2 * n - 1]
            return att, dfn, p[-2], p[-1]

        def nll(p):
            att, dfn, home, rho = unpack(p)
            lam = np.exp(home + att[hi] + dfn[ai])
            mu = np.exp(att[ai] + dfn[hi])
            tau = _tau(hg, ag, lam, mu, rho)
            if np.any(tau <= 0):
                return 1e12
            ll = (np.log(tau) + hg * np.log(lam) - lam - lgam_h
                  + ag * np.log(mu) - mu - lgam_a)
            pen = self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))
            return -np.sum(w * ll) + pen

        x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n - 1) + [(-1, 1), (-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        att, dfn, home, rho = unpack(res.x)

        self.teams = teams
        self.attack = dict(zip(teams, att))
        self.defence = dict(zip(teams, dfn))
        self.home_adv = float(home)
        self.rho = float(rho)
        self.fitted_at = as_of
        self._opt = res
        return self

    # --------------------------------------------------------------- prediction
    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        self._check(home, away)
        lam = np.exp(self.home_adv + self.attack[home] + self.defence[away])
        mu = np.exp(self.attack[away] + self.defence[home])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.expected_goals(home, away)
        g = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def predict(self, home: str, away: str) -> dict:
        m = self.score_matrix(home, away)
        lam, mu = self.expected_goals(home, away)
        return {"home": home, "away": away, "xg_home": lam, "xg_away": mu,
                **markets_from_matrix(m)}

    def predict_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = [self.predict(r.home, r.away) for r in fixtures.itertuples()
                if r.home in self.attack and r.away in self.attack]
        return pd.DataFrame(rows)

    def ratings(self) -> pd.DataFrame:
        """Team strength table: higher attack and lower defence = better."""
        df = pd.DataFrame({"attack": self.attack, "defence": self.defence})
        df["overall"] = df.attack - df.defence
        return df.sort_values("overall", ascending=False).round(3)

    def _check(self, *teams):
        for t in teams:
            if t not in self.attack:
                raise KeyError(f"Unknown team {t!r}. Known: {', '.join(self.teams)}")


def markets_from_matrix(m: np.ndarray) -> dict:
    """1X2, O/U 1.5/2.5/3.5, BTTS, most likely score and AH lines from a score matrix."""
    h, a = np.indices(m.shape)
    total = h + a
    diff = h - a
    out = {
        "p_home": float(m[diff > 0].sum()),
        "p_draw": float(m[diff == 0].sum()),
        "p_away": float(m[diff < 0].sum()),
        "p_btts": float(m[(h > 0) & (a > 0)].sum()),
    }
    for line in (1.5, 2.5, 3.5):
        key = str(line).replace(".", "")
        out[f"p_over{key}"] = float(m[total > line].sum())
        out[f"p_under{key}"] = float(m[total < line].sum())
    i, j = np.unravel_index(np.argmax(m), m.shape)
    out["top_score"] = f"{i}-{j}"
    out["p_top_score"] = float(m[i, j])
    # Asian handicap on the home team: -0.5, -1.5 (win prob), and 0 (draw-no-bet)
    out["p_home_ah_m15"] = float(m[diff >= 2].sum())
    out["p_home_dnb"] = out["p_home"] / max(out["p_home"] + out["p_away"], 1e-12)
    for k in ("home", "draw", "away"):
        out[f"fair_{k}"] = 1 / out[f"p_{k}"] if out[f"p_{k}"] > 0 else np.inf
    return out


def _log_factorial(k: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(np.asarray(k) + 1)


# ------------------------------------------------------------------ evaluation

def walk_forward(matches: pd.DataFrame, start: str | pd.Timestamp, refit_days: int = 7,
                 **model_kw) -> pd.DataFrame:
    """Out-of-sample predictions: refit every ``refit_days`` using only past data.

    Returns predictions merged with actual results and any odds columns, ready
    for ``odds.backtest``.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(start)
    test = matches[matches.date >= start]
    out = []
    cursor = start
    model = None
    while cursor <= test.date.max():
        window_end = cursor + pd.Timedelta(days=refit_days)
        chunk = test[(test.date >= cursor) & (test.date < window_end)]
        if len(chunk):
            train = matches[matches.date < cursor]
            model = DixonColes(**model_kw).fit(train, as_of=cursor - pd.Timedelta(days=1))
            preds = model.predict_fixtures(chunk)
            if len(preds):
                merged = chunk.reset_index(drop=True).merge(preds, on=["home", "away"], how="inner")
                out.append(merged)
        cursor = window_end
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def log_loss_1x2(preds: pd.DataFrame) -> float:
    p = np.select([preds.result == "H", preds.result == "D"], [preds.p_home, preds.p_draw], preds.p_away)
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1))))


def brier_1x2(preds: pd.DataFrame) -> float:
    y = np.stack([(preds.result == r).astype(float) for r in "HDA"], axis=1)
    p = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))
