"""Export the data behind the EPL Edge dashboard as JSON files ({rows, meta}).

Usage:  python scripts/export_dashboard.py [--season 2026] [--out data/dashboard]

Each file carries this script's own source in ``meta`` so readers can see how
the numbers were produced.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from football_edge import data, fpl, model, odds, stats

FPL_TO_FD = {"Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry",
             "Hull City": "Hull", "Ipswich Town": "Ipswich"}
SRC = Path(__file__).read_text(encoding="utf-8")


def clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if hasattr(v, "item"):
        return clean(v.item())
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def write(out: Path, name: str, df: pd.DataFrame, inputs: str):
    rows = [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    meta = [{"label": "export_dashboard.py", "value": SRC, "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs},
            {"label": "Generated", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}]
    (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--history", type=int, default=3, help="previous seasons used to fit the model")
    ap.add_argument("--out", default="data/dashboard")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fd_in = "football-data.co.uk EPL CSVs, seasons " + ", ".join(
        data.season_code(y) for y in range(a.season - a.history, a.season + 1))

    seasons = list(range(a.season - a.history, a.season + 1))
    allm = data.load_seasons(seasons, refresh=a.refresh)
    cur = allm[allm.season == data.season_code(a.season)]

    # 1. League table + form + profiles
    t = stats.league_table(cur).reset_index().rename(columns={"index": "team"})
    f = stats.form(cur, 5)[["team", "form", "pts"]].rename(columns={"pts": "form_pts"})
    p = stats.team_profiles(cur).reset_index()[["team", "gf_pg", "ga_pg", "over25_pct", "btts_pct", "clean_sheet_pct"]]
    table = t.merge(f, on="team").merge(p, on="team").sort_values("Pos")
    write(out, "table", table, fd_in)

    # 2. League summary (one row)
    s = stats.league_summary(cur)
    s["last_match"] = cur.date.max().strftime("%Y-%m-%d")
    s["rounds"] = int(len(cur) // 10)
    write(out, "summary", pd.DataFrame([s]), fd_in)

    # 3. Predictions for upcoming fixtures (current + next gameweek, unplayed only)
    d = data.load_fpl(refresh=a.refresh)
    names = d["teams"].set_index("id")["name"].map(lambda n: FPL_TO_FD.get(n, n))
    gw = fpl.next_gameweek(d["events"])
    fx = d["fixtures"]
    kick = pd.to_datetime(fx.kickoff_time, utc=True)
    fx = fx[fx.event.isin([gw - 1, gw]) & (fx.finished == False)  # noqa: E712
            & (kick > pd.Timestamp.now(tz="UTC"))].copy()
    fx["home"] = fx.team_h.map(names)
    fx["away"] = fx.team_a.map(names)
    dc = model.DixonColes().fit(allm)
    preds = dc.predict_fixtures(fx[["home", "away"]])
    preds = preds.merge(fx[["home", "away", "event", "kickoff_time"]], on=["home", "away"])
    preds["match"] = preds.home + " v " + preds.away
    preds["kickoff"] = pd.to_datetime(preds.kickoff_time, utc=True).dt.strftime("%Y-%m-%dT%H:%MZ")
    keep = ["match", "event", "kickoff", "home", "away", "xg_home", "xg_away", "p_home", "p_draw", "p_away",
            "fair_home", "fair_draw", "fair_away", "p_over25", "p_btts", "top_score", "p_top_score"]
    write(out, "predictions", preds.sort_values("kickoff")[keep], fd_in + "; FPL API fixtures")

    # 4. Team strength ratings
    r = dc.ratings().reset_index().rename(columns={"index": "team"})
    r = r[r.team.isin(set(cur.home))]
    write(out, "ratings", r, fd_in)

    # 5. Backtest: model vs market, last two seasons out of sample
    start = pd.Timestamp(f"{a.season - 1}-08-01")
    wf = model.walk_forward(allm, start)
    mk = wf.dropna(subset=["odds_h_avg"])
    imp = mk[["odds_h_avg", "odds_d_avg", "odds_a_avg"]].apply(
        lambda x: pd.Series(odds.remove_margin(x.values), index=["p_home", "p_draw", "p_away"]), axis=1)
    wf_old = model.walk_forward(allm, start, xi=0.0019, sot_weight=0.0, handle_promoted=False, l2_promoted=0.0)
    bets = odds.value_bets(wf, min_edge=0.05)
    bt = odds.backtest(bets, "flat")
    write(out, "backtest", pd.DataFrame([{
        "from": start.strftime("%Y-%m-%d"), "matches": len(wf),
        "model_logloss": model.log_loss_1x2(wf),
        "old_model_logloss": model.log_loss_1x2(wf_old),
        "market_logloss": model.log_loss_1x2(pd.concat([mk[["result"]], imp], axis=1)),
        "bets": bt.get("bets", 0), "hit_rate": bt.get("hit_rate"), "roi": bt.get("roi")}]), fd_in)

    # 6. FPL squad + fixture ticker
    fin = int(d["events"].finished.sum())
    proj = fpl.project_points(d, horizon=5)
    plan = fpl.optimise(proj)
    sq = plan.squad.copy()
    sq["role"] = sq.in_xi.map({True: "XI", False: "Bench"})
    sq["captain"] = sq.web_name.map({plan.captain: "C", plan.vice: "V"}).fillna("")
    sq["player"] = sq.web_name
    write(out, "fpl_squad", sq[["player", "team_name", "pos", "price", "role", "captain", "xp_next", "xp_total"]],
          f"FPL API (bootstrap-static, fixtures), {fin} gameweeks finished, horizon GW{gw}-{gw + 4}")
    tk = fpl.fixture_ticker(d, horizon=5).reset_index().rename(columns={"team": "club"})
    tk.columns = [f"gw{c}" if isinstance(c, (int,)) else str(c) for c in tk.columns]
    write(out, "ticker", tk, f"FPL API fixtures, GW{gw}-{gw + 4}")


if __name__ == "__main__":
    main()
