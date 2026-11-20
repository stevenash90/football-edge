"""Command line interface.

Examples
--------
    football-edge stats --seasons 2024 2025
    football-edge predict --seasons 2023 2024 2025 --match Arsenal "Man City"
    football-edge backtest --seasons 2022 2023 2024 --start 2024-08-01
    football-edge fpl-draft --horizon 5
    football-edge fpl-transfers --entry 123456 --free-transfers 2
    football-edge demo     # offline, synthetic data
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data, fpl, model, odds, stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)


def _matches(args) -> pd.DataFrame:
    if getattr(args, "csv", None):
        return pd.concat([data.load_matches_csv(p) for p in args.csv]).sort_values("date")
    return data.load_seasons(args.seasons, refresh=args.refresh)


def cmd_stats(args):
    m = _matches(args)
    cur = m[m.season == m.season.max()] if "season" in m else m
    print("== League summary ==")
    for k, v in stats.league_summary(cur).items():
        print(f"  {k}: {v}")
    print("\n== Table ==")
    print(stats.league_table(cur).to_string())
    print(f"\n== Form (last {args.last}) ==")
    print(stats.form(cur, args.last).to_string(index=False))
    print("\n== Team profiles ==")
    print(stats.team_profiles(cur).to_string())


def _fmt_pred(p: dict) -> str:
    return (f"{p['home']} vs {p['away']}: xG {p['xg_home']:.2f}-{p['xg_away']:.2f} | "
            f"1X2 {p['p_home']:.1%}/{p['p_draw']:.1%}/{p['p_away']:.1%} "
            f"(fair {p['fair_home']:.2f}/{p['fair_draw']:.2f}/{p['fair_away']:.2f}) | "
            f"O2.5 {p['p_over25']:.1%} | BTTS {p['p_btts']:.1%} | top {p['top_score']} ({p['p_top_score']:.1%})")


def cmd_predict(args):
    m = _matches(args)
    dc = model.DixonColes(xi=args.xi).fit(m)
    if args.ratings:
        print(dc.ratings().to_string())
    for i in range(0, len(args.match or []), 2):
        print(_fmt_pred(dc.predict(args.match[i], args.match[i + 1])))
        if args.odds:
            h, d, a = args.odds[3 * (i // 2): 3 * (i // 2) + 3]
            p = dc.predict(args.match[i], args.match[i + 1])
            for sel, o, pr in zip("HDA", (h, d, a), (p["p_home"], p["p_draw"], p["p_away"])):
                print(f"   {sel} @ {o}: edge {odds.edge(pr, o):+.1%}, kelly {odds.kelly(pr, o):.2%}")


def cmd_backtest(args):
    m = _matches(args)
    preds = model.walk_forward(m, args.start, xi=args.xi)
    print(f"Out-of-sample matches: {len(preds)}")
    print(f"Log loss: {model.log_loss_1x2(preds):.4f} | Brier: {model.brier_1x2(preds):.4f}")
    col = f"odds_h_{args.book}"
    if col in preds:
        mk = preds.dropna(subset=[col]).copy()
        imp = mk[[f"odds_{s}_{args.book}" for s in "hda"]].apply(lambda r: pd.Series(odds.remove_margin(r.values),
                                                                                    index=["p_home", "p_draw", "p_away"]), axis=1)
        mk_ll = model.log_loss_1x2(pd.concat([mk[["result"]], imp], axis=1))
        print(f"Market ({args.book}) log loss: {mk_ll:.4f}  (lower is better; beating this is hard)")
        bets = odds.value_bets(preds, odds_prefix=f"odds_{{}}_{args.book}", min_edge=args.min_edge, blend=args.blend)
        print("Flat staking:", odds.backtest(bets, "flat"))
        print("Kelly staking:", odds.backtest(bets, "kelly"))
        if args.show:
            print(bets.head(args.show).to_string(index=False))


def _fpl_data(args):
    return data.load_fpl(refresh=args.refresh)


def _print_plan(plan: fpl.Plan, gw_cols: list[str]):
    cols = ["web_name", "team_name", "pos", "price", "xp_next", "xp_total"]
    print(plan.summary())
    print("\nStarting XI:")
    print(plan.xi[cols].to_string(index=False))
    print("\nBench (in order):")
    print(plan.bench[cols].to_string(index=False))


def cmd_fpl_draft(args):
    d = _fpl_data(args)
    proj = fpl.project_points(d, horizon=args.horizon)
    plan = fpl.optimise(proj, budget=args.budget, bench_weight=args.bench_weight)
    _print_plan(plan, proj.attrs["gw_cols"])
    _print_fpl_context(d, args)


def cmd_fpl_transfers(args):
    d = _fpl_data(args)
    proj = fpl.project_points(d, horizon=args.horizon)
    gw = fpl.next_gameweek(d["events"])
    picks = data.load_fpl_entry(args.entry, gw - 1)
    current = [p["element"] for p in picks["picks"]]
    bank = picks.get("entry_history", {}).get("bank", 0) / 10.0 if args.bank is None else args.bank
    print(f"GW{gw} | bank £{bank:.1f}m | free transfers {args.free_transfers}")
    print(fpl.compare_transfer_options(proj, current, bank, args.free_transfers,
                                       max_options=args.max_transfers).to_string(index=False))
    plan = fpl.optimise(proj, current=current, bank=bank, free_transfers=args.free_transfers,
                        max_transfers=args.max_transfers)
    print()
    _print_plan(plan, proj.attrs["gw_cols"])
    _print_fpl_context(d, args)


def _print_fpl_context(d, args):
    print("\n== Fixture ticker (easiest first) ==")
    print(fpl.fixture_ticker(d, horizon=args.horizon).head(10).to_string())
    bd = fpl.blank_double(d, horizon=max(args.horizon, 8))
    bd = bd[(bd.blank != "") | (bd.double != "")]
    if len(bd):
        print("\n== Blank / double gameweeks (chip planning) ==")
        print(bd.to_string(index=False))


def cmd_demo(args):
    from . import sample
    m, _ = sample.synthetic_matches(seasons=2)
    print("== Synthetic data demo (not real EPL) ==")
    print(stats.league_table(m[m.date >= m.date.quantile(0.5)]).head(6).to_string())
    dc = model.DixonColes().fit(m)
    print("\n" + _fmt_pred(dc.predict("TeamA", "TeamT")))
    print(_fmt_pred(dc.predict("TeamJ", "TeamK")))
    preds = model.walk_forward(m, m.date.quantile(0.6))
    bets = odds.value_bets(preds, min_edge=0.05)
    print(f"\nWalk-forward log loss {model.log_loss_1x2(preds):.4f}; value bets: {odds.backtest(bets)}")
    d = sample.synthetic_fpl()
    proj = fpl.project_points(d, horizon=5)
    plan = fpl.optimise(proj)
    print()
    _print_plan(plan, proj.attrs["gw_cols"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="football-edge", description="EPL stats, predictions, value bets and FPL planning")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data(sp, default_seasons):
        sp.add_argument("--seasons", type=int, nargs="+", default=default_seasons,
                        help="season start years, e.g. 2024 2025")
        sp.add_argument("--csv", nargs="+", help="use local football-data CSV files instead")
        sp.add_argument("--refresh", action="store_true", help="re-download data")

    s = sub.add_parser("stats", help="table, form and team profiles")
    add_data(s, [2025])
    s.add_argument("--last", type=int, default=5)
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("predict", help="Dixon-Coles predictions")
    add_data(s, [2023, 2024, 2025])
    s.add_argument("--match", nargs="+", help="pairs: HOME AWAY [HOME AWAY ...]")
    s.add_argument("--odds", type=float, nargs="+", help="bookmaker H D A odds per match")
    s.add_argument("--ratings", action="store_true")
    s.add_argument("--xi", type=float, default=0.0019)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("backtest", help="walk-forward model evaluation and value-bet backtest")
    add_data(s, [2022, 2023, 2024, 2025])
    s.add_argument("--start", required=True, help="first date to predict, e.g. 2024-08-01")
    s.add_argument("--book", default="avg", choices=["avg", "max", "b365", "ps", "psc"])
    s.add_argument("--min-edge", type=float, default=0.03)
    s.add_argument("--blend", type=float, default=0.0, help="mix with market probs (0..1)")
    s.add_argument("--xi", type=float, default=0.0019)
    s.add_argument("--show", type=int, default=0)
    s.set_defaults(func=cmd_backtest)

    for name, fn, hlp in (("fpl-draft", cmd_fpl_draft, "best 15-man squad from scratch"),
                          ("fpl-transfers", cmd_fpl_transfers, "transfer plan for your team")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--horizon", type=int, default=5)
        s.add_argument("--refresh", action="store_true")
        if name == "fpl-draft":
            s.add_argument("--budget", type=float, default=100.0)
            s.add_argument("--bench-weight", type=float, default=0.15)
        else:
            s.add_argument("--entry", type=int, required=True, help="your FPL team id")
            s.add_argument("--free-transfers", type=int, default=1)
            s.add_argument("--max-transfers", type=int, default=3)
            s.add_argument("--bank", type=float, help="override bank (£m)")
        s.set_defaults(func=fn)

    s = sub.add_parser("demo", help="run everything offline on synthetic data")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
