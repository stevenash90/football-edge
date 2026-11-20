import numpy as np
import pandas as pd
import pytest

from football_edge import data, fpl, model, odds, sample, stats


@pytest.fixture(scope="module")
def matches():
    m, truth = sample.synthetic_matches(seasons=2, seed=1)
    return m, truth


# ------------------------------------------------------------------- data

def test_normalize_football_data_columns():
    raw = pd.DataFrame({
        "Date": ["16/08/2025", "17/08/2025", "30/05/2026"],
        "HomeTeam": ["Liverpool", "Arsenal", "Chelsea"],
        "AwayTeam": ["Bournemouth", "Man United", "Everton"],
        "FTHG": [4, 1, None], "FTAG": [2, 0, None], "FTR": ["H", "H", None],
        "AvgH": [1.3, 1.9, 1.7], "AvgD": [6.0, 3.6, 3.9], "AvgA": [9.0, 4.2, 5.0],
    })
    df = data.normalize_matches(raw, season="2526")
    assert len(df) == 2                                 # unplayed match dropped
    assert {"home", "away", "hg", "ag", "odds_h_avg"} <= set(df.columns)
    assert df.date.iloc[0] == pd.Timestamp("2025-08-16")
    assert data.season_code(2025) == "2526"


# ------------------------------------------------------------------- stats

def test_league_table_points_consistent(matches):
    m, _ = matches
    t = stats.league_table(m)
    assert (t.W + t.D + t.L == t.P).all()
    assert t.GF.sum() == t.GA.sum() == m.hg.sum() + m.ag.sum()
    assert t.Pts.iloc[0] >= t.Pts.iloc[-1]
    home, away = stats.league_table(m, "H"), stats.league_table(m, "A")
    assert home.P.sum() == away.P.sum() == len(m)


def test_profiles_and_summary(matches):
    m, _ = matches
    prof = stats.team_profiles(m)
    assert prof.over25_pct.between(0, 100).all()
    s = stats.league_summary(m)
    assert abs(s["home_win_pct"] + s["draw_pct"] + s["away_win_pct"] - 100) < 0.5
    f = stats.form(m, 5)
    assert f.form.str.len().eq(5).all()


# ------------------------------------------------------------------- model

def test_dixon_coles_recovers_strengths(matches):
    m, truth = matches
    dc = model.DixonColes(xi=0.0).fit(m)
    est = pd.Series(dc.attack)
    tru = pd.Series(truth["attack"])
    assert np.corrcoef(est[tru.index], tru)[0, 1] > 0.85
    assert abs(dc.home_adv - truth["home_adv"]) < 0.12


def test_predictions_are_valid_probabilities(matches):
    m, _ = matches
    dc = model.DixonColes().fit(m)
    p = dc.predict("TeamA", "TeamT")
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-9
    assert abs(p["p_over25"] + p["p_under25"] - 1) < 1e-9
    assert p["p_home"] > p["p_away"]                   # strongest vs weakest at home
    assert dc.score_matrix("TeamA", "TeamT").sum() == pytest.approx(1.0)
    with pytest.raises(KeyError):
        dc.predict("TeamA", "Nope")


def test_walk_forward_beats_uniform(matches):
    m, _ = matches
    preds = model.walk_forward(m, m.date.quantile(0.7), refit_days=28)
    assert len(preds) > 50
    assert model.log_loss_1x2(preds) < np.log(3)
    assert model.brier_1x2(preds) < 2 / 3


# ------------------------------------------------------------------- odds

def test_margin_removal():
    o = [2.0, 3.4, 4.0]
    assert odds.overround(o) > 0
    for method in ("proportional", "power"):
        p = odds.remove_margin(o, method)
        assert p.sum() == pytest.approx(1.0)
        assert p[0] > p[1] > p[2]


def test_edge_and_kelly():
    assert odds.edge(0.55, 2.0) == pytest.approx(0.10)
    assert odds.kelly(0.40, 2.0) == 0.0
    assert 0 < odds.kelly(0.60, 2.0, fraction=1.0, cap=1.0) == pytest.approx(0.2)


def test_value_bets_and_backtest(matches):
    m, _ = matches
    preds = model.walk_forward(m, m.date.quantile(0.7), refit_days=28)
    bets = odds.value_bets(preds, min_edge=0.02)
    assert (bets.edge >= 0.02).all()
    res = odds.backtest(bets)
    assert res["bets"] == len(bets)


# ------------------------------------------------------------------- FPL

@pytest.fixture(scope="module")
def fpl_proj():
    d = sample.synthetic_fpl(seed=2)
    return d, fpl.project_points(d, horizon=4)


def _valid_squad(squad, xi, budget):
    assert len(squad) == 15 and len(xi) == 11
    assert squad.pos.value_counts().to_dict() == fpl.SQUAD_SLOTS
    assert squad.team.value_counts().max() <= 3
    assert squad.sel_price.sum() <= budget + 1e-6
    c = xi.pos.value_counts()
    assert c["GK"] == 1 and c["DEF"] >= 3 and c["MID"] >= 2 and c["FWD"] >= 1


def test_projection_handles_injury_blank_double(fpl_proj):
    d, proj = fpl_proj
    gws = proj.attrs["gws"]
    injured = proj[proj.status == "i"]
    assert (injured[f"xp_gw{gws[0]}"] == 0).all()
    team1 = proj[(proj.team == 1) & (proj.base_ppm > 0) & (proj.play_prob > 0)]
    assert (team1[f"xp_gw{gws[1]}"] == 0).all()          # blank gameweek
    bd = fpl.blank_double(d, horizon=3)
    row = bd[bd.gw == gws[1]].iloc[0]
    assert "AAA" in row.blank and row.double != ""


def test_draft_squad_respects_rules(fpl_proj):
    _, proj = fpl_proj
    plan = fpl.optimise(proj, budget=100.0)
    _valid_squad(plan.squad, plan.xi, 100.0)
    assert plan.captain in set(plan.xi.web_name)


def test_transfers_respect_limits(fpl_proj):
    _, proj = fpl_proj
    cheap = fpl.optimise(proj, budget=85.0)
    current = cheap.squad.id.tolist()
    plan = fpl.optimise(proj, current=current, bank=15.0, free_transfers=1, max_transfers=2)
    _valid_squad(plan.squad, plan.xi, 100.0)
    assert len(plan.transfers_in) == len(plan.transfers_out) <= 2
    assert plan.hits == max(0, len(plan.transfers_in) - 1)
    table = fpl.compare_transfer_options(proj, current, 15.0, free_transfers=1, max_options=2)
    assert table.objective.is_monotonic_increasing
    roll = fpl.optimise(proj, current=current, bank=15.0, max_transfers=0)
    assert set(roll.squad.id) == set(current)


def test_fixture_ticker(fpl_proj):
    d, _ = fpl_proj
    t = fpl.fixture_ticker(d, horizon=4)
    assert len(t) == 20 and t.avg_fdr.is_monotonic_increasing
