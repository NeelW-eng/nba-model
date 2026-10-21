"""Player-game features built only from games before the one being predicted.

Player form: per-game EWMAs (half-life 6 games), per-minute rates (half-life 12), season-to-date rates.
Role: minutes trend, starter rate, share of the team's available minutes.
Team context: pace and net rating (EWMA of possessions and margin per 100), rest, back-to-backs.
Matchup: what the opponent allows per minute to the player's position (G/F/C) versus league average.
Injuries: minutes, usage, points, rebounds and assists of rotation teammates who are not playing.
"""
import numpy as np, pandas as pd

BASE = ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'oreb', 'fga', 'fta', 'fg3a']
OPP_STATS = ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov']
RENAME = {'points': 'pts', 'rebounds': 'reb', 'assists': 'ast', 'three_point_field_goals_made': 'fg3m',
          'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'offensive_rebounds': 'oreb',
          'field_goals_attempted': 'fga', 'free_throws_attempted': 'fta',
          'three_point_field_goals_attempted': 'fg3a', 'minutes': 'min'}
TEAMS = ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU', 'IND', 'LAC', 'LAL', 'MEM',
         'MIA', 'MIL', 'MIN', 'NO', 'NY', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SA', 'SAC', 'TOR', 'UTAH', 'WSH']
POS = {'G': 'G', 'PG': 'G', 'SG': 'G', 'F': 'F', 'SF': 'F', 'PF': 'F', 'GF': 'F', 'C': 'C', 'FC': 'C'}


def add_targets(d):
    d['pra'] = d.pts + d.reb + d.ast
    d['pr'] = d.pts + d.reb
    d['pa'] = d.pts + d.ast
    d['ra'] = d.reb + d.ast
    d['stocks'] = d.stl + d.blk
    d['fant'] = d.pts + 1.2 * d.reb + 1.5 * d.ast + 3 * d.blk + 3 * d.stl - d.tov
    tens = sum((d[c] >= 10).astype(int) for c in ['pts', 'reb', 'ast', 'stl', 'blk'])
    d['dd'] = (tens >= 2).astype(float)
    d['td'] = (tens >= 3).astype(float)
    return d


def tidy(raw):
    """ESPN player box rows -> one row per player-game with standard column names."""
    d = raw.rename(columns=RENAME)
    d = d[d.team_abbreviation.isin(TEAMS) & d.opponent_team_abbreviation.isin(TEAMS)]
    d = d[d.season_type.isin([2, 3, 5]) & d.athlete_id.notna()]
    out = pd.DataFrame({
        'game_id': d.game_id.astype('int64'), 'date': pd.to_datetime(d.game_date).dt.normalize(),
        'season': d.season.astype(int), 'stype': d.season_type.astype(int),
        'pid': d.athlete_id.astype('int64'), 'name': d.athlete_display_name,
        'team': d.team_abbreviation, 'opp': d.opponent_team_abbreviation,
        'home': (d.home_away == 'home').astype(int), 'pos': d.athlete_position_abbreviation.map(POS).fillna('F'),
        'starter': d.starter.fillna(False).astype(int), 'team_score': d.team_score, 'opp_score': d.opponent_team_score,
        'min': d['min'],
    })
    for c in BASE:
        out[c] = d[c].astype(float)
    out['played'] = (~d.did_not_play.fillna(False).astype(bool)) & (out['min'].fillna(0) > 0)
    out = out.drop_duplicates(['game_id', 'pid'])
    return add_targets(out).reset_index(drop=True)


def _ewm(g, cols, hl):
    return g[cols].transform(lambda x: x.ewm(halflife=hl, min_periods=1).mean())


def player_states(box):
    """State of each player after each game he played (includes that game)."""
    p = box[box.played].sort_values(['pid', 'date']).copy()
    g = p.groupby('pid')
    e6 = _ewm(g, ['min'] + BASE, 6)
    e12 = _ewm(g, ['min'] + BASE, 12)
    st = pd.DataFrame({'pid': p.pid, 'date': p.date, 'team': p.team, 'season': p.season, 'pos': p.pos})
    st['min_ewm6'] = e6['min']
    st['min_l1'] = p['min']
    st['min_last3'] = g['min'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    st['starter_ewm'] = g['starter'].transform(lambda x: x.ewm(halflife=4, min_periods=1).mean())
    st['n_career'] = g.cumcount() + 1
    gs = p.groupby(['pid', 'season'])
    st['n_season'] = gs.cumcount() + 1
    smin = gs['min'].cumsum()
    st['min_season'] = smin / st.n_season
    for c in BASE:
        st[f'{c}_ewm6'] = e6[c]
        st[f'{c}_rate12'] = e12[c] / e12['min'].clip(lower=1)
        st[f'{c}_rate_season'] = gs[c].cumsum() / smin.clip(lower=1)
    st['usg_ewm'] = st.fga_ewm6 + 0.44 * st.fta_ewm6 + st.tov_ewm6
    # longer-run per-game averages for volatility
    for c in ['pts', 'reb', 'ast']:
        st[f'{c}_sd'] = g[c].transform(lambda x: x.rolling(20, min_periods=3).std())
    return st


def team_games(box):
    p = box[box.played]
    t = p.groupby(['game_id', 'date', 'season', 'team', 'opp', 'home']).agg(
        fga=('fga', 'sum'), oreb=('oreb', 'sum'), tov=('tov', 'sum'), fta=('fta', 'sum'),
        pts=('pts', 'sum')).reset_index()
    t['poss'] = t.fga - t.oreb + t.tov + 0.44 * t.fta
    o = t[['game_id', 'team', 'pts', 'poss']].rename(columns={'team': 'opp', 'pts': 'pts_against', 'poss': 'opp_poss'})
    t = t.merge(o, on=['game_id', 'opp'])
    t['pace'] = (t.poss + t.opp_poss) / 2
    t['net100'] = 100 * (t.pts - t.pts_against) / t.pace.clip(lower=60)
    t = t.sort_values(['team', 'date'])
    g = t.groupby('team')
    t['pace_ewm'] = g['pace'].transform(lambda x: x.ewm(halflife=10, min_periods=1).mean())
    t['net_ewm'] = g['net100'].transform(lambda x: x.ewm(halflife=12, min_periods=1).mean())
    return t


def opp_allowance(box):
    """Per-minute rate each defense allows to each position, EWMA over its games, as a factor of league average."""
    p = box[box.played]
    a = p.groupby(['game_id', 'date', 'opp', 'pos'])[['min'] + OPP_STATS].sum().reset_index()
    a = a.rename(columns={'opp': 'defense'}).sort_values(['defense', 'pos', 'date'])
    g = a.groupby(['defense', 'pos'])
    em = g['min'].transform(lambda x: x.ewm(halflife=15, min_periods=1).mean())
    league = {pos: (a[a.pos == pos][OPP_STATS].sum() / a[a.pos == pos]['min'].sum()) for pos in a.pos.unique()}
    out = a[['defense', 'pos', 'date']].copy()
    n = g.cumcount() + 1
    for c in OPP_STATS:
        rate = g[c].transform(lambda x: x.ewm(halflife=15, min_periods=1).mean()) / em.clip(lower=1)
        lg = a.pos.map(lambda q: league[q][c])
        f = rate / lg
        w = (n / (n + 8)).clip(upper=1)  # shrink early-season factors toward 1
        out[f'opp_{c}'] = 1 + w * (f - 1)
    return out


def _asof(left, right, by, cols, lcol='date'):
    """For each left row, the latest right row with the same keys strictly before the left date."""
    left = left.copy()
    left[lcol] = pd.to_datetime(left[lcol]).astype('datetime64[ns]')
    right = right.copy()
    right['date'] = pd.to_datetime(right['date']).astype('datetime64[ns]')
    left['_i'] = np.arange(len(left))
    l = left.sort_values(lcol)
    r = right.sort_values('date')[by + ['date'] + cols].rename(columns={'date': '_rdate'})
    m = pd.merge_asof(l, r, left_on=lcol, right_on='_rdate', by=by, allow_exact_matches=False)
    return m.sort_values('_i').drop(columns=['_i', '_rdate'])


def vacated(box):
    """Production of rotation teammates who are not playing, per historical team-game.
    Rotation = players on that team's roster this season (EWMA minutes >= 10) who played for it within its
    last 10 games; absent = rotation players not in the box score."""
    p = box[box.played].sort_values('date')
    st = player_states(box)
    cols = ['min_ewm6', 'usg_ewm', 'pts_ewm6', 'reb_ewm6', 'ast_ewm6']
    on_team = p.groupby(['pid', 'season'])['team'].apply(set).to_dict()
    by_date_state = st[['pid', 'date', 'team', 'season'] + cols].groupby('date')
    games = p[['date', 'season', 'team', 'game_id']].drop_duplicates(['date', 'team'])
    played = p.groupby(['date', 'team'])['pid'].apply(set).to_dict()
    tcount, state, rows = {}, {}, []
    for d, gd in games.groupby('date'):
        for r in gd.itertuples(index=False):
            n = tcount.get(r.team, 0)
            vm = vu = vp = vr = va = avail = 0.0
            pids = played[(d, r.team)]
            for pid, (tm, idx, v) in state.items():
                if tm != r.team or n - idx > 10 or v[0] < 10 or r.team not in on_team.get((pid, r.season), ()):
                    continue
                if pid in pids:
                    avail += v[0]
                else:
                    vm += v[0]; vu += v[1]; vp += v[2]; vr += v[3]; va += v[4]
            rows.append((d, r.team, vm, vu, vp, vr, va, avail))
        for r in gd.itertuples(index=False):
            tcount[r.team] = tcount.get(r.team, 0) + 1
        for r in by_date_state.get_group(d).itertuples(index=False):
            state[r.pid] = (r.team, tcount.get(r.team, 0), (r.min_ewm6, r.usg_ewm, r.pts_ewm6, r.reb_ewm6, r.ast_ewm6))
    return pd.DataFrame(rows, columns=VAC_COLS)


VAC_COLS = ['date', 'team', 'vac_min', 'vac_usg', 'vac_pts', 'vac_reb', 'vac_ast', 'team_avail_min']


def build(box, upcoming=None, up_vac=None):
    """Feature table for every historical played row (training) plus the upcoming rows (prediction).
    upcoming: DataFrame with pid, date, team, opp, home, pos (players to predict).
    up_vac: VAC_COLS rows for the upcoming team-games (built from the injury report)."""
    st = player_states(box)
    tg = team_games(box)
    oa = opp_allowance(box)
    vac = vacated(box)
    if up_vac is not None and len(up_vac):
        vac = pd.concat([vac, up_vac[VAC_COLS]], ignore_index=True).drop_duplicates(['date', 'team'], keep='last')

    tgt = box[box.played][['game_id', 'date', 'season', 'pid', 'name', 'team', 'opp', 'home', 'pos', 'min',
                            'pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa', 'ra', 'stocks',
                            'fant', 'dd', 'td']].copy()
    tgt['upcoming'] = 0
    if upcoming is not None and len(upcoming):
        u = upcoming.copy()
        u['upcoming'] = 1
        tgt = pd.concat([tgt, u], ignore_index=True)
    pcols = [c for c in st.columns if c not in ('pid', 'date', 'team', 'season', 'pos')]
    st = st.rename(columns={'date': 'date'})
    st['last_date'] = st['date']
    f = _asof(tgt, st, ['pid'], pcols + ['last_date'])
    f['rest_days'] = (f.date - f.last_date).dt.days.clip(upper=10)
    tcols = ['pace_ewm', 'net_ewm']
    tg2 = tg[['team', 'date'] + tcols].copy()
    tg2['team_last'] = tg2.date
    f = _asof(f, tg2, ['team'], tcols + ['team_last'])
    f = f.rename(columns={'pace_ewm': 'team_pace', 'net_ewm': 'team_net'})
    f['team_rest'] = (f.date - f.team_last).dt.days.clip(upper=5)
    o2 = tg[['team', 'date'] + tcols].rename(columns={'team': 'opp', 'pace_ewm': 'opp_pace', 'net_ewm': 'opp_net'})
    o2['opp_last'] = o2.date
    f = _asof(f, o2, ['opp'], ['opp_pace', 'opp_net', 'opp_last'])
    f['opp_rest'] = (f.date - f.opp_last).dt.days.clip(upper=5)
    f['exp_margin'] = (f.team_net - f.opp_net) * (f.team_pace + f.opp_pace) / 200 + np.where(f.home == 1, 2.5, -2.5)
    f['abs_margin'] = f.exp_margin.abs()
    f['game_pace'] = (f.team_pace + f.opp_pace) / 2
    oa2 = oa.rename(columns={'defense': 'opp'})
    f = _asof(f, oa2, ['opp', 'pos'], [f'opp_{c}' for c in OPP_STATS])
    vac['date'] = pd.to_datetime(vac.date).astype('datetime64[ns]')
    f['date'] = pd.to_datetime(f.date).astype('datetime64[ns]')
    f = f.merge(vac, on=['date', 'team'], how='left')
    for c in ['vac_min', 'vac_usg', 'vac_pts', 'vac_reb', 'vac_ast']:
        f[c] = f[c].fillna(0)
    f['share_min'] = f.min_ewm6 / f.team_avail_min.clip(lower=60)
    f['pos_code'] = f.pos.map({'G': 0, 'F': 1, 'C': 2})
    f = f.drop(columns=['last_date', 'team_last', 'opp_last'])
    return f[f.n_career.notna()].reset_index(drop=True)


def feature_cols(f):
    skip = {'game_id', 'date', 'season', 'pid', 'name', 'team', 'opp', 'pos', 'min', 'upcoming',
            'pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa', 'ra', 'stocks', 'fant', 'dd', 'td'}
    return [c for c in f.columns if c not in skip and pd.api.types.is_numeric_dtype(f[c])]
