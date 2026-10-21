"""Score every Underdog line and choose the day's Top 10 and each game's top 3.

Win chance for a side:
  * market: Underdog's own two-sided price for the line, de-vigged
  * model: the projection's full outcome distribution at Underdog's line (calibrated)
  * blend: start from the market, move toward the model by up to 40%, less the more they disagree
    (a big gap usually means news the model hasn't seen). With no price, the model is shrunk halfway to 50%.
Value = win chance x Underdog's payout multiplier for that side. Standard 3-pick entries break even at ~0.55
per pick, 4-pick at ~0.56, so picks are ranked by value.
"""
import json, os, datetime
import numpy as np, pandas as pd
from scipy.stats import norm, nbinom
import features as F, model as M, data as D
from paths import DATA

TOP_N, PER_GAME_TOP, PER_PLAYER_TOP, GAME_N = 10, 2, 1, 3
BREAKEVEN = {'3-pick': 0.550, '4-pick': 0.562}
OPP_KEY = {'pts': 'pts', 'reb': 'reb', 'ast': 'ast', 'fg3m': 'fg3m', 'stl': 'stl', 'blk': 'blk', 'tov': 'tov',
           'pra': 'pts', 'pr': 'pts', 'pa': 'pts', 'ra': 'reb', 'stocks': 'stl', 'fant': 'pts', 'dd': 'reb',
           'td': 'ast'}
LATE_LAST_SEASON = pd.Timestamp('2026-03-15')


def model_weight(n_cur):
    """How far the model may pull the price: 40% once a player has 8 games this season, 16% with none
    (early on, projections still reflect last season's role while Underdog's price knows the new one)."""
    return 0.4 * (0.4 + 0.6 * min(1.0, (n_cur or 0) / 8))


def blend(pc, pm, wmax=0.4):
    pm = min(max(pm, 0.02), 0.98)
    if pc is None or not np.isfinite(pc):
        return 0.5 + 0.5 * (pm - 0.5)
    pc = min(max(pc, 0.02), 0.98)
    d = abs(norm.ppf(pm) - norm.ppf(pc))
    return pc + wmax * float(np.exp(-(d / 0.8) ** 2)) * (pm - pc)


def match_players(lines, box):
    played = box[box.played].sort_values('date')
    latest = played.groupby('pid').tail(1)[['pid', 'name', 'team', 'pos', 'date']].copy()
    latest['key'] = latest.name.map(D.norm_name)
    alias_p = os.path.join(DATA, 'name_aliases.json')
    aliases = json.load(open(alias_p)) if os.path.exists(alias_p) else {}
    by_key = {k: g for k, g in latest.groupby('key')}
    out = {}
    for (player, team), _ in lines.groupby(['player', 'team']):
        if player in aliases:
            out[(player, team)] = int(aliases[player]); continue
        k = D.norm_name(player)
        c = by_key.get(k)
        if c is None:  # first initial + last name on the same team
            parts = k.split()
            if len(parts) >= 2:
                c = latest[latest.key.str.endswith(' ' + parts[-1]) & latest.key.str.startswith(parts[0][0])]
                c = c if len(c) == 1 else None
        if c is None or not len(c):
            continue
        same = c[c.team == team]
        out[(player, team)] = int((same if len(same) else c).sort_values('date').pid.iloc[-1])
    return out


def upcoming_vacated(slate_players, states, out_ids, today):
    """VAC rows for tonight's teams from the injury report: absent = rotation players listed Out."""
    rows = []
    st = states.set_index('pid')
    recent_cut = pd.Timestamp(today) - pd.Timedelta(days=21)
    early = pd.Timestamp(today) <= pd.Timestamp(D.SEASON_START) + pd.Timedelta(days=30)
    for (date, team), g in slate_players.groupby(['date', 'team']):
        vm = vu = vp = vr = va = avail = 0.0
        for pid in set(g.pid) - set(out_ids.get(team, set())):
            if pid in st.index and st.at[pid, 'min_ewm6'] >= 10:
                avail += st.at[pid, 'min_ewm6']
        for pid in out_ids.get(team, set()):
            if pid not in st.index:
                continue
            r = st.loc[pid]
            if r.min_ewm6 < 10:
                continue
            if not (r.date >= recent_cut or (early and r.date >= LATE_LAST_SEASON)):
                continue
            vm += r.min_ewm6; vu += r.usg_ewm; vp += r.pts_ewm6; vr += r.reb_ewm6; va += r.ast_ewm6
        rows.append((pd.Timestamp(date), team, vm, vu, vp, vr, va, avail))
    return pd.DataFrame(rows, columns=F.VAC_COLS)


def recent_games(box, pids, n=10):
    p = box[box.played & box.pid.isin(pids)].sort_values('date')
    return {pid: g.tail(n) for pid, g in p.groupby('pid')}


def quantiles(stat, mu, d, qs=(0.25, 0.5, 0.75)):
    if stat in M.COUNT:
        a = max(d, 1e-3); n = 1 / a; p = n / (n + max(mu, 1e-3))
        return [float(nbinom.ppf(q, n, p)) for q in qs]
    return [float(mu + norm.ppf(q) * d) for q in qs]


def score(lines, box, models, meta, inj, now):
    shrink = meta.get('shrink', 1.0)
    today = now.astimezone(D.ET).date()
    lines = lines.copy()
    pidmap = match_players(lines, box)
    lines['pid'] = [pidmap.get((p, t)) for p, t in zip(lines.player, lines.team)]
    states = F.player_states(box).sort_values('date').groupby('pid').tail(1)
    pos = states.set_index('pid').pos.to_dict()

    # injuries: Out removes the player and shifts his production to teammates; Day-To-Day is Questionable
    inj = inj.copy() if len(inj) else pd.DataFrame(columns=['team', 'pid', 'name', 'status', 'detail'])
    out_rows = inj[inj.status.isin(['Out', 'Doubtful'])]
    out_ids = {t: set(g.pid.dropna().astype(int)) for t, g in out_rows.groupby('team')}
    out_all = set(out_rows.pid.dropna().astype(int))
    q_all = set(inj[inj.status.isin(['Day-To-Day', 'Questionable'])].pid.dropna().astype(int))
    out_names = {D.norm_name(n) for n in out_rows.name.dropna()}

    sl = lines[lines.pid.notna()].drop_duplicates(['pid'])[['pid', 'player', 'team', 'opp', 'home', 'date', 'match_id']]
    sl = sl.assign(pid=sl.pid.astype(int), date=pd.to_datetime(sl.date))
    sl['pos'] = sl.pid.map(pos).fillna('F')
    up = sl.rename(columns={'player': 'name', 'match_id': 'game_id'})[['pid', 'name', 'team', 'opp', 'home', 'date', 'pos', 'game_id']]
    up_vac = upcoming_vacated(up, states, out_ids, today)
    f = F.build(box, up, up_vac)
    fu = f[f.upcoming == 1].copy()
    cols = meta['cols']
    P = M.predict(models, {k: v for k, v in meta['disp'].items()}, fu, cols)
    fu = pd.concat([fu.reset_index(drop=True), P.reset_index(drop=True).add_prefix('mu_')], axis=1)
    fu = fu.set_index('pid')

    # score lines
    n_cur = box[box.played & (box.season == D.SEASON)].groupby('pid').size().to_dict()
    recs = []
    hist = recent_games(box, set(fu.index))
    for r in lines.itertuples(index=False):
        pid = None if r.pid is None or (isinstance(r.pid, float) and np.isnan(r.pid)) else int(r.pid)
        if pid in out_all or D.norm_name(r.player) in out_names:
            continue
        rec = r._asdict()
        rec['questionable'] = bool(pid in q_all) if pid else False
        rec['has_model'] = pid is not None and pid in fu.index and f'mu_{r.stat}' in fu.columns
        pm = None
        if rec['has_model']:
            x = fu.loc[pid]
            mu = float(x[f'mu_{r.stat}'])
            if r.stat in M.BINARY:
                pm = mu if r.line < 1 else 0.0
            else:
                pm = float(M.p_over(r.stat, mu, x[f'mu_{r.stat}_disp'], r.line))
            pm = 0.5 + shrink * (pm - 0.5)
            rec['proj'] = mu
            rec['min_proj'] = float(x['mu_min'])
            rec['n_career'] = int(x.n_career)
        rec['p_model'] = pm
        rec['n_cur'] = n_cur.get(pid, 0) if pid else 0
        wmax = model_weight(rec['n_cur'])
        best = None
        for side, sgn in (('higher', 1), ('lower', -1)):
            mult = getattr(r, f'{side}_mult')
            if mult is None or not np.isfinite(mult):
                continue
            pc = r.p_market if side == 'higher' else (None if r.p_market is None else 1 - r.p_market)
            pms = None if pm is None else (pm if side == 'higher' else 1 - pm)
            if pms is None:
                p = pc if pc is not None else None
            else:
                p = blend(pc, pms, wmax)
            if p is None:
                continue
            v = p * mult
            if best is None or v > best['value']:
                best = {'side': side, 'mult': mult, 'p': p, 'p_market_side': pc, 'p_model_side': pms, 'value': v}
        if best is None:
            continue
        rec.update(best)
        recs.append(rec)
    S = pd.DataFrame(recs)
    if S.empty:
        return S, fu, up_vac, hist
    S['word'] = S.side.map({'higher': 'Higher', 'lower': 'Lower'})
    # Only lines Underdog prices on both sides, at near-standard payouts, can be picked: one-sided long shots
    # (double-doubles at 5x, etc.) have no reliable fair price to anchor to.
    S['pickable'] = S.higher_imp.notna() & S.lower_imp.notna() & S.mult.between(0.6, 1.6)
    S['agrees'] = S.p_model_side.fillna(0) >= 0.5
    S['started'] = pd.to_datetime(S.kickoff, utc=True) <= pd.Timestamp(now)
    S['reasons'] = [reasons(s, fu, hist, inj, box) for s in S.itertuples(index=False)]
    return S.sort_values('value', ascending=False).reset_index(drop=True), fu, up_vac, hist


def reasons(s, fu, hist, inj, box):
    """Evidence chips; ok=True when the item supports the chosen side."""
    out = []
    if not s.has_model:
        return [{'t': 'No NBA history for this player yet: priced from Underdog only', 'ok': False}]
    x = fu.loc[int(s.pid)]
    up = s.side == 'higher'
    if s.stat not in M.BINARY:
        out.append({'t': f'Projection {s.proj:.1f} vs line {s.line:g}', 'ok': (s.proj > s.line) == up})
    else:
        out.append({'t': f'Model chance {100 * s.p_model:.0f}%', 'ok': (s.p_model > 0.5) == up})
    g = hist.get(int(s.pid))
    if g is not None and len(g) and s.stat in g.columns:
        last = g[s.stat].tail(10)
        n_over = int((last > s.line).sum())
        k = len(last)
        out.append({'t': f'Over in {n_over} of last {k} (avg {last.mean():.1f})',
                    'ok': (n_over / k > 0.5) == up if k else False})
    if s.p_market_side is not None and np.isfinite(s.p_market_side):
        out.append({'t': f"Underdog's price {100 * s.p_market_side:.0f}%", 'ok': s.p_market_side >= 0.5})
    ok_key = OPP_KEY.get(s.stat)
    fac = x.get(f'opp_{ok_key}')
    if fac is not None and np.isfinite(fac) and abs(fac - 1) >= 0.05:
        out.append({'t': f"{s.opp} allows {abs(fac - 1) * 100:.0f}% {'more' if fac > 1 else 'fewer'} {ok_key.upper() if ok_key == 'fg3m' else ok_key} to {x.pos}s",
                    'ok': (fac > 1) == up})
    gp = x.get('game_pace')
    if gp is not None and np.isfinite(gp) and abs(gp - 100) >= 1.5:
        out.append({'t': f"{'Fast' if gp > 100 else 'Slow'} game pace ({gp:.0f})", 'ok': (gp > 100) == up})
    if x.vac_min >= 15:
        names = inj[(inj.team == s.team) & inj.status.isin(['Out', 'Doubtful'])].name.head(3).tolist()
        out.append({'t': f"Teammates out ({', '.join(names)}): {x.vac_min:.0f} min to fill", 'ok': up})
    if np.isfinite(x.get('min_last3', np.nan)) and abs(x.min_last3 - x.min_ewm6) >= 3:
        out.append({'t': f"Minutes {'up' if x.min_last3 > x.min_ewm6 else 'down'} lately ({x.min_last3:.0f} last 3)",
                    'ok': (x.min_last3 > x.min_ewm6) == up})
    if x.team_rest == 1:
        out.append({'t': 'Second night of a back-to-back', 'ok': not up})
    if abs(x.exp_margin) >= 10:
        out.append({'t': f"Blowout risk (expected margin {x.exp_margin:+.0f})", 'ok': not up})
    if s.n_cur < 3:
        out.append({'t': f"Only {int(s.n_cur)} game{'s' if s.n_cur != 1 else ''} this season: leaning on Underdog's price", 'ok': False})
    if s.questionable:
        out.append({'t': 'Questionable on the injury report', 'ok': False})
    return out


def select(S, locked=None):
    """Top 10 (max 2 per game, 1 per player) and each game's top 3 (1 per player)."""
    if S.empty:
        return S.assign(top10=False, rank=np.nan, game3=False)
    S = S.copy()
    S['top10'] = False; S['rank'] = np.nan; S['game3'] = False
    elig = S[S.pickable & S.has_model & S.agrees & ~S.questionable & ~S.started &
             (S.n_career.fillna(0) >= 10)]
    per_game, players, rank = {}, set(), 0
    lk = locked if locked is not None and len(locked) else None
    if lk is not None:
        for r in lk[lk.top10].itertuples(index=False):
            rank += 1; players.add(int(r.pid)); per_game[r.match_id] = per_game.get(r.match_id, 0) + 1
    for i, r in elig.iterrows():
        if rank >= TOP_N:
            break
        if per_game.get(r.match_id, 0) >= PER_GAME_TOP or r.pid in players:
            continue
        rank += 1
        S.at[i, 'top10'] = True; S.at[i, 'rank'] = rank
        per_game[r.match_id] = per_game.get(r.match_id, 0) + 1
        players.add(r.pid)
    g_elig = S[S.pickable & S.has_model & S.agrees & ~S.started & (S.n_career.fillna(0) >= 5)]
    for mid, g in g_elig.groupby('match_id'):
        seen = set()
        for i, r in g.sort_values('value', ascending=False).iterrows():
            if r.pid in seen:
                continue
            S.at[i, 'game3'] = True; seen.add(r.pid)
            if len(seen) >= GAME_N:
                break
    return S
