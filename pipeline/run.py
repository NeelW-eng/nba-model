"""Daily pipeline. Every run: refresh this season's box scores, grade finished picks, pull Underdog lines and
the injury report, score every line, log the slate's picks, and write data/board.json + site/tipoff_edge.html.
Models retrain when older than 18 hours (the morning run); the pre-tip run reuses them."""
import os, sys, json, pickle, datetime, traceback
import numpy as np, pandas as pd
from paths import DATA, SITE
import data as D, features as F, model as M, underdog as U, picks as PK, track as T

MODELS = os.path.join(DATA, 'models.pkl')


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
        return x.isoformat()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_models(box, force=False):
    meta = json.load(open(os.path.join(DATA, 'model_meta.json')))
    meta['shrink'] = json.load(open(os.path.join(DATA, 'calibration.json'))).get('shrink', 1.0)
    if not force and os.path.exists(MODELS) and (datetime.datetime.now().timestamp() - os.path.getmtime(MODELS)) < 18 * 3600:
        models, cols = pickle.load(open(MODELS, 'rb'))
    else:
        print('training models...')
        f = F.build(box)
        models, cols = M.train_all(f, meta)
        pickle.dump((models, cols), open(MODELS, 'wb'))
    meta['cols'] = cols
    return models, meta


def pick_dict(r, reasons=None):
    keys = ['player', 'team', 'opp', 'game', 'kickoff', 'match_id', 'stat', 'stat_label', 'line', 'word', 'mult',
            'p', 'p_model', 'p_market', 'value', 'proj', 'rank', 'top10', 'game3', 'questionable', 'result', 'actual']
    d = {k: r.get(k) for k in keys}
    d['reasons'] = reasons if reasons is not None else []
    return d


def main(mode='run', offline=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.astimezone(D.ET).date()
    status = {}
    hist = D.history()
    try:
        cur = pd.read_parquet(D.CUR) if offline and os.path.exists(D.CUR) else (pd.DataFrame() if offline else D.update_current(today))
        status['espn_box'] = 'ok'
    except Exception as e:
        traceback.print_exc(); cur = pd.read_parquet(D.CUR) if os.path.exists(D.CUR) else pd.DataFrame()
        status['espn_box'] = f'failed: {e}'[:200]
    raw = pd.concat([hist, cur], ignore_index=True) if len(cur) else hist
    box = F.tidy(raw)
    box_cur = box[box.season == D.SEASON]

    T.grade(box_cur, now)
    models, meta = load_models(box, force=(mode in ('am', 'train')))

    try:
        inj = D.injuries() if not offline else pd.read_json(os.path.join(DATA, 'injuries.json'))
        inj.to_json(os.path.join(DATA, 'injuries.json'), orient='records')
        status['injuries'] = f'ok ({len(inj)} listed)'
    except Exception as e:
        traceback.print_exc(); inj = pd.read_json(os.path.join(DATA, 'injuries.json')) if os.path.exists(os.path.join(DATA, 'injuries.json')) else pd.DataFrame()
        status['injuries'] = f'failed, using last saved: {e}'[:200]

    try:
        js = D.underdog_raw() if not offline else json.load(open(os.path.join(DATA, 'underdog_raw.json')))
        if not offline:
            json.dump(js, open(os.path.join(DATA, 'underdog_raw.json'), 'w'))
        lines, games = U.parse(js)
        status['underdog'] = f'ok ({len(lines)} lines)'
    except Exception as e:
        traceback.print_exc(); lines, games = pd.DataFrame(), pd.DataFrame()
        status['underdog'] = f'failed: {e}'[:200]

    slate = None
    if len(lines):
        upcoming_dates = sorted(set(lines.date))
        slate = upcoming_dates[0]
        lines = lines[lines.date == slate]
    S = pd.DataFrame()
    fu = pd.DataFrame(); up_vac = pd.DataFrame()
    if len(lines):
        S, fu, up_vac, _ = PK.score(lines, box, models, meta, inj, now)
        log = T.load()
        locked = log[(log.date.astype(str) == slate) & (pd.to_datetime(log.kickoff, utc=True) <= pd.Timestamp(now))]
        S = PK.select(S, locked)
        T.log(S, now)
    log = T.load()
    write_board(now, slate, S, fu, up_vac, inj, games, log, status, box)


def write_board(now, slate, S, fu, up_vac, inj, games, log, status, box):
    rmap = {}
    if len(S):
        for r in S.itertuples(index=False):
            rmap[(int(r.pid) if r.pid == r.pid and r.pid is not None else r.player, r.stat, float(r.line))] = r.reasons
    day = log[log.date.astype(str) == str(slate)] if slate else log.iloc[0:0]

    def from_log(r):
        d = pick_dict(r.to_dict())
        key = (int(r.pid) if pd.notna(r.pid) else r.player, r.stat, float(r.line))
        d['reasons'] = rmap.get(key, [])
        d['locked'] = pd.to_datetime(r.kickoff, utc=True) <= pd.Timestamp(now)
        return d
    top10 = [from_log(r) for _, r in day[day.top10].sort_values('rank').iterrows()]
    gl = []
    if len(games):
        for g in games[games.date == slate].sort_values('kickoff').itertuples(index=False):
            ps = day[day.game3 & (day.match_id.astype(str) == str(g.match_id))].sort_values('value', ascending=False)
            gl.append({'match_id': g.match_id, 'game': g.game, 'kickoff': g.kickoff,
                       'picks': [from_log(r) for _, r in ps.iterrows()]})
    # every line
    all_lines = []
    if len(S):
        for r in S.itertuples(index=False):
            all_lines.append({'player': r.player, 'team': r.team, 'opp': r.opp, 'stat': r.stat_label, 'line': r.line,
                              'word': r.word, 'mult': r.mult, 'p': r.p, 'pm': r.p_model_side, 'pc': r.p_market_side,
                              'value': r.value, 'proj': getattr(r, 'proj', None), 'q': r.questionable, 'ok': r.pickable,
                              'started': r.started})
    # projections
    proj = []
    if len(fu):
        for pid, x in fu.iterrows():
            row = {'player': x['name'], 'team': x.team, 'opp': x.opp, 'pos': x.pos, 'min': x.mu_min,
                   'vac': x.vac_min}
            for s in ['pts', 'reb', 'ast', 'fg3m', 'pra', 'stl', 'blk']:
                row[s] = PK.quantiles(s, x[f'mu_{s}'], x[f'mu_{s}_disp'])
            proj.append(row)
        proj.sort(key=lambda r: -r['pra'][1])
    # injuries on tonight's teams
    teams = set(games[games.date == slate][['away', 'home']].values.ravel()) if len(games) and slate else set()
    st = F.player_states(box).sort_values('date').groupby('pid').tail(1).set_index('pid')
    injl = []
    if len(inj):
        for r in inj[inj.team.isin(teams)].itertuples(index=False):
            m = st.min_ewm6.get(int(r.pid)) if pd.notna(r.pid) and int(r.pid) in st.index else None
            injl.append({'team': r.team, 'name': r.name, 'status': r.status, 'detail': r.detail, 'min': m})
        injl.sort(key=lambda r: (r['team'], -(r['min'] or 0)))
    vac = up_vac[['team', 'vac_min']].to_dict('records') if len(up_vac) else []
    recent = log[log.result.isin(['win', 'loss', 'push', 'void'])].sort_values('kickoff', ascending=False).head(60)
    bt = json.load(open(os.path.join(DATA, 'backtest_summary.json')))
    board = {
        'generated': now.isoformat(timespec='minutes'), 'slate': slate, 'status': status,
        'breakeven': PK.BREAKEVEN, 'top10': top10, 'games': gl, 'lines': all_lines, 'proj': proj,
        'injuries': injl, 'vacated': vac, 'tracker': T.summary(log),
        'recent': [pick_dict(r.to_dict()) for _, r in recent.iterrows()],
        'backtest': {k: {kk: v[kk] for kk in ('n', 'mae_model', 'mae_line', 'hit_at_55', 'hit_at_60')} for k, v in bt['by_stat'].items()},
        'calibration': bt['calibration'],
    }
    board = clean(board)
    # The dashboard's database caps each document at 256 KB, so the board ships as three documents:
    # board/current (picks, tracker, injuries), board/lines (every line, as compact rows), board/proj.
    keys = ['player', 'team', 'opp', 'stat', 'line', 'word', 'mult', 'p', 'pm', 'pc', 'value', 'proj', 'q', 'ok', 'started']
    rows = [[l.get(k) for k in keys] for l in board['lines']]
    core = {k: v for k, v in board.items() if k not in ('lines', 'proj')}
    core['n_lines'] = len(rows)
    docs = {'board.json': core,
            'board_lines.json': {'generated': board['generated'], 'keys': keys, 'rows': rows},
            'board_proj.json': {'generated': board['generated'], 'proj': board['proj']}}
    for fn, doc in docs.items():
        txt = json.dumps(doc, allow_nan=False, separators=(',', ':'))
        open(os.path.join(DATA, fn), 'w').write(txt)
        print(fn, len(txt.encode()), 'bytes')
    tpl = open(os.path.join(os.path.dirname(__file__), 'dashboard_template.html')).read()
    html = tpl.replace('/*DATA*/null', json.dumps(board, allow_nan=False))
    open(os.path.join(SITE, 'tipoff_edge.html'), 'w').write(html)
    print(f"slate {slate}: {len(top10)} top picks, {len(gl)} games, {len(all_lines)} lines; status {status}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'run', offline='--offline' in sys.argv)
