"""Season tracker: logs the Top 10 and each game's top 3, locks them at tip-off, grades them from box scores.
A player who doesn't play is void (as on Underdog); a whole-number line hit exactly is a push."""
import os, datetime
import numpy as np, pandas as pd
from paths import DATA
import data as D

LOG = os.path.join(DATA, 'picks_log.csv')
COLS = ['date', 'logged', 'kickoff', 'match_id', 'game', 'player', 'pid', 'team', 'opp', 'stat', 'stat_label', 'line',
        'side', 'word', 'mult', 'p', 'p_model', 'p_market', 'value', 'proj', 'top10', 'rank', 'game3',
        'questionable', 'result', 'actual']


def load():
    if not os.path.exists(LOG):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(LOG).reindex(columns=COLS)
    for c in ('top10', 'game3', 'questionable'):
        d[c] = d[c].fillna(False).astype(str).str.lower().isin(['true', '1'])
    d['result'] = d['result'].astype(object)
    return d


def log(S, now):
    """Replace this slate's not-yet-started picks with the new ones; started games stay as logged."""
    d = load()
    kick = pd.to_datetime(d.kickoff, utc=True, errors='coerce')
    dates = set(S.date.astype(str)) if len(S) else set()
    keep = d[~d.date.astype(str).isin(dates) | (kick <= pd.Timestamp(now))]
    sel = S[(S.top10 | S.game3) & ~S.started].copy()
    sel['logged'] = now.isoformat(timespec='minutes')
    sel['p_model'] = sel.p_model_side
    sel['p_market'] = sel.p_market_side
    locked = set(zip(keep.date.astype(str), keep.match_id.astype(str)))
    sel = sel[[(str(a), str(b)) not in locked for a, b in zip(sel.date, sel.match_id)]]
    out = pd.concat([keep, sel.reindex(columns=COLS)], ignore_index=True)
    out['kickoff'] = pd.to_datetime(out.kickoff, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    out.to_csv(LOG, index=False)
    return out


def grade(box_cur, now):
    d = load()
    if d.empty or box_cur is None or not len(box_cur):
        return d
    b = box_cur.copy()
    b['date'] = pd.to_datetime(b.date).dt.date.astype(str)
    finals = set(b.game_id.unique())
    by = {(int(r.pid), r.date): r for r in b.itertuples(index=False)}
    team_done = set(zip(b.team, b.date))
    for i, r in d.iterrows():
        if isinstance(r.result, str) and r.result in ('win', 'loss', 'push', 'void'):
            continue
        if pd.to_datetime(r.kickoff, utc=True) > pd.Timestamp(now) - pd.Timedelta(hours=3):
            continue
        key = (int(r.pid), str(r.date))
        if key not in by:
            if (r.team, str(r.date)) in team_done:
                d.at[i, 'result'] = 'void'
            continue
        x = by[key]
        if not x.played:
            d.at[i, 'result'] = 'void'; continue
        a = float(getattr(x, r.stat))
        d.at[i, 'actual'] = a
        if a == r.line:
            d.at[i, 'result'] = 'push'
        else:
            d.at[i, 'result'] = 'win' if (a > r.line) == (r.side == 'higher') else 'loss'
    d.to_csv(LOG, index=False)
    return d


def _rec(g):
    w, l = int((g.result == 'win').sum()), int((g.result == 'loss').sum())
    n = w + l
    pay = float(g[g.result == 'win'].mult.sum() / n) if n else None
    return {'w': w, 'l': l, 'v': int(g.result.isin(['void', 'push']).sum()), 'n': n,
            'pct': round(w / n, 3) if n else None, 'pay': round(pay, 3) if pay is not None else None,
            'exp': round(float(g[g.result.isin(['win', 'loss'])].p.mean()), 3) if n else None}


def summary(d):
    g = d[d.result.isin(['win', 'loss', 'push', 'void'])].copy()
    out = {'top10': _rec(g[g.top10]), 'game3': _rec(g[g.game3]), 'all': _rec(g)}
    out['by_day'] = [{'date': k, **_rec(x[x.top10]), 'g3': _rec(x[x.game3])} for k, x in g.groupby('date')]
    out['by_stat'] = sorted([{'stat': k, **_rec(x)} for k, x in g.groupby('stat_label')], key=lambda r: -r['n'])
    pl = [{'player': k, **_rec(x)} for k, x in g.groupby('player')]
    out['by_player'] = sorted([p for p in pl if p['n'] >= 2], key=lambda r: (-(r['pct'] or 0), -r['n']))
    g['bucket'] = (np.floor(g.p.astype(float) * 20) / 20).round(2)
    out['by_chance'] = [{'bucket': float(k), **_rec(x)} for k, x in g.groupby('bucket')]
    out['by_side'] = [{'side': k, **_rec(x)} for k, x in g.groupby('word')]
    return out
