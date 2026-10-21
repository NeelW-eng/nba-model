"""Walk-forward check: train 2021-22..2023-24, tune on 2024-25, test on every 2025-26 game.
Underdog doesn't publish past lines, so test lines are rebuilt from what a bettor could see before the game
(blend of season average and last 5, rounded to .5)."""
import json, sys, numpy as np, pandas as pd
import features as F, model as M
from paths import DATA

LINE_STATS = ['pts', 'reb', 'ast', 'fg3m', 'pra', 'pr', 'pa', 'ra', 'stl', 'blk', 'tov', 'stocks']


def proxy_lines(f):
    """Pre-game line proxy: 0.6 x season average + 0.4 x last-5 average, to the nearest .5 below/above."""
    f = f.sort_values(['pid', 'date'])
    g = f.groupby(['pid', 'season'])
    out = {}
    for s in LINE_STATS:
        sea = g[s].transform(lambda x: x.shift().expanding().mean())
        l5 = g[s].transform(lambda x: x.shift().rolling(5, min_periods=3).mean())
        out[s] = (np.floor(0.6 * sea + 0.4 * l5) + 0.5).where(g.cumcount() >= 5)
    return pd.DataFrame(out, index=f.index).reindex(f.index)


def run(f):
    cols = F.feature_cols(f)
    tr, va, te = f[f.season <= 2024], f[f.season == 2025], f[f.season == 2026]
    import pickle, os
    cache = '/tmp/claude-0/-home-claude/57010de7-92b9-53e0-b8cf-7e975f5d9a40/scratchpad/bt_models.pkl'
    if os.path.exists(cache):
        models, rounds, disp = pickle.load(open(cache, 'rb'))
    else:
        models, rounds, disp = M.fit(tr, va, cols)
        pickle.dump((models, rounds, disp), open(cache, 'wb'))
    P = M.predict(models, disp, te, cols)
    L = proxy_lines(te).loc[te.index]
    res, rows = {}, []
    for s in LINE_STATS:
        ok = L[s].notna() & (te.min_ewm6 >= 15)
        line, y = L[s][ok].values, te[s][ok].values
        p = M.p_over(s, P[s][ok].values, P[s + '_disp'][ok].values, line)
        hit = (y > line).astype(float)
        ll = -np.mean(hit * np.log(np.clip(p, 1e-6, 1)) + (1 - hit) * np.log(np.clip(1 - p, 1e-6, 1)))
        side_p = np.maximum(p, 1 - p); side_hit = np.where(p >= 0.5, hit, 1 - hit)
        r = {'n': int(ok.sum()), 'over_rate': round(float(hit.mean()), 3), 'logloss': round(float(ll), 4),
             'logloss_coinflip': round(float(np.log(2)), 4),
             'mae_model': round(float(np.mean(np.abs(P[s][ok].values - y))), 3),
             'mae_line': round(float(np.mean(np.abs(line - y))), 3)}
        for th in (0.55, 0.6, 0.65):
            m = side_p >= th
            r[f'hit_at_{int(th*100)}'] = [int(m.sum()), round(float(side_hit[m].mean()), 3) if m.sum() else None]
        res[s] = r
        rows.append(pd.DataFrame({'stat': s, 'p': side_p, 'hit': side_hit}))
    allr = pd.concat(rows)
    allr['bucket'] = (allr.p * 20).astype(int) / 20
    cal = allr.groupby('bucket').agg(n=('hit', 'size'), pred=('p', 'mean'), actual=('hit', 'mean')).reset_index()
    return res, cal, rounds, disp


if __name__ == '__main__':
    f = pd.read_parquet(sys.argv[1])
    res, cal, rounds, disp = run(f)
    json.dump({'rounds': rounds, 'disp': disp}, open(f'{DATA}/model_meta.json', 'w'))
    print(json.dumps(res, indent=1)); print(cal.round(3).to_string()); print(rounds)
    json.dump({'by_stat': res, 'calibration': cal.round(4).to_dict('records'), 'rounds': rounds},
              open(f'{DATA}/backtest_summary.json', 'w'), indent=1)
