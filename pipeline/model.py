"""Projection models: one LightGBM per stat (Poisson for counts, L2 for fantasy points, logistic for
double/triple-doubles), each turned into a full outcome distribution with a negative binomial whose
dispersion is fitted out of sample by projection size."""
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats as S
import features as F

COUNT = ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa', 'ra', 'stocks']
BINARY = ['dd', 'td']
CONT = ['fant']
TARGETS = COUNT + CONT + BINARY + ['min']
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)


def _obj(t):
    return 'poisson' if t in COUNT else ('binary' if t in BINARY else 'regression')


def fit(train, valid, cols):
    """Train each target with early stopping on valid; returns models, best rounds, dispersion tables."""
    models, rounds, disp = {}, {}, {}
    for t in TARGETS:
        tr, va = train[train[t].notna()], valid[valid[t].notna()]
        ds = lgb.Dataset(tr[cols], tr[t]); dv = lgb.Dataset(va[cols], va[t])
        m = lgb.train({**PARAMS, 'objective': _obj(t)}, ds, 3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        models[t], rounds[t] = m, m.best_iteration
        if t in COUNT or t in CONT or t == 'min':
            mu = m.predict(va[cols], num_iteration=m.best_iteration)
            disp[t] = dispersion(mu, va[t].values, t)
    return models, rounds, disp


def refit(data, cols, rounds):
    return {t: lgb.train({**PARAMS, 'objective': _obj(t)}, lgb.Dataset(data[data[t].notna()][cols],
                         data[data[t].notna()][t]), max(rounds[t], 50)) for t in TARGETS}


def dispersion(mu, y, t, bins=12):
    """Spread of outcomes by projection size: NB alpha for counts (var = mu + a mu^2), sd for continuous."""
    q = np.unique(np.quantile(mu, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(q, mu, side='right') - 1, 0, len(q) - 2)
    out = []
    for b in range(len(q) - 1):
        s = idx == b
        if s.sum() < 50:
            continue
        m, v = mu[s].mean(), np.mean((y[s] - mu[s]) ** 2)
        val = max((v - m) / max(m, 1e-6) ** 2, 0.01) if t in COUNT else np.sqrt(v)
        out.append((float(m), float(val)))
    return out


def _interp(table, mu):
    xs, ys = np.array([a for a, _ in table]), np.array([b for _, b in table])
    return np.interp(mu, xs, ys)


def predict(models, disp, X, cols):
    P = pd.DataFrame(index=X.index)
    for t, m in models.items():
        P[t] = m.predict(X[cols])
        if t in disp:
            P[t + '_disp'] = _interp(disp[t], P[t].values)
    return P


def p_over(stat, mu, d, line):
    """P(outcome > line), with pushes (whole-number lines) removed as Underdog voids them."""
    mu = np.maximum(np.asarray(mu, float), 1e-3); line = np.asarray(line, float)
    if stat in COUNT:
        a = np.maximum(np.asarray(d, float), 1e-3)
        n = 1 / a; p = n / (n + mu)
        k = np.floor(line)
        over = 1 - S.nbinom.cdf(k, n, p)
        whole = np.isclose(line, np.round(line))
        under = np.where(whole, S.nbinom.cdf(k - 1, n, p), S.nbinom.cdf(k, n, p))
        return over / np.maximum(over + under, 1e-9)
    sd = np.maximum(np.asarray(d, float), 0.5)
    return 1 - S.norm.cdf((line - mu) / sd)


def train_all(f, meta):
    """Production models: every past game, fixed rounds from the walk-forward run."""
    cols = F.feature_cols(f)
    data = f[f.upcoming == 0] if 'upcoming' in f else f
    return refit(data, cols, meta['rounds']), cols
