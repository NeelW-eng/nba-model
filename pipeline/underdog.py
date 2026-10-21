"""Underdog NBA lines -> one row per (player, stat, line) with each side's price and payout multiplier."""
import datetime
import numpy as np, pandas as pd
import data as D

STAT = {'points': 'pts', 'rebounds': 'reb', 'assists': 'ast', 'three_points_made': 'fg3m',
        'pts_rebs_asts': 'pra', 'pts_rebs': 'pr', 'pts_asts': 'pa', 'rebs_asts': 'ra', 'steals': 'stl',
        'blocks': 'blk', 'turnovers': 'tov', 'blks_stls': 'stocks', 'fantasy_points': 'fant',
        'double_doubles': 'dd', 'triple_doubles': 'td'}
LABEL = {'pts': 'Points', 'reb': 'Rebounds', 'ast': 'Assists', 'fg3m': '3-Pointers Made', 'pra': 'Pts+Rebs+Asts',
         'pr': 'Pts+Rebs', 'pa': 'Pts+Asts', 'ra': 'Rebs+Asts', 'stl': 'Steals', 'blk': 'Blocks',
         'tov': 'Turnovers', 'stocks': 'Blks+Stls', 'fant': 'Fantasy Points', 'dd': 'Double-Double',
         'td': 'Triple-Double'}


def implied(am):
    try:
        a = float(am)
    except (TypeError, ValueError):
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def parse(js):
    games = {}
    team_of = {}
    for g in js.get('games', []):
        if g.get('sport_id') != 'NBA':
            continue
        t = g.get('abbreviated_title') or ''
        if ' @ ' not in t:
            continue
        a, h = [D.team_abbr(x.strip()) for x in t.split(' @ ')]
        team_of[g['away_team_id']], team_of[g['home_team_id']] = a, h
        kick = datetime.datetime.fromisoformat(g['scheduled_at'].replace('Z', '+00:00'))
        games[g['id']] = {'match_id': g['id'], 'away': a, 'home': h, 'game': f'{a} @ {h}', 'kickoff': kick,
                          'date': kick.astimezone(D.ET).date().isoformat(), 'status': g.get('status')}
    players = {p['id']: p for p in js.get('players', [])}
    apps = {a['id']: a for a in js.get('appearances', [])}
    rows = []
    for l in js.get('over_under_lines', []):
        if l.get('status') != 'active' or l.get('live_event'):
            continue
        ou = l.get('over_under') or {}
        st = (ou.get('appearance_stat') or {})
        stat = STAT.get(st.get('stat'))
        app = apps.get(st.get('appearance_id'))
        if not stat or not app or app.get('match_id') not in games:
            continue
        g = games[app['match_id']]
        team = team_of.get(app.get('team_id'))
        if team is None:
            continue
        pl = players.get(app.get('player_id'), {})
        sides = {}
        for o in l.get('options', []):
            if o.get('status', 'active') != 'active':
                continue
            sides[o.get('choice')] = (implied(o.get('american_price')), float(o.get('payout_multiplier') or 1))
        if not sides:
            continue
        name = f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip()
        r = {**g, 'line_id': l['id'], 'player': name, 'ud_player_id': app.get('player_id'), 'team': team,
             'opp': g['home'] if team == g['away'] else g['away'], 'home': int(team == g['home']),
             'stat': stat, 'stat_label': LABEL[stat], 'line': float(l['stat_value']),
             'image': pl.get('image_url')}
        for side in ('higher', 'lower'):
            imp, mult = sides.get(side, (None, None))
            r[f'{side}_imp'], r[f'{side}_mult'] = imp, mult
        # market probability of Higher: de-vig both sides; one-sided lines assume a 4.5% margin
        hi, lo = r['higher_imp'], r['lower_imp']
        if hi and lo:
            r['p_market'] = hi / (hi + lo)
        elif hi:
            r['p_market'] = min(max(hi / 1.045, 0.02), 0.98)
        elif lo:
            r['p_market'] = 1 - min(max(lo / 1.045, 0.02), 0.98)
        else:
            r['p_market'] = None
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(games.values())
