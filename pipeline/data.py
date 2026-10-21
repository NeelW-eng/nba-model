"""Data sources (all free, no keys):
  * history: sportsdataverse ESPN player box scores, 2021-22 to 2025-26 (data/hist, downloaded once)
  * this season: ESPN scoreboard + game summaries, cached in data/box_current.parquet
  * injuries: ESPN NBA injuries feed
  * lines: Underdog pick'em (api.underdogfantasy.com/v1/over_under_lines with the web app's version header)
"""
import os, re, json, time, uuid, datetime, unicodedata
import numpy as np, pandas as pd, requests
from zoneinfo import ZoneInfo
from paths import DATA, HIST
import features as F

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'
ET = ZoneInfo('America/New_York')
SEASON = 2027                      # ESPN/sportsdataverse label for 2026-27
SEASON_START = datetime.date(2026, 10, 20)
HIST_YEARS = range(2022, 2027)
CUR = os.path.join(DATA, 'box_current.parquet')
SDV = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{y}.parquet'
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
TEAM_NAMES = {'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
              'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
              'Detroit Pistons': 'DET', 'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU',
              'Indiana Pacers': 'IND', 'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
              'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
              'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NO', 'New York Knicks': 'NY',
              'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
              'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
              'San Antonio Spurs': 'SA', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH', 'Washington Wizards': 'WSH'}
ABBR = {'GSW': 'GS', 'NOP': 'NO', 'NYK': 'NY', 'SAS': 'SA', 'UTA': 'UTAH', 'WAS': 'WSH', 'PHO': 'PHX', 'BRK': 'BKN',
        'NOR': 'NO', 'CHO': 'CHA', 'GOS': 'GS'}


def team_abbr(a):
    a = (a or '').upper()
    return ABBR.get(a, a)


def norm_name(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"[.'`\-]", ' ', s)
    toks = [t for t in s.split() if t not in ('jr', 'sr', 'ii', 'iii', 'iv', 'v')]
    return ' '.join(toks)


def get(url, params=None, headers=None, tries=3, timeout=30):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={'User-Agent': UA, **(headers or {})}, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 401, 403, 404, 426):
                r.raise_for_status()
        except requests.HTTPError:
            raise
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f'GET failed: {url}')


# ---------- history ----------
def history():
    frames = []
    for y in HIST_YEARS:
        p = os.path.join(HIST, f'player_box_{y}.parquet')
        if not os.path.exists(p):
            open(p, 'wb').write(get(SDV.format(y=y), timeout=120).content)
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


# ---------- this season (ESPN) ----------
def _et_date(iso):
    return datetime.datetime.fromisoformat(iso.replace('Z', '+00:00')).astimezone(ET).date()


def scoreboard(day):
    return get(f'{ESPN}/scoreboard', params={'dates': day.strftime('%Y%m%d'), 'limit': 50}).json()


def parse_summary(js, gid):
    comp = js['header']['competitions'][0]
    teams = {c['team']['abbreviation']: c for c in comp['competitors']}
    stype = js['header'].get('season', {}).get('type', 2)
    gdate = _et_date(comp['date'])
    rows = []
    for tb in js.get('boxscore', {}).get('players', []):
        ab = tb['team']['abbreviation']
        opp = [a for a in teams if a != ab][0]
        for blk in tb.get('statistics', []):
            keys = blk.get('keys') or []
            for a in blk.get('athletes', []):
                st = dict(zip(keys, a.get('stats') or []))

                def num(k):
                    v = st.get(k)
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return np.nan

                def pair(k, i):
                    v = st.get(k)
                    try:
                        return float(str(v).split('-')[i])
                    except (TypeError, ValueError, IndexError):
                        return np.nan
                ath = a.get('athlete', {})
                rows.append({
                    'game_id': int(gid), 'season': SEASON, 'season_type': int(stype), 'game_date': gdate.isoformat(),
                    'athlete_id': float(ath.get('id')), 'athlete_display_name': ath.get('displayName'),
                    'team_abbreviation': ab, 'opponent_team_abbreviation': opp,
                    'home_away': teams[ab].get('homeAway'),
                    'athlete_position_abbreviation': (ath.get('position') or {}).get('abbreviation'),
                    'starter': bool(a.get('starter')), 'did_not_play': bool(a.get('didNotPlay')) or not a.get('stats'),
                    'reason': a.get('reason'), 'active': a.get('active'),
                    'team_score': float(teams[ab].get('score') or 0), 'opponent_team_score': float(teams[opp].get('score') or 0),
                    'minutes': num('minutes'), 'points': num('points'), 'rebounds': num('rebounds'),
                    'assists': num('assists'), 'steals': num('steals'), 'blocks': num('blocks'),
                    'turnovers': num('turnovers'), 'offensive_rebounds': num('offensiveRebounds'),
                    'field_goals_attempted': pair('fieldGoalsMade-fieldGoalsAttempted', 1),
                    'three_point_field_goals_made': pair('threePointFieldGoalsMade-threePointFieldGoalsAttempted', 0),
                    'three_point_field_goals_attempted': pair('threePointFieldGoalsMade-threePointFieldGoalsAttempted', 1),
                    'free_throws_attempted': pair('freeThrowsMade-freeThrowsAttempted', 1),
                })
    return rows


def update_current(today):
    """Fetch box scores for every finished game this season not yet cached."""
    cur = pd.read_parquet(CUR) if os.path.exists(CUR) else pd.DataFrame()
    have = set(cur.game_id.unique()) if len(cur) else set()
    new, day = [], SEASON_START
    while day < today + datetime.timedelta(days=1):
        try:
            sb = scoreboard(day)
        except Exception as e:
            print('scoreboard failed', day, e); day += datetime.timedelta(days=1); continue
        for ev in sb.get('events', []):
            if not ev['status']['type'].get('completed') or int(ev['id']) in have:
                continue
            try:
                js = get(f'{ESPN}/summary', params={'event': ev['id']}).json()
                new += parse_summary(js, ev['id'])
            except Exception as e:
                print('summary failed', ev['id'], e)
        day += datetime.timedelta(days=1)
    if new:
        cur = pd.concat([cur, pd.DataFrame(new)], ignore_index=True)
        cur.to_parquet(CUR, index=False)
    print(f'current season: {cur.game_id.nunique() if len(cur) else 0} games ({len(set(r["game_id"] for r in new))} new)')
    return cur


# ---------- injuries ----------
def injuries():
    js = get(f'{ESPN}/injuries').json()
    out = []
    for t in js.get('injuries', []):
        ab = TEAM_NAMES.get(t.get('displayName'))
        for i in t.get('injuries', []):
            a = i.get('athlete', {})
            pid = None
            for l in a.get('links', []):
                m = re.search(r'/id/(\d+)', l.get('href', ''))
                if m:
                    pid = int(m.group(1)); break
            out.append({'team': ab, 'pid': pid, 'name': a.get('displayName'), 'status': i.get('status'),
                        'detail': (i.get('shortComment') or '')[:240], 'date': i.get('date')})
    return pd.DataFrame(out)


# ---------- Underdog ----------
def underdog_raw():
    cv = get('https://app.underdogsports.com/client-version').json()['clientVersion']
    h = {'Accept': 'application/json', 'Client-Type': 'web', 'Client-Version': cv,
         'Client-Device-Id': str(uuid.uuid4()), 'Origin': 'https://app.underdogsports.com',
         'Referer': 'https://app.underdogsports.com/'}
    return get('https://api.underdogfantasy.com/v1/over_under_lines', params={'sport_id': 'NBA'}, headers=h,
               timeout=60).json()
