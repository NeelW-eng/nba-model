import json, os, requests
os.makedirs('probe/out', exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
      'Accept': 'application/json, text/plain, */*'}
NBA = dict(UA, **{'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com', 'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true'})
T = {
 'ud_v6': ('https://api.underdogfantasy.com/beta/v6/over_under_lines', UA),
 'ud_v5': ('https://api.underdogfantasy.com/beta/v5/over_under_lines', UA),
 'ud_v3': ('https://api.underdogfantasy.com/beta/v3/over_under_lines', UA),
 'espn_inj': ('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries', UA),
 'espn_sb': ('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard', UA),
 'nba_cdn_sched': ('https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json', UA),
 'nba_cdn_sb': ('https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json', UA),
 'nba_stats': ('https://stats.nba.com/stats/leaguegamelog?Counter=0&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=2025-26&SeasonType=Regular%20Season&Sorter=DATE', NBA),
 'nba_inj_pdf_idx': ('https://official.nba.com/nba-injury-report-2025-26-season/', UA),
 'bref': ('https://www.basketball-reference.com/leagues/NBA_2026_per_game.html', UA),
}
res = {}
for k, (u, h) in T.items():
    try:
        r = requests.get(u, headers=h, timeout=25)
        res[k] = {'status': r.status_code, 'bytes': len(r.content)}
        if r.ok:
            open(f'probe/out/{k}.txt', 'wb').write(r.content[:3_000_000])
    except Exception as e:
        res[k] = {'error': repr(e)[:200]}
    print(k, res[k])
json.dump(res, open('probe/out/status.json', 'w'), indent=1)
