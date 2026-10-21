import json, os, requests
os.makedirs('probe/out', exist_ok=True)
B = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
     'Accept': 'application/json', 'Origin': 'https://app.underdogfantasy.com', 'Referer': 'https://app.underdogfantasy.com/'}
variants = {
 'sp_v6': ('https://api.underdogsports.com/beta/v6/over_under_lines', {}),
 'sp_v5': ('https://api.underdogsports.com/beta/v5/over_under_lines', {}),
 'sp_v6_web': ('https://api.underdogsports.com/beta/v6/over_under_lines', {'client-type': 'web', 'client-version': '20261020000000'}),
 'sp_v6_origin': ('https://api.underdogsports.com/beta/v6/over_under_lines', {'Origin': 'https://app.underdogsports.com', 'Referer': 'https://app.underdogsports.com/'}),
 'sp_stats': ('https://stats.underdogsports.com/v1/sports/nba/players', {}),
}
res = {}
for k, (u, h) in variants.items():
    try:
        r = requests.get(u, headers={**B, **h}, timeout=25)
        res[k] = {'status': r.status_code, 'bytes': len(r.content), 'head': r.text[:400], 'hdrs': dict(list(r.headers.items())[:25])}
        if r.ok: open(f'probe/out/ud_{k}.txt', 'wb').write(r.content[:4_000_000])
    except Exception as e:
        res[k] = {'error': repr(e)[:200]}
json.dump(res, open('probe/out/ud2_status.json', 'w'), indent=1)
