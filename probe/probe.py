import json, os, requests
os.makedirs('probe/out', exist_ok=True)
B = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36',
     'Accept': 'application/json', 'Origin': 'https://app.underdogfantasy.com', 'Referer': 'https://app.underdogfantasy.com/'}
variants = {
 'v6_plain': ('https://api.underdogfantasy.com/beta/v6/over_under_lines', {}),
 'v6_web': ('https://api.underdogfantasy.com/beta/v6/over_under_lines', {'client-type': 'web', 'client-version': '20261020000000', 'client-device-id': 'probe'}),
 'v7_web': ('https://api.underdogfantasy.com/beta/v7/over_under_lines', {'client-type': 'web', 'client-version': '20261020000000'}),
 'v8_web': ('https://api.underdogfantasy.com/beta/v8/over_under_lines', {'client-type': 'web', 'client-version': '20261020000000'}),
 'v1_pickem': ('https://api.underdogfantasy.com/v1/pickem_searches', {'client-type': 'web', 'client-version': '20261020000000'}),
 'stats_v1': ('https://stats.underdogfantasy.com/v1/sports/nba/players', {}),
 'app_html': ('https://app.underdogfantasy.com/pick-em/higher-lower/all/nba', {}),
}
res = {}
for k, (u, h) in variants.items():
    try:
        r = requests.get(u, headers={**B, **h}, timeout=25)
        res[k] = {'status': r.status_code, 'bytes': len(r.content), 'head': r.text[:400], 'hdrs': dict(list(r.headers.items())[:25])}
        if r.ok: open(f'probe/out/ud_{k}.txt', 'wb').write(r.content[:4_000_000])
    except Exception as e:
        res[k] = {'error': repr(e)[:200]}
json.dump(res, open('probe/out/ud_status.json', 'w'), indent=1)
