import json, os, uuid, requests
os.makedirs('probe/out', exist_ok=True)
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'
res = {}
try:
    cv = requests.get('https://app.underdogsports.com/client-version', headers={'User-Agent': UA}, timeout=20).json()['clientVersion']
    res['cv'] = cv
    h = {'User-Agent': UA, 'Accept': 'application/json', 'Client-Type': 'web', 'Client-Version': cv,
         'Client-Device-Id': str(uuid.uuid4()), 'Origin': 'https://app.underdogsports.com', 'Referer': 'https://app.underdogsports.com/'}
    r = requests.get('https://api.underdogfantasy.com/v1/over_under_lines', params={'sport_id': 'NBA'}, headers=h, timeout=40)
    res['lines'] = [r.status_code, len(r.content), r.text[:200]]
    if r.ok:
        open('probe/out/ud_nba.json', 'wb').write(r.content)
except Exception as e:
    res['error'] = repr(e)[:300]
json.dump(res, open('probe/out/ud3_status.json', 'w'), indent=1)
