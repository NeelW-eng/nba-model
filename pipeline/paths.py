import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
HIST = os.path.join(DATA, 'hist')
SITE = os.path.join(ROOT, 'site')
for p in (DATA, HIST, SITE):
    os.makedirs(p, exist_ok=True)
