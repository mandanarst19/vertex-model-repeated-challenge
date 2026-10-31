import json, numpy as np
from vm_core import relax
EPS, TARGET = 0.04, 3.842
Ds, sims = [0.25, 0.35, 0.45, 0.55, 0.65, 0.70], [1000, 1001, 1002]
res = {}
for D in Ds:
    ps, dh = [], []
    for s in sims:
        c, _ = relax(D, s, EPS); m = c.mesh; a = m.area; al = a > 0
        n = np.bincount(m.face_id_by_edge, minlength=len(a))
        ps.append(np.mean(m.perimeter[al] / np.sqrt(a[al]))); dh.append(np.abs(n[al] - 6).mean())
    res[D] = (float(np.mean(ps)), float(np.mean(dh))); print(D, res[D], flush=True)
x = np.array(Ds); y = np.array([res[d][0] for d in Ds])
ok = np.all(np.diff(y) > 0)
Dstar = float(np.interp(TARGET, y, x)) if (y.min() <= TARGET <= y.max()) else None
json.dump(dict(table=res, target=TARGET, monotonic=bool(ok), D_star=Dstar), open('../results/stage2_calibration.json', 'w'), indent=1)
print('monotonic:', ok, ' D*=', Dstar)
