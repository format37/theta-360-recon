#!/usr/bin/env python3
"""Step 5, the check. Distances you can verify with a tape, read off the fitted planes; and the scale error.

    python3 -I scripts/measure.py data/pass-0/rig.yaml

The scale comes from one number, the lens height (camera_height_cm). Everything here is independent of it:
  ceiling      floor plane to ceiling plane (indoors)
  wall pairs   distance between two fitted walls that face each other (same room axis, within 6 deg of each other,
               at least 2 lens heights apart and overlapping by at least 2)
Each gets a label, its length in model units and, once camera_height_cm is set, in cm.
Put tape numbers into the YAML as  checks_cm: {"<label>": cm}  and run again: the script prints the error of each.
A wall plane is fitted to what PatchMatch saw of the wall: skirting, sockets, frames. It can sit 1-2 cm
in front of the plaster.
Writes product/measure.json and product/plan.png (top view with the walls lettered). With camera_height_cm set it
also writes copies in metres: product/mesh_metric.ply and <splat>/splat_metric.ply.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
import pycolmap

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1])
M = json.load(open(p.product / 'mesh.json')); A = json.load(open(p.product / 'align.json'))
h = M['lens_height']
cm_per_unit = float(p.camera_height_cm) if p.camera_height_cm else None    # align_scale.py made one unit = one lens height
planes = M['planes']; out = []
floor = next((pl for pl in planes if pl['kind'] == 'floor'), None); ceil = next((pl for pl in planes if pl['kind'] == 'ceiling'), None)
walls = [pl for pl in planes if pl['kind'] == 'wall']
for i, pl in enumerate(walls): pl['letter'] = chr(ord('A') + i)
if ceil and floor:
    out.append(dict(label='ceiling height', units=ceil['z'] - floor['z'], note='floor plane to ceiling plane'))
for i, a in enumerate(walls):
    for b in walls[i + 1:]:
        na, nb = np.array(a['normal']), np.array(b['normal'])
        if a['normal_axis'] != b['normal_axis'] or na @ nb < 0.995: continue
        d = abs(a['d'] - b['d']); overlap = min(a['s1'], b['s1']) - max(a['s0'], b['s0'])
        if d >= 2 * h and overlap >= 2 * h:
            out.append(dict(label=f"wall {a['letter']} to wall {b['letter']}", units=float(d),
                            note=f"{'x' if a['normal_axis'] == 'x' else 'y'} = {-a['d'] / h:+.2f} and {-b['d'] / h:+.2f}; facing each other over "
                                 f"{overlap / h:.1f} lens heights; {np.degrees(np.arccos(min(1, na @ nb))):.1f} deg from parallel"))
# Outdoors: the building of the crop box, facade to facade. Its facades stand as dense lines in a top view of the PatchMatch
# points (0.5 to 8 lens heights up); the strongest line in the outer third at each end of the box, along x and along y, is a
# facade (the box is drawn tight round the building; in halves, an inner structure outnumbered a weakly seen facade).
box = p.cfg.get('crop') or {}
if p.outdoor and box.get('x') and box.get('y'):
    P_ = np.asarray(o3d.io.read_point_cloud(str(p.dense / 'fused.ply')).points); (x0, x1), (y0, y1) = box['x'], box['y']
    P_ = P_[(P_[:, 0] > x0) & (P_[:, 0] < x1) & (P_[:, 1] > y0) & (P_[:, 1] < y1) & (P_[:, 2] > 0.5 * h) & (P_[:, 2] < 8 * h)]
    def peaks(v, lo_, hi_):
        hist, e = np.histogram(v, bins=np.arange(lo_, hi_ + 0.25, 0.25)); th = len(hist) // 3
        a_, b_ = int(np.argmax(hist[:th])), len(hist) - th + int(np.argmax(hist[-th:]))
        return (e[a_] + e[a_ + 1]) / 2, (e[b_] + e[b_ + 1]) / 2, int(hist[a_]), int(hist[b_])
    xa, xb, na, nb = peaks(P_[:, 0], x0, x1); ya, yb, ma, mb = peaks(P_[:, 1], y0, y1)
    out.append(dict(label='building length, x', units=float(xb - xa), note=f'facade lines at x = {xa:+.2f} and {xb:+.2f} ({na} and {nb} points in their 0.25 bins)'))
    out.append(dict(label='building width, y', units=float(yb - ya), note=f'facade lines at y = {ya:+.2f} and {yb:+.2f} ({ma} and {mb} points in their 0.25 bins)'))
print(f"unit: one lens height" + (f" = {p.camera_height_cm} cm" if p.camera_height_cm else ' (camera_height_cm is empty: lengths below are in lens heights)'))
errs = []
for o in out:
    o['lens_heights'] = o['units']; o['cm'] = o['units'] * cm_per_unit if cm_per_unit else None
    tape = p.checks_cm.get(o['label'])
    line = f"  {o['label']:24s} {o['lens_heights']:7.3f} lens heights" + (f" = {o['cm']:7.1f} cm" if o['cm'] else '') + f"   ({o['note']})"
    if tape and o['cm']:
        o['tape_cm'] = float(tape); o['error_pct'] = 100 * (o['cm'] - tape) / tape; errs.append(o['error_pct'])
        line += f"   tape {tape} cm -> error {o['error_pct']:+.2f}%"
    print(line)
R = dict(lens_height_units=h, camera_height_cm=p.camera_height_cm, lengths=out,
         scale_error_pct=dict(mean_abs=float(np.mean(np.abs(errs))), worst=float(max(errs, key=abs)), n=len(errs)) if errs else None)
if errs: print(f"scale check on {len(errs)} tape numbers: mean |error| {np.mean(np.abs(errs)):.2f}%, worst {max(errs, key=abs):+.2f}%")
else: print('no scale error yet: needs camera_height_cm and at least one entry in checks_cm')
json.dump(R, open(p.product / 'measure.json', 'w'), indent=1)

# ---- top view
pts = np.asarray(o3d.io.read_point_cloud(str(p.dense / 'fused.ply')).points)
top = M['room_top']
sl = pts[(pts[:, 2] > 0.3 * h) & (pts[:, 2] < top - 0.5 * h)]
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
C = np.array([im.projection_center() for im in sorted(rec.images.values(), key=lambda im: im.name) if im.has_pose and im.name.startswith('v00/')])
fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(sl[::max(1, len(sl) // 400000), 0], sl[::max(1, len(sl) // 400000), 1], s=0.05, c='#555555', linewidths=0)
ax.plot(C[:, 0], C[:, 1], '-', color='#1f77b4', lw=2.5, label=f'{"walked" if p.outdoor else "rover"} path, {len(C)} shots'); ax.plot(*C[0, :2], 'o', color='#1f77b4')
for pl in walls:
    n, t = np.array(pl['normal']), np.array(pl['t']); a, b = -pl['d'] * n + pl['s0'] * t, -pl['d'] * n + pl['s1'] * t
    ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#d62728', lw=2.5)
    ax.text(*((a + b)[:2] / 2 + 0.35 * h * n[:2]), pl['letter'], color='#d62728', fontsize=14, weight='bold', ha='center', va='center')
lim = max(14 * h, 0.6 * float(np.ptp(C[:, :2], axis=0).max()) + 4 * h)            # outdoors: the whole walked area
ax.set(xlim=(C[:, 0].mean() - lim, C[:, 0].mean() + lim), ylim=(C[:, 1].mean() - lim, C[:, 1].mean() + lim), aspect='equal',
       xlabel=f'x, {M["unit"]}', ylabel=f'y, {M["unit"]}', title=('top view: PatchMatch points from 0.3 lens heights up, fitted facades in red' if p.outdoor else 'top view: PatchMatch points between floor and ceiling, fitted walls in red'))
ax.legend(loc='upper right'); plt.savefig(p.product / 'plan.png', dpi=110, bbox_inches='tight')
print('wrote', p.product / 'measure.json', p.product / 'plan.png')

# ---- copies in metres
if cm_per_unit:
    k = cm_per_unit / 100
    m = p.product / 'mesh.ply'
    if m.exists():
        mesh = o3d.io.read_triangle_mesh(str(m)); mesh.scale(k, center=(0, 0, 0)); o3d.io.write_triangle_mesh(str(p.product / 'mesh_metric.ply'), mesh)
        print('wrote', p.product / 'mesh_metric.ply')
    for f in p.run.glob('splat*/splat.ply'):                          # INRIA layout: x y z first, three log scales before the quaternion
        raw = open(f, 'rb').read(); head = raw[:raw.index(b'end_header\n') + 11]
        cols = [ln.split()[-1].decode() for ln in head.split(b'\n') if ln.startswith(b'property')]
        data = np.frombuffer(raw[len(head):], '<f4').reshape(-1, len(cols)).copy()
        data[:, [cols.index(c) for c in 'xyz']] *= k; data[:, [cols.index(f'scale_{i}') for i in range(3)]] += np.log(k)
        open(f.with_name('splat_metric.ply'), 'wb').write(head + data.astype('<f4').tobytes()); print('wrote', f.with_name('splat_metric.ply'))
