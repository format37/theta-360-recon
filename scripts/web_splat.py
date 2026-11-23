#!/usr/bin/env python3
"""A light copy of a splat for a web page: cropped round the walk, thinned, 32 bytes a Gaussian.

    python3 -I scripts/web_splat.py data/pool/rig.yaml OUT_DIR [--tag splat] [--count 350000] [--reach 14]

Keeps the Gaussians within --reach lens heights of the walk (in plan), between z -3 and 13 lens heights and at opacity
0.05 or more; of those the --count with the largest opacity x area, which are what a view from above is made of. Colour is
the degree-0 part only (the view-dependent part is left out). Writes OUT_DIR/scene.splat (the antimatter15 layout:
position 3 x f32, scale 3 x f32, RGBA 4 x u8, rotation w x y z 4 x u8, little endian; read by gaussian-splats-3d and most
web viewers), OUT_DIR/sky.png (the learned sky, if any) and OUT_DIR/meta.json: the opening view (as the figures' raised view from the south-west), the building's
box, the sky colours (from the learned sky: zenith and horizon) and the counts. Units: lens heights, z up.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from project import load

ap = argparse.ArgumentParser(); ap.add_argument('yaml'); ap.add_argument('out'); ap.add_argument('--tag', default='splat')
ap.add_argument('--count', type=int, default=350_000); ap.add_argument('--reach', type=float, default=14)
ap.add_argument('--zmin', type=float, default=-3); ap.add_argument('--zmax', type=float, default=13)
a = ap.parse_args()
p = load(a.yaml); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
C = np.array([im.projection_center() for im in sorted(rec.images.values(), key=lambda im: im.name) if im.has_pose and im.name.startswith('v00/')])
ck = torch.load(p.run / a.tag / 'ckpt.pt', map_location='cpu')
M = ck['means'].numpy(); op = torch.sigmoid(ck['opacities']).numpy(); S = torch.exp(ck['scales']).numpy()
near = cKDTree(C[:, :2]).query(M[:, :2])[0] < a.reach
keep = near & (M[:, 2] > a.zmin) & (M[:, 2] < a.zmax) & (op >= 0.05)
ss = np.sort(S, 1); score = np.where(keep, op * ss[:, 1] * ss[:, 2], -1)
idx = np.argsort(-score)[:min(a.count, int(keep.sum()))]
idx = idx[np.argsort(np.lexsort((M[idx, 1], M[idx, 0])))]               # a stable order: the file compresses a little better

C0 = 0.28209479177387814
rgb = np.clip(0.5 + C0 * ck['sh0'][idx, 0].numpy(), 0, 1)
q = ck['quats'][idx].numpy(); q /= np.linalg.norm(q, axis=1, keepdims=True)
rec_ = np.zeros(len(idx), dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
rec_['p'] = M[idx]; rec_['s'] = S[idx]
rec_['c'] = np.round(np.c_[rgb, op[idx]] * 255).astype(np.uint8)
rec_['r'] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
(out / 'scene.splat').write_bytes(rec_.tobytes())

# the opening view: from the south-west, 30 deg above, looking at the middle of the walked area (as figures.py)
centre = np.r_[(C[:, :2].min(0) + C[:, :2].max(0)) / 2, 0.0]; span = float(np.ptp(C[:, :2], axis=0).max()) + 2
box = p.cfg.get('crop') or {}
look = np.array([np.mean(box['x']), np.mean(box['y']), 2.0]) if box.get('x') else centre + [0, 0, 2.0]     # the building, if a crop box is set
fov_v = 40.0; az, el = math.radians(225), math.radians(32); dist = 0.78 * span                 # about the framing of the figures' raised view
eye = look + dist * np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
sky = None
if 'env' in ck:                                                          # zenith and horizon colours of the learned sky
    env = torch.sigmoid(ck['env'][0]).permute(1, 2, 0).numpy(); hh = env.shape[0]
    sky = dict(zenith=[round(float(x), 3) for x in env[:hh // 8].reshape(-1, 3).mean(0)],
               horizon=[round(float(x), 3) for x in env[hh * 7 // 16:hh // 2].reshape(-1, 3).mean(0)])
meta = dict(count=int(len(idx)), of=int(len(M)), bytes=int(len(idx) * 32), unit='lens height', up=[0, 0, 1], fov=fov_v,
            eye=[round(float(x), 3) for x in eye], target=[round(float(x), 3) for x in look],
            extent=[[round(float(x), 2) for x in M[idx].min(0)], [round(float(x), 2) for x in M[idx].max(0)]],
            walk=[[round(float(x), 2) for x in c] for c in C[::2]], building=box or None, sky=sky)
json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
if (p.run / a.tag / 'sky.png').exists():                                # the learned sky as it is: u = atan2(y, x) / 2pi + 0.5, top row = up
    cv2.imwrite(str(out / 'sky.png'), cv2.imread(str(p.run / a.tag / 'sky.png')))
print(f'{out / "scene.splat"}: {len(idx):,} of {int(keep.sum()):,} kept ({len(M):,} in the splat), {len(idx) * 32 / 2**20:.1f} MB')
