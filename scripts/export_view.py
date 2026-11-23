#!/usr/bin/env python3
"""Light copies for the web viewers: the splat without its near-transparent Gaussians, the mesh thinned.

    python3 -I scripts/export_view.py data/pool/rig.yaml [--tag splat]

MCMC training keeps its whole budget of Gaussians alive and a large part of it ends up almost transparent.
Writes <tag>/splat_view.ply for every splat (or the one --tag), product/mesh_view.ply (600k triangles indoors,
1.5M outdoors) and product/view.json (where the viewers put the eye).
Indoors the splat opens on the path at the lens height, looking along the drive. Outdoors it opens at the pose of
the face that saw the most 3D points (a level view of a facade, not of the sky), and the mesh viewer from 30 deg
above the walked area.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1])
tags = [sys.argv[sys.argv.index('--tag') + 1]] if '--tag' in sys.argv else sorted(d.name for d in p.run.glob('splat*') if (d / 'ckpt.pt').exists())
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
C = np.array([im.projection_center() for im in sorted(rec.images.values(), key=lambda im: im.name) if im.has_pose and im.name.startswith('v00/')])
h = float(np.median(C[:, 2])); mid = C[len(C) // 2]
if p.outdoor:
    lower = {f'v{v:02d}' for v, (_, pt) in enumerate(p.views) if pt < 0}
    cand = [im for im in rec.images.values() if im.has_pose and im.name[:3] in lower]
    box = p.cfg.get('crop') or {}
    if box.get('x') and box.get('y'):                                   # a face that looks at the building, 10-25 lens heights from its middle
        bc = np.array([np.mean(box['x']), np.mean(box['y'])])
        def aims(im):
            c = im.projection_center()[:2]; fw = im.cam_from_world().rotation.matrix()[2][:2]; to = bc - c
            return 10 * h < np.linalg.norm(to) < 25 * h and fw @ to / (np.linalg.norm(fw) * np.linalg.norm(to) + 1e-9) > np.cos(np.radians(25))
        cand = [im for im in cand if aims(im)] or cand
    im = max(cand, key=lambda im: im.num_points3D)
    eye = im.projection_center(); fwd = im.cam_from_world().rotation.matrix()[2]; fwd = fwd - fwd[2] * np.array([0, 0, 1.0]); fwd /= np.linalg.norm(fwd)
    ctr = np.r_[(C[:, :2].min(0) + C[:, :2].max(0)) / 2, 0.0]; span = float(np.ptp(C[:, :2], axis=0).max())
    view = dict(lens_height=h, eye=[float(x) for x in eye], look=[float(x) for x in eye + 6 * h * fwd], start_face=im.name,
                mesh_eye=[float(x) for x in ctr + np.array([-0.55, -0.55, 0.6]) * span], mesh_look=[float(x) for x in ctr])
else:
    d = (C[-1] - C[0]) / np.linalg.norm(C[-1] - C[0])                   # the splat opens where it is at its best: on the path, at the lens height
    view = dict(lens_height=h, eye=[float(x) for x in C[1]], look=[float(x) for x in C[1] + 10 * h * d],
                mesh_eye=[float(mid[0] - 10 * h), float(mid[1] - 22 * h), 30 * h], mesh_look=[float(mid[0]), float(mid[1]), 2 * h])
json.dump(view, open(p.product / 'view.json', 'w'), indent=1)

for tag in tags:
    ck = p.run / tag / 'ckpt.pt'
    if not ck.exists(): continue
    d = torch.load(ck, map_location='cpu'); keep = torch.sigmoid(d['opacities']) >= 0.05; n = int(keep.sum())
    K_sh = d['shN'].shape[1] + 1
    data = torch.cat([d['means'][keep], torch.zeros(n, 3), d['sh0'][keep].reshape(n, 3), d['shN'][keep].permute(0, 2, 1).reshape(n, -1),
                      d['opacities'][keep].reshape(n, 1), d['scales'][keep], F.normalize(d['quats'][keep], dim=1)], 1).float().numpy()
    props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{j}' for j in range(3 * (K_sh - 1))] \
            + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    out = p.run / tag / 'splat_view.ply'
    with open(out, 'wb') as fh:
        fh.write(('ply\nformat binary_little_endian 1.0\n' + f'element vertex {n}\n' + ''.join(f'property float {q}\n' for q in props) + 'end_header\n').encode())
        fh.write(data.astype('<f4').tobytes())
    print(f'{out}: {n:,} of {len(keep):,} Gaussians (opacity >= 0.05), {out.stat().st_size / 2**20:.0f} MB')
m = p.product / 'mesh.ply'
if m.exists():
    mesh = o3d.io.read_triangle_mesh(str(m)); n0 = len(mesh.triangles); cap = 1_500_000 if p.outdoor else 600_000
    lite = mesh.simplify_quadric_decimation(cap) if n0 > cap else mesh
    lite.remove_unreferenced_vertices(); lite.compute_vertex_normals(); o3d.io.write_triangle_mesh(str(p.product / 'mesh_view.ply'), lite)
    print(f'{p.product / "mesh_view.ply"}: {len(lite.triangles):,} of {n0:,} triangles')
