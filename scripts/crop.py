#!/usr/bin/env python3
"""The building alone: the mesh, the voxel cubes and the splats cut to a box.

    python3 -I scripts/crop.py data/pool/rig.yaml

The box is crop: {x: [x0, x1], y: [y0, y1], z: [z0, z1]} in the YAML, in the scene frame of colmap/aligned (lens
heights, z up, the ground at z = 0); read it off product/plan.png, where the facades are lettered. z may be left out.
Everything else of the run stays as it is: the poses and the dense cloud need the whole scene.
Writes product/mesh_building.ply, product/voxels_cubes_building.ply, <splat>/splat_building.ply (from splat_view.ply)
and product/crop.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1]); box = p.cfg.get('crop') or {}
if not box: raise SystemExit('no crop: {x: [..], y: [..], z: [..]} in the YAML')
lo = np.array([box.get(k, [-1e9, 1e9])[0] for k in 'xyz'], float); hi = np.array([box.get(k, [-1e9, 1e9])[1] for k in 'xyz'], float)
inside = lambda X: np.all((X >= lo) & (X <= hi), axis=1)
J = dict(box=dict(lo=lo.tolist(), hi=hi.tolist()))
for src, dst in (('mesh.ply', 'mesh_building.ply'), ('voxels_cubes.ply', 'voxels_cubes_building.ply')):
    f = p.product / src
    if not f.exists(): continue
    m = o3d.io.read_triangle_mesh(str(f)); n0 = len(m.triangles)
    V = np.asarray(m.vertices); T = np.asarray(m.triangles)
    m.remove_triangles_by_mask(~inside(V[T].mean(1))); m.remove_unreferenced_vertices()
    if src == 'mesh.ply':                                                # loose bits left at the cut
        tc, nt, _ = m.cluster_connected_triangles(); m.remove_triangles_by_mask(np.asarray(nt)[np.asarray(tc)] < 2000); m.remove_unreferenced_vertices()
    if len(m.triangles) > 1_500_000: m = m.simplify_quadric_decimation(1_500_000); m.remove_unreferenced_vertices()   # light enough for the browser
    m.compute_vertex_normals(); o3d.io.write_triangle_mesh(str(p.product / dst), m)
    J[dst] = dict(triangles=len(m.triangles), of=n0); print(f'{p.product / dst}: {len(m.triangles):,} of {n0:,} triangles')
for f in sorted(p.run.glob('splat*/splat_view.ply')):
    raw = open(f, 'rb').read(); head = raw[:raw.index(b'end_header\n') + 11]
    cols = [ln.split()[-1].decode() for ln in head.split(b'\n') if ln.startswith(b'property')]
    data = np.frombuffer(raw[len(head):], '<f4').reshape(-1, len(cols))
    keep = inside(data[:, [cols.index(c) for c in 'xyz']]); out = f.with_name('splat_building.ply')
    head2 = head.replace(f'element vertex {len(data)}'.encode(), f'element vertex {int(keep.sum())}'.encode())
    open(out, 'wb').write(head2 + data[keep].astype('<f4').tobytes())
    J[str(out.relative_to(p.run))] = dict(gaussians=int(keep.sum()), of=len(data)); print(f'{out}: {int(keep.sum()):,} of {len(data):,} Gaussians')
json.dump(J, open(p.product / 'crop.json', 'w'), indent=1)
