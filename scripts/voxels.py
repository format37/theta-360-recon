#!/usr/bin/env python3
"""Step 6. The voxel model: the scene as coloured cubes of voxels.size lens heights. From galaxy-s26-ultra-stereo.

    python3 -I scripts/voxels.py data/pool/rig.yaml
    python3 scripts/voxview.py data/pool/rig.yaml          # look at it

Built from the depth maps of the dense faces (voxels.source: filled = PatchMatch + prior + planes, as in the
mesh; mvs = PatchMatch only), on the GPU, in a grid over the box of the mesh (cut at mesh.reach from the walk):
1. fuse      every face looks at every voxel of the grid. Along the ray to a voxel the face has a depth D;
             the voxel is at depth z. Their difference, cut off at a few voxels, is averaged over the faces
             (a truncated signed distance, TSDF):
               z well in front of D   the face saw THROUGH the voxel: a vote for "free"  (the carving of
                                      ~/projects/carving, with depth maps in place of silhouettes)
               z near D               the face saw the surface here: colour, kind of depth, one view
               z well behind D        hidden: no vote
             A vote weighs less the further the face stood (depth error grows with distance) and by the
             kind of depth: PatchMatch 1, fitted plane 0.6, monocular prior 0.25. The band "near D" is
             3 voxels, widened to 3% of D (8% for prior-only pixels).
2. surface   a voxel is surface where the averaged distance changes sign towards a neighbour: one layer,
             however many slightly different depth maps voted. Free votes that outweigh the surface votes
             leave no sign change: what one face hallucinated and five others saw through is gone.
3. keep      surface voxels seen from voxels.min_views different panoramas (a prior-only view counts half),
             then groups of fewer than 50 connected voxels are dropped.
Every voxel keeps its colour from the photos, the number of panoramas that saw it and where its depth came from.
Writes product/voxels.npz (ijk, rgb, views, source, origin, size), product/voxels.ply (centres with colours, for
any point-cloud viewer), product/voxels_cubes.ply (the cubes as a mesh, only the faces between a voxel and empty
space: for the web mesh viewer and the renders) and product/voxels.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1]); dev = 'cuda'
M = json.load(open(p.product / 'mesh.json')); h = M['lens_height']
size = p.voxel_size * h; FAR = p.mesh_max_depth * h; TRUNC = 3 * size; MIN_CLUSTER = 50
rec = pycolmap.Reconstruction(str(p.dense / 'sparse'))
ims = sorted((im for im in rec.images.values() if im.has_pose and (p.dense / 'final' / (im.name + '.npy')).exists()), key=lambda im: im.name)
cam = rec.cameras[ims[0].camera_id]
H, W = np.load(p.dense / 'final' / (ims[0].name + '.npy')).shape
fx, fy, cx, cy = np.array(cam.params[:4]) * (W / cam.width)

# ---- the box: the mesh, without its few lowest and highest vertices; the walk, for the reach
V = np.asarray(o3d.io.read_triangle_mesh(str(p.product / 'mesh.ply')).vertices)
lo = np.percentile(V, 0.2, axis=0) - 2 * size; hi = np.percentile(V, 99.8, axis=0) + 2 * size
if p.voxel_top is not None: hi[2] = min(hi[2], float(p.voxel_top) * h)
if p.voxel_bottom is not None: lo[2] = max(lo[2], float(p.voxel_bottom) * h)            # stray prior depth below the ground
dims = np.ceil((hi - lo) / size).astype(int); N = int(dims.prod())
full = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
path = np.array([im.projection_center() for im in full.images.values() if im.has_pose and im.name.startswith('v00/')])
print(f'grid {dims[0]} x {dims[1]} x {dims[2]} = {N / 1e6:.1f}M voxels of {p.voxel_size} lens heights over {(dims * size)[0] / h:.1f} x '
      f'{(dims * size)[1] / h:.1f} x {(dims * size)[2] / h:.1f} lens heights; {len(ims)} faces {W}x{H}, source: {p.voxel_source}', flush=True)
if N * 32 > 0.8 * torch.cuda.mem_get_info()[0]: raise SystemExit(f'{N / 1e6:.0f}M voxels need {N * 32 / 2**30:.1f} GB on the GPU: raise voxels.size or set voxels.top')

lo_t = torch.tensor(lo, dtype=torch.float32, device=dev); dy, dz = int(dims[1]), int(dims[2])
tsdf = torch.ones(N, device=dev); wsum = torch.zeros(N, device=dev)
col = torch.zeros(N, 3, device=dev); wcol = torch.zeros(N, device=dev); cnt = torch.zeros(N, 3, dtype=torch.int16, device=dev)
WTAB = torch.tensor([0.0, 1.0, 0.6, 0.25], device=dev); BAND = torch.tensor([0.0, 0.03, 0.03, 0.08], device=dev); CH = 6_000_000
pano_idx = {n: i for i, n in enumerate(sorted({Path(im.name).stem for im in ims}))}
last = torch.full((N,), -1, dtype=torch.int16, device=dev)                  # the panorama that voted last: its other faces do not count again
for k, im in enumerate(sorted(ims, key=lambda im: (Path(im.name).stem, im.name))):
    T = im.cam_from_world().matrix(); pi = pano_idx[Path(im.name).stem]
    R = torch.tensor(T[:3, :3], dtype=torch.float32, device=dev); t = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev)
    z = torch.tensor(np.load(p.dense / 'final' / (im.name + '.npy')).astype(np.float32), device=dev)
    L = torch.tensor(cv2.imread(str(p.dense / 'final' / (im.name + '.label.png')), cv2.IMREAD_GRAYSCALE), device=dev).long()
    if p.voxel_source == 'mvs': z = torch.where(L == 1, z, torch.zeros_like(z))
    z = torch.where(z < FAR, z, torch.zeros_like(z))
    img = cv2.imread(str(p.dense / 'images' / im.name))
    if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=torch.float32, device=dev)
    for s in range(0, N, CH):
        idx = torch.arange(s, min(s + CH, N), device=dev)
        X = (torch.stack([idx // (dy * dz), (idx // dz) % dy, idx % dz], 1).float() + 0.5) * size + lo_t
        Xc = X @ R.T + t; zc = Xc[:, 2]
        u = torch.floor(Xc[:, 0] / zc.clamp_min(1e-6) * fx + cx).long(); v = torch.floor(Xc[:, 1] / zc.clamp_min(1e-6) * fy + cy).long()
        ok = (zc > 0.05 * h) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx, zc, u, v = idx[ok], zc[ok], u[ok], v[ok]
        D = z[v, u]; Lb = L[v, u]; band = torch.maximum(BAND[Lb] * D, torch.full_like(D, TRUNC)); sdf = D - zc
        ok = (D > 0) & (sdf > -band)                                      # behind the surface: hidden, no vote
        idx, sdf, band, Lb, D, u, v = idx[ok], sdf[ok], band[ok], Lb[ok], D[ok], u[ok], v[ok]
        w = WTAB[Lb] / (D / h).clamp_min(1.0) ** 2
        tsdf[idx] = (wsum[idx] * tsdf[idx] + w * (sdf / band).clamp(max=1.0)) / (wsum[idx] + w); wsum[idx] += w
        near = sdf < band
        i2, w2, l2 = idx[near], w[near], Lb[near]
        col[i2] += w2[:, None] * rgb[v[near], u[near]]; wcol[i2] += w2
        new = last[i2] != pi                                              # the faces of one panorama count as one view
        cnt[i2[new], l2[new] - 1] += 1; last[i2] = pi
    if (k + 1) % 50 == 0: print(f'  {k + 1}/{len(ims)} faces', flush=True)
del z, L, rgb, last

# ---- surface: the voxel nearer to the sign change, along each axis. On the CPU: the temporaries of a 200M+ grid do not
# fit next to a splat in training on the GPU.
G = tsdf.cpu().numpy().reshape(*dims); has = (wsum > 0).cpu().numpy().reshape(*dims); del tsdf, wsum
torch.cuda.empty_cache(); surf = np.zeros(dims, bool)
for ax in range(3):
    n = int(dims[ax]) - 1
    sl0 = tuple(slice(0, n) if k == ax else slice(None) for k in range(3)); sl1 = tuple(slice(1, None) if k == ax else slice(None) for k in range(3))
    a, b = G[sl0], G[sl1]
    cross = has[sl0] & has[sl1] & ((a > 0) != (b > 0)); first = np.abs(a) <= np.abs(b)
    surf[sl0] |= cross & first; surf[sl1] |= cross & ~first
    del a, b, cross, first
del G, has
cnt_c = cnt.cpu().numpy()
eff = cnt_c[:, 0].astype(np.float32) + cnt_c[:, 1] + 0.5 * cnt_c[:, 2]
seen = int((cnt_c.sum(1) > 0).sum()); n_surf = int(surf.sum())
trade = {t: int((surf.reshape(-1) & (eff >= t)).sum()) for t in (1, 1.5, 2, 3)}
print('surface voxels seen from at least 1 / 1.5 / 2 / 3 panoramas (a prior-only view counts half): ' + ' / '.join(f'{v:,}' for v in trade.values()), flush=True)
keep = surf.reshape(-1) & (eff >= p.voxel_min_views); n_keep = int(keep.sum()); del eff, surf

# ---- small groups out, and what lies beyond mesh.reach of the walk
occ = keep.reshape(*dims)
cc, n_cc = ndimage.label(occ, structure=np.ones((3, 3, 3), bool)); sz = np.bincount(cc.ravel()); sz[0] = 0
occ &= sz[cc] >= MIN_CLUSTER; n_cl = int(occ.sum())
ijk = np.argwhere(occ).astype(np.int32)
C = (ijk.astype(np.float32) + 0.5) * size + lo.astype(np.float32)
if p.mesh_reach:
    inside = cKDTree(path[:, :2]).query(C[:, :2])[0] <= p.mesh_reach * h; ijk, C = ijk[inside], C[inside]
lin = torch.tensor((ijk[:, 0].astype(np.int64) * dy + ijk[:, 1]) * dz + ijk[:, 2], device=dev)
rgb = (col[lin] / wcol[lin].clamp_min(1e-9)[:, None]).clamp(0, 255).round().byte().cpu().numpy()
views = cnt[lin].sum(1).clamp(max=32767).cpu().numpy().astype(np.uint16)
source = ((cnt[lin].float() * WTAB[1:]).argmax(1) + 1).cpu().numpy().astype(np.uint8)
np.savez_compressed(p.product / 'voxels.npz', ijk=ijk.astype(np.int16), rgb=rgb, views=views, source=source, origin=lo, size=size, dims=dims,
                    unit=M['unit'], lens_height=h)
with open(p.product / 'voxels.ply', 'wb') as fh:
    fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(C)}\nproperty float x\nproperty float y\nproperty float z\n'
             f'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'.encode())
    rec_ = np.zeros(len(C), dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); rec_['p'] = C; rec_['c'] = rgb; fh.write(rec_.tobytes())

# ---- the cubes as a mesh: a square wherever a voxel borders empty space
occ = np.zeros(dims, bool); occ[tuple(ijk.T)] = True
quads, qcol = [], []
corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
FACES = {(-1, 0, 0): [0, 4, 7, 3], (1, 0, 0): [1, 2, 6, 5], (0, -1, 0): [0, 1, 5, 4], (0, 1, 0): [3, 7, 6, 2], (0, 0, -1): [0, 3, 2, 1], (0, 0, 1): [4, 5, 6, 7]}
pad = np.pad(occ, 1)
for d, cidx in FACES.items():
    nb = pad[1 + d[0]:1 + d[0] + dims[0], 1 + d[1]:1 + d[1] + dims[1], 1 + d[2]:1 + d[2] + dims[2]]
    exposed = ~nb[tuple(ijk.T)]
    q = (ijk[exposed][:, None, :] + corner[cidx][None]) * size + lo        # 4 corners, counter-clockwise seen from outside
    quads.append(q.astype(np.float32)); qcol.append(np.repeat(rgb[exposed], 4, axis=0))
Q = np.concatenate(quads).reshape(-1, 3); QC = np.concatenate(qcol); nq = len(Q) // 4
tri = np.arange(nq)[:, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]]).reshape(1, 6)
with open(p.product / 'voxels_cubes.ply', 'wb') as fh:
    fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(Q)}\nproperty float x\nproperty float y\nproperty float z\n'
             f'property uchar red\nproperty uchar green\nproperty uchar blue\nelement face {2 * nq}\nproperty list uchar int vertex_indices\nend_header\n'.encode())
    vr = np.zeros(len(Q), dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); vr['p'] = Q; vr['c'] = QC; fh.write(vr.tobytes())
    fr = np.zeros(2 * nq, dtype=[('n', 'u1'), ('i', '<i4', 3)]); fr['n'] = 3; fr['i'] = tri.reshape(-1, 3); fh.write(fr.tobytes())

share = {k: round(100 * float((source == i).mean()), 1) for i, k in ((1, 'patchmatch'), (2, 'plane'), (3, 'prior'))}
ext = [float(x) for x in (C.max(0) - C.min(0) + size) / h]
J = dict(voxel_lens_heights=p.voxel_size, voxel_model_units=size, unit=M['unit'], source=p.voxel_source, grid=[int(x) for x in dims], origin=[float(x) for x in lo],
         faces=len(ims), panoramas=len(pano_idx), voxels_near_a_measured_surface=seen, on_the_fused_surface=n_surf, dropped_under_min_views=n_surf - n_keep,
         dropped_small_groups=n_keep - n_cl, dropped_beyond_reach=n_cl - int(len(C)), voxels=int(len(C)), extent_lens_heights=ext,
         surface_area_lens_heights2_approx=round(float(len(C) * p.voxel_size ** 2), 1), views_median=int(np.median(views)), share_by_source_pct=share,
         min_views=p.voxel_min_views, surface_voxels_by_min_views={str(k_): v for k_, v in trade.items()},
         cube_mesh_triangles=int(2 * nq))
json.dump(J, open(p.product / 'voxels.json', 'w'), indent=1)
print(f"{seen:,} voxels lie near a measured surface in some face; {n_surf:,} on the fused surface; {n_surf - n_keep:,} of them seen from under "
      f"{p.voxel_min_views} panoramas, {n_keep - n_cl:,} in groups under {MIN_CLUSTER}, {n_cl - len(C):,} beyond the reach -> {len(C):,} voxels, "
      f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} lens heights; from PatchMatch {share['patchmatch']}%, planes {share['plane']}%, prior {share['prior']}%")
print('wrote', p.product / 'voxels.npz', p.product / 'voxels.ply', f'{p.product / "voxels_cubes.ply"} ({2 * nq:,} triangles)', p.product / 'voxels.json')
