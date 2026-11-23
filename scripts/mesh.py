#!/usr/bin/env python3
"""Step 4b. A mesh with walls: PatchMatch depth + the aligned prior + fitted planes -> TSDF fusion.

    python3 -I scripts/mesh.py data/pass-0/rig.yaml

1. planes    from the fused PatchMatch cloud, thinned to one point per 5% of the lens height (a plane counts by
             its area, not by its texture). Floor: the level plane at z = 0. Ceiling: the level plane where the
             points stop going up (a white ceiling shows only by its edges and lamps). Walls: vertical RANSAC
             planes within 4 deg of a room axis that are one piece (filled columns along a run of at least
             4 lens heights) and do not cross the driven path. A wall exists only along its run, floor to ceiling.
2. depth     per face: PatchMatch depth where it exists. Elsewhere the aligned prior; a low-texture pixel
             whose prior lies within 8% of a plane is moved onto the plane.
3. check     a depth counts as confirmed when the same 3D point, seen from a panorama two dense steps away,
             has the same depth there within 5%. This is the test PatchMatch applies to itself, applied to the
             filled pixels too: a prior that fills a wall with a wrong depth does not pass.
4. coverage  share of pixels with a depth / with a confirmed depth, by the pixel classes of kitchen-recon's
             dense_report.py (texture quartiles; blank bright = white wall, ceiling, cabinet fronts).
5. mesh      TSDF fusion of the depth maps: product/mesh_mvs.ply (PatchMatch only) and product/mesh.ply.
Outdoors (data/pool/rig.yaml) mesh.planes is [walls]: the ground slopes and there is no ceiling, so only facades
are snapped to; mesh.reach cuts the cloud and the meshes to that many lens heights from the walked path (depth beyond it
is not even fused: Open3D's TSDF keeps 32 bytes a voxel, and prior-filled far surfaces took 30 GB on the pool walk), and
mesh.max_depth, mesh.voxel set the TSDF (depth farther than that from the camera is not fused). Lengths in lens heights.
Writes dense/final/<face>.npy (depth) and .label.png (1 PatchMatch, 2 plane, 3 prior), product/mesh*.ply, product/mesh.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from depth_prior import read_array
from project import load

p = load(sys.argv[1]); S = p.dense_size
rec = pycolmap.Reconstruction(str(p.dense / 'sparse'))
full = pycolmap.Reconstruction(str(p.colmap / 'aligned'))               # the whole walk; a panorama taken out of it (misplaced) is out here too
in_model = {im.name for im in full.images.values() if im.has_pose}
ims = sorted((im for im in rec.images.values() if im.has_pose and im.name in in_model and (p.dense / 'stereo' / 'depth_maps' / (im.name + '.geometric.bin')).exists()),
             key=lambda im: (Path(im.name).stem, im.name))                 # faces left out as sky (dense.max_sky) have no depth map
panos = sorted({Path(im.name).stem for im in ims})
h = 1.0 if p.outdoor else float(np.median([im.projection_center()[2] for im in ims]))   # lens height, model units (outdoors the unit itself: the
                                                                         # cameras' z also holds the slope of the ground)
unit = json.load(open(p.product / 'align.json'))['unit']
print(f'{len(ims)} faces of {len(panos)} panoramas at {S} px; lens height {h:.3f} {unit}; planes {p.mesh_planes}')
path_xy = cKDTree(np.array([im.projection_center() for im in full.images.values() if im.has_pose and im.name.startswith('v00/')])[:, :2])
reach = (lambda X: path_xy.query(X[:, :2])[0] <= p.mesh_reach * h) if p.mesh_reach else (lambda X: np.ones(len(X), bool))

# ------------------------------------------------------------------ 1. planes
pcd = o3d.io.read_point_cloud(str(p.dense / 'fused.ply'))
A = np.asarray(pcd.voxel_down_sample(0.05 * h).points); A = A[reach(A)]; n_down = len(A)


def ransac(pts, thickness):
    (a, b, c, d), inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).segment_plane(thickness, 3, 3000)
    n = np.array([a, b, c]); return n / np.linalg.norm(n), np.asarray(inl)


def refit(pts, n0):
    cen = pts.mean(0); _, _, vt = np.linalg.svd(pts - cen, full_matrices=False); n = vt[2] * np.sign(vt[2] @ n0)
    return n, float(-n @ cen), float(np.sqrt((((pts - cen) @ n) ** 2).mean()))


planes = []
# Floor: the level plane at z = 0 (the frame was built on it). Ceiling: the fullest 0.25 h layer of points in the
# upper 40% of the room's height; a white ceiling shows by its edges and lamps, which is enough for a plane.
z995 = np.percentile(A[:, 2], 99.5)
hist, edges = np.histogram(A[A[:, 2] > 0.6 * z995, 2], bins=np.arange(0.6 * z995, z995 + 0.5 * h, 0.25 * h))
lid = [int(hist.argmax())] if len(hist) and hist.max() > 0.005 * n_down else []
for kind, zc in (('floor', 0.0), ('ceiling', edges[lid[-1]] + 0.125 * h if lid else None)):
    if zc is None or kind not in p.mesh_planes: continue
    slab = A[np.abs(A[:, 2] - zc) < 0.5 * h]; n, inl = ransac(slab, 0.1 * h)
    if abs(n[2]) < 0.97: continue
    n, d0, rms = refit(slab[inl], np.array([0, 0, 1.0]))
    if kind == 'ceiling':                                                # few points, far away: keep it parallel to the floor
        zc = float(np.median(slab[inl][:, 2])); n, d0 = np.array([0, 0, 1.0]), -zc; rms = float(np.sqrt(((slab[inl][:, 2] - zc) ** 2).mean()))
    planes.append(dict(kind=kind, n=n, d=d0, rms=rms, share=len(inl) / n_down, z=float(-d0 / n[2])))
ztop = next((pl['z'] for pl in planes if pl['kind'] == 'ceiling'), float(np.percentile(A[:, 2], 99.7)))

# Walls. RANSAC alone also returns planes through unrelated things that happen to line up across the room.
# A wall has to be one piece: columns 0.25 h wide with at least 1.25 h of their height filled, along a run of
# 4 lens heights or more, gaps of at most 1, at least half of the run's columns filled. And the rover cannot have
# driven through it: a plane that passes within one lens height of a camera inside its run is dropped.
Cw = np.array([im.projection_center() for im in ims])
rest = A[(A[:, 2] > 0.3 * h) & (A[:, 2] < ztop - 0.3 * h)]; cell = 0.25 * h; tried = 0
for tried in range(1, (61 if p.outdoor else 41) if 'walls' in p.mesh_planes else 1):
    if len(rest) < 0.02 * n_down or sum(pl['kind'] == 'wall' for pl in planes) >= p.mesh_max_walls: break
    n, inl = ransac(rest, 0.1 * h); gone = np.zeros(len(rest), bool); gone[inl] = True
    ax = int(np.argmax(np.abs(n[:2])))
    if abs(n[2]) < 0.07 and abs(n[ax]) > np.cos(np.radians(4)):
        n = n * np.sign(n[ax]); pts = rest[inl]; t = np.cross(n, [0, 0, 1.0]); t /= np.linalg.norm(t); sp = pts @ t
        filled = {}
        for c_, z_ in set(zip(np.floor(sp / cell).astype(int).tolist(), np.floor(pts[:, 2] / cell).astype(int).tolist())):
            filled[c_] = filled.get(c_, 0) + 1
        runs = []
        for c_ in sorted(c_ for c_, m in filled.items() if m >= 5):
            if runs and c_ - runs[-1][1] <= 4: runs[-1][1] = c_; runs[-1][2] += filled[c_]; runs[-1][3] += 1
            else: runs.append([c_, c_, filled[c_], 1])
        run = max(runs, key=lambda r: r[2], default=None)
        if run and (run[1] - run[0] + 1) * cell >= 4 * h and run[2] >= 150 and run[3] >= 0.5 * (run[1] - run[0] + 1):
            s0, s1 = run[0] * cell, (run[1] + 1) * cell; inside = (sp >= s0) & (sp <= s1)
            n, d0, rms = refit(pts[inside], n); t = np.cross(n, [0, 0, 1.0]); t /= np.linalg.norm(t)
            if ((np.abs(Cw @ n + d0) < h) & (Cw @ t > s0) & (Cw @ t < s1)).any():
                rest = rest[~gone]; continue                             # crosses the driven path: not a wall
            twin = next((w for w in planes if w['kind'] == 'wall' and w['n'] @ n > 0.995 and abs(w['d'] - d0) < 0.35 * h
                         and min(s1, w['s1']) > max(s0, w['s0'])), None)
            if twin: twin['s0'], twin['s1'] = min(twin['s0'], s0), max(twin['s1'], s1)      # the same wall again, a few cm off
            else: planes.append(dict(kind='wall', n=n, d=d0, rms=rms, share=int(inside.sum()) / n_down, t=t, s0=float(s0), s1=float(s1),
                                     axis='xy'[ax], cells=int(run[2]), zlo=float(np.percentile(pts[inside][:, 2], 1)) if p.outdoor else 0.0))
            gone[inl[~inside]] = False                                   # the rest of that infinite plane stays in play
    rest = rest[~gone]
use = planes
# The up of align_scale.py against the facades: if the frame leans by a small horizontal vector g, a facade with normal n
# leans by -(n_h . g). Least squares over the facades, weighted by their points (needs two that face different ways).
W_ = [pl for pl in planes if pl['kind'] == 'wall']; frame_tilt = None
if len(W_) >= 2:
    Nh = np.array([pl['n'][:2] for pl in W_]); nz = np.array([pl['n'][2] for pl in W_]); w_ = np.sqrt([pl['cells'] for pl in W_])
    if np.linalg.matrix_rank(Nh * w_[:, None], tol=0.2) == 2:
        g_ = np.linalg.lstsq(Nh * w_[:, None], -nz * w_, rcond=None)[0]; resid = nz + Nh @ g_
        frame_tilt = dict(deg=float(np.degrees(np.linalg.norm(g_))), towards_deg=float(np.degrees(np.arctan2(g_[1], g_[0]))),
                          facade_lean_deg_after=float(np.degrees(np.sqrt(np.average(resid ** 2, weights=w_ ** 2)))))
print(f'fused cloud {len(pcd.points):,} points ({n_down:,} after thinning); ' + '; '.join(
    f"{pl['kind']} at z {pl['z'] / h:.2f} h (rms {pl['rms'] / h:.3f} h)" for pl in planes if pl['kind'] != 'wall')
    + f"; {sum(pl['kind'] == 'wall' for pl in planes)} walls from {tried} RANSAC planes: " + ', '.join(
    f"normal {pl['axis']} at {-pl['d'] / h:+.2f} h, {(pl['s1'] - pl['s0']) / h:.1f} h long (rms {pl['rms'] / h:.3f})" for pl in planes if pl['kind'] == 'wall'))

# ------------------------------------------------------------------ 2. depth per face
cam = rec.cameras[ims[0].camera_id]
fx, fy, cx, cy = np.array(cam.params) * (S / cam.width)                  # PatchMatch writes depth maps at S px; the workspace images can be larger


def photo(name, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(str(p.dense / 'images' / name), flag)
    return img if img.shape[0] == S else cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)


xs = (np.arange(S) + 0.5 - cx) / fx; ys = (np.arange(S) + 0.5 - cy) / fy
gx, gy = np.meshgrid(xs, ys); rays = np.stack([gx, gy, np.ones_like(gx)], -1).astype(np.float32)
F = []                                                                  # per face: everything the later steps need
stds = []
for im in ims[::5]:
    g = photo(im.name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    m = cv2.blur(g, (9, 9)); stds.append(np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - m * m, 0))[::4, ::4].ravel())
q = np.percentile(np.concatenate(stds), [25, 50, 75])
(p.dense / 'final').mkdir(exist_ok=True)
for im in ims:
    T = im.cam_from_world().matrix(); R, t = T[:3, :3], T[:3, 3]; C = -R.T @ t
    mask = cv2.imread(str(p.dense / 'masks' / (im.name + '.png')), cv2.IMREAD_GRAYSCALE) > 127
    mvs = read_array(p.dense / 'stereo' / 'depth_maps' / (im.name + '.geometric.bin')); mvs_ok = (mvs > 0) & (mvs < 400 * h) & mask
    pf = p.dense / 'prior' / (im.name + '.npy')
    prior = np.load(pf).astype(np.float32) if pf.exists() else np.zeros((S, S), np.float32)
    g = photo(im.name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mean = cv2.blur(g, (9, 9)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mean * mean, 0))
    fill = mask & ~mvs_ok & (prior > 0)
    dw = rays @ R                                                       # ray directions in the world (rows: R.T @ ray)
    best = np.full((S, S), np.inf, np.float32); zbest = np.zeros((S, S), np.float32)
    for pl in use:
        den = dw @ pl['n']
        z = np.where(np.abs(den) > 1e-6, -(pl['n'] @ C + pl['d']) / np.where(np.abs(den) > 1e-6, den, 1), -1)
        if pl['kind'] == 'wall':                                         # a wall exists only along its run, floor to ceiling
            sw = pl['t'] @ C + z * (dw @ pl['t']); zw = C[2] + z * dw[..., 2]
            z = np.where((sw > pl['s0'] - 0.5 * h) & (sw < pl['s1'] + 0.5 * h) & (zw > pl['zlo'] - 0.2 * h) & (zw < ztop + 0.3 * h), z, -1)
        z = z.astype(np.float32)
        rel = np.where(z > 0, np.abs(z - prior) / np.maximum(prior, 1e-6), np.inf)
        better = rel < best; best[better] = rel[better]; zbest[better] = z[better]
    snap = fill & (best < 0.08) & (sd < q[1])
    final = np.where(mvs_ok, mvs, np.where(snap, zbest, np.where(fill, prior, 0))).astype(np.float32)
    label = np.where(mvs_ok, 1, np.where(snap, 2, np.where(fill, 3, 0))).astype(np.uint8)       # 1 PatchMatch, 2 plane, 3 prior
    out = p.dense / 'final' / (im.name + '.npy'); out.parent.mkdir(parents=True, exist_ok=True); np.save(out, final.astype(np.float16))
    cv2.imwrite(str(out.with_suffix('.label.png')), label)                # for voxels.py: where every depth came from
    cls = np.packbits(np.stack([sd < q[0], (sd >= q[0]) & (sd < q[1]), (sd >= q[1]) & (sd < q[2]), sd >= q[2],
                                (sd < q[0]) & (mean > 150) & (mean < 250), mean < 45], -1), axis=-1)[..., 0]     # 6 classes as bits: 1 byte a pixel
    F.append(dict(name=im.name, pano=Path(im.name).stem, R=R, t=t, C=C, mask=mask, final=final, label=label, cls=cls))

# ------------------------------------------------------------------ 3. confirmed by another panorama?
by_pano = {n: [f for f in F if f['pano'] == n] for n in panos}
st = 4; sub = (slice(st // 2, None, st), slice(st // 2, None, st)); rs = rays[sub].reshape(-1, 3)
for f in F:
    i = panos.index(f['pano']); partners = [panos[j] for j in {max(0, i - 2), min(len(panos) - 1, i + 2)} if j != i]
    z = f['final'][sub].ravel(); X = (rs * z[:, None] - f['t']) @ f['R']                        # world points: R.T @ (x_cam - t)
    conf = np.zeros(len(z), bool); seen = np.zeros(len(z), bool)
    for pn in partners:
        for b in by_pano[pn]:
            Xc = X @ b['R'].T + b['t']; zc = Xc[:, 2]
            u = np.floor(Xc[:, 0] / np.maximum(zc, 1e-6) * fx + cx).astype(int); v = np.floor(Xc[:, 1] / np.maximum(zc, 1e-6) * fy + cy).astype(int)
            ok = (z > 0) & (zc > 0.05 * h) & (u >= 0) & (u < S) & (v >= 0) & (v < S)
            ok[ok] &= b['mask'][v[ok], u[ok]]
            zb = np.zeros(len(z), np.float32); zb[ok] = b['final'][v[ok], u[ok]]
            ok &= zb > 0; seen |= ok
            conf |= ok & (np.abs(zb - zc) < 0.05 * zc)
    f['conf'] = conf.reshape(f['final'][sub].shape); f['seen'] = seen.reshape(f['conf'].shape)

# ------------------------------------------------------------------ 4. coverage by pixel class
names = ['texture q1 (flattest)', 'texture q2', 'texture q3', 'texture q4 (busiest)', 'blank bright (white wall, ceiling, fronts)', 'dark']
tot = np.zeros((7, 6)); keys = ['pixels', 'patchmatch', 'patchmatch_confirmed', 'filled', 'filled_confirmed', 'on_plane', 'prior_only']
for f in F:
    m = f['mask'][sub]; lab = f['label'][sub]; cf = f['conf']
    for k in range(6):
        c = ((f['cls'][sub] >> (7 - k)) & 1).astype(bool) & m
        tot[:, k] += [c.sum(), (c & (lab == 1)).sum(), (c & (lab == 1) & cf).sum(), (c & (lab > 0)).sum(), (c & (lab > 0) & cf).sum(),
                      (c & (lab == 2)).sum(), (c & (lab == 3)).sum()]
allm = sum(f['mask'][sub].sum() for f in F)
if frame_tilt: print(f"the facades say the frame leans {frame_tilt['deg']:.2f} deg (towards {frame_tilt['towards_deg']:.0f} deg); "
                     f"what they lean apart from that: {frame_tilt['facade_lean_deg_after']:.2f} deg rms")
R = dict(faces=len(F), panoramas=len(panos), size=S, lens_height=h, unit=unit, texture_std_quartile_edges=[float(x) for x in q], frame_tilt_from_facades=frame_tilt,
         planes=[dict(kind=pl['kind'], share_of_cloud_pct=round(100 * pl['share'], 2), rms_over_lens_height=round(pl['rms'] / h, 4),
                      normal=[float(x) for x in pl['n']], d=pl['d'],
                      **(dict(t=[float(x) for x in pl['t']], s0=pl['s0'], s1=pl['s1'], normal_axis=pl['axis'], filled_cells=pl['cells'])
                         if pl['kind'] == 'wall' else dict(z=pl['z']))) for pl in planes], room_top=ztop, coverage={})
print(f'{"pixel class":44s} share  PatchMatch  confirmed | +prior+planes  confirmed  (on a plane / prior only)')
for k, nm in enumerate(names):
    n = max(tot[0, k], 1); r = dict(share_pct=round(100 * tot[0, k] / allm, 1), **{keys[j]: round(100 * tot[j, k] / n, 1) for j in range(1, 7)})
    R['coverage'][nm] = r
    print(f"{nm:44s} {r['share_pct']:4.1f}%  {r['patchmatch']:8.1f}%  {r['patchmatch_confirmed']:8.1f}% | {r['filled']:11.1f}%  "
          f"{r['filled_confirmed']:8.1f}%   ({r['on_plane']:.1f}% / {r['prior_only']:.1f}%)")
lab_all = np.concatenate([f['label'][sub][f['mask'][sub]] for f in F]); cf_all = np.concatenate([f['conf'][f['mask'][sub]] for f in F])
R['all_pixels'] = dict(patchmatch=round(100 * float((lab_all == 1).mean()), 1), patchmatch_confirmed=round(100 * float(((lab_all == 1) & cf_all).mean()), 1),
                       filled=round(100 * float((lab_all > 0).mean()), 1), filled_confirmed=round(100 * float(((lab_all > 0) & cf_all).mean()), 1),
                       confirmed_share_of_plane_pixels=round(100 * float(cf_all[lab_all == 2].mean()), 1) if (lab_all == 2).any() else None,
                       confirmed_share_of_prior_pixels=round(100 * float(cf_all[lab_all == 3].mean()), 1) if (lab_all == 3).any() else None,
                       confirmed_share_of_patchmatch_pixels=round(100 * float(cf_all[lab_all == 1].mean()), 1))
print('all pixels outside the rover:', R['all_pixels'])


# ------------------------------------------------------------------ 5. TSDF meshes
def fuse(pick, path):
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=p.mesh_voxel * h, sdf_trunc=4 * p.mesh_voxel * h,
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    K = o3d.camera.PinholeCameraIntrinsic(S, S, fx, fy, cx - 0.5, cy - 0.5)
    for f in F:
        z = np.where(pick(f['label']), f['final'], 0).astype(np.float32)
        if p.mesh_reach:                                                # what lies beyond the reach of the walk is not fused at all (memory)
            v_ = z > 0; X = (rays[v_] * z[v_][:, None] - f['t']) @ f['R']; z[v_] = np.where(reach(X), z[v_], 0)
        rgb = cv2.cvtColor(photo(f['name']), cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(z),
                                                                  depth_scale=1.0, depth_trunc=p.mesh_max_depth * h, convert_rgb_to_intensity=False)
        E = np.eye(4); E[:3, :3] = f['R']; E[:3, 3] = f['t']; vol.integrate(rgbd, K, E)
    mesh = vol.extract_triangle_mesh()
    if p.mesh_reach: mesh.remove_vertices_by_mask(~reach(np.asarray(mesh.vertices)))       # beyond mesh.reach of the walked path
    tc, n_tri, _ = mesh.cluster_connected_triangles(); n_tri = np.asarray(n_tri)
    mesh.remove_triangles_by_mask(n_tri[np.asarray(tc)] < 500); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    V = np.asarray(mesh.vertices); Tr = np.asarray(mesh.triangles)
    e1, e2 = V[Tr[:, 1]] - V[Tr[:, 0]], V[Tr[:, 2]] - V[Tr[:, 0]]; cr = np.cross(e1, e2); area = np.linalg.norm(cr, axis=1) / 2
    nrm = cr / np.maximum(2 * area[:, None], 1e-12); cen = V[Tr].mean(1)
    wall = np.zeros(len(Tr), bool); level = np.zeros(len(Tr), bool)
    for pl in use:
        on = (np.abs(cen @ pl['n'] + pl['d']) < 0.2 * h) & (np.abs(nrm @ pl['n']) > 0.9)
        if pl['kind'] == 'wall': wall |= on & (cen @ pl['t'] > pl['s0'] - 0.5 * h) & (cen @ pl['t'] < pl['s1'] + 0.5 * h)
        else: level |= on
    return dict(triangles=int(len(Tr)), area=float(area.sum()), wall_area=float(area[wall].sum()), floor_ceiling_area=float(area[level & ~wall].sum()))


R['mesh_patchmatch_only'] = fuse(lambda lab: lab == 1, p.product / 'mesh_mvs.ply')
R['mesh'] = fuse(lambda lab: lab > 0, p.product / 'mesh.ply')
for k in ('mesh_patchmatch_only', 'mesh'):
    m = R[k]; print(f"{k}: {m['triangles']:,} triangles, area {m['area']:.1f} {unit}^2, of it on wall planes {m['wall_area']:.1f}, "
                    f"on floor / ceiling planes {m['floor_ceiling_area']:.1f}")
json.dump(R, open(p.product / 'mesh.json', 'w'), indent=1)
print('wrote', p.product / 'mesh.ply', p.product / 'mesh_mvs.ply', p.product / 'mesh.json')
