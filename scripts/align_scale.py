#!/usr/bin/env python3
"""Step 5. Put the sparse model into a room frame and give it a scale from the camera height.

    python3 -I scripts/align_scale.py data/pass-0/rig.yaml

  z up, floor at z = 0     the floor is a RANSAC plane among the well-tracked points below the cameras,
                           its normal within 15 deg of the cameras' own "up" (the rover rides level)
  x, y                     the yaw that makes the top-down histograms of the points most peaked (walls on the axes)
  origin                   on the floor under the middle of the path
  unit                     one lens height: the median height of the lens above the floor is 1. Every later
                           step works in this unit, so nothing has to be run again when camera_height_cm arrives
                           or changes; measure.py converts lengths and writes metric copies of the results.
The camera height is one number for the whole path, so the spread of the 49 heights above the fitted floor
is the internal check of the scale: on a flat floor it should be zero.

scene: outdoor (a camera carried by hand round a building; the ground is not one plane)
  z up                     the camera's gravity sensor: the THETA S does not level its stills, it writes its tilt into the
                           XMP (PosePitchDegrees, PoseRollDegrees). Turned into the model's frame, every shot gives an up;
                           in a good model they agree, and their robust mean is used (if they agree within 2.5 deg, median).
                           Else the panoramas' own up, if they agree within 1.5 deg (stills levelled by the camera); else the
                           normal of the largest ground plane near the walk. That plane is reported against the up either way.
  ground under each shot   the sparse points within 2.5 lens heights of it (horizontally) and 0.3 to 3 below it:
                           the median of their depths. A hand-held camera has little sparse ground right round it
                           (the people hide it; on the pool walk the next shot stood 4-5 lens heights away), so shots
                           with fewer than 4 such points take their neighbours' value
                  z = 0    the median of the ground heights under the shots
  x, y                     the yaw that makes the top-down histograms of facade points (0.5 to 8 lens heights up,
                           within 15 of the path) most peaked; +x along the longer side of the walked area
  origin                   the middle of the walked area, on z = 0
  unit                     one lens height: the median of the lens heights above the local ground. Their spread is
                           how steadily the pole was carried plus the error of the ground planes.
Writes colmap/aligned/ (the model), product/align.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from pano import face_rotation
from project import load

p = load(sys.argv[1])
rec = pycolmap.Reconstruction(str(p.colmap / 'sparse' / '0'))
# Panoramas sfm_report.py suspects to be misplaced, and those listed in the YAML (exclude: [R0013079]), are taken out of the
# model first: a wrong pose puts phantom surfaces into the mesh and contradicts the splat.
sus = json.load(open(p.product / 'sfm.json'))['path'].get('suspected_misplaced', []) if (p.product / 'sfm.json').exists() else []
drop = sorted(set(sus) | set(p.cfg.get('exclude') or []))
for fid in sorted({im.frame_id for im in rec.images.values() if im.has_pose and Path(im.name).stem in drop}): rec.deregister_frame(fid)
if drop: print(f'taken out before aligning: {" ".join(drop)}')


def outdoor():
    from collections import defaultdict
    from scipy.spatial import cKDTree
    R0 = face_rotation(*p.views[0]); up_cam = R0 @ np.array([0, -1.0, 0])
    frames = defaultdict(list)
    for im in rec.images.values():
        if im.has_pose: frames[Path(im.name).stem].append(im)
    names = sorted(frames)
    ref = {n: next(im for im in frames[n] if im.name.startswith('v00/')) for n in names}
    C = np.array([ref[n].projection_center() for n in names])
    ups = np.array([ref[n].cam_from_world().rotation.matrix().T @ up_cam for n in names])
    up0 = ups.mean(0); up0 /= np.linalg.norm(up0)
    dev = np.degrees(np.arccos(np.clip(ups @ up0, -1, 1)))
    P = np.array([q.xyz for q in rec.points3D.values()]); ntrack = np.array([q.track.length() for q in rec.points3D.values()])
    pid = {k: i for i, k in enumerate(rec.points3D.keys())}
    lower = [v for v, (_, pitch) in enumerate(p.views) if pitch < 0]

    step = float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))

    def ground(up):
        """Per shot: lens height above the ground (model units), ground height along up, points used, filled or not; h1.
        The sparse model has little ground right round a hand-held camera (the people hide it, and on the pool walk the
        next shot stood 4-5 lens heights away), so in two passes. 1: the sparse points within half a step of their nearest
        shot (horizontally) and below it: the median of their depths is a first lens height h1. 2: per shot, the points
        within 2.5 h1 of it horizontally and 0.3 to 3 h1 below it: the median of their depths. A shot with fewer than 4 such
        points takes the median of the 2 shots on either side."""
        e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
        Ch, Ph = np.c_[C @ e1, C @ e2], np.c_[P @ e1, P @ e2]; zc, zp = C @ up, P @ up
        d, j = cKDTree(Ch).query(Ph); dz = zc[j] - zp
        for r in (0.5, 1, 2):                                            # widen until there is something to measure
            m = (d < r * step) & (ntrack >= 3) & (dz > 0) & (dz < 2 * step)
            if m.sum() >= 50: break
        h1 = float(np.median(dz[m]))
        tree = cKDTree(Ph); h, n = np.full(len(C), np.nan), np.zeros(len(C), int)
        for i in range(len(C)):
            idx = np.array(tree.query_ball_point(Ch[i], 2.5 * h1), int)
            if not len(idx): continue
            dd = zc[i] - zp[idx]; dd = dd[(dd > 0.3 * h1) & (dd < 3 * h1) & (ntrack[idx] >= 3)]; n[i] = len(dd)
            if len(dd) >= 4: h[i] = np.median(dd)
        filled = ~np.isfinite(h); h0_ = h.copy()
        for i in np.flatnonzero(filled):
            nb = h0_[max(0, i - 2):i + 3]; nb = nb[np.isfinite(nb)]
            h[i] = np.median(nb) if len(nb) else h1
        return h, zc - h, n, filled, h1, (Ch, Ph, d, dz)

    # The THETA S does not level its stills: it writes its own tilt (gravity sensor) into the XMP, GPano:PosePitchDegrees
    # and PoseRollDegrees, for viewers to apply. Turned into the model's frame, the 144 "ups" it gives must agree: the
    # sign convention that makes them agree best is taken (on pass-0: both negated), and their robust mean is the up.
    xmp = {r['name'].rsplit('.', 1)[0]: (r['pitch'], r['roll']) for r in json.load(open(p.product / 'inspect.json'))['per_shot']} \
        if (p.product / 'inspect.json').exists() else {}
    Rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    grav = None
    if all(n in xmp for n in names):
        best = None
        for sp in (1, -1):
            for sr in (1, -1):
                g = np.array([ref[n].cam_from_world().rotation.matrix().T @ (R0 @ (Rx(np.radians(sp * xmp[n][0])) @ Rz(np.radians(sr * xmp[n][1])) @ [0, -1.0, 0]))
                              for n in names])
                m = g.mean(0); m /= np.linalg.norm(m); s = np.degrees(np.arccos(np.clip(g @ m, -1, 1)))
                if best is None or np.median(s) < best[0]: best = (float(np.median(s)), sp, sr, g)
        s0, sp, sr, g = best
        m = g.mean(0); m /= np.linalg.norm(m)
        for _ in range(3):                                               # robust mean: drop shots far from the others (a jolt of the pole)
            s = np.degrees(np.arccos(np.clip(g @ m, -1, 1))); m = g[s < max(3 * np.median(s), 1.0)].mean(0); m /= np.linalg.norm(m)
        s = np.degrees(np.arccos(np.clip(g @ m, -1, 1)))
        grav = dict(convention=dict(pitch_sign=sp, roll_sign=sr), spread_deg_median=float(np.median(s)), spread_deg_p90=float(np.percentile(s, 90)),
                    used=int((s < max(3 * np.median(s), 1.0)).sum()), up=m)
        print(f"camera gravity sensor (XMP pitch/roll, signs {sp:+d} {sr:+d}): the {len(names)} ups lie within {np.median(s):.2f} deg of their robust mean (median), "
              f"90% within {np.percentile(s, 90):.2f}; the raw panorama ups within {np.median(dev):.2f} (median)")
    if grav and grav['spread_deg_median'] < 2.5:                         # a camera fixed at a tilt (pass-0) gives ups that agree but are not level
        src, up0 = 'camera gravity sensor (XMP)', grav['up']
    elif np.median(dev) <= 8:
        # Levelled by the camera, but each with the jolt of a walking step in it (pool: 4.6 deg median, 24 at most): their
        # robust mean, the jolts averaging out over the walk.
        m = up0
        for _ in range(3):
            s = np.degrees(np.arccos(np.clip(ups @ m, -1, 1))); m = ups[s < 3 * np.median(s)].mean(0); m /= np.linalg.norm(m)
        src, up0 = f'panorama zenith (robust mean of {int((s < 3 * np.median(s)).sum())} levelled stills)', m
    else:                                                                # neither: the ground near the cameras
        src = 'ground plane'
    # the ground near the walk: points 0.3 to 3 first lens heights under their nearest shot, within 10 of it; the largest plane
    _, _, _, _, h1, (_, _, d, dz) = ground(up0)
    near = np.flatnonzero((ntrack >= 3) & (d < 10 * h1) & (dz > 0.3 * h1) & (dz < 3 * h1))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[near]))
    (a, b, c_, d_), inl_g = pc.segment_plane(0.1 * h1, 3, 4000); gn = np.array([a, b, c_]) / np.linalg.norm([a, b, c_]); gn *= np.sign(gn @ up0)
    ground_vs_up = float(np.degrees(np.arccos(np.clip(gn @ up0, -1, 1))))
    print(f'the largest ground plane near the walk ({len(inl_g):,} of {len(near):,} ground points) is {ground_vs_up:.2f} deg from that up')
    if src == 'ground plane': up0 = gn
    # Facades stand plumb: the up is the direction most at right angles to their normals. Vertical planes (within 10 deg)
    # among the sparse points 0.5 to 8 first lens heights above the ground near the walk, by RANSAC; used when two or more
    # of them, 300+ points each, face directions at least 30 deg apart. It wins over the ground and the camera's sensor.
    _, gz1, _, _, _, (_, _, d, _) = ground(up0); _, j = cKDTree(C).query(P)
    zr = P @ up0 - gz1[j]
    cand = P[(ntrack >= 3) & (zr > 0.5 * h1) & (zr < 8 * h1) & (d < 15 * h1)]; pcf = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand)); fac = []
    for _ in range(25):
        if len(pcf.points) < 300: break
        (a, b, c_, d_), inl = pcf.segment_plane(0.05 * h1, 3, 3000); n_ = np.array([a, b, c_]) / np.linalg.norm([a, b, c_])
        if abs(n_ @ up0) < np.sin(np.radians(10)) and len(inl) >= 300:
            Xi = np.asarray(pcf.points)[inl]; _, _, vt = np.linalg.svd(Xi - Xi.mean(0), full_matrices=False); fac.append((vt[2], len(inl)))
        pcf = pcf.select_by_index(inl, invert=True)
    hz = lambda v: np.arctan2(*(v - (v @ up0) * up0)[:2][::-1])
    spread = max((abs(np.degrees((hz(a_) - hz(b_)) % np.pi)) for (a_, _), (b_, _) in [(x, y) for x in fac for y in fac]), default=0)
    spread = min(spread, 180 - spread) if spread else 0
    facade_up = None
    if len(fac) >= 2 and spread >= 30:
        Mf = sum(w * np.outer(n_, n_) for n_, w in fac); ev, evec = np.linalg.eigh(Mf); facade_up = evec[:, 0] * np.sign(evec[:, 0] @ up0)
        print(f'{len(fac)} facades ({sum(w for _, w in fac):,} sparse points, facing directions up to {spread:.0f} deg apart): their up is '
              f'{np.degrees(np.arccos(np.clip(facade_up @ gn, -1, 1))):.2f} deg from the ground plane, {np.degrees(np.arccos(np.clip(facade_up @ (ups.mean(0) / np.linalg.norm(ups.mean(0))), -1, 1))):.2f} from the panoramas\' mean up')
        src, up0 = f'facades ({len(fac)} vertical planes)', facade_up
    up = up0
    hs, gz, npts, filled, h1, _ = ground(up)
    ok = ~filled; hm = float(np.median(hs[ok]))
    z0 = float(np.median(gz))
    tilt = np.degrees(np.arccos(np.clip(ups @ up, -1, 1)))
    print(f'up: {src}; the {len(names)} panoramas\' own up directions lie within {np.median(dev):.2f} deg of their mean (median), {dev.max():.2f} at most')
    print(f'lens height above the ground, measured at {ok.sum()}/{len(names)} shots (the rest from their neighbours; first estimate {h1:.4f}): '
          f'median {hm:.4f} model units = {hm / step:.2f} steps of the walk; 10-90%: {np.percentile(hs[ok], 10) / hm:.3f} .. {np.percentile(hs[ok], 90) / hm:.3f} '
          f'of the median, std {100 * np.std(hs[ok]) / hm:.1f}%; ground height over the walk {(gz.min() - z0) / hm:+.2f} .. {(gz.max() - z0) / hm:+.2f} lens heights')

    # ---- yaw: facades on the axes
    x0 = np.cross(up, [1.0, 0, 0]); x0 /= np.linalg.norm(x0); Rl = np.stack([x0, np.cross(up, x0), up])
    Ch = (C @ Rl.T)[:, :2]; Q = P @ Rl.T
    zr = Q[:, 2] - z0
    sel = (ntrack >= 3) & (zr > 0.5 * hm) & (zr < 8 * hm) & (cKDTree(Ch).query(Q[:, :2])[0] < 15 * hm)
    Q = Q[sel, :2] - Ch.mean(0); lim = np.abs(Q).max(); nb = int(min(4000, 2 * lim / (0.1 * hm)))

    def entropy(th):
        c, s = np.cos(th), np.sin(th); e = 0
        for v in (c * Q[:, 0] + s * Q[:, 1], -s * Q[:, 0] + c * Q[:, 1]):
            hist, _ = np.histogram(v, bins=nb, range=(-lim, lim)); q = hist[hist > 0] / hist.sum(); e -= (q * np.log(q)).sum()
        return e
    ths = np.radians(np.arange(0, 90, 0.25)); E = np.array([entropy(t) for t in ths]); yaw = ths[E.argmin()]
    c, s = np.cos(yaw), np.sin(yaw); Rw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ Rl
    ext = np.ptp(C @ Rw.T, axis=0)
    if ext[1] > ext[0]: Rw = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]) @ Rw        # the longer side of the walked area along x
    print(f'facades: yaw {np.degrees(yaw):.2f} deg (entropy {E.min():.3f} vs median {np.median(E):.3f}) from {len(Q):,} points')

    # ---- scale and origin
    scale = 1.0 / hm
    Cw = C @ Rw.T; mid = (Cw.min(0) + Cw.max(0)) / 2
    t = -scale * np.array([mid[0], mid[1], (Rw @ (z0 * up))[2]])
    rec.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(Rw), t))
    out = p.colmap / 'aligned'; out.mkdir(exist_ok=True); rec.write(str(out))
    T = np.eye(4); T[:3, :3] = scale * Rw; T[:3, 3] = t
    Ca = Cw * scale + t; L = float(np.linalg.norm(np.diff(Ca, axis=0), axis=1).sum())
    cm = p.camera_height_cm
    J = dict(unit='lens heights', metres_per_unit=float(cm) / 100 if cm else None, camera_height_cm=cm, scene='outdoor', up_source=src,
             panorama_up_deg_from_mean=dict(median=float(np.median(dev)), max=float(dev.max())),
             gravity_sensor={k: v for k, v in grav.items() if k != 'up'} if grav else None, ground_plane_deg_from_up=ground_vs_up,
             T_aligned_from_colmap=T.tolist(), lens_height_colmap_units=dict(median=hm, std=float(np.std(hs[ok])), first_estimate=h1),
             lens_height_spread_pct=float(100 * np.std(hs[ok]) / hm), lens_height_p10_p90=[float(np.percentile(hs[ok], 10) / hm), float(np.percentile(hs[ok], 90) / hm)],
             shots_with_ground=int(ok.sum()), step_median_lens_heights=step / hm,
             ground_height_range=[float((gz.min() - z0) / hm), float((gz.max() - z0) / hm)],
             camera_tilt_deg_median=float(np.median(tilt)), path_length=L, path_end_to_end=float(np.linalg.norm(Ca[-1] - Ca[0])),
             walked_area=[float(x) for x in np.ptp(Ca[:, :2], axis=0)],
             per_shot={n: dict(lens_height=round(float(h * scale), 4), ground_z=round(float((g - z0) * scale), 4), points=int(k), measured=bool(o))
                       for n, h, g, k, o in zip(names, hs, gz, npts, ok)},
             note='aligned = T @ colmap (homogeneous). z up, z = 0 the median ground under the shots, facades on the axes, unit one lens height.')
    json.dump(J, open(p.product / 'align.json', 'w'), indent=1)
    print(f'unit: one lens height' + (f' = {cm} cm' if cm else ' (camera_height_cm is empty in the YAML: no metres yet)')
          + f'; path {L:.1f} lens heights long, walked area {J["walked_area"][0]:.1f} x {J["walked_area"][1]:.1f}, start to end {J["path_end_to_end"]:.1f}; written {out}')


if p.outdoor:
    outdoor(); sys.exit()
ref = sorted((im for im in rec.images.values() if im.has_pose and im.name.startswith('v00/')), key=lambda im: im.name)
C = np.array([im.projection_center() for im in ref])
R0 = face_rotation(*p.views[0])
up_cam = R0 @ np.array([0, -1.0, 0])                                   # the panorama's up, in face 0
ups = np.array([im.cam_from_world().rotation.matrix().T @ up_cam for im in ref])
up0 = ups.mean(0); up0 /= np.linalg.norm(up0)
P = np.array([q.xyz for q in rec.points3D.values()]); ntrack = np.array([q.track.length() for q in rec.points3D.values()])
L = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
c0 = C.mean(0)
hor = lambda X, u: (X - c0) - np.outer((X - c0) @ u, u)

# ---- floor
h = (P - c0) @ up0
pool = np.flatnonzero((ntrack >= 3) & (h < 0) & (np.linalg.norm(hor(P, up0), axis=1) < 3 * L))
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[pool]))
floor = None
for _ in range(6):
    if len(pc.points) < 200: break
    (a, b, c, d), inl = pc.segment_plane(0.004 * L, 3, 4000)
    n = np.array([a, b, c]); s = np.linalg.norm(n); n, d = n / s, d / s
    if n @ up0 < 0: n, d = -n, -d
    ang = np.degrees(np.arccos(np.clip(n @ up0, -1, 1)))
    print(f'  plane candidate: {len(inl)} inliers, {ang:.1f} deg from the cameras\' up')
    if ang < 15 and len(inl) >= 200: floor = (n, d, len(inl)); break
    pc = pc.select_by_index(inl, invert=True)
if floor is None: raise SystemExit('no floor plane found; refusing to guess')
up, d_floor, n_floor = floor
inl_pts = P[pool][np.abs(P[pool] @ up + d_floor) < 0.004 * L]
cen = inl_pts.mean(0); _, _, vt = np.linalg.svd(inl_pts - cen, full_matrices=False)       # least-squares refit on the inliers
n = vt[2] * np.sign(vt[2] @ up); up, d_floor = n, -n @ cen
heights = C @ up + d_floor
hm = float(np.median(heights))
tilt = np.degrees(np.arccos(np.clip(ups @ up, -1, 1)))
along = (C - C[0]) @ ((C[-1] - C[0]) / np.linalg.norm(C[-1] - C[0]))
slope = np.polyfit(along, heights, 1)[0]
print(f'floor: {n_floor} sparse points (thickness +-{0.004 * L:.4f} units), normal {np.degrees(np.arccos(np.clip(up @ up0, -1, 1))):.2f} deg '
      f'from the cameras\' mean up; camera tilt from vertical: median {np.median(tilt):.2f} deg, max {tilt.max():.2f}')
print(f'lens height above the floor over {len(C)} shots: median {hm:.4f} units, std {heights.std():.4f} = {100 * heights.std() / hm:.2f}%, '
      f'min {heights.min():.4f}, max {heights.max():.4f}; slope along the path {100 * slope:.2f}% ({np.degrees(np.arctan(slope)):.2f} deg)')

# ---- level, then yaw so that walls lie along x and y
x0 = np.cross(up, [1.0, 0, 0]); x0 /= np.linalg.norm(x0); Rl = np.stack([x0, np.cross(up, x0), up])
near = (ntrack >= 3) & (np.linalg.norm(hor(P, up), axis=1) < 3 * L)
Q = (P[near] - c0) @ Rl.T


def entropy(th):
    c, s = np.cos(th), np.sin(th); e = 0
    for v in (c * Q[:, 0] + s * Q[:, 1], -s * Q[:, 0] + c * Q[:, 1]):
        hist, _ = np.histogram(v, bins=240, range=(-3 * L, 3 * L)); q = hist[hist > 0] / hist.sum(); e -= (q * np.log(q)).sum()
    return e


ths = np.radians(np.arange(0, 90, 0.25)); E = np.array([entropy(t) for t in ths]); yaw = ths[E.argmin()]
c, s = np.cos(yaw), np.sin(yaw); Rw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ Rl
d_path = Rw @ (C[-1] - C[0]); path_deg = np.degrees(np.arctan2(d_path[1], d_path[0]))
k = int(np.round(path_deg / 90)); c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
Rw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ Rw                  # the quarter turn that points +x along the drive
d_path = Rw @ (C[-1] - C[0])
print(f'walls: yaw {np.degrees(yaw):.2f} deg (entropy {E.min():.3f} vs median {np.median(E):.3f}); the path runs '
      f'{np.degrees(np.arctan2(d_path[1], d_path[0])):.1f} deg off the +x wall direction')

# ---- scale and origin
cm = p.camera_height_cm
scale = 1.0 / hm
unit = 'lens heights'
mid = (C[0] + C[-1]) / 2; mid_floor = mid - (mid @ up + d_floor) * up
t = -scale * (Rw @ mid_floor)
rec.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(Rw), t))
out = p.colmap / 'aligned'; out.mkdir(exist_ok=True); rec.write(str(out))
T = np.eye(4); T[:3, :3] = scale * Rw; T[:3, 3] = t
J = dict(unit=unit, metres_per_unit=float(cm) / 100 if cm else None, camera_height_cm=cm, T_aligned_from_colmap=T.tolist(),
         lens_height_colmap_units=dict(median=hm, std=float(heights.std()), min=float(heights.min()), max=float(heights.max())),
         lens_height_spread_pct=float(100 * heights.std() / hm), floor_points=int(n_floor),
         floor_slope_along_path_pct=float(100 * slope), path_length=float(L * scale), path_end_to_end=float(np.linalg.norm(C[-1] - C[0]) * scale),
         path_length_in_camera_heights=float(L / hm), camera_tilt_deg_median=float(np.median(tilt)),
         note='aligned = T @ colmap (homogeneous). z up, floor z = 0, walls on the axes, +x = the wall direction closest to the drive.')
json.dump(J, open(p.product / 'align.json', 'w'), indent=1)
print(f'unit: one lens height' + (f' = {cm} cm' if cm else ' (camera_height_cm is empty in the YAML: no metres yet)')
      + f'; path {L / hm:.2f} lens heights long' + (f' = {L / hm * float(cm):.1f} cm' if cm else '') + f'; written {out}')
