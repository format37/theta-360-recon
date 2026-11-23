#!/usr/bin/env python3
"""Numbers of a sparse model: what registered, the reprojection error in pixels AND in degrees, the path,
and where on the sphere the observations are bad (stitching seam, near objects).

    python3 -I scripts/sfm_report.py data/pass-0/rig.yaml [--model colmap/sparse/0] [--tag sfm]
    python3 -I scripts/sfm_report.py --plain ~/projects/kitchen-recon/data/ws/sparse_aligned     # any COLMAP model

Degrees: the angle between the ray of the measured keypoint and the ray to its 3D point. Pixels of a 4K phone
frame and pixels of a 1600 px face of a panorama are different sizes; degrees are the same thing for both.
Writes product/<tag>.json and product/<tag>.png.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))


def observations(rec):
    """Per image: keypoints with a 3D point -> pixel error, angular error (deg), depth, pixel xy."""
    out = {}
    for im in rec.images.values():
        if not im.has_pose: continue
        cam = rec.cameras[im.camera_id]
        p2 = [q for q in im.points2D if q.has_point3D()]
        n_all = len(im.points2D)
        if not p2: out[im.name] = dict(n_all=n_all, xy=np.zeros((0, 2)), px=np.zeros(0), deg=np.zeros(0), depth=np.zeros(0)); continue
        xy = np.array([q.xy for q in p2]); X = np.array([rec.points3D[q.point3D_id].xyz for q in p2])
        T = im.cam_from_world().matrix(); Xc = X @ T[:3, :3].T + T[:3, 3]
        proj = np.array([cam.img_from_cam(x) for x in Xc])
        obs_n = np.array([cam.cam_from_img(q) for q in xy])                       # normalised image coordinates
        r_obs = np.c_[obs_n, np.ones(len(obs_n))]; r_obs /= np.linalg.norm(r_obs, axis=1, keepdims=True)
        r_3d = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        out[im.name] = dict(n_all=n_all, xy=xy, px=np.linalg.norm(proj - xy, axis=1),
                            deg=np.degrees(np.arccos(np.clip((r_obs * r_3d).sum(1), -1, 1))), depth=np.linalg.norm(Xc, axis=1))
    return out


def headline(rec, obs, total=None):
    px = np.concatenate([o['px'] for o in obs.values()]); deg = np.concatenate([o['deg'] for o in obs.values()])
    tl = np.array([p.track.length() for p in rec.points3D.values()])
    return dict(images_registered=len(obs), images_total=total or rec.num_images(), points=len(rec.points3D),
                observations=int(len(px)), track_length_mean=round(float(tl.mean()), 2),
                reproj_px_mean=round(float(px.mean()), 3), reproj_px_median=round(float(np.median(px)), 3),
                reproj_deg_mean=round(float(deg.mean()), 4), reproj_deg_median=round(float(np.median(deg)), 4),
                reproj_deg_p95=round(float(np.percentile(deg, 95)), 4))


if __name__ == '__main__':
    if sys.argv[1] == '--plain':
        rec = pycolmap.Reconstruction(sys.argv[2]); print(json.dumps(headline(rec, observations(rec)), indent=1)); sys.exit()
    from pano import face_rotation, pixel_lonlat
    from project import load
    p = load(sys.argv[1]); a = sys.argv[2:]
    model = p.run / (a[a.index('--model') + 1] if '--model' in a else 'colmap/sparse/0')
    tag = a[a.index('--tag') + 1] if '--tag' in a else 'sfm'
    rec = pycolmap.Reconstruction(str(model)); obs = observations(rec)
    R = headline(rec, obs, total=len(p.photos) * len(p.views))
    panos = defaultdict(list)
    for name in obs: panos[Path(name).stem].append(name)
    R['panoramas_registered'] = sum(len(v) == len(p.views) for v in panos.values()); R['panoramas_total'] = len(p.photos)
    R['focal_px'] = p.focal
    print(f"registered {R['images_registered']}/{R['images_total']} faces = {R['panoramas_registered']}/{R['panoramas_total']} panoramas; "
          f"{R['points']:,} points, {R['observations']:,} observations, track length {R['track_length_mean']}")
    print(f"reprojection error: {R['reproj_px_mean']} px mean / {R['reproj_px_median']} median at focal {p.focal:.0f} px "
          f"= {R['reproj_deg_mean']} deg mean / {R['reproj_deg_median']} median / {R['reproj_deg_p95']} at the 95th percentile")

    # ---- per face direction, and over the sphere (longitude: the stitching seam is at +-90 deg)
    rots = [face_rotation(y, pt) for y, pt in p.views]
    lon, lat, px, deg, depth, lon_all, lat_all = [], [], [], [], [], [], []
    R['per_view'] = {}
    for v in range(len(p.views)):
        names = [n for n in obs if n.startswith(f'v{v:02d}/')]
        e = np.concatenate([obs[n]['px'] for n in names]) if names else np.zeros(0)
        R['per_view'][f'v{v:02d} yaw {p.views[v][0]:.0f} pitch {p.views[v][1]:.0f}'] = dict(
            faces=len(names), observations=int(len(e)), px_mean=round(float(e.mean()), 3) if len(e) else None,
            triangulated_share=round(float(len(e) / max(1, sum(obs[n]['n_all'] for n in names))), 3))
        for n in names:
            lo, la = pixel_lonlat(obs[n]['xy'], rots[v], p.size, p.focal)
            lon.append(lo); lat.append(la); px.append(obs[n]['px']); deg.append(obs[n]['deg']); depth.append(obs[n]['depth'])
            allxy = np.array([q.xy for q in rec.image(rec.find_image_with_name(n).image_id).points2D]).reshape(-1, 2)
            lo, la = pixel_lonlat(allxy, rots[v], p.size, p.focal); lon_all.append(lo); lat_all.append(la)
    lon, lat, px, deg, depth = map(np.concatenate, (lon, lat, px, deg, depth)); lon_all, lat_all = np.concatenate(lon_all), np.concatenate(lat_all)
    for k, r in R['per_view'].items(): print(f"  {k}: {r['observations']:7,d} observations, {r['px_mean']} px, {100 * r['triangulated_share']:.0f}% of keypoints triangulated")
    seam = np.abs(np.abs(lon) - 90); seam_all = np.abs(np.abs(lon_all) - 90)
    R['seam'] = {}
    for lo_, hi_ in ((0, 5), (5, 10), (10, 20), (20, 45), (45, 90)):
        m = (seam >= lo_) & (seam < hi_); ma = (seam_all >= lo_) & (seam_all < hi_)
        R['seam'][f'{lo_}-{hi_} deg from the seam'] = dict(
            observations=int(m.sum()), deg_mean=round(float(deg[m].mean()), 4), deg_median=round(float(np.median(deg[m])), 4),
            triangulated_share=round(float(m.sum() / max(1, ma.sum())), 3),
            keypoints_per_sq_deg=round(float(ma.sum() / len(obs) * len(p.views) / (4 * (hi_ - lo_) * 180)), 2))
    print('by distance from the stitching seam (longitude +-90 deg):')
    for k, r in R['seam'].items(): print(f"  {k:26s} {r['observations']:8,d} obs, error {r['deg_mean']:.4f} deg mean / {r['deg_median']:.4f} median, "
                                         f"{100 * r['triangulated_share']:.0f}% of keypoints triangulated")
    R['by_depth'] = {}
    dq = np.percentile(depth, [0, 10, 25, 50, 75, 90, 100])
    for a_, b_ in zip(dq[:-1], dq[1:]):
        m = (depth >= a_) & (depth <= b_)
        R['by_depth'][f'{a_:.2f}-{b_:.2f} units'] = dict(deg_mean=round(float(deg[m].mean()), 4), deg_median=round(float(np.median(deg[m])), 4))
    print('by distance of the point from the camera (model units): ' + '; '.join(f"{k}: {r['deg_mean']:.4f}" for k, r in R['by_depth'].items()))

    # ---- path of the rig centres
    ref = sorted(n for n in obs if n.startswith('v00/'))
    C = np.array([rec.image(rec.find_image_with_name(n).image_id).projection_center() for n in ref])
    Rw = [rec.image(rec.find_image_with_name(n).image_id).cam_from_world().rotation.matrix() for n in ref]
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    turn = np.array([np.degrees(np.arccos(np.clip((np.trace(Rw[i + 1] @ Rw[i].T) - 1) / 2, -1, 1))) for i in range(len(Rw) - 1)])
    c0 = C.mean(0); _, s, vt = np.linalg.svd(C - c0); off = np.linalg.norm((C - c0) - np.outer((C - c0) @ vt[0], vt[0]), axis=1)
    L = float(step.sum())
    R['path'] = dict(length_units=round(L, 4), end_to_end_units=round(float(np.linalg.norm(C[-1] - C[0])), 4),
                     step_median=round(float(np.median(step)), 5), step_min=round(float(step.min()), 5), step_max=round(float(step.max()), 5),
                     step_max_over_median=round(float(step.max() / np.median(step)), 2),
                     off_line_max_pct_of_length=round(float(100 * off.max() / L), 2),
                     turn_between_shots_deg_median=round(float(np.median(turn)), 3), turn_max=round(float(turn.max()), 3),
                     total_turn_first_to_last_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(Rw[-1] @ Rw[0].T) - 1) / 2, -1, 1)))), 2))
    print(f"path: {len(C)} centres, length {L:.3f} units, steps median {np.median(step):.4f} (min {step.min():.4f}, max {step.max():.4f} = "
          f"{step.max() / np.median(step):.1f}x median); off a straight line by at most {100 * off.max() / L:.1f}% of the length; "
          f"turn between shots median {np.median(turn):.2f} deg, max {turn.max():.2f}, first to last {R['path']['total_turn_first_to_last_deg']} deg")
    # a panorama placed wrongly shows as a jump out of the walk and back: both of its steps much longer than the steps around
    local = np.array([np.median(step[max(0, k - 5):k + 6]) for k in range(len(step))])
    big = step > 4 * local
    R['path']['steps_over_4x_local_median'] = [[Path(ref[k]).stem, Path(ref[k + 1]).stem, round(float(step[k] / local[k]), 1)] for k in np.flatnonzero(big)]
    # A spike: a shot far off the line between its two neighbours, against the local step. On the pool walk R0013079 (under
    # the pines, 97 sparse points) sat 17 lens heights off it with steps of 13.6 and 21.0 against a median of 5: only one
    # step over 4x, so the jump test alone missed it; OpenMVG put it back on the walk.
    spike = np.zeros(len(C))
    for k in range(1, len(C) - 1):
        a_, b_ = C[k - 1], C[k + 1]; t_ = np.clip((C[k] - a_) @ (b_ - a_) / max((b_ - a_) @ (b_ - a_), 1e-12), 0, 1)
        spike[k] = np.linalg.norm(C[k] - (a_ + t_ * (b_ - a_))) / np.median(step[max(0, k - 6):k + 6])
    R['path']['spike_over_local_step'] = {Path(ref[k]).stem: round(float(spike[k]), 2) for k in np.flatnonzero(spike > 1.5)}
    R['path']['suspected_misplaced'] = sorted({Path(ref[k]).stem for k in range(1, len(C) - 1) if (big[k - 1] and big[k]) or spike[k] > 2.5})
    print(f"steps over 4x the local median: {len(R['path']['steps_over_4x_local_median'])}; suspected misplaced panoramas (a jump out and back, or "
          f"more than 2.5 local steps off the line between its neighbours): " + (' '.join(R['path']['suspected_misplaced']) or 'none'))
    json.dump(R, open(p.product / f'{tag}.json', 'w'), indent=1)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    bins = np.arange(-180, 181, 10); idx = np.digitize(lon, bins) - 1
    ax[0].bar(bins[:-1] + 5, [deg[idx == i].mean() if (idx == i).any() else 0 for i in range(len(bins) - 1)], width=9, color='#3b6ea5')
    for x in (-90, 90): ax[0].axvline(x, color='r', ls='--', lw=1)
    ax[0].set(xlabel='longitude in the panorama, deg (red: stitching seam)', ylabel='mean reprojection error, deg', title='error over the sphere')
    ax0b = ax[0].twinx(); ax0b.plot(bins[:-1] + 5, np.histogram(lon, bins)[0], color='#e08a00', lw=1.5); ax0b.set_ylabel('observations per 10 deg (orange)')
    Q = (C - c0) @ vt[:2].T; ax[1].plot(Q[:, 0], Q[:, 1], '.-', color='#3b6ea5'); ax[1].set_aspect('equal', 'datalim')
    ax[1].set(title=f'rig centres in their best plane ({len(C)} shots)', xlabel='along the path, units', ylabel='across, units')
    ax[2].plot(step, '.-', color='#3b6ea5'); ax[2].set(title='step between consecutive shots', xlabel='shot', ylabel='units'); ax[2].set_ylim(0)
    plt.tight_layout(); plt.savefig(p.product / f'{tag}.png', dpi=110)
    print('wrote', p.product / f'{tag}.json', p.product / f'{tag}.png')
