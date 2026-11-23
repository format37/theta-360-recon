#!/usr/bin/env python3
"""Numbers of the OpenMVG spherical model, and how far it is from the rig model of the same panoramas.

    python3 -I scripts/spherical_report.py data/pass-0/spherical.yaml data/pass-0/rig.yaml

Reprojection error in degrees: angle between the ray of the keypoint in the equirectangular picture and the
ray to its 3D point. The two models are compared after a similarity transform fitted on the camera centres:
distance between matching centres (as % of the path length) and the angle between matching orientations.
Writes product/spherical.json in the spherical variant's folder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from pano import face_rotation
from project import load

ps, pr = load(sys.argv[1]), load(sys.argv[2])
J = json.load(open(ps.run / 'openmvg' / 'sfm' / 'sfm_data.json'))
views = {v['key']: v['value']['ptr_wrapper']['data'] for v in J['views']}
W, H = views[0]['width'], views[0]['height']
poses = {e['key']: (np.array(e['value']['rotation']), np.array(e['value']['center'])) for e in J['extrinsics']}
ang, ntrack = [], []
for s in J['structure']:
    X = np.array(s['value']['X']); obs = s['value']['observations']; ntrack.append(len(obs))
    for o in obs:
        Rm, C = poses[views[o['key']]['id_pose']]; u, v = o['value']['x']
        lon = (u - W / 2) / W * 2 * np.pi; lat = -(v - H / 2) / W * 2 * np.pi
        b = np.array([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)])
        xc = Rm @ (X - C); ang.append(np.degrees(np.arccos(np.clip(b @ xc / np.linalg.norm(xc), -1, 1))))
ang = np.array(ang)
R = dict(panoramas_registered=len(poses), panoramas_total=len(ps.photos), points=len(J['structure']), observations=int(len(ang)),
         track_length_mean=round(float(np.mean(ntrack)), 2),
         reproj_deg_mean=round(float(ang.mean()), 4), reproj_deg_median=round(float(np.median(ang)), 4), reproj_deg_p95=round(float(np.percentile(ang, 95)), 4),
         reproj_px_equirect_mean=round(float(np.radians(ang.mean()) * W / (2 * np.pi)), 3),
         reproj_px_at_face_focal_mean=round(float(np.radians(ang.mean()) * pr.focal), 3))
print(f"OpenMVG spherical: {R['panoramas_registered']}/{R['panoramas_total']} panoramas, {R['points']:,} points, {R['observations']:,} observations, "
      f"track length {R['track_length_mean']}; reprojection error {R['reproj_deg_mean']} deg mean / {R['reproj_deg_median']} median "
      f"(= {R['reproj_px_equirect_mean']} px of the panorama, {R['reproj_px_at_face_focal_mean']} px at the faces' focal length)")

# ---- against the rig model
rec = pycolmap.Reconstruction(str(pr.colmap / 'sparse' / '0'))
R0 = face_rotation(*pr.views[0])
rig = {Path(im.name).stem: (R0.T @ im.cam_from_world().rotation.matrix(), im.projection_center())
       for im in rec.images.values() if im.has_pose and im.name.startswith('v00/')}
name_of = {d['id_pose']: Path(d['filename']).stem for d in views.values()}
common = sorted(n for k, n in name_of.items() if k in poses and n in rig)
inv = {n: k for k, n in name_of.items()}
A = np.array([poses[inv[n]][1] for n in common]); B = np.array([rig[n][1] for n in common])       # OpenMVG centres -> rig centres
ma, mb = A.mean(0), B.mean(0); U, D, Vt = np.linalg.svd((B - mb).T @ (A - ma)); S_ = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
Ra = U @ S_ @ Vt; s = np.trace(np.diag(D) @ S_) / ((A - ma) ** 2).sum(); ta = mb - s * Ra @ ma
res = np.linalg.norm((s * (Ra @ A.T).T + ta) - B, axis=1)
L = float(np.linalg.norm(np.diff(B, axis=0), axis=1).sum())
# Orientations: the centres lie on a straight line, so they do not fix the roll about that line. The rotation
# between the two world frames is therefore taken from the orientations themselves (their mean), and what is
# reported is how far each panorama is from that one common rotation.
D = [rig[n][0].T @ poses[inv[n]][0] for n in common]                     # rig world <- OpenMVG world, once per panorama
U2, _, Vt2 = np.linalg.svd(np.sum(D, axis=0)); Mrot = U2 @ np.diag([1, 1, np.sign(np.linalg.det(U2 @ Vt2))]) @ Vt2
rot = np.array([np.degrees(np.arccos(np.clip((np.trace(d @ Mrot.T) - 1) / 2, -1, 1))) for d in D])
stepA = np.linalg.norm(np.diff(A, axis=0), axis=1) * s; stepB = np.linalg.norm(np.diff(B, axis=0), axis=1)
R['vs_rig'] = dict(common_panoramas=len(common), centre_rms_pct_of_path=round(float(100 * np.sqrt((res ** 2).mean()) / L), 3),
                   centre_max_pct_of_path=round(float(100 * res.max() / L), 3), rotation_deg_median=round(float(np.median(rot)), 3),
                   rotation_deg_max=round(float(rot.max()), 3), step_ratio_median=round(float(np.median(stepA / stepB)), 4),
                   step_ratio_min_max=[round(float((stepA / stepB).min()), 3), round(float((stepA / stepB).max()), 3)])
print(f"against the rig model on {len(common)} shared panoramas: centres differ by {R['vs_rig']['centre_rms_pct_of_path']}% of the path length (rms), "
      f"{R['vs_rig']['centre_max_pct_of_path']}% at most; orientations by {R['vs_rig']['rotation_deg_median']} deg (median), {R['vs_rig']['rotation_deg_max']} at most")
ps.product.mkdir(parents=True, exist_ok=True)
json.dump(R, open(ps.product / 'spherical.json', 'w'), indent=1); print('wrote', ps.product / 'spherical.json')
