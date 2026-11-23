#!/usr/bin/env python3
"""Experiment: does a stitched THETA panorama have one centre? Let bundle adjustment move the faces in the rig.

    python3 -I scripts/rig_refine.py data/pass-0/rig.yaml

The THETA S has two fisheye lenses back to back, about 2 cm apart, and stitches their pictures inside the
camera. The rig model says all faces share one centre. Here the faces' poses in the rig are freed (rotation
and translation of faces 1..N against face 0; intrinsics stay fixed) and the model is adjusted again.
If the two lenses show up, faces cut from the rear lens move along the lens axis against faces from the front one.
Reported: where each face's centre ends up (panorama frame: x right, y down, z = front lens axis), how far its
rotation moved, and the reprojection error before and after.
Writes colmap/sparse_twocentre/ and product/rig_refine.json. The main pipeline does not use this model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from pano import face_rotation
from project import load
from sfm_report import headline, observations

p = load(sys.argv[1])
rec = pycolmap.Reconstruction(str(p.colmap / 'sparse' / '0'))
before = headline(rec, observations(rec))
al = json.load(open(p.product / 'align.json')) if (p.product / 'align.json').exists() else None
h = al['lens_height_colmap_units']['median'] if al else None                 # lens height in the units of this model
opts = pycolmap.BundleAdjustmentOptions(refine_focal_length=False, refine_principal_point=False, refine_extra_params=False,
                                        refine_sensor_from_rig=True, refine_rig_from_world=True)
pycolmap.bundle_adjustment(rec, opts)
after = headline(rec, observations(rec))
out = p.colmap / 'sparse_twocentre'; out.mkdir(exist_ok=True); rec.write(str(out))

rots = [face_rotation(y, pt) for y, pt in p.views]; R0 = rots[0]
rig = next(iter(rec.rigs.values())); rows = {}
cam_view = {c: int(Path(next(im.name for im in rec.images.values() if im.camera_id == c)).parent.name[1:]) for c in rec.cameras}
print(f"reprojection error {before['reproj_deg_mean']} -> {after['reproj_deg_mean']} deg mean, {before['reproj_px_mean']} -> {after['reproj_px_mean']} px")
print('face centres after the faces were freed (panorama frame; z = axis of the front lens), in ' + ('lens heights' if h else 'model units')
      + (f" and cm at {p.camera_height_cm} cm lens height" if h and p.camera_height_cm else '') + ':')
for s, T in sorted(rig.non_ref_sensors.items(), key=lambda kv: kv[0].id):
    v = cam_view[s.id]
    c_rig = -T.rotation.matrix().T @ T.translation                           # the face's centre in the rig frame (= face 0)
    c_pano = R0.T @ c_rig
    dR = T.rotation.matrix() @ (rots[v] @ R0.T).T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    c = c_pano / h if h else c_pano
    rows[f'v{v:02d}'] = dict(yaw=p.views[v][0], pitch=p.views[v][1], centre=[float(x) for x in c], rotation_moved_deg=ang)
    cm = f"  = [{', '.join(f'{x * p.camera_height_cm:+.2f}' for x in c)}] cm" if h and p.camera_height_cm else ''
    print(f"  v{v:02d} yaw {p.views[v][0]:5.0f} pitch {p.views[v][1]:4.0f}: centre [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}]{cm}, rotation moved {ang:.3f} deg")
json.dump(dict(before=before, after=after, unit='lens heights' if h else 'model units', faces=rows), open(p.product / 'rig_refine.json', 'w'), indent=1)
