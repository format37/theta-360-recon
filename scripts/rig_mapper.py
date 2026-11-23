#!/usr/bin/env python3
"""Incremental SfM where the faces of one panorama move as one rigid rig (pycolmap >= 3.12).

    python3 -I scripts/rig_mapper.py data/pass-0/rig.yaml [--free]

The database comes from the COLMAP 3.11 CLI (GPU SIFT and matching); pycolmap adds the rig and frame tables to it.
Fixed in bundle adjustment: the faces' rotations in the rig, their zero translations, and the intrinsics
(they are exact by construction: we rendered the faces ourselves). Only the 6 numbers of each panorama's
pose and the 3D points are solved.
sfm.mapper in the YAML passes options through to pycolmap.IncrementalPipelineOptions (or its .mapper), e.g.
{ba_global_frames_ratio: 1.3, ba_global_ignore_redundant_points3D: true} for fewer and lighter global adjustments.
If the walk falls apart into several models, all are written; the largest becomes sparse/0.
--free: no rig, every face is its own image (the comparison run -> colmap/sparse_free).
"""
import shutil
import sys
from pathlib import Path

import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1]); free = '--free' in sys.argv
src = p.colmap / 'database.db'
db_path = p.colmap / ('database_free.db' if free else 'database_rig.db')
out = p.colmap / ('sparse_free' if free else 'sparse')
shutil.copy(src, db_path); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)

if not free:
    db = pycolmap.Database.open(str(db_path))
    pycolmap.apply_rig_config(pycolmap.read_rig_config(str(p.product / 'rig.json')), db)
    print(f'rigs {db.num_rigs()}, frames {db.num_frames()}, images {db.num_images()}', flush=True)
    db.close()

opts = pycolmap.IncrementalPipelineOptions(
    ba_refine_focal_length=False, ba_refine_principal_point=False, ba_refine_extra_params=False,
    ba_refine_sensor_from_rig=False)
for k, v in p.mapper.items():                       # straight from the YAML
    target = opts if hasattr(opts, k) else opts.mapper if hasattr(opts.mapper, k) else None
    if target is None: raise SystemExit(f'sfm.mapper.{k}: pycolmap.IncrementalPipelineOptions has no such option')
    setattr(target, k, v); print(f'option {k} = {v}', flush=True)
recs = pycolmap.incremental_mapping(str(db_path), str(p.faces), str(out), opts)
best = max(recs, key=lambda i: recs[i].num_reg_frames()) if recs else None
if best not in (None, 0):                            # every later step reads sparse/0: make it the largest model
    tmp = out / 'tmp'; (out / '0').rename(tmp); (out / str(best)).rename(out / '0'); tmp.rename(out / str(best))
    recs[0], recs[best] = recs[best], recs[0]
for i, rec in sorted(recs.items()):
    print(f'model {i}: {rec.num_reg_frames()} panoramas; {rec.summary()}', flush=True)
