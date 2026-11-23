#!/usr/bin/env python3
"""Read a session YAML: settings and the paths of the run.

    from project import load;  p = load('data/pass-0/rig.yaml')
    eval "$(python3 scripts/project.py data/pass-0/rig.yaml)"     # the same as shell variables

A session is a folder data/<session>/ with the panoramas in photos/ and one or more YAML files.
Every YAML is one variant of the run and writes into its own folder:
    data/<session>/photos/*.JPG          (input: equirectangular panoramas)
    data/<session>/<variant>.yaml
    data/<session>/out/<variant>/        faces/ masks/ labels/ colmap/ dense/ splat/ product/

scene: indoor (the default, pass-0) or outdoor. Every default below is the indoor one, so an old YAML runs as
it did; data/pool/rig.yaml is the outdoor example with every key commented.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_VIEWS = [[45, -10], [135, -10], [225, -10], [315, -10], [0, 50], [90, 50], [180, 50], [270, 50]]
INDOOR_PRIOR = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf'


def load(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(open(path)) or {}
    session = path.parent
    run = session / 'out' / path.stem
    g = lambda k: cfg.get(k) or {}
    faces, mask, match, sfm, dense, mesh, vox, splat = (g(k) for k in ('faces', 'mask', 'matching', 'sfm', 'dense', 'mesh', 'voxels', 'splat'))
    fov, size = float(faces.get('fov', 90)), int(faces.get('size', 1600))
    outdoor = cfg.get('scene', 'indoor') == 'outdoor'
    p = SimpleNamespace(
        yaml=path, session=session, name=f'{session.name}/{path.stem}', run=run, cfg=cfg,
        photos=sorted(session.glob(cfg.get('photos', 'photos/*.JPG'))), photos_glob=cfg.get('photos', 'photos/*.JPG'),
        method=cfg.get('method', 'rig'), outdoor=outdoor,
        faces=run / 'faces', masks=run / 'masks', labels=run / 'labels', masks_equirect=run / 'masks_equirect',
        colmap=run / 'colmap', dense=run / 'dense', splat=run / 'splat', product=run / 'product',
        fov=fov, size=size, focal=size / 2 / math.tan(math.radians(fov) / 2),
        views=[tuple(map(float, v)) for v in faces.get('views', DEFAULT_VIEWS)],
        # masks: the platform (rover: auto | none | <png in the session>), then classes of a segmentation network per face
        mask_rover=mask.get('rover', 'auto'), mask_grow_deg=float(mask.get('grow_deg', 1.0)),
        mask_seam_deg=float(mask.get('seam_deg', 0)), mask_semantic=list(mask.get('semantic') or []),
        mask_given=mask.get('given'),          # a folder in the session: <photo stem>.png drawn on the panorama, white = hide (scripts/anonymise.py)
        seg_model=mask.get('seg_model', 'openmmlab/upernet-convnext-large'), seg_size=int(mask.get('seg_size', 768)),
        features=int(cfg.get('sift_features', 8192)),
        # matching: no window = every face against every face of every other panorama (pass-0)
        match_window=match.get('window'), match_retrieval=int(match.get('retrieval', 0)),
        match_min=int(match.get('min_face_matches', 6)),
        mapper=dict(sfm.get('mapper') or {}),
        camera_height_cm=cfg.get('camera_height_cm'), checks_cm=cfg.get('checks_cm') or {},
        dense_size=int(dense.get('size', 1024)), dense_every=int(dense.get('every', 2)),
        dense_sources=int(dense.get('sources', 12)), dense_max_sky=float(dense.get('max_sky', 1.0)),
        dense_keep_maps=bool(dense.get('keep_maps', True)),
        prior_model=dense.get('prior_model', INDOOR_PRIOR),
        # mesh, voxels: lengths in lens heights (the unit of the aligned model)
        mesh_planes=list(mesh.get('planes', ['floor', 'ceiling', 'walls'])), mesh_max_depth=float(mesh.get('max_depth', 30)),
        mesh_reach=mesh.get('reach'), mesh_voxel=float(mesh.get('voxel', 0.07)), mesh_max_walls=int(mesh.get('max_walls', 12)),
        voxel_size=float(vox.get('size', 0.1)), voxel_min_views=int(vox.get('min_views', 3)), voxel_source=vox.get('source', 'filled'),
        voxel_top=vox.get('top'), voxel_bottom=vox.get('bottom'),
        splat_size=int(splat.get('size', 1200)), splat_steps=int(splat.get('steps', 30000)),
        splat_cap=int(splat.get('cap', 1500000)), holdout_every=int(splat.get('holdout_every', 6)),
        holdout_tail=int(splat.get('holdout_tail', 5)), splat_sky=splat.get('sky', 'none'), splat_init=splat.get('init', 'sparse'))
    return p


def pano_id(face_name):
    """'v03/R0012938.jpg' -> 'R0012938'"""
    return Path(face_name).stem


if __name__ == '__main__':
    p = load(sys.argv[1])
    print(f'RUN="{p.run}"; FACES="{p.faces}"; MASKS="{p.masks}"; WS="{p.colmap}"; DENSE="{p.dense}"; '
          f'SPLAT="{p.splat}"; PRODUCT="{p.product}"; SESSION="{p.session}"; PHOTOS_GLOB="{p.photos_glob}"; FEATURES={p.features}; '
          f'FOCAL={p.focal:.4f}; SIZE={p.size}; DENSE_SIZE={p.dense_size}; DENSE_SOURCES={p.dense_sources}; '
          f'DENSE_KEEP_MAPS={int(p.dense_keep_maps)}; METHOD={p.method}; NPHOTOS={len(p.photos)}')
