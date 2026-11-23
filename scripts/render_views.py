#!/usr/bin/env python3
"""Pictures to judge the mesh, the voxel model and the splat by eye, from poses the capture did and did not visit.

    python3 -I scripts/render_views.py data/pool/rig.yaml

  product/renders/mesh_check.jpg     photo | PatchMatch-only mesh | mesh with prior and planes, from four real poses
  product/renders/mesh_top.jpg       both meshes from above (indoors: everything above 80% of the room height removed)
  product/renders/*_aerial.jpg       outdoors: the mesh, the voxel cubes and every splat from four sides, 30 deg from
                                     above, looking at the middle of the walked area: views nobody took a photo from
  product/renders/voxels_*.png       the voxel model in voxview (needs a display): whole, and coloured by source
  product/renders/<splat>_eye.jpg    each trained splat from a spot on the path at 1x, 3x and 6x the lens height,
                                     looking along +x, +y, -x, -y. The photos were taken from 1x only.
  product/mesh_coverage.json         how much of the real views (outside the masks) each mesh covers
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from gsplat import rasterization

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1]); out = p.product / 'renders'; out.mkdir(parents=True, exist_ok=True)
Rr = o3d.visualization.rendering
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
ref = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
C = np.array([im.projection_center() for im in ref if im.name.startswith('v00/')])
h = float(np.median(C[:, 2])); mid = C[len(C) // 2]
if p.outdoor:
    spot = C[np.argmin(np.linalg.norm(C[:, :2] - np.median(C[:, :2], 0), axis=1))]     # a place somebody stood at, near the middle of the walk
    centre = np.r_[(C[:, :2].min(0) + C[:, :2].max(0)) / 2, 0.0]; span = float(np.ptp(C[:, :2], axis=0).max()) + 2 * h
M = json.load(open(p.product / 'mesh.json')) if (p.product / 'mesh.json').exists() else None


def look_at(eye, target, up=(0, 0, 1.0)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); d = np.cross(f, r)
    E = np.eye(4); E[:3, :3] = np.stack([r, d, f]); E[:3, 3] = -E[:3, :3] @ eye
    return E


def label(img, txt):
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4); cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return img


def aerial_poses(Wd, Hd, fov=50, centre=None, span=None):
    """Four views from 30 deg above, one per diagonal, far enough to see the whole walked area (or the given box)."""
    centre = globals()['centre'] if centre is None else centre; span = globals()['span'] if span is None else span
    fd = Wd / 2 / np.tan(np.radians(fov / 2)); Kd = np.array([[fd, 0, Wd / 2], [0, fd, Hd / 2], [0, 0, 1.0]]); dist = 0.6 * span / np.tan(np.radians(fov / 2))
    res = []
    for name, az in (('from the south-west (-x -y)', 225), ('from the south-east (+x -y)', 315), ('from the north-east (+x +y)', 45), ('from the north-west (-x +y)', 135)):
        d = np.array([np.cos(np.radians(az)) * np.cos(np.radians(30)), np.sin(np.radians(az)) * np.cos(np.radians(30)), np.sin(np.radians(30))])
        res.append((name, look_at(centre + dist * d, centre), Kd))
    return res


# ------------------------------------------------------------------ meshes and voxel cubes (all Open3D rendering stays ahead of the
# first CUDA call below: a Filament renderer created after torch has taken the GPU crashes here)
meshes = {k: o3d.io.read_triangle_mesh(str(p.product / f)) for k, f in (('PatchMatch only', 'mesh_mvs.ply'), ('with prior + planes', 'mesh.ply'))
          if (p.product / f).exists()}
mat = Rr.MaterialRecord(); mat.shader = 'defaultUnlit'


def shot(mesh, E, w, hh, Kx, bg=(0.25, 0.0, 0.25)):
    r = Rr.OffscreenRenderer(w, hh); r.scene.set_background([*bg, 1.0]); r.scene.add_geometry('g', mesh, mat)
    r.scene.view.set_post_processing(False); r.setup_camera(Kx, E, w, hh)
    img = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR); hole = np.asarray(r.render_to_depth_image()) >= 1.0
    del r
    return img, 100 * (1 - hole.mean())


if meshes:
    W = 800; f = W / 2; K = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1.0]])
    drec = pycolmap.Reconstruction(str(p.dense / 'sparse'))
    dims = sorted((im for im in drec.images.values() if im.has_pose and (p.dense / 'final' / (im.name + '.npy')).exists()), key=lambda im: im.name)
    if p.outdoor:                                                        # in each quarter of the walk, the face that saw the most structure
        stems = sorted({Path(im.name).stem for im in dims}); lower = {f'v{v:02d}' for v, (_, pt) in enumerate(p.views) if pt < 0}
        picks = []
        for qa in range(4):
            part = set(stems[qa * len(stems) // 4:(qa + 1) * len(stems) // 4])
            picks.append(max((im for im in dims if Path(im.name).stem in part and im.name[:3] in lower), key=lambda im: im.num_points3D).name)
    else:
        dense = sorted({Path(q.name).stem for q in dims})
        picks = [f'v00/{dense[1]}.jpg', f'v01/{dense[len(dense) // 2]}.jpg', f'v02/{dense[-2]}.jpg', f'v05/{dense[len(dense) // 2]}.jpg']
    rows = []
    for name in picks:
        im = rec.image(rec.find_image_with_name(name).image_id); E = np.eye(4); E[:3] = im.cam_from_world().matrix()
        tiles = [label(cv2.resize(cv2.imread(str(p.faces / name)), (W, W), interpolation=cv2.INTER_AREA), f'photo {name}')]
        for k, m in meshes.items():
            img, cov = shot(m, E, W, W, K); tiles.append(label(img, f'{k}: {cov:.0f}% of the view'))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / 'mesh_check.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    tiles = []
    for k, m in meshes.items():
        V = np.asarray(m.vertices); Wd, Hd = 1100, 900; fd = Wd / 2 / np.tan(np.radians(25)); Kd = np.array([[fd, 0, Wd / 2], [0, fd, Hd / 2], [0, 0, 1.0]])
        if p.outdoor:
            cut = m; eye = centre + np.array([0, -0.05 * span, 1.2 * span / 2 / np.tan(np.radians(25))]); target = centre
        else:
            top = np.percentile(V[:, 2], 99.5); cut = o3d.geometry.TriangleMesh(m); cut.remove_vertices_by_mask(V[:, 2] > 0.8 * top)
            eye = np.array([mid[0], mid[1] - 6 * h, 60 * h]); target = [mid[0], mid[1], 0]
        img, _ = shot(cut, look_at(eye, target, up=(0, 1.0, 0)), Wd, Hd, Kd); tiles.append(label(img, f'{k}, from above'))
    cv2.imwrite(str(out / 'mesh_top.jpg'), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote', out / 'mesh_check.jpg', out / 'mesh_top.jpg', flush=True)
    # how much of each real view does a mesh cover, outside the masks
    Wc = 400; Kc = np.array([[Wc / 2, 0, Wc / 2], [0, Wc / 2, Wc / 2], [0, 0, 1.0]]); cov = {}
    for k, m in meshes.items():
        r = Rr.OffscreenRenderer(Wc, Wc); r.scene.add_geometry('g', m, mat); vals = []
        for im in ref[::2]:                                              # every second face of every panorama
            keep = cv2.resize(cv2.imread(str(p.masks / (im.name + '.png')), cv2.IMREAD_GRAYSCALE), (Wc, Wc), interpolation=cv2.INTER_NEAREST) > 127
            if keep.mean() < 0.05: continue                              # all sky or people: nothing to cover
            E = np.eye(4); E[:3] = im.cam_from_world().matrix(); r.setup_camera(Kc, E, Wc, Wc)
            hole = np.asarray(r.render_to_depth_image()) >= 1.0
            vals.append(100 * (1 - hole[keep].mean()))
        cov[k] = dict(views=len(vals), mean=round(float(np.mean(vals)), 1), median=round(float(np.median(vals)), 1), worst=round(float(np.min(vals)), 1))
        print(f'mesh "{k}" seen from {len(vals)} real poses covers {cov[k]["mean"]}% of the pixels outside the masks '
              f'(median {cov[k]["median"]}, worst {cov[k]["worst"]})', flush=True)
        del r                                                            # two live Filament renderers crash: drop this one before the next
    json.dump(cov, open(p.product / 'mesh_coverage.json', 'w'), indent=1)

if p.outdoor:
    Wd, Hd = 960, 640; things = {}
    if 'with prior + planes' in meshes: things['mesh'] = meshes['with prior + planes']
    if (p.product / 'voxels_cubes.ply').exists():
        vm = o3d.io.read_triangle_mesh(str(p.product / 'voxels_cubes.ply')); vm.compute_vertex_normals(); things['voxels'] = vm      # no vertex is shared between squares: flat shading
    for k, m in things.items():
        if k == 'voxels':                                                # lit, so that the cubes read as cubes
            lit = Rr.MaterialRecord(); lit.shader = 'defaultLit'
        tiles = []
        for name, E, Kd in aerial_poses(Wd, Hd):
            if k == 'voxels':
                r = Rr.OffscreenRenderer(Wd, Hd); r.scene.set_background([0.12, 0.13, 0.15, 1.0]); r.scene.add_geometry('g', m, lit)
                r.scene.scene.set_sun_light(np.array([-0.4, -0.3, -0.85], np.float32), np.array([1, 1, 1], np.float32), 60000); r.scene.scene.enable_sun_light(True)
                r.setup_camera(Kd, E, Wd, Hd); img = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR); del r
            else:
                img, _ = shot(m, E, Wd, Hd, Kd, bg=(0.12, 0.13, 0.15))
            tiles.append(label(img, f'{k} {name}'))
        cv2.imwrite(str(out / f'{k}_aerial.jpg'), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print('wrote', out / f'{k}_aerial.jpg', flush=True)

# the building alone (crop.py), framed on its box
box = p.cfg.get('crop') or {}
bc = bs = None
if p.outdoor and box.get('x') and box.get('y'):
    bc = np.array([np.mean(box['x']), np.mean(box['y']), 0.0]); bs = 0.85 * max(np.ptp(box['x']), np.ptp(box['y']))
    for k, fn in (('mesh', 'mesh_building.ply'), ('voxels', 'voxels_cubes_building.ply')):
        if not (p.product / fn).exists(): continue
        m = o3d.io.read_triangle_mesh(str(p.product / fn)); m.compute_vertex_normals(); tiles = []
        for name, E, Kd in aerial_poses(960, 640, centre=bc, span=bs):
            if k == 'voxels':
                lit = Rr.MaterialRecord(); lit.shader = 'defaultLit'
                r = Rr.OffscreenRenderer(960, 640); r.scene.set_background([0.12, 0.13, 0.15, 1.0]); r.scene.add_geometry('g', m, lit)
                r.scene.scene.set_sun_light(np.array([-0.4, -0.3, -0.85], np.float32), np.array([1, 1, 1], np.float32), 60000); r.scene.scene.enable_sun_light(True)
                r.setup_camera(Kd, E, 960, 640); img = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR); del r
            else:
                img, _ = shot(m, E, 960, 640, Kd, bg=(0.12, 0.13, 0.15))
            tiles.append(label(img, f'building, {k}, {name}'))
        cv2.imwrite(str(out / f'building_{k}.jpg'), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print('wrote', out / f'building_{k}.jpg', flush=True)

# ------------------------------------------------------------------ voxels in voxview (a hidden window: needs a display)
if (p.product / 'voxels.npz').exists() and os.environ.get('DISPLAY'):
    vv = [sys.executable, str(Path(__file__).parent / 'voxview.py'), str(p.yaml), '--size', '1500x1000', '--zoom', '0.45']
    for name, extra in (('voxels_whole.png', []), ('voxels_source.png', ['--color', 'source', '--no-path']), ('voxels_top.png', ['--view=0,-0.05,1', '--no-path'])):
        r = subprocess.run(vv + extra + ['--screenshot', str(out / name)], capture_output=True, text=True)
        print('wrote' if r.returncode == 0 else 'voxview failed for', out / name, '' if r.returncode == 0 else r.stderr[-300:], flush=True)

# ------------------------------------------------------------------ splats
dev = 'cuda'


def splat(ck, E, K, w, hh):
    colors = torch.cat([ck['sh0'], ck['shN']], 1); sh = int(round(colors.shape[1] ** 0.5)) - 1
    with torch.no_grad():
        o, a_, _ = rasterization(ck['means'], ck['quats'], torch.exp(ck['scales']), torch.sigmoid(ck['opacities']), colors,
                                 torch.tensor(E, dtype=torch.float32, device=dev)[None], torch.tensor(K, dtype=torch.float32, device=dev)[None],
                                 w, hh, sh_degree=sh, packed=False, rasterize_mode='antialiased', near_plane=0.01)
        o, a_ = o[0], a_[0]
        if 'env' in ck:                                                  # the learned sky behind the Gaussians
            u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(hh) + 0.5)
            rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
            dw = torch.tensor(rays @ E[:3, :3], dtype=torch.float32, device=dev); dw = dw / dw.norm(dim=-1, keepdim=True)
            g = torch.stack([torch.atan2(dw[..., 1], dw[..., 0]) / math.pi, -torch.asin(dw[..., 2].clamp(-1, 1)) / (math.pi / 2)], -1)
            sky = torch.nn.functional.grid_sample(torch.sigmoid(ck['env']), g[None], mode='bilinear', padding_mode='border', align_corners=False)[0].permute(1, 2, 0)
            o = o + (1 - a_) * sky
    return cv2.cvtColor(o.clamp(0, 1).mul(255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)


W, Hh = 720, 540; f = W / 2 / np.tan(np.radians(45)); K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])
eye0 = spot if p.outdoor else mid
for tag in sorted(d.name for d in p.run.glob('splat*') if (d / 'ckpt.pt').exists()):
    ck = {k: v.to(dev) for k, v in torch.load(p.run / tag / 'ckpt.pt', map_location='cpu').items()}
    rows = []
    for mult in (1, 3, 6):
        tiles = []
        for name, d in (('+x', (1, 0)), ('+y', (0, 1)), ('-x', (-1, 0)), ('-y', (0, -1))):
            z_eye = eye0[2] + (mult - 1) * h if p.outdoor else mult * h      # outdoors the ground is not at z = 0 everywhere: from the shot's own height
            eye = np.array([eye0[0], eye0[1], z_eye]); E = look_at(eye, eye + np.array([d[0], d[1], -0.08 * (mult - 1)]))
            tiles.append(label(splat(ck, E, K, W, Hh), f'{mult}x lens height, looking {name}'))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / f'{tag}_eye.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('wrote', out / f'{tag}_eye.jpg')
    if p.outdoor:
        tiles = [label(splat(ck, E, Kd, 960, 640), f'{tag} {name}') for name, E, Kd in aerial_poses(960, 640)]
        cv2.imwrite(str(out / f'{tag}_aerial.jpg'), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print('wrote', out / f'{tag}_aerial.jpg')
        if bc is not None:                                               # the building alone: Gaussians inside the box, the sky behind
            lo_ = torch.tensor([box['x'][0], box['y'][0], box.get('z', [-1e9])[0]], device=dev); hi_ = torch.tensor([box['x'][1], box['y'][1], box.get('z', [0, 1e9])[1]], device=dev)
            inb = ((ck['means'] >= lo_) & (ck['means'] <= hi_)).all(1); cb = {k: (v[inb] if v.shape[:1] == inb.shape else v) for k, v in ck.items()}
            tiles = [label(splat(cb, E, Kd, 960, 640), f'building, {tag}, {name}') for name, E, Kd in aerial_poses(960, 640, centre=bc, span=bs)]
            cv2.imwrite(str(out / f'building_{tag}.jpg'), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('wrote', out / f'building_{tag}.jpg')
