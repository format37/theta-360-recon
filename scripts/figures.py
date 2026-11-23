#!/usr/bin/env python3
"""Pictures for a write-up of an outdoor run: no captions burnt in, no people in any of them.

    python3 -I scripts/figures.py data/pool/rig.yaml [--tag splat] [--pass-tag splat_pass] [--shot R0013101] [--orbit]

  product/post/hero.jpg        the splat from 30 deg above, a view nobody took a photo from
  product/post/walk.jpg        the splat from straight above with the walk on it, one colour per time round the building
  product/post/faces.jpg       the 8 faces of one panorama with what the segmentation network takes out (sky, people)
  product/post/products.jpg    mesh | voxel cubes | splat from the same raised view
  product/post/depth.jpg       one face | the depth PatchMatch kept | the depth after the prior filled it
  product/post/heldout.jpg     faces of the held-out pass: photo | splat that never trained on that pass (needs --pass-tag run)
  product/post/orbit.mp4       --orbit: the camera once round the site (needs ffmpeg)
"""
import argparse
import json
import math
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

ap = argparse.ArgumentParser(); ap.add_argument('yaml'); ap.add_argument('--tag', default='splat'); ap.add_argument('--pass-tag', default='splat_pass')
ap.add_argument('--shot', help='the panorama of faces.jpg and depth.jpg (default: the one whose level faces have the least people)')
ap.add_argument('--depth-face', help='the face of depth.jpg, e.g. v03/R0013101.jpg'); ap.add_argument('--heldout', nargs='*', help='faces of heldout.jpg')
ap.add_argument('--azimuth', type=float, default=225, help='where the raised views stand, degrees from +x (225 = south-west)')
ap.add_argument('--orbit', action='store_true')
a = ap.parse_args()
p = load(a.yaml); out = p.product / 'post'; out.mkdir(parents=True, exist_ok=True)
Rr = o3d.visualization.rendering
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
ref = sorted((im for im in rec.images.values() if im.has_pose and im.name.startswith('v00/')), key=lambda im: im.name)
stems = [Path(im.name).stem for im in ref]; C = np.array([im.projection_center() for im in ref]); h = float(np.median(C[:, 2]))
centre = np.r_[(C[:, :2].min(0) + C[:, :2].max(0)) / 2, 0.0]; span = float(np.ptp(C[:, :2], axis=0).max()) + 2 * h
box = p.cfg.get('crop') or {}
bc = np.array([np.mean(box['x']), np.mean(box['y']), 0.0]) if box.get('x') else centre
save = lambda name, img, q=90: (cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, q]), print('wrote', out / name, img.shape[1], 'x', img.shape[0], flush=True))


def look_at(eye, target, up=(0, 0, 1.0)):
    f = np.asarray(target, float) - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); d = np.cross(f, r)
    E = np.eye(4); E[:3, :3] = np.stack([r, d, f]); E[:3, 3] = -E[:3, :3] @ eye
    return E


def raised(w, hh, az, fov=50, elev=30, fill=0.6):
    fd = w / 2 / np.tan(np.radians(fov / 2)); K = np.array([[fd, 0, w / 2], [0, fd, hh / 2], [0, 0, 1.0]]); dist = fill * span / np.tan(np.radians(fov / 2))
    d = np.array([np.cos(np.radians(az)) * np.cos(np.radians(elev)), np.sin(np.radians(az)) * np.cos(np.radians(elev)), np.sin(np.radians(elev))])
    return look_at(centre + dist * d, centre), K


# ------------------------------------------------------------------ mesh and voxel cubes (Open3D before the first CUDA call)
W3, H3 = 800, 600; E3, K3 = raised(W3, H3, a.azimuth); BG = (0.04, 0.04, 0.05); tiles = {}
for k, fn in (('mesh', 'mesh_view.ply'), ('voxels', 'voxels_cubes.ply')):
    if not (p.product / fn).exists(): continue
    m = o3d.io.read_triangle_mesh(str(p.product / fn)); mat = Rr.MaterialRecord(); mat.shader = 'defaultLit' if k == 'voxels' else 'defaultUnlit'
    if k == 'voxels': m.compute_vertex_normals()
    r = Rr.OffscreenRenderer(W3, H3); r.scene.set_background([*BG, 1.0]); r.scene.add_geometry('g', m, mat); r.scene.view.set_post_processing(False)
    if k == 'voxels':
        r.scene.scene.set_sun_light(np.array([-0.4, -0.3, -0.85], np.float32), np.array([1, 1, 1], np.float32), 60000); r.scene.scene.enable_sun_light(True)
    r.setup_camera(K3, E3, W3, H3); tiles[k] = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR); del r, m

# ------------------------------------------------------------------ the splat
dev = 'cuda'
ck = {k: v.to(dev) for k, v in torch.load(p.run / a.tag / 'ckpt.pt', map_location='cpu').items()}


def splat(E, K, w, hh, sky=True, bg=BG):
    colors = torch.cat([ck['sh0'], ck['shN']], 1); sh = int(round(colors.shape[1] ** 0.5)) - 1
    with torch.no_grad():
        o, al, _ = rasterization(ck['means'], ck['quats'], torch.exp(ck['scales']), torch.sigmoid(ck['opacities']), colors,
                                 torch.tensor(E, dtype=torch.float32, device=dev)[None], torch.tensor(K, dtype=torch.float32, device=dev)[None],
                                 w, hh, sh_degree=sh, packed=False, rasterize_mode='antialiased', near_plane=0.01)
        o, al = o[0], al[0]
        if sky and 'env' in ck:                                          # the learned sky behind the Gaussians
            u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(hh) + 0.5)
            rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
            dw = torch.tensor(rays @ E[:3, :3], dtype=torch.float32, device=dev); dw = dw / dw.norm(dim=-1, keepdim=True)
            g = torch.stack([torch.atan2(dw[..., 1], dw[..., 0]) / math.pi, -torch.asin(dw[..., 2].clamp(-1, 1)) / (math.pi / 2)], -1)
            o = o + (1 - al) * torch.nn.functional.grid_sample(torch.sigmoid(ck['env']), g[None], mode='bilinear', padding_mode='border', align_corners=False)[0].permute(1, 2, 0)
        else:
            o = o + (1 - al) * torch.tensor(bg[::-1], device=dev)
    return cv2.cvtColor(o.clamp(0, 1).mul(255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)


E, K = raised(1600, 840, a.azimuth, fill=0.52); save('hero.jpg', splat(E, K, 1600, 840))
tiles['splat'] = splat(E3, K3, W3, H3)
if len(tiles) == 3: save('products.jpg', np.hstack([tiles['mesh'], np.full((H3, 6, 3), 255, np.uint8), tiles['voxels'], np.full((H3, 6, 3), 255, np.uint8), tiles['splat']]))

# the walk from straight above: once round the building per colour (the angle round the middle of the crop box)
Wt = 1400; fov = 30; ft = Wt / 2 / np.tan(np.radians(fov / 2)); Kt = np.array([[ft, 0, Wt / 2], [0, ft, Wt / 2], [0, 0, 1.0]])
Et = look_at(centre + np.array([0, 0, 0.56 * span / np.tan(np.radians(fov / 2))]), centre, up=(0, 1.0, 0)); top = splat(Et, Kt, Wt, Wt, sky=False)
ang = np.unwrap(np.arctan2(C[:, 1] - bc[1], C[:, 0] - bc[0])); turn = np.abs(ang - ang[0]) / (2 * np.pi)
px = (np.c_[C, np.ones(len(C))] @ Et[:3].T) @ Kt.T; px = px[:, :2] / px[:, 2:]
top = (top * 0.75).astype(np.uint8); cols = [(60, 200, 255), (255, 170, 60), (120, 255, 120), (255, 120, 255)]       # BGR: amber, blue, green, pink
passes = {}
for n in range(int(turn.max()) + 1):
    idx = np.flatnonzero(turn.astype(int) == n); idx = np.r_[idx, min(idx[-1] + 1, len(C) - 1)]; passes[n + 1] = [stems[idx[0]], stems[idx[-1]], len(idx) - 1]
    cv2.polylines(top, [np.round(px[idx] * 16).astype(np.int32)], False, (0, 0, 0), 9, cv2.LINE_AA, shift=4)
    cv2.polylines(top, [np.round(px[idx] * 16).astype(np.int32)], False, cols[n % 4], 4, cv2.LINE_AA, shift=4)
for n in range(int(turn.max()) + 1):
    idx = np.flatnonzero(turn.astype(int) == n)
    for q in px[idx]: cv2.circle(top, tuple(np.round(q * 16).astype(int)), 6 * 16, cols[n % 4], -1, cv2.LINE_AA, shift=4)
    q = px[idx[len(idx) // 5]] + np.array([14, -14])
    cv2.putText(top, str(n + 1), tuple(np.round(q).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 9, cv2.LINE_AA)
    cv2.putText(top, str(n + 1), tuple(np.round(q).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, cols[n % 4], 3, cv2.LINE_AA)
cv2.circle(top, tuple(np.round(px[0]).astype(int)), 15, (255, 255, 255), 3, cv2.LINE_AA)                             # the first shot
save('walk.jpg', top)
json.dump(dict(passes={k: dict(first=v[0], last=v[1], panoramas=v[2]) for k, v in passes.items()}, turns=round(float(turn.max()), 2)), open(out / 'walk.json', 'w'), indent=1)
print('times round the building:', round(float(turn.max()), 2), passes, flush=True)

# ------------------------------------------------------------------ the faces of one panorama, and the depth of one face
nv = len(p.views); lower = [v for v, (_, pt) in enumerate(p.views) if pt < 0]
lab = lambda name, s: cv2.resize(cv2.imread(str(p.labels / (str(Path(name).with_suffix('')) + '.png')), cv2.IMREAD_GRAYSCALE), (s, s), interpolation=cv2.INTER_NEAREST)
if a.shot is None:                                                       # few people, and a facade in view (many 3D points)
    ppl = lambda s: np.mean([(lab(f'v{v:02d}/{s}.jpg', 200) == 2).mean() for v in lower])
    pts = {s: sum(rec.image(rec.find_image_with_name(f'v{v:02d}/{s}.jpg').image_id).num_points3D for v in lower) for s in stems}
    a.shot = max((s for s in stems if ppl(s) < 0.02), key=lambda s: pts[s], default=stems[len(stems) // 2])
T = 400; grid = []
for v in range(nv):
    name = f'v{v:02d}/{a.shot}.jpg'; img = cv2.resize(cv2.imread(str(p.faces / name)), (T, T), interpolation=cv2.INTER_AREA); L = lab(name, T)
    img[L == 1] = (0.45 * img[L == 1] + 0.55 * np.array([60, 200, 255])).astype(np.uint8)                              # sky: amber wash
    img[L == 2] = (200, 60, 200)                                                                                        # people: solid, nothing shows through
    grid.append(img)
gap = np.full((T, 4, 3), 255, np.uint8); rows = [np.hstack(sum(([g, gap] for g in grid[i:i + nv // 2]), [])[:-1]) for i in (0, nv // 2)]
save('faces.jpg', np.vstack([rows[0], np.full((4, rows[0].shape[1], 3), 255, np.uint8), rows[1]]))
print('faces of', a.shot, flush=True)


def depth_colour(z, keep):
    inv = np.where(keep, 1 / np.maximum(z, 1e-6), 0); lo, hi = np.percentile(inv[keep], [2, 98]) if keep.any() else (0, 1)
    img = cv2.applyColorMap((255 * np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO); img[~keep] = 0
    return img


fin = p.dense / 'final'
if fin.is_dir():
    if a.depth_face is None:                                             # a level face with nobody in it and little sky: the median one by PatchMatch share
        share = {}
        for v in lower:
            for s in stems:
                n = f'v{v:02d}/{s}.jpg'
                if not (fin / (n + '.npy')).exists(): continue
                L0 = lab(n, 200)
                if (L0 == 2).any() or (L0 == 1).mean() > 0.35: continue
                share[n] = (cv2.imread(str(fin / (n + '.label.png')), cv2.IMREAD_GRAYSCALE) == 1).mean()
        a.depth_face = sorted(share, key=share.get)[len(share) // 2] if share else None
    if a.depth_face:
        z = np.load(fin / (a.depth_face + '.npy')).astype(np.float32); L = cv2.imread(str(fin / (a.depth_face + '.label.png')), cv2.IMREAD_GRAYSCALE); s = z.shape[0]
        scene = lab(a.depth_face, s) == 0; pm = (L == 1) & (z > 0); allz = (z > 0) & scene
        photo = cv2.resize(cv2.imread(str(p.faces / a.depth_face)), (s, s), interpolation=cv2.INTER_AREA); photo[lab(a.depth_face, s) == 2] = (200, 60, 200); gap = np.full((s, 6, 3), 255, np.uint8)
        save('depth.jpg', np.hstack([photo, gap, depth_colour(z, pm), gap, depth_colour(z, allz)]))
        print(f'depth of {a.depth_face}: PatchMatch on {100 * pm.sum() / scene.sum():.1f}% of the scene pixels, filled {100 * allz.sum() / scene.sum():.1f}%', flush=True)

# ------------------------------------------------------------------ the held-out pass: photo | render, as train_splat.py scored them
ev = p.run / a.pass_tag / 'eval'
if ev.is_dir() and (a.heldout or list(ev.glob('pass_*.jpg'))):
    files = {f.name.split('_psnr')[0][5:]: f for f in ev.glob('pass_*.jpg')}                                            # 'v03_R0013104' -> file
    picks = [n.replace('/', '_').replace('.jpg', '') for n in a.heldout] if a.heldout else sorted(files)[:3]
    rows = []
    for n in picks:
        both = cv2.imread(str(files[n])); s = both.shape[0]; s2 = 600
        rows.append(np.hstack([cv2.resize(both[:, :s], (s2, s2), interpolation=cv2.INTER_AREA), np.full((s2, 6, 3), 255, np.uint8), cv2.resize(both[:, s:], (s2, s2), interpolation=cv2.INTER_AREA)]))
        print('held out:', files[n].name, flush=True)
    save('heldout.jpg', np.vstack(sum(([r, np.full((6, r.shape[1], 3), 255, np.uint8)] for r in rows), [])[:-1]))

if a.orbit:                                                              # once round the site, 12 s
    w, hh, n = 1280, 720, 360; tmp = out / 'orbit'; tmp.mkdir(exist_ok=True)
    for i in range(n):
        E, K = raised(w, hh, a.azimuth + 360 * i / n, fill=0.5); cv2.imwrite(str(tmp / f'{i:04d}.jpg'), splat(E, K, w, hh), [cv2.IMWRITE_JPEG_QUALITY, 95])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', str(tmp / '%04d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-movflags', '+faststart', str(out / 'orbit.mp4')], check=True)
    for f in tmp.glob('*.jpg'): f.unlink()
    tmp.rmdir(); print('wrote', out / 'orbit.mp4', flush=True)
