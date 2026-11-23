#!/usr/bin/env python3
"""Steps 2 and 3. Masks, then every panorama cut into perspective faces of one rigid rig.

    python3 -I scripts/split_faces.py data/pool/rig.yaml

What is masked (black in masks/: never matched, fused or scored):
  platform  mask.rover: auto (pass-0) finds the rover bolted to the camera: the only thing that stays in place while
            the floor moves. Each shot is compared with the median of all shots by local correlation (15 x 15 window;
            blind to exposure changes); pixels that correlate in most shots, in the lower part of the sphere, connected
            to the bottom edge, are the rover, grown by mask.grow_deg. A PNG path takes a fixed mask; none = nothing.
  semantic  mask.semantic: ADE20K classes of a segmentation network (mask.seg_model) run on every face.
            person, animal: whoever carries the camera or walks next to it stays in the same place in the picture,
            like a point at infinity; their features would be matched from shot to shot. Grown by mask.grow_deg.
            sky: nothing to triangulate. Grown by 0.5 deg in masks/, so that no keypoint or fused point sits on its edge.
  seam      mask.seam_deg > 0: a strip around the stitching seam (longitude +-90 deg).
Faces: faces.views = [yaw, pitch] of each pinhole view, all with one centre, faces.fov and faces.size.
Writes faces/vNN/<shot>.jpg, masks/vNN/<shot>.jpg.png (255 = use), product/mask_equirect.png (the platform),
product/rig.json (COLMAP rig config: rotations fixed, translations zero), product/faces.json and product/faces.jpg.
With mask.semantic also labels/vNN/<shot>.png (0 scene, 1 sky, 2 people, 3 platform; the splat learns the sky from 1
and ignores 2 and 3) and masks_equirect/<shot>.png (the masks back on the whole panorama, for OpenMVG and inspect).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from pano import face_maps, face_rotation, lonlat_to_rays, sample
from project import load

p = load(sys.argv[1])
for d in (p.faces, p.masks, p.product): d.mkdir(parents=True, exist_ok=True)
first = cv2.imread(str(p.photos[0])); H, W = first.shape[:2]
print(f'{len(p.photos)} panoramas {W}x{H} -> {len(p.views)} faces each, {p.size} px, fov {p.fov:.0f} deg, focal {p.focal:.1f} px '
      f'(the panorama has {W / 360:.1f} px/deg, a face has {p.focal * np.pi / 180:.1f} px/deg in its centre)', flush=True)

# ---- platform mask on the panorama (1 = keep)
keep = np.ones((H, W), np.uint8)
if p.mask_rover == 'auto':
    w, h = 2688, 1344
    G = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(f)), (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                  for f in p.photos]).astype(np.float32)
    med = np.median(G, 0); k = (15, 15)
    mb = cv2.blur(med, k); vb = cv2.blur(med * med, k) - mb * mb
    scores = []
    for g in G[::max(1, len(G) // 16)]:
        ma = cv2.blur(g, k); va = cv2.blur(g * g, k) - ma * ma
        scores.append((cv2.blur(g * med, k) - ma * mb) / np.sqrt(np.maximum(va, 1e-3) * np.maximum(vb, 1e-3)))
    static = (np.median(scores, 0) > 0.6).astype(np.uint8)
    static[:int(0.60 * h)] = 0                                           # only below latitude -18 deg
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    _, lab = cv2.connectedComponents(np.hstack([static, static[:, :50]]))  # the extra columns join left and right
    ids = np.unique(lab[-1]); rover = np.isin(lab, ids[ids > 0])[:, :w]
    ff = (~rover).astype(np.uint8); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    rover |= ff == 1                                                     # holes inside the rover
    top = np.array([np.argmax(rover[:, x]) if rover[:, x].any() else h for x in range(w)])
    lat = 90 - (top + 0.5) / h * 180
    g = max(1, int(round(p.mask_grow_deg * w / 360)))
    rover = cv2.dilate(rover.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    keep = 1 - cv2.resize(rover, (W, H), interpolation=cv2.INTER_NEAREST)
    print(f'rover mask: {100 * (1 - keep.mean()):.1f}% of the panorama, its top edge at latitude {lat.max():.1f} deg '
          f'(highest) / {np.median(lat):.1f} (median) / {lat.min():.1f} (lowest), grown by {p.mask_grow_deg} deg')
elif p.mask_rover and p.mask_rover != 'none':
    keep = (cv2.imread(str(p.session / p.mask_rover), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
if p.mask_seam_deg > 0:
    lon = (np.arange(W) + 0.5) / W * 360 - 180
    keep[:, np.abs(np.abs(lon) - 90) < p.mask_seam_deg] = 0
    print(f'seam mask: +-{p.mask_seam_deg} deg around longitude +-90')
cv2.imwrite(str(p.product / 'mask_equirect.png'), keep * 255)

# ---- segmentation network
seg = None
if p.mask_semantic:
    import torch
    sys.modules.setdefault('OpenSSL', None)     # a broken pyOpenSSL in the environment must fail as "not installed", not crash transformers
    from transformers import AutoModelForSemanticSegmentation
    net = AutoModelForSemanticSegmentation.from_pretrained(p.seg_model).cuda().eval().half()
    names = {v.strip().lower(): int(k) for k, v in net.config.id2label.items()}
    unknown = [c for c in p.mask_semantic if c not in names]
    if unknown: raise SystemExit(f'mask.semantic: {unknown} are not classes of {p.seg_model}')
    SKY = names['sky'] if 'sky' in p.mask_semantic else None
    MOV = [names[c] for c in p.mask_semantic if c != 'sky']
    OTH = [i for i in range(len(names)) if i != SKY and i not in MOV]
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225], device='cuda')[:, None, None]
    neg = torch.full((1,), -1e4, device='cuda')

    def seg(faces_bgr):
        """Per face: sky and people as boolean masks at the face size. The decision (class score minus the best other
        class) is upsampled, not the class index, so the edges come out smooth at 1600 px."""
        x = torch.stack([torch.from_numpy(cv2.resize(f, (p.seg_size, p.seg_size), interpolation=cv2.INTER_AREA)[..., ::-1].copy())
                         for f in faces_bgr]).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            lg = net(pixel_values=((x - mean) / std).half()).logits.float()
            other = lg[:, OTH].max(1).values
            sky = lg[:, SKY] if SKY is not None else neg.expand_as(other)
            mov = lg[:, MOV].max(1).values if MOV else neg.expand_as(other)
            m = torch.stack([sky - torch.maximum(other, mov), mov - torch.maximum(other, sky)], 1)
            m = torch.nn.functional.interpolate(m, (p.size, p.size), mode='bilinear', align_corners=False) > 0
        return m[:, 0].cpu().numpy(), m[:, 1].cpu().numpy()
    print(f'segmentation: {p.seg_model} at {p.seg_size} px, masking {p.mask_semantic}', flush=True)

# ---- faces
rots = [face_rotation(y, pt) for y, pt in p.views]
maps = [face_maps(R, p.size, p.focal, W, H) for R in rots]
for d in (p.faces, p.masks) + ((p.labels,) if seg else ()):
    for v in range(len(p.views)): (d / f'v{v:02d}').mkdir(parents=True, exist_ok=True)
fmask = [sample(keep * 255, mx, my, cv2.INTER_NEAREST) > 127 for mx, my in maps]
share = [float(m.mean()) for m in fmask]
disk = lambda deg: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, round(p.focal * np.radians(deg))) + 1,) * 2)
grow, sky_edge = disk(p.mask_grow_deg), disk(0.5)
if seg:
    # where every panorama pixel is seen best (nearest a face centre), to put the face masks back on the panorama
    p.masks_equirect.mkdir(exist_ok=True)
    ray = lonlat_to_rays(*np.meshgrid((np.arange(W) + 0.5) / W * 360 - 180, 90 - (np.arange(H) + 0.5) / H * 180)).astype(np.float32)
    best = np.full((H, W), -1, np.int8); bz = np.zeros((H, W), np.float32); bu = np.zeros((H, W), np.int16); bv = np.zeros((H, W), np.int16)
    for v, R in enumerate(rots):
        c = ray @ R.T.astype(np.float32)                                 # rows: R @ ray, the ray in the face's frame
        z = c[..., 2]; u = np.floor(c[..., 0] / np.maximum(z, 1e-6) * p.focal + p.size / 2); w_ = np.floor(c[..., 1] / np.maximum(z, 1e-6) * p.focal + p.size / 2)
        ok = (z > bz) & (u >= 0) & (u < p.size) & (w_ >= 0) & (w_ < p.size)
        best[ok] = v; bz[ok] = z[ok]; bu[ok] = u[ok]; bv[ok] = w_[ok]
        del c, z, u, w_
    covered = best >= 0; del ray
    print(f'the faces cover {100 * covered.mean():.1f}% of the panorama (the rest, round the nadir, is masked on the panorama too)', flush=True)
stats = {k: np.zeros(len(p.views)) for k in ('sky', 'people', 'kept')}; per_shot = {}
for i, f in enumerate(p.photos):
    pano = cv2.imread(str(f))
    faces = [sample(pano, mx, my) for mx, my in maps]
    if seg: sky, mov = seg(faces)
    km = []
    for v, face in enumerate(faces):
        cv2.imwrite(str(p.faces / f'v{v:02d}' / f'{f.stem}.jpg'), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        k = fmask[v].copy()
        if seg:
            m = cv2.dilate(mov[v].astype(np.uint8), grow) > 0
            label = np.where(~fmask[v], 3, np.where(m, 2, np.where(sky[v], 1, 0))).astype(np.uint8)
            cv2.imwrite(str(p.labels / f'v{v:02d}' / f'{f.stem}.png'), label)
            k &= ~m & ~(cv2.dilate(sky[v].astype(np.uint8), sky_edge) > 0)
            stats['sky'][v] += sky[v].mean(); stats['people'][v] += m.mean()
        stats['kept'][v] += k.mean()
        cv2.imwrite(str(p.masks / f'v{v:02d}' / f'{f.stem}.jpg.png'), k.astype(np.uint8) * 255); km.append(k)
    if seg:
        eq = np.zeros((H, W), np.uint8); kk = np.stack(km)
        eq[covered] = kk[best[covered], bv[covered], bu[covered]]
        eq &= keep
        cv2.imwrite(str(p.masks_equirect / f'{f.stem}.png'), eq * 255)
        per_shot[f.stem] = dict(kept_pct=round(100 * float(eq.mean()), 1), people_pct_of_faces=round(100 * float(np.mean([x.mean() for x in mov])), 2))
    if i == len(p.photos) // 2:                                          # a sheet of one shot's faces with the masks
        tiles = []
        for v, face in enumerate(faces):
            t = face.copy()
            if seg:
                t[sky[v]] = (0.35 * t[sky[v]] + (0, 160, 160)).astype(np.uint8)
                mm = cv2.dilate(mov[v].astype(np.uint8), grow) > 0; t[mm] = (0.4 * t[mm] + (150, 0, 150)).astype(np.uint8)
            t[~fmask[v]] = (0.4 * t[~fmask[v]] + (150, 0, 150)).astype(np.uint8)
            t = cv2.resize(t, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f'v{v:02d} yaw {p.views[v][0]:.0f} pitch {p.views[v][1]:.0f}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(p.product / 'faces.jpg'), np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    if (i + 1) % 24 == 0: print(f'  {i + 1}/{len(p.photos)} panoramas', flush=True)
n = len(p.photos)
print('usable share of each face: ' + ' '.join(f'v{v:02d} {100 * stats["kept"][v] / n:.0f}%' for v in range(len(p.views))))
if seg:
    print('sky:    ' + ' '.join(f'v{v:02d} {100 * stats["sky"][v] / n:.0f}%' for v in range(len(p.views))))
    print('people: ' + ' '.join(f'v{v:02d} {100 * stats["people"][v] / n:.1f}%' for v in range(len(p.views))))

# ---- rig config for COLMAP: one camera per face folder, the first face is the reference sensor
cams = []
for v, R in enumerate(rots):
    c = {'image_prefix': f'v{v:02d}/'}
    if v == 0: c['ref_sensor'] = True
    else:
        R_rel = R @ rots[0].T                                            # face v from face 0
        q = pycolmap.Rotation3d(R_rel).quat                              # x, y, z, w
        c['cam_from_rig_rotation'] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        c['cam_from_rig_translation'] = [0.0, 0.0, 0.0]
    cams.append(c)
json.dump([{'cameras': cams}], open(p.product / 'rig.json', 'w'), indent=1)
J = {'views': p.views, 'face0_from_pano': rots[0].tolist(), 'size': p.size, 'focal': p.focal,
     'usable_share': [float(x / n) for x in stats['kept']], 'platform_share': share, 'pano_size': [W, H]}
if seg:
    J |= {'semantic': p.mask_semantic, 'seg_model': p.seg_model, 'sky_share': [float(x / n) for x in stats['sky']],
          'people_share': [float(x / n) for x in stats['people']], 'faces_cover_pct_of_panorama': round(100 * float(covered.mean()), 2),
          'per_shot': per_shot}
json.dump(J, open(p.product / 'faces.json', 'w'), indent=1)
print(f'wrote {n * len(p.views)} faces to {p.faces}, rig config {p.product / "rig.json"}')
