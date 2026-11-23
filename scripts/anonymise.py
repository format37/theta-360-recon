#!/usr/bin/env python3
"""A copy of a session's photos fit to publish: people blacked out, vehicles blurred, no GPS in the files.

    python3 -I scripts/anonymise.py data/pool/rig.yaml share/pool [--blur car bus truck van minibike] [--static 0.2]

Needs the faces of the run (split_faces.py). For every panorama:
  people     found on the faces by two networks, the pipeline's segmentation network (person, animal) and Mask R-CNN (COCO
             person), grown by 1.5 deg, put back on the panorama and painted black. Whoever carries the camera stays at the
             same place in the picture: where a person was found in more than --static of the shots, every shot is painted.
             What no face covers below -40 deg (the hand, the pole, the head from above) is painted too.
  vehicles   where both networks see one (the --blur classes; COCO car, motorcycle, bus, truck), blurred with a 48 px
             Gaussian: number plates go, the scene keeps its cars for the reconstruction.
  second look  Mask R-CNN once more on the painted photo (twice at most): a person it still finds is painted, a car-sized
             vehicle blurred.
  metadata   kept: camera, firmware, time, exposure (EXIF) and the camera's own tilt (XMP GPano). Dropped: the GPS
             position, the maker note, the user comment, the embedded thumbnail (it would show the unpainted photo).
Writes OUT/photos/*.JPG (same quantisation tables as the originals), OUT/people/*.png (white = painted: give it to the
pipeline as mask.given, the networks cannot find people who are not there any more), the session's YAML files with
that key set, OUT/check/*.jpg (every shot's lower half, to look through before uploading) and OUT/anonymise.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, JpegImagePlugin

sys.path.insert(0, str(Path(__file__).parent))
from pano import face_maps, face_rotation, lonlat_to_rays, sample
from project import load

ap = argparse.ArgumentParser(); ap.add_argument('yaml'); ap.add_argument('out')
ap.add_argument('--blur', nargs='*', default=['car', 'bus', 'truck', 'van', 'minibike'], help='classes of the segmentation network to blur')
ap.add_argument('--people', nargs='*', default=['person', 'animal']); ap.add_argument('--grow', type=float, default=1.5, help='degrees')
ap.add_argument('--static', type=float, default=0.2, help='a pixel that is a person in this share of the shots is painted in all of them')
ap.add_argument('--nadir', type=float, default=-40, help='below this latitude, what no face covers is painted')
a = ap.parse_args()
p = load(a.yaml); out = Path(a.out); work = out / '.work'
for d in (out / 'photos', out / 'people', out / 'check', work): d.mkdir(parents=True, exist_ok=True)
H, W = cv2.imread(str(p.photos[0])).shape[:2]; nv = len(p.views)

# ------------------------------------------------------------------ where every panorama pixel is seen best (as split_faces.py)
rots = [face_rotation(y, pt) for y, pt in p.views]
lat = 90 - (np.arange(H) + 0.5) / H * 180
ray = lonlat_to_rays(*np.meshgrid((np.arange(W) + 0.5) / W * 360 - 180, lat)).astype(np.float32)
best = np.full((H, W), -1, np.int8); bz = np.zeros((H, W), np.float32); bu = np.zeros((H, W), np.int16); bv = np.zeros((H, W), np.int16)
for v, R in enumerate(rots):
    c = ray @ R.T.astype(np.float32); z = c[..., 2]
    u = np.floor(c[..., 0] / np.maximum(z, 1e-6) * p.focal + p.size / 2); w_ = np.floor(c[..., 1] / np.maximum(z, 1e-6) * p.focal + p.size / 2)
    ok = (z > bz) & (u >= 0) & (u < p.size) & (w_ >= 0) & (w_ < p.size)
    best[ok] = v; bz[ok] = z[ok]; bu[ok] = u[ok]; bv[ok] = w_[ok]
covered = best >= 0; del ray, c, z, u, w_
nadir = ~covered & (lat[:, None] < a.nadir)
to_pano = lambda faces: np.where(covered, np.stack(faces)[best, bv, bu], False)

# ------------------------------------------------------------------ the two networks, on the faces
todo = [f for f in p.photos if not (work / f'{f.stem}.png').exists()]
if todo:
    sys.modules.setdefault('OpenSSL', None)     # a broken pyOpenSSL in the environment must fail as "not installed", not crash transformers
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from transformers import AutoModelForSemanticSegmentation
    net = AutoModelForSemanticSegmentation.from_pretrained(p.seg_model).cuda().eval().half()
    names = {v.strip().lower(): int(k) for k, v in net.config.id2label.items()}
    PPL = [names[c] for c in a.people]; VEH = [names[c] for c in a.blur]; OTH = [i for i in range(len(names)) if i not in PPL + VEH]
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225], device='cuda')[:, None, None]
    det = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda().eval()
    COCO_PPL, COCO_VEH = {1}, {3, 4, 6, 8}                               # person; car, motorcycle, bus, truck
    disk = lambda deg: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, round(p.focal * np.radians(deg))) + 1,) * 2)
    maps = None
    for i, f in enumerate(todo):
        if all((p.faces / f'v{v:02d}' / f'{f.stem}.jpg').exists() for v in range(nv)):
            faces = [cv2.imread(str(p.faces / f'v{v:02d}' / f'{f.stem}.jpg')) for v in range(nv)]
        else:
            maps = maps or [face_maps(R, p.size, p.focal, W, H) for R in rots]; pano = cv2.imread(str(f)); faces = [sample(pano, mx, my) for mx, my in maps]
        rgb = torch.stack([torch.from_numpy(x[..., ::-1].copy()) for x in faces]).cuda().permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            x = torch.nn.functional.interpolate(rgb, (p.seg_size, p.seg_size), mode='area')
            lg = net(pixel_values=((x - mean) / std).half()).logits.float(); oth = lg[:, OTH].max(1).values
            pp = lg[:, PPL].max(1).values; vv = lg[:, VEH].max(1).values if VEH else torch.full_like(oth, -1e4)
            m = torch.stack([pp - torch.maximum(oth, vv), vv - torch.maximum(oth, pp)], 1)
            m = (torch.nn.functional.interpolate(m, (p.size, p.size), mode='bilinear', align_corners=False) > 0).cpu().numpy()
            ppl, veh = [m[v, 0] for v in range(nv)], [m[v, 1] for v in range(nv)]
            for v, r in enumerate(det(list(rgb))):                       # Mask R-CNN: anything it calls a person at 0.3, a vehicle at 0.5
                veh2 = np.zeros_like(veh[v])
                for lab, sc, mk in zip(r['labels'].tolist(), r['scores'].tolist(), r['masks']):
                    if lab in COCO_PPL and sc >= 0.3: ppl[v] = ppl[v] | (mk[0] > 0.5).cpu().numpy()
                    elif lab in COCO_VEH and sc >= 0.5: veh2 |= (mk[0] > 0.5).cpu().numpy()
                # a person is painted if either network says so; a vehicle is blurred only where both do (each alone takes a
                # dark facade for a bus now and then): the connected pieces of their union in which each has a tenth or more
                n, cc = cv2.connectedComponents((veh[v] | veh2).astype(np.uint8)); both = np.zeros(n, bool)
                if n > 1:
                    area = np.bincount(cc.ravel(), minlength=n); a1 = np.bincount(cc[veh[v]], minlength=n); a2 = np.bincount(cc[veh2], minlength=n)
                    both = (a1 >= 0.1 * area) & (a2 >= 0.1 * area); both[0] = False
                veh[v] = both[cc]
        ppl = [cv2.dilate(x.astype(np.uint8), disk(a.grow)) > 0 for x in ppl]; veh = [cv2.dilate(x.astype(np.uint8), disk(0.5)) > 0 for x in veh]
        P_, V_ = to_pano(ppl), to_pano(veh)                             # 255 person, 100 vehicle (a person in front of a car is a person)
        cv2.imwrite(str(work / f'{f.stem}.png'), np.where(P_, 255, np.where(V_, 100, 0)).astype(np.uint8))
        if (i + 1) % 24 == 0: print(f'  masks {i + 1}/{len(todo)}', flush=True)
    del net, det; torch.cuda.empty_cache()

# ------------------------------------------------------------------ who is in the same place in every shot
raw = {f.stem: cv2.imread(str(work / f'{f.stem}.png'), cv2.IMREAD_GRAYSCALE) for f in p.photos}
freq = np.mean([(m >= 200) for m in raw.values()], 0, dtype=np.float32)
static = (freq >= a.static) & (lat[:, None] < 0)
cv2.imwrite(str(out / 'check' / 'people_frequency.jpg'), cv2.resize((np.clip(freq, 0, 1) * 255).astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA))
print(f'in the same place in {100 * a.static:.0f}%+ of the shots: {100 * static.mean():.1f}% of the panorama; no face covers {100 * nadir.mean():.1f}% of it below {a.nadir} deg', flush=True)

# ------------------------------------------------------------------ a second look: the detector on the painted photo
# Whatever Mask R-CNN still calls a person (0.5) and is not mostly black already is painted too; a vehicle (0.7) that is not
# blurred yet and is car-sized (under 40% of the face high and 60% wide: a facade it takes for a bus is larger) is blurred.
_det = None
def second_look(pano, done):
    global _det
    if _det is None:
        sys.modules.setdefault('OpenSSL', None)
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
        _det = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda().eval()
        globals()['_maps'] = [face_maps(R, p.size, p.focal, W, H) for R in rots]
    faces = [sample(pano, mx, my) for mx, my in _maps]; fd = [sample(done.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0 for mx, my in _maps]
    g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, round(p.focal * np.radians(a.grow))) + 1,) * 2)
    P, V = [np.zeros((p.size, p.size), bool) for _ in range(nv)], [np.zeros((p.size, p.size), bool) for _ in range(nv)]
    with torch.no_grad():
        rs = _det([torch.from_numpy(x[..., ::-1].copy()).cuda().permute(2, 0, 1).float() / 255 for x in faces])
    for v, r in enumerate(rs):
        for lab, sc, b, mk in zip(r['labels'].tolist(), r['scores'].tolist(), r['boxes'].tolist(), r['masks']):
            m = (mk[0] > 0.5).cpu().numpy()
            if not m.any() or (faces[v][m].max(1) <= 12).mean() > 0.65 or fd[v][m].mean() > 0.65: continue      # already painted or blurred
            if lab == 1 and sc >= 0.5: P[v] |= cv2.dilate(m.astype(np.uint8), g) > 0
            elif lab in (3, 4, 6, 8) and sc >= 0.7 and b[3] - b[1] < 0.4 * p.size and b[2] - b[0] < 0.6 * p.size: V[v] |= m
    return to_pano(P), to_pano(V)


# ------------------------------------------------------------------ the photos
KEEP0 = (0x010F, 0x0110, 0x0131, 0x0112, 0x0132)                         # Make, Model, Software, Orientation, DateTime
KEEPX = (0x9000, 0x9003, 0x9004, 0x829A, 0x829D, 0x8822, 0x8827, 0x9202, 0x9203, 0x9204, 0x9207, 0x920A, 0xA001, 0xA402, 0xA403)
stats = {}; sheet = []
for i, f in enumerate(p.photos):
    src = Image.open(f); orig = cv2.imread(str(f)); m = raw[f.stem]
    people = (m >= 200) | static | nadir; veh = (m == 100) & ~people
    small = cv2.GaussianBlur(cv2.resize(orig, (W // 16, H // 16), interpolation=cv2.INTER_AREA), (0, 0), 3)             # 48 px at full size
    soft = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC); added = [0, 0]
    for _ in range(2):                                                   # paint, look again, paint what is left
        pano = orig.copy(); pano[veh] = soft[veh]; pano[people] = 0
        P2, V2 = second_look(pano, people | veh)
        if not P2.any() and not V2.any(): break
        added[0] += int(P2.sum()); added[1] += int(V2.sum()); people |= P2; veh = (veh | V2) & ~people
    pano = orig.copy(); pano[veh] = soft[veh]; pano[people] = 0
    ex = Image.Exif(); old = src.getexif()
    for k in KEEP0:
        if k in old: ex[k] = old[k]
    sub = old.get_ifd(0x8769); ex[0x8769] = {k: sub[k] for k in KEEPX if k in sub}
    tmp = out / 'photos' / f.name
    Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)).save(tmp, 'JPEG', qtables=src.quantization, subsampling=JpegImagePlugin.get_sampling(src), exif=ex.tobytes())
    blob = open(f, 'rb').read(); s0 = blob.find(b'\xff\xe1', 4)        # the XMP packet (GPano: projection, the camera's tilt), verbatim
    k0 = blob.find(b'http://ns.adobe.com/xap/1.0/\x00'); xmp = b''
    if k0 > 0:
        L = int.from_bytes(blob[k0 - 2:k0], 'big'); xmp = blob[k0 - 4:k0 - 2 + L]
        if b'GPS' in xmp or b'exif:' in xmp: raise SystemExit(f'{f.name}: the XMP packet holds more than GPano tags, look at it first')
    new = open(tmp, 'rb').read(); e0 = 4 + int.from_bytes(new[4:6], 'big') if new[2:4] == b'\xff\xe1' else 2             # after the EXIF segment
    open(tmp, 'wb').write(new[:e0] + xmp + new[e0:])
    cv2.imwrite(str(out / 'people' / f'{f.stem}.png'), people.astype(np.uint8) * 255)
    stats[f.stem] = dict(people_pct=round(100 * float(people.mean()), 2), vehicles_pct=round(100 * float(veh.mean()), 2),
                         second_look_people_px=added[0], second_look_vehicle_px=added[1])
    sheet.append(cv2.resize(pano[H // 2 - H // 12:H * 5 // 6], (W // 6, (H * 5 // 6 - H // 2 + H // 12) // 6), interpolation=cv2.INTER_AREA))     # latitudes +15 .. -60
    if len(sheet) == 12 or i == len(p.photos) - 1:
        for t, n in zip(sheet, [x.stem for x in p.photos[i + 1 - len(sheet):i + 1]]): cv2.putText(t, n, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / 'check' / f'{p.photos[i + 1 - len(sheet)].stem}.jpg'), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85]); sheet = []
    if (i + 1) % 24 == 0: print(f'  photos {i + 1}/{len(p.photos)}', flush=True)

# ------------------------------------------------------------------ the YAML files, with the given masks switched on
import re
for y in sorted(p.session.glob('*.yaml')):
    t = y.read_text()
    if 'given:' not in t:
        t = re.sub(r'(?m)^mask:\n', 'mask:\n  given: people          # per-photo masks drawn on the panorama (white = hide): what scripts/anonymise.py painted\n', t, count=1) \
            if re.search(r'(?m)^mask:\n', t) else t + '\nmask:\n  given: people\n'
    (out / y.name).write_text(t)
chk = Image.open(out / 'photos' / p.photos[0].name); e = chk.getexif()
json.dump(dict(photos=len(p.photos), painted=a.people, blurred=a.blur, grow_deg=a.grow, static_share=a.static, nadir_deg=a.nadir,
               people_pct_mean=round(float(np.mean([s['people_pct'] for s in stats.values()])), 2),
               vehicles_pct_mean=round(float(np.mean([s['vehicles_pct'] for s in stats.values()])), 2),
               exif_kept=sorted(hex(k) for k in list(e) + list(e.get_ifd(0x8769))), gps_in_copy=bool(e.get_ifd(0x8825)), per_shot=stats),
          open(out / 'anonymise.json', 'w'), indent=1)
print(f'{len(p.photos)} photos -> {out / "photos"}: people {np.mean([s["people_pct"] for s in stats.values()]):.1f}% of a panorama painted, '
      f'vehicles {np.mean([s["vehicles_pct"] for s in stats.values()]):.1f}% blurred; GPS in the copy: {bool(e.get_ifd(0x8825))}; look through {out / "check"}')
