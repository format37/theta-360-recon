#!/usr/bin/env python3
"""Step 1. Look at the photos before running anything: type, size, EXIF, and how much neighbours share.

    python3 -I scripts/inspect_panos.py data/pass-0/rig.yaml

- stitched equirectangular or raw dual fisheye: the XMP ProjectionType tag and the 2:1 aspect
- EXIF per shot: time step, exposure, ISO, the camera's own pitch / roll
- overlap: a 360 photo always sees the whole room, so "overlap" is measured as SIFT matches between
  shots i and i+k (ratio test + mutual check, rover and poles excluded), the median angle the matched
  directions moved by, and the share of 30 x 30 deg cells of the sphere that hold a match.
  If split_faces.py has made masks_equirect/ (mask.semantic), they are used too: people who walk with the
  camera stay in place in the picture and would count as overlap.
Writes product/inspect.json and product/contact.jpg.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

sys.path.insert(0, str(Path(__file__).parent))
from pano import lonlat_to_rays
from project import load

p = load(sys.argv[1])
p.product.mkdir(parents=True, exist_ok=True)
rows = []
for f in p.photos:
    im = Image.open(f)
    ex = {ExifTags.TAGS.get(k, k): v for k, v in (im._getexif() or {}).items()}
    raw = open(f, 'rb').read(200000)
    xmp = raw[raw.find(b'<x:xmpmeta'):raw.find(b'</x:xmpmeta>')].decode('utf8', 'replace')
    tag = lambda k: (re.search(k + r'>([^<]+)<', xmp) or [None, None])[1]
    rows.append(dict(name=f.name, w=im.size[0], h=im.size[1], model=str(ex.get('Model', '')).strip(),
                     firmware=str(ex.get('Software', '')).strip(), time=ex.get('DateTimeOriginal'),
                     exposure_s=float(ex.get('ExposureTime', 0)), iso=ex.get('ISOSpeedRatings'),
                     f_number=float(ex.get('FNumber', 0)), program=ex.get('ExposureProgram'),
                     projection=tag('GPano:ProjectionType'), pitch=float(tag('GPano:PosePitchDegrees') or 0),
                     roll=float(tag('GPano:PoseRollDegrees') or 0), has_gps='GPSInfo' in ex))
t = [datetime.strptime(r['time'], '%Y:%m:%d %H:%M:%S') for r in rows]
dt = np.diff([x.timestamp() for x in t])
exp = np.array([r['exposure_s'] * r['iso'] for r in rows])
R = dict(shots=len(rows), size=sorted({(r['w'], r['h']) for r in rows}), camera=sorted({r['model'] for r in rows}),
         firmware=sorted({r['firmware'] for r in rows}), projection=sorted({str(r['projection']) for r in rows}),
         stitched_equirectangular=all(r['projection'] == 'equirectangular' and r['w'] == 2 * r['h'] for r in rows),
         interval_s=[float(dt.min()), float(np.median(dt)), float(dt.max())], duration_s=float(dt.sum()),
         exposure_program=sorted({r['program'] for r in rows}),
         shutter_s=[min(r['exposure_s'] for r in rows), max(r['exposure_s'] for r in rows)],
         iso=sorted({r['iso'] for r in rows}), exposure_range_stops=round(float(np.log2(exp.max() / exp.min())), 2),
         camera_pitch_deg=[min(r['pitch'] for r in rows), max(r['pitch'] for r in rows)],
         camera_roll_deg=[min(r['roll'] for r in rows), max(r['roll'] for r in rows)],
         gps_in_exif=sum(r['has_gps'] for r in rows))
print(f"{R['shots']} shots {R['size']} {R['camera']} {R['firmware']}; projection tag {R['projection']} -> "
      f"{'stitched equirectangular' if R['stitched_equirectangular'] else 'NOT a stitched 2:1 panorama'}")
print(f"interval {R['interval_s']} s (min, median, max), {R['duration_s']:.0f} s in all; exposure program {R['exposure_program']} "
      f"(2 = auto), shutter {R['shutter_s'][0]:.4f}-{R['shutter_s'][1]:.4f} s, ISO {R['iso']}: {R['exposure_range_stops']} stops of drift")
print(f"camera's own pitch {R['camera_pitch_deg']} roll {R['camera_roll_deg']} deg; GPS position in {R['gps_in_exif']} files"
      + (' (strip it before sharing a photo)' if R['gps_in_exif'] else ''))

# ---- overlap between shots
W, H = 2688, 1344
band = slice(int(H * (90 - 60) / 180), int(H * (90 + 32) / 180))        # latitudes +60 .. -32 deg: no poles, no rover
sift = cv2.SIFT_create(6000)
kps, des, thumbs = [], [], []
for i, f in enumerate(p.photos):
    img = cv2.imread(str(f))
    g = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m = np.zeros((H, W), np.uint8); m[band] = 255
    if (p.masks_equirect / f'{f.stem}.png').exists():
        m &= cv2.resize(cv2.imread(str(p.masks_equirect / f'{f.stem}.png'), cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST)
    k, d = sift.detectAndCompute(g, m)
    xy = np.array([x.pt for x in k])
    kps.append(lonlat_to_rays((xy[:, 0] / W - 0.5) * 360, (0.5 - xy[:, 1] / H) * 180)); des.append(d)
    th = cv2.resize(img, (384, 192), interpolation=cv2.INTER_AREA)
    cv2.putText(th, f'{i} {f.stem[-4:]}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); thumbs.append(th)
while len(thumbs) % 7: thumbs.append(np.zeros_like(thumbs[0]))
cv2.imwrite(str(p.product / 'contact.jpg'), np.vstack([np.hstack(thumbs[r:r + 7]) for r in range(0, len(thumbs), 7)]),
            [cv2.IMWRITE_JPEG_QUALITY, 80])
bf = cv2.BFMatcher(cv2.NORM_L2)


def match(i, j):
    a = bf.knnMatch(des[i], des[j], k=2); b = bf.knnMatch(des[j], des[i], k=2)
    fwd = {m.queryIdx: m.trainIdx for m, n in a if m.distance < 0.75 * n.distance}
    back = {m.queryIdx: m.trainIdx for m, n in b if m.distance < 0.75 * n.distance}
    pairs = np.array([(q, t) for q, t in fwd.items() if back.get(t) == q]).reshape(-1, 2)
    if not len(pairs): return 0, 0.0, 0.0
    ra, rb = kps[i][pairs[:, 0]], kps[j][pairs[:, 1]]
    ang = np.degrees(np.arccos(np.clip((ra * rb).sum(1), -1, 1)))
    lon = np.degrees(np.arctan2(ra[:, 0], ra[:, 2])); lat = np.degrees(-np.arcsin(ra[:, 1]))
    cells = {(int((x + 180) // 30), int((y + 32) // 30.7)) for x, y in zip(lon, lat)}      # 12 x 3 cells in the band
    return len(pairs), float(np.median(ang)), len(cells) / 36


n = len(p.photos); R['overlap'] = {}
print('overlap (SIFT, ratio test + mutual check at 2688 px; no geometric check yet):')
for k in (1, 2, 4, 8, 16, n - 1):
    res = np.array([match(i, i + k) for i in range(0, n - k, max(1, (n - k) // 12))])
    R['overlap'][f'i_vs_i+{k}'] = dict(pairs=len(res), matches_median=float(np.median(res[:, 0])), matches_min=float(res[:, 0].min()),
                                       moved_deg_median=float(np.median(res[:, 1])), cells_with_matches=float(np.median(res[:, 2])))
    print(f"  shot i vs i+{k:<2d}: {np.median(res[:, 0]):5.0f} matches (min {res[:, 0].min():.0f}), matched directions moved "
          f"{np.median(res[:, 1]):.2f} deg (median), matches in {100 * np.median(res[:, 2]):.0f}% of the 30-deg cells")
json.dump(R | {'per_shot': rows}, open(p.product / 'inspect.json', 'w'), indent=1, default=str)
print('wrote', p.product / 'inspect.json', p.product / 'contact.jpg')
