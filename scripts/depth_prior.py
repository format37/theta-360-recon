#!/usr/bin/env python3
"""Step 4a. A monocular depth prior for every dense face, aligned to the PatchMatch depth of the same face.

    python3 -I scripts/depth_prior.py data/pass-0/rig.yaml

White walls give PatchMatch nothing to match (kitchen-recon: 8% of blank bright pixels got a depth).
A monocular network (dense.prior_model, Depth Anything V2 metric indoor) gives a depth for every pixel, in its
own approximate scale and with slow warps. Per face it is tied to the multi-view geometry in two steps:
  1. one scale: the median of PatchMatch depth / network depth over the pixels PatchMatch solved
  2. a smooth correction field: the same ratio, blurred over 6% of the face width (normalised convolution),
     so the prior follows the measured depth where there is any and bends slowly in between.
The honest error of step 2 comes from a checkerboard: the field is fitted on the white 64 px blocks and
compared with PatchMatch on the black ones.
Writes dense/prior/<face>.npy (float16, model units) and product/prior.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.modules.setdefault('OpenSSL', None)     # a broken pyOpenSSL in the environment must fail as "not installed", not crash transformers
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).parent))
from project import load


def read_array(path):                                   # COLMAP binary array: "W&H&C&" + float32, column-major
    with open(path, 'rb') as f:
        head = b''
        while head.count(b'&') < 3: head += f.read(1)
        w, h, c = map(int, head.split(b'&')[:3]); arr = np.fromfile(f, np.float32)
    return arr.reshape((w, h, c), order='F').transpose(1, 0, 2).squeeze()


def field(ratio, valid, sigma):
    """Smooth ratio field from scattered valid pixels; 1 where there is no support nearby."""
    w = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    r = cv2.GaussianBlur(np.where(valid, ratio, 0).astype(np.float32), (0, 0), sigma)
    w2 = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 4 * sigma)
    r2 = cv2.GaussianBlur(np.where(valid, ratio, 0).astype(np.float32), (0, 0), 4 * sigma)
    far = (r2 + 1e-4) / (w2 + 1e-4)                      # wide blur: the fallback away from any support
    return (r + 0.02 * far) / (w + 0.02)


if __name__ == '__main__':
    p = load(sys.argv[1])
    proc = AutoImageProcessor.from_pretrained(p.prior_model)
    net = AutoModelForDepthEstimation.from_pretrained(p.prior_model).cuda().eval().half()
    files = sorted((p.dense / 'stereo' / 'depth_maps').glob('*/*.geometric.bin'))
    rows = {}
    for f in files:
        name = f'{f.parent.name}/{f.name[:-len(".geometric.bin")]}'
        img = cv2.cvtColor(cv2.imread(str(p.dense / 'images' / name)), cv2.COLOR_BGR2RGB)
        mvs = read_array(f); S = mvs.shape[0]; sig = 0.06 * S
        yy, xx = np.mgrid[:S, :S]; white = ((yy // 64 + xx // 64) % 2) == 0
        mask = cv2.resize(cv2.imread(str(p.dense / 'masks' / (name + '.png')), cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_NEAREST) > 127
        ok = (mvs > 0) & mask
        with torch.no_grad():
            x = proc(images=img, return_tensors='pt')['pixel_values'].cuda().half()
            d = net(pixel_values=x).predicted_depth[None].float()
            mono = torch.nn.functional.interpolate(d, (S, S), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
        mono = np.maximum(mono, 1e-3)
        r = dict(mvs_valid=float(ok[mask].mean()))
        if ok.sum() < 2000:                              # nothing to tie the prior to: the face is skipped
            rows[name] = r | dict(skipped=True); continue
        s = float(np.median(mvs[ok] / mono[ok]))
        ratio = mvs / (s * mono)
        okr = ok & (ratio > 0.5) & (ratio < 2.0)
        test = okr & ~white
        r |= dict(scale=s, err_scale_only=float(np.median(np.abs(ratio[test] - 1))))
        r['err_field_heldout'] = float(np.median(np.abs(ratio[test] / field(ratio, okr & white, sig)[test] - 1)))
        prior = s * mono * field(ratio, okr, sig)
        prior[~mask] = 0
        out = p.dense / 'prior' / (name + '.npy'); out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, prior.astype(np.float16)); rows[name] = r
    good = [r for r in rows.values() if 'scale' in r]
    sc = np.array([r['scale'] for r in good])
    R = dict(model=p.prior_model, faces=len(rows), faces_aligned=len(good),
             scale_units_per_network_metre=dict(median=float(np.median(sc)), p10=float(np.percentile(sc, 10)), p90=float(np.percentile(sc, 90))),
             median_rel_error_one_scale_pct=float(100 * np.median([r['err_scale_only'] for r in good])),
             median_rel_error_with_field_heldout_pct=float(100 * np.median([r['err_field_heldout'] for r in good])),
             per_face=rows)
    json.dump(R, open(p.product / 'prior.json', 'w'), indent=1)
    print(f"{len(good)}/{len(rows)} faces aligned; one model unit = {1 / np.median(sc):.3f} network metres "
          f"(10-90%: {1 / np.percentile(sc, 90):.3f}-{1 / np.percentile(sc, 10):.3f}); prior vs PatchMatch, median relative error: "
          f"{R['median_rel_error_one_scale_pct']:.1f}% with one scale per face, {R['median_rel_error_with_field_heldout_pct']:.1f}% "
          f"with the smooth field on held-out blocks")
