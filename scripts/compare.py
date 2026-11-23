#!/usr/bin/env python3
"""This run next to the phone run of the same kitchen (kitchen-recon), and next to the spherical variant.

    python3 -I scripts/compare.py data/pass-0/rig.yaml

Reads product/*.json of the session's variants and, for the phone run, the model and the dense report named
in the YAML (compare: {model: ..., dense_report: ...}). Writes product/compare.md.
"""
import json
import sys
from pathlib import Path

import pycolmap

sys.path.insert(0, str(Path(__file__).parent))
from project import load
from sfm_report import headline, observations

p = load(sys.argv[1])
cmpc = p.cfg.get('compare') or {}
J = lambda f: json.load(open(f)) if Path(f).exists() else {}
sfm, mesh, al, sph = J(p.product / 'sfm.json'), J(p.product / 'mesh.json'), J(p.product / 'align.json'), J(p.session / 'out' / 'spherical' / 'product' / 'spherical.json')
splat = {t: J(p.run / t / 'metrics.json') for t in ('splat', 'splat_depth')}
bbj = J(p.product / 'blank_bright.json')          # the same class at the phone run's texture threshold, when computed
phone, phone_dense = {}, {}
if cmpc.get('model') and Path(cmpc['model']).expanduser().exists():
    rec = pycolmap.Reconstruction(str(Path(cmpc['model']).expanduser())); phone = headline(rec, observations(rec))
    phone_dense = J(Path(cmpc.get('dense_report', '')).expanduser())
g = lambda d, *ks: (g(d.get(ks[0], {}), *ks[1:]) if len(ks) > 1 else d.get(ks[0], '-')) if isinstance(d, dict) else '-'
bb = 'blank bright (white wall, ceiling, fronts)'
rows = [
    ('shots', phone.get('images_total', '-'), sfm.get('panoramas_total', '-'), sph.get('panoramas_total', '-')),
    ('registered', f"{phone.get('images_registered', '-')} frames", f"{sfm.get('panoramas_registered', '-')} panoramas ({sfm.get('images_registered', '-')} faces)",
     f"{sph.get('panoramas_registered', '-')} panoramas"),
    ('3D points', phone.get('points', '-'), sfm.get('points', '-'), sph.get('points', '-')),
    ('mean track length', phone.get('track_length_mean', '-'), sfm.get('track_length_mean', '-'), sph.get('track_length_mean', '-')),
    ('reprojection error, px (mean)', f"{phone.get('reproj_px_mean', '-')} at 4K, f 2621 px", f"{sfm.get('reproj_px_mean', '-')} at f {sfm.get('focal_px', 0):.0f} px",
     f"{sph.get('reproj_px_equirect_mean', '-')} px of the panorama"),
    ('reprojection error, deg (mean / median)', f"{phone.get('reproj_deg_mean', '-')} / {phone.get('reproj_deg_median', '-')}",
     f"{sfm.get('reproj_deg_mean', '-')} / {sfm.get('reproj_deg_median', '-')}", f"{sph.get('reproj_deg_mean', '-')} / {sph.get('reproj_deg_median', '-')}"),
    ('PatchMatch depth on blank bright pixels, % (share of all pixels)',
     f"{g(phone_dense, 'valid_pct_by_pixel_type', 'blank bright (white wall, ceiling, cabinet fronts)')} ({g(phone_dense, 'pixel_share_pct', 'blank bright (white wall, ceiling, cabinet fronts)')})",
     f"{g(mesh, 'coverage', bb, 'patchmatch')} ({g(mesh, 'coverage', bb, 'share_pct')})", 'not run'),
    ('the same with the phone run\'s texture threshold', '-',
     f"{g(bbj, 'same threshold as the phone run (std < 1.08)', 'patchmatch_pct')} ({g(bbj, 'same threshold as the phone run (std < 1.08)', 'share_of_pixels_pct')})", 'not run'),
    ('the same after prior + planes, % (confirmed by a second panorama)', 'not tried', f"{g(mesh, 'coverage', bb, 'filled')} ({g(mesh, 'coverage', bb, 'filled_confirmed')})", 'not run'),
    ('PatchMatch depth on all pixels, %', phone_dense.get('valid_depth_pct_mean', '-'), g(mesh, 'all_pixels', 'patchmatch'), 'not run'),
    ('scale', 'none (model units)', f"lens height {p.camera_height_cm or 'not given'} cm; spread of the 49 lens heights above the floor {al.get('lens_height_spread_pct', 0):.2f}%", 'none'),
]
lines = ['| | phone video (kitchen-recon) | THETA S, faces as a rig (COLMAP) | THETA S, spherical (OpenMVG) |', '|---|---|---|---|']
lines += [f'| {a} | {b} | {c} | {d} |' for a, b, c, d in rows]
if sph.get('vs_rig'):
    v = sph['vs_rig']; lines += ['', f"Rig model against spherical model, {v['common_panoramas']} panoramas: centres {v['centre_rms_pct_of_path']}% of the path length apart (rms), "
                                     f"orientations {v['rotation_deg_median']} deg (median)."]
for t, m in splat.items():
    if m: lines += ['', f"Splat `{t}`: held-out panoramas inside the path {g(m, 'inside', 'psnr')} dB / SSIM {g(m, 'inside', 'ssim')}, beyond the end of the path "
                        f"{g(m, 'beyond', 'psnr')} dB / {g(m, 'beyond', 'ssim')}; {m.get('gaussians', 0):,} Gaussians, depth weight {m.get('depth_weight')}."]
(p.product / 'compare.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
