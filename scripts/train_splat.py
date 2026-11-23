#!/usr/bin/env python3
"""3D Gaussian splatting (gsplat, MCMC) on the faces of the panoramas, with the poses of the aligned rig model.

    python3 -I scripts/train_splat.py data/pool/rig.yaml [--tag splat] [--depth 0.2] [--steps N]
    python3 -I scripts/train_splat.py data/pool/rig.yaml --tag splat_pass --holdout-from R0013103    # train on the first passes, test on the last
    python3 -I scripts/train_splat.py data/pool/rig.yaml --eval-only                                 # score splat/ckpt.pt again

Taken from kitchen-recon/scripts/train_splat.py; what a 360 capture changes:
  - what does not belong to the scene is in every face: the rover (pass-0), the operator and whoever walks along
    (pool). Those pixels take no part in the loss or in the scores (labels/ from split_faces.py, else masks/)
  - exposure is automatic: one learned RGB gain per panorama, shared by its 8 faces
  - outdoors, splat.sky: env. The sky is a panorama at infinity (an equirectangular texture of 256 x 128, in the
    model's frame) seen wherever the Gaussians leave light through: colour = splat + (1 - alpha) * sky. It learns from
    the sky pixels only: trained on every pixel, it explained a blurred average of the whole scene by direction, the
    Gaussians faded out and MCMC had none left to resample (pool, step 2000). Alpha on the sky is pushed to zero
    (weight 0.05, on sky pixels at least 0.5 deg from anything else), so no Gaussians float in front of it.
  - whole panoramas are held out, never single faces. Two test sets:
      inside   every splat.holdout_every-th panorama (between trained neighbours: the easy test)
      beyond   the last splat.holdout_tail panoramas (the model has to extrapolate along the walk)
    Scores are on the scene pixels (no sky, no people); with the sky in the scores too as psnr_with_sky.
    --holdout-from NAME replaces both by one set, `pass`: every panorama from NAME on (a later pass over ground the
    earlier ones walked: views from another time, another line and another exposure). The poses and the starting
    cloud still come from all panoramas; only the training leaves them out.
    LPIPS (AlexNet, if the lpips package is there) is on the whole face with the sky, people and platform blanked.
  - --depth W: L1 loss in inverse depth against dense/final/ (PatchMatch depth, filled with the aligned
    monocular prior and the fitted planes on blank walls). Only the faces of the dense subset have it.
    On pass-0 it cost 0.3 dB inside the path and 1.0 dB beyond it and did not help raised viewpoints: off by default.
  - splat.init: dense starts from the fused PatchMatch cloud (thinned) besides the sparse points: needed when the shots are
    far apart for the area they cover (pool).
  - The opacity and size penalties act only on the Gaussians the current view sees (see the loss).
Writes <tag>/splat.ply (INRIA 3DGS format), sky.png (the learned sky), ckpt.pt, metrics.json, eval/*.jpg.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from project import load

ap = argparse.ArgumentParser(); ap.add_argument('yaml'); ap.add_argument('--tag', default='splat')
ap.add_argument('--depth', type=float, default=0.0); ap.add_argument('--steps', type=int); ap.add_argument('--sh', type=int, default=2)
ap.add_argument('--no-gain', action='store_true'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--noise-lr', type=float, help='MCMC exploration noise (gsplat default 5e5); default: scaled to the scene, see below')
ap.add_argument('--sky-alpha', type=float, default=0.05, help='weight of the opacity penalty on sure sky pixels')
ap.add_argument('--holdout-from', help='hold out every panorama from this one on (a later pass), instead of the inside and beyond sets')
ap.add_argument('--eval-only', action='store_true', help='no training: score <tag>/ckpt.pt again (same hold-out flags as when it was trained)')
ap.add_argument('--eval-save', type=int, default=8, help='comparison pictures per test set in <tag>/eval/')
a = ap.parse_args()
p = load(a.yaml); steps = a.steps or p.splat_steps; S = p.splat_size
out = p.run / a.tag; (out / 'eval').mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
use_labels = p.labels.is_dir(); use_env = p.splat_sky == 'env'
if use_env and not use_labels: raise SystemExit('splat.sky: env needs the sky labels of split_faces.py (mask.semantic with sky)')

# ------------------------------------------------------------------ data
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: (Path(im.name).stem, im.name))
panos = sorted({Path(im.name).stem for im in ims}); pidx = {n: i for i, n in enumerate(panos)}
tail = set(panos[len(panos) - p.holdout_tail:]) if p.holdout_tail else set()
inside = {n for i, n in enumerate(panos) if p.holdout_every and i % p.holdout_every == p.holdout_every // 2 and n not in tail}
later = set()
if a.holdout_from:
    if a.holdout_from not in pidx: raise SystemExit(f'--holdout-from {a.holdout_from}: no such panorama in the model')
    later = set(panos[pidx[a.holdout_from]:]); tail = inside = set()
k = S / p.size; f = p.focal * k
K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, round(f * math.radians(0.5))) + 1,) * 2)
names, viewmats, imgs, labs, pano_of, depths = [], [], [], [], [], {}
for im in ims:
    E = np.eye(4); E[:3] = im.cam_from_world().matrix()
    bgr = cv2.resize(cv2.imread(str(p.faces / im.name)), (S, S), interpolation=cv2.INTER_AREA)
    if use_labels:            # 0 scene, 1 sky, 2 people, 3 platform; +16 = sky at least 0.5 deg from anything else
        lab = cv2.resize(cv2.imread(str(p.labels / (str(Path(im.name).with_suffix('')) + '.png')), cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_NEAREST)
        lab = lab | (cv2.erode((lab == 1).astype(np.uint8), erode) * 16)
    else:
        m = cv2.resize(cv2.imread(str(p.masks / (im.name + '.png')), cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_NEAREST) > 127
        lab = np.where(m, 0, 3).astype(np.uint8)
    names.append(im.name); viewmats.append(E); pano_of.append(pidx[Path(im.name).stem])
    imgs.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))); labs.append(torch.from_numpy(lab))
    d = p.dense / 'final' / (im.name + '.npy')
    if a.depth and d.exists():
        z = np.load(d).astype(np.float32); z = cv2.resize(z, (S, S), interpolation=cv2.INTER_NEAREST)
        lp = d.with_suffix('.label.png')                                # 1 PatchMatch, 2 plane, 3 prior: the prior counts 0.3
        wz = np.array([0, 1, 1, 0.3], np.float32)[cv2.resize(cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE), (S, S), interpolation=cv2.INTER_NEAREST)] \
            if lp.exists() else (z > 0).astype(np.float32)
        depths[len(names) - 1] = (torch.from_numpy(z).half().to(dev), torch.from_numpy(wz).half().to(dev))
N = len(names)
imgs = torch.stack(imgs).to(dev); labs = torch.stack(labs).to(dev)
viewmats = torch.tensor(np.array(viewmats), dtype=torch.float32, device=dev)
Ks = torch.tensor(K, dtype=torch.float32, device=dev)[None]
pano_of = torch.tensor(pano_of, device=dev)
sets = {'inside': [i for i, n in enumerate(names) if Path(n).stem in inside], 'beyond': [i for i, n in enumerate(names) if Path(n).stem in tail],
        'pass': [i for i, n in enumerate(names) if Path(n).stem in later]}
held = set(sets['inside']) | set(sets['beyond']) | set(sets['pass'])
train_idx = [i for i in range(N) if i not in held]
train_panos = sorted({int(pano_of[i]) for i in train_idx})
centres = torch.linalg.inv(viewmats)[:, :3, 3]; cen = centres.cpu().numpy()
xyz = np.array([q.xyz for q in rec.points3D.values()]); rgb = np.array([q.color for q in rec.points3D.values()])
lens = float(centres[:, 2].median())
if p.outdoor:                                                           # the walk is long: distance to the nearest camera, not to the middle
    dist = cKDTree(cen).query(xyz)[0]; near = dist < 40 * lens
else:
    dist = np.linalg.norm(xyz - cen.mean(0), axis=1); near = dist < 60 * lens   # drop stray far points
xyz, rgb, dist = xyz[near], rgb[near], dist[near]
# the step size of the means: 1.6e-4 of this length. Indoors the room round the middle of the path; outdoors the distance at
# which the cameras saw things (pool: 63 lens heights from the middle of the walk against 9 from the nearest shot: at the
# larger one, Gaussians moved a unit per 100 steps, many times their size)
scene_scale = float(np.percentile(dist if p.outdoor else np.linalg.norm(xyz - cen.mean(0), axis=1), 75))
unit = float(np.median(dist if p.outdoor else np.linalg.norm(xyz - cen.mean(0), axis=1))) / 2     # a scene-sized length: makes the size penalty unit-free
n_sparse = len(xyz)
if p.splat_init == 'dense':
    # Start from the fused PatchMatch cloud as well, thinned to 60% of the budget. The sparse points of a walk with shots
    # 5 lens heights apart are too few for its area (pool: 92k over 175 x 145 m); the splat then spends its steps growing.
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(p.dense / 'fused.ply')); D = np.asarray(pc.points)
    pc = pc.select_by_index(np.flatnonzero(cKDTree(cen).query(D)[0] < 40 * lens if p.outdoor else np.linalg.norm(D - cen.mean(0), axis=1) < 60 * lens))
    vs, target = 0.01 * lens, int(0.6 * p.splat_cap)
    while len(pd_ := pc.voxel_down_sample(vs).points) > target: vs *= 1.25
    pd = pc.voxel_down_sample(vs)
    xyz = np.r_[xyz, np.asarray(pd.points)]; rgb = np.r_[rgb, np.asarray(pd.colors) * 255]
    print(f'dense start: {len(D):,} fused points -> {len(pd.points):,} after thinning to {vs / lens:.3f} lens heights, plus {n_sparse:,} sparse', flush=True)
print(f'{N} faces {S}x{S} of {len(panos)} panoramas: {len(train_idx)} train, {len(sets["inside"])} held out inside, '
      f'{len(sets["beyond"])} held out beyond the path, {len(sets["pass"])} of a held-out pass; depth maps for {len(depths)} faces; scene scale {scene_scale:.2f}; '
      f'labels {"yes" if use_labels else "no (masks)"}, sky {"learned panorama" if use_env else "none"}', flush=True)

# ------------------------------------------------------------------ Gaussians
# Outdoors the training runs in a frame scaled by 1 / scene scale, as gsplat's own trainer normalises its scenes: its learning
# rates and MCMC's exploration noise (lr_means * noise_lr * covariance, which grows with the cube of the unit) are tuned for
# that size. In lens heights the pool walk is 25 units across and its Gaussians 10-100x larger than gsplat expects: faded
# Gaussians were shaken by many times their size at every step, and the training loss rose from step 8000 on (0.080 -> 0.112).
# The checkpoint and the PLY are written back in the scene's units.
nrm = scene_scale if p.outdoor else 1.0
viewmats[:, :3, 3] /= nrm; depths = {k_: ((z_.float() / nrm).half(), w_) for k_, (z_, w_) in depths.items()}; unit /= nrm
xyz_t = torch.tensor(xyz / nrm, dtype=torch.float32); rgb_t = torch.tensor(rgb, dtype=torch.float32) / 255
d, _ = cKDTree(xyz).query(xyz, k=4); dist = torch.tensor(d[:, 1:].mean(1) / nrm, dtype=torch.float32).clamp_min(1e-7)
C0 = 0.28209479177387814; n0 = len(xyz); K_sh = (a.sh + 1) ** 2
params = torch.nn.ParameterDict({
    'means': torch.nn.Parameter(xyz_t), 'scales': torch.nn.Parameter(torch.log(dist * 0.1).unsqueeze(1).repeat(1, 3)),
    'quats': torch.nn.Parameter(torch.rand(n0, 4)), 'opacities': torch.nn.Parameter(torch.logit(torch.full((n0,), 0.5))),
    'sh0': torch.nn.Parameter(((rgb_t - 0.5) / C0).unsqueeze(1)), 'shN': torch.nn.Parameter(torch.zeros(n0, K_sh - 1, 3))}).to(dev)
if a.eval_only:
    ck = torch.load(out / 'ckpt.pt', map_location='cpu')
    params = torch.nn.ParameterDict({k_: torch.nn.Parameter(ck[k_]) for k_ in params}).to(dev)
    with torch.no_grad(): params['means'].div_(nrm); params['scales'].sub_(math.log(nrm))
lrs = {'means': 1.6e-4 * scene_scale / nrm, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'sh0': 2.5e-3, 'shN': 2.5e-3 / 20}
opt = {k_: torch.optim.Adam([{'params': params[k_], 'lr': lr, 'name': k_}], eps=1e-15) for k_, lr in lrs.items()}
sched = torch.optim.lr_scheduler.ExponentialLR(opt['means'], gamma=0.01 ** (1.0 / steps))
noise_lr = a.noise_lr or 5e5                                           # gsplat's own, for a normalised scene (indoors: pass-0 as it was)
strat = MCMCStrategy(cap_max=p.splat_cap, refine_stop_iter=int(steps * 25 / 30), noise_lr=noise_lr); strat.check_sanity(params, opt)
state = strat.initialize_state()
expo = torch.nn.Parameter(torch.zeros(len(panos), 3, device=dev))       # log gain per panorama
extra = [expo]
if use_env:                                                             # the sky: logits of an equirectangular texture, x east-ish, z up
    sky_rgb = torch.stack([imgs[i][(labs[i] & 15) == 1].float().mean(0) for i in range(0, N, 7) if ((labs[i] & 15) == 1).any()]).mean(0) / 255
    env = torch.nn.Parameter(torch.logit(sky_rgb.clamp(0.02, 0.98))[:, None, None].repeat(1, 128, 256)[None].contiguous())
    extra.append(env)
    c_ = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5 - S / 2) / f
    gy, gx = torch.meshgrid(c_, c_, indexing='ij')
    rays_cam = F.normalize(torch.stack([gx, gy, torch.ones_like(gx)], -1), dim=-1)    # pixel rays in the face's frame
if a.eval_only:
    expo.data = ck['expo'].to(dev)
    if use_env: env.data = ck['env'].to(dev)
expo_opt = torch.optim.Adam([{'params': [expo], 'lr': 2e-3}] + ([{'params': [env], 'lr': 1e-2}] if use_env else []))
print(f'init {n0:,} Gaussians ({n_sparse:,} from sparse points' + (f', {n0 - n_sparse:,} from the dense cloud' if n0 > n_sparse else '') + f'), cap {p.splat_cap:,}, {steps} steps, SH degree {a.sh}, MCMC noise {noise_lr:.3g}', flush=True)


def sky(i, lab=None):
    """The learned sky in the directions of face i's pixels. With lab (training) it learns from the sky pixels only."""
    dw = rays_cam @ viewmats[i, :3, :3]                                 # world directions: rows of R.T @ ray
    g = torch.stack([torch.atan2(dw[..., 1], dw[..., 0]) / math.pi, -torch.asin(dw[..., 2].clamp(-1, 1)) / (math.pi / 2)], -1)
    s = F.grid_sample(torch.sigmoid(env), g[None], mode='bilinear', padding_mode='border', align_corners=False)[0].permute(1, 2, 0)
    return s if lab is None else torch.where(((lab & 15) == 1)[..., None], s, s.detach())


def render(i, sh_degree, mode='RGB', lab=None):
    colors = torch.cat([params['sh0'], params['shN']], 1)
    o, alpha, info = rasterization(params['means'], params['quats'], torch.exp(params['scales']), torch.sigmoid(params['opacities']),
                                   colors, viewmats[i:i + 1], Ks, S, S, sh_degree=sh_degree, packed=False,
                                   rasterize_mode='antialiased', near_plane=0.01 / nrm, render_mode=mode)
    o, alpha = o[0], alpha[0]
    if use_env: o = torch.cat([o[..., :3] + (1 - alpha) * sky(i, lab), o[..., 3:]], -1)
    return o, alpha, info


def gain(i):                                                            # a held-out panorama: its nearest trained ones
    j = int(pano_of[i])
    if j in train_panos: return torch.exp(expo[j])
    if a.holdout_from: nb = [train_panos[t] for t in pano_tree.query(pano_cen[j], k=2)[1]]      # a held-out pass: the two nearest in space
    else: nb = sorted(train_panos, key=lambda t: abs(t - j))[:2]
    return torch.exp(expo[nb].mean(0))


po_ = pano_of.cpu().numpy()
pano_cen = np.array([cen[np.flatnonzero(po_ == j)[0]] for j in range(len(panos))])      # the faces of a panorama share their centre
pano_tree = cKDTree(pano_cen[train_panos])
try:
    import lpips
    lp_net = lpips.LPIPS(net='alex', verbose=False).to(dev)
except Exception as e:
    lp_net = None; print(f'no LPIPS ({type(e).__name__}): pip install lpips', flush=True)

_win = None
def ssim_map(x, y):
    global _win
    if _win is None:
        g = torch.exp(-(torch.arange(11, device=dev) - 5.0) ** 2 / (2 * 1.5 ** 2)); g /= g.sum()
        _win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    mx, my = F.conv2d(x, _win, groups=3, padding=5), F.conv2d(y, _win, groups=3, padding=5)
    sx = F.conv2d(x * x, _win, groups=3, padding=5) - mx * mx; sy = F.conv2d(y * y, _win, groups=3, padding=5) - my * my
    sxy = F.conv2d(x * y, _win, groups=3, padding=5) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (sx + sy + 9e-4)))[0].permute(1, 2, 0)


# ------------------------------------------------------------------ train
t0 = time.time(); ema = None; perm = []; bad = 0
for step in range(0 if a.eval_only else steps):
    if not perm: perm = torch.randperm(len(train_idx)).tolist()
    i = train_idx[perm.pop()]
    gt = imgs[i].float() / 255; lab = labs[i]; m = (lab & 15) <= (1 if use_env else 0)       # pixels that count: scene (and sky)
    use_d = a.depth and i in depths
    o, alpha, info = render(i, min(step // 1000, a.sh), 'RGB+ED' if use_d else 'RGB', lab)
    if use_d: dep, o = o[..., 3], o[..., :3]
    if not a.no_gain: o = o * gain(i)
    o = o.clamp(0, 1)
    l1 = (o - gt).abs()[m].mean()
    loss = 0.8 * l1 + 0.2 * (1 - ssim_map(o, gt)[m].mean())
    # the opacity and size penalties only on the Gaussians this view sees. On all of them, every step, Adam turns the tiny
    # penalty of a Gaussian that is off screen in 98% of the views (a long walk) into full-size steps: pool, 47-56% of them
    # faded below 0.005 by step 4000. In pass-0's room every view saw most of the Gaussians and it made no difference.
    vis = (info['radii'][0] > 0) if info['radii'].dim() == 2 else (info['radii'][0] > 0).all(-1)
    nG = len(params['means'])
    loss = loss + 0.01 * torch.sigmoid(params['opacities'][vis]).sum() / nG + 0.01 * torch.exp(params['scales'][vis]).sum() / (3 * nG) / unit
    if use_env:
        sk = (lab & 16) > 0
        if sk.any() and a.sky_alpha: loss = loss + a.sky_alpha * alpha[..., 0][sk].mean()
    if not a.no_gain: loss = loss + 0.1 * (expo[train_panos].mean(0) ** 2).sum()
    if use_d:
        z, wz = depths[i]; z = z.float(); ok = m & (z > 0) & (dep > 1e-3)
        if ok.any():                                                    # inverse depth, unit-free: (scene scale / nrm) is 1 / its own unit
            loss = loss + a.depth * scene_scale / nrm * (wz[ok].float() * (1 / dep[ok] - 1 / z[ok]).abs()).sum() / wz[ok].float().sum().clamp_min(1)
    if not torch.isfinite(loss):                                        # never seen yet; a NaN would poison every parameter
        bad += 1
        for o_ in opt.values(): o_.zero_grad(set_to_none=True)
        expo_opt.zero_grad(set_to_none=True); continue
    loss.backward()
    strat.step_post_backward(params, opt, state, step, info, lr=sched.get_last_lr()[0])
    for o_ in opt.values(): o_.step(); o_.zero_grad(set_to_none=True)
    expo_opt.step(); expo_opt.zero_grad(set_to_none=True)
    sched.step()
    ema = l1.item() if ema is None else 0.99 * ema + 0.01 * l1.item()
    if step % 2000 == 0 or step == steps - 1:
        op_ = torch.sigmoid(params['opacities'].detach())
        print(f'step {step:6d}  L1(ema) {ema:.4f}  {len(params["means"]):,} Gaussians (opacity mean {op_.mean():.3f}, {100 * (op_ < 0.005).float().mean():.0f}% '
              f'under 0.005)  {time.time() - t0:.0f}s  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB' + (f'  {bad} steps skipped (not finite)' if bad else ''), flush=True)
train_time = time.time() - t0


# ------------------------------------------------------------------ scores on the held-out panoramas: scene pixels, and with the sky
@torch.no_grad()
def evaluate(idx, save):
    ps, ss, pa, lp = [], [], [], []
    for k_, i in enumerate(idx):
        gt = imgs[i].float() / 255; lab = labs[i] & 15; o, _, _ = render(i, a.sh)
        if not a.no_gain: o = o * gain(i)
        o = o.clamp(0, 1); m = lab == 0; ma = lab <= 1
        if m.sum() < 1000: continue                                     # a face that is all sky or people has no scene to score
        ps.append(float(-10 * torch.log10(((o - gt) ** 2)[m].mean()))); ss.append(float(ssim_map(o, gt)[m].mean()))
        pa.append(float(-10 * torch.log10(((o - gt) ** 2)[ma].mean())))
        if lp_net is not None:                                          # the whole face with the sky; what is not scored is the same in both
            x = torch.where(ma[..., None], o, gt).permute(2, 0, 1)[None] * 2 - 1
            lp.append(float(lp_net(x, gt.permute(2, 0, 1)[None] * 2 - 1)))
        if save and k_ % max(1, len(idx) // max(1, a.eval_save)) == 0:
            both = torch.cat([gt, o * ma[..., None]], 1).mul(255).byte().cpu().numpy()
            cv2.imwrite(str(out / 'eval' / f'{save}_{names[i].replace("/", "_")[:-4]}_psnr{ps[-1]:.1f}.jpg'),
                        cv2.cvtColor(both, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return ps, ss, pa, lp


M = json.load(open(out / 'metrics.json')) if a.eval_only else \
    dict(faces=N, train_faces=len(train_idx), size=S, steps=steps, gaussians=len(params['means']), train_time_s=round(train_time),
         depth_weight=a.depth, faces_with_depth=len(depths), per_panorama_gain=not a.no_gain, sh_degree=a.sh, sky=p.splat_sky,
         scene_scale=scene_scale, trained_at_scale=1 / nrm, mcmc_noise_lr=noise_lr, init=p.splat_init, init_gaussians=n0, init_sparse=n_sparse,
         held_out_inside_panoramas=sorted(inside), held_out_beyond_panoramas=sorted(tail), held_out_pass_panoramas=sorted(later))
near_tr = pano_tree.query(pano_cen)[0] / lens                           # every panorama: lens heights to the nearest trained one
for name, idx in sets.items():
    if not idx: continue
    ps, ss, pa, lp = evaluate(idx, name)
    sc = [i for i in idx if int(((labs[i] & 15) == 0).sum()) >= 1000]
    M[name] = dict(faces=len(sc), psnr=round(float(np.mean(ps)), 2), ssim=round(float(np.mean(ss)), 4), psnr_with_sky=round(float(np.mean(pa)), 2),
                   psnr_min=round(min(ps), 2), worst=names[sc[int(np.argmin(ps))]])
    if lp: M[name]['lpips'] = round(float(np.mean(lp)), 4)
    for v in range(len(p.views)):
        q = [x for x, i in zip(ps, sc) if names[i].startswith(f'v{v:02d}/')]
        if q: M[name][f'psnr_v{v:02d}'] = round(float(np.mean(q)), 2)
    per = {}                                                            # per panorama, with its distance to the nearest trained one
    for t, i in enumerate(sc): per.setdefault(int(pano_of[i]), []).append(t)
    M[name]['panoramas'] = {panos[j]: dict(nearest_trained=round(float(near_tr[j]), 2), psnr=round(float(np.mean([ps[t] for t in ts])), 2),
                                           ssim=round(float(np.mean([ss[t] for t in ts])), 4), psnr_with_sky=round(float(np.mean([pa[t] for t in ts])), 2))
                            | ({'lpips': round(float(np.mean([lp[t] for t in ts])), 4)} if lp else {}) for j, ts in sorted(per.items())}
    M[name]['nearest_trained_median'] = round(float(np.median([near_tr[j] for j in per])), 2)
ps, ss, pa, lp = evaluate(train_idx[::8], None)
M['train'] = dict(psnr=round(float(np.mean(ps)), 2), ssim=round(float(np.mean(ss)), 4), psnr_with_sky=round(float(np.mean(pa)), 2)) | ({'lpips': round(float(np.mean(lp)), 4)} if lp else {})
if lp: M['lpips'] = 'AlexNet, whole face with the sky; people and platform pixels blanked'
if a.eval_only:
    json.dump(M, open(out / 'metrics.json', 'w'), indent=1)
    print(json.dumps({k_: ({q: w for q, w in v.items() if q != 'panoramas'} if isinstance(v, dict) else v) for k_, v in M.items() if not k_.startswith('held_out')}, indent=1))
    sys.exit(0)
g = torch.exp(expo.detach()[train_panos]).mean(1).cpu().numpy()
M['gain_range_stops'] = round(float(np.log2(g.max() / g.min())), 2)
op = torch.sigmoid(params['opacities']).detach(); ls = params['scales'].detach().sort(dim=1).values; vis = op > 0.05
M |= {'opacity_lt_0.05_pct': round(float((op < 0.05).float().mean()) * 100, 1),
      'needles_pct_of_visible': round(float(((ls[:, 2] - ls[:, 1]) > math.log(10.0))[vis].float().mean()) * 100, 1)}
json.dump(M, open(out / 'metrics.json', 'w'), indent=1)
print(json.dumps({k_: ({q: w for q, w in v.items() if q != 'panoramas'} if isinstance(v, dict) else v) for k_, v in M.items() if not k_.startswith('held_out')}, indent=1))

# ------------------------------------------------------------------ export
with torch.no_grad():                                                   # back to the scene's units
    params['means'].mul_(nrm); params['scales'].add_(math.log(nrm))
torch.save({k_: v.detach().cpu() for k_, v in params.items()} | {'expo': expo.detach().cpu()} | ({'env': env.detach().cpu()} if use_env else {}), out / 'ckpt.pt')
if use_env:
    cv2.imwrite(str(out / 'sky.png'), cv2.cvtColor(torch.sigmoid(env.detach())[0].permute(1, 2, 0).mul(255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR))
with torch.no_grad():
    n = len(params['means'])
    data = torch.cat([params['means'], torch.zeros(n, 3, device=dev), params['sh0'].reshape(n, 3),
                      params['shN'].permute(0, 2, 1).reshape(n, -1), params['opacities'].reshape(n, 1), params['scales'],
                      F.normalize(params['quats'], dim=1)], 1).float().cpu().numpy()
props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{j}' for j in range(3 * (K_sh - 1))] \
        + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
with open(out / 'splat.ply', 'wb') as fh:
    fh.write(('ply\nformat binary_little_endian 1.0\n' + f'element vertex {n}\n' + ''.join(f'property float {q}\n' for q in props) + 'end_header\n').encode())
    fh.write(data.astype('<f4').tobytes())
print(f'wrote {out / "splat.ply"} ({n:,} Gaussians, {os.path.getsize(out / "splat.ply") / 2**20:.0f} MB)')
