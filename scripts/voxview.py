#!/usr/bin/env python3
"""voxview - the voxel model in a native window (Open3D). From galaxy-s26-ultra-stereo (and drone-recon before it).

    python3 scripts/voxview.py data/pool/rig.yaml            # the run's product/voxels.npz
    python3 scripts/voxview.py path/to/voxels.npz
    python3 scripts/voxview.py data/pool/rig.yaml --cut 4 --screenshot shot.png

The model is already in the scene frame: z up, the ground at z = 0, the unit one lens height (metres once
camera_height_cm is in the YAML: the ruler then prints both).

Keys:  [ / ]  finer / coarser voxels (whole multiples of the model's own size)
       C      colour: photo RGB -> height -> source (green PatchMatch, blue plane, orange prior) -> views
       K      cut the top off (at 70% of the height), to look in from above; again = whole
       , / .  lower / raise the cut by 0.25 lens heights
       T      the walked path with camera frusta on/off    B  background dark/light
       L      lighting on/off             R  reset view        H  help        Q/Esc  quit
       X      clear the ruler
Mouse: left-drag rotate, right/ctrl-drag pan, alt-drag roll, wheel zoom.
Ruler: Shift + left click on a voxel, then on another: the distance is printed in the terminal, with its horizontal
       and vertical parts. It cannot be finer than the voxel.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('input', help='session YAML or a voxels.npz')
ap.add_argument('--model', help='COLMAP model for the walked path (default: the run\'s colmap/aligned)')
ap.add_argument('--times', type=int, default=1, help='start with voxels this many times the model\'s size')
ap.add_argument('--cut', type=float, help='hide everything above this height, model units')
ap.add_argument('--color', choices=['rgb', 'height', 'source', 'views'], default='rgb')
ap.add_argument('--no-path', action='store_true')
ap.add_argument('--screenshot', help='render one frame to this PNG and exit')
ap.add_argument('--size', default='1600x1000')
ap.add_argument('--view', default='-0.45,-0.75,0.55', help='initial view direction (front vector, z up)')
ap.add_argument('--zoom', type=float, default=0.55, help='smaller = closer')
a = ap.parse_args()

# ---------------------------------------------------------------- load
model = a.model; metres = None
if a.input.endswith(('.yaml', '.yml')):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from project import load
    p = load(a.input); src = p.product / 'voxels.npz'; model = model or str(p.colmap / 'aligned')
    al = p.product / 'align.json'
    metres = json.load(open(al)).get('metres_per_unit') if al.exists() else None
else:
    src = Path(a.input)
if not Path(src).exists(): raise SystemExit(f'{src} not found: run scripts/voxels.py first')
Z = np.load(src); IJK = Z['ijk'].astype(np.int64); RGB = Z['rgb'].astype(np.float64) / 255; VIEWS = Z['views'].astype(np.float64); SRC = Z['source']
ORIGIN = Z['origin'].astype(np.float64); SIZE = float(Z['size']); UNIT = str(Z['unit']) if 'unit' in Z else 'm'
ztop = float((IJK[:, 2].max() + 1) * SIZE + ORIGIN[2])
fmt = lambda x: f'{x:.2f} {UNIT}' + (f' = {x * metres:.2f} m' if metres else '')
print(f'{src}: {len(IJK):,} voxels of {SIZE:.3f} {UNIT}, extent {np.ptp(IJK[:, 0]) * SIZE:.1f} x {np.ptp(IJK[:, 1]) * SIZE:.1f} x {np.ptp(IJK[:, 2]) * SIZE:.1f} {UNIT}')

cams = None
if model and not a.no_path and os.path.exists(os.path.join(model, 'images.bin')):
    import pycolmap
    rec = pycolmap.Reconstruction(model)
    ims = sorted((im for im in rec.images.values() if im.has_pose and im.name.startswith('v00/')), key=lambda im: im.name)
    if ims:
        cams = np.array([im.projection_center() for im in ims]); cam_R = [im.cam_from_world().rotation.matrix() for im in ims]
        cam_K = rec.cameras[ims[0].camera_id]

state = dict(times=max(1, a.times), cmode=['rgb', 'height', 'source', 'views'].index(a.color), traj=cams is not None, dark=True, light=True, cut=a.cut)
SRC_COL = np.array([[0.5, 0.5, 0.5], [0.30, 0.69, 0.31], [0.25, 0.47, 0.85], [0.96, 0.60, 0.15]])       # -, PatchMatch, plane, prior


# ---------------------------------------------------------------- voxels at the current size, cut and colour
def rebin(k):
    """Voxels k times the model's size: mean colour, summed views, most frequent source."""
    q = IJK // k; qmin = q.min(0); q = q - qmin; d = q.max(0) + 1
    key = (q[:, 0] * d[1] + q[:, 1]) * d[2] + q[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    col = np.stack([np.bincount(inv, RGB[:, c], len(uk)) for c in range(3)], 1) / cnt[:, None]
    views = np.bincount(inv, VIEWS, len(uk))
    votes = np.stack([np.bincount(inv, SRC == s, len(uk)) for s in (1, 2, 3)], 1); srcm = votes.argmax(1) + 1
    qi = np.stack([uk // (d[1] * d[2]), (uk // d[2]) % d[1], uk % d[2]], 1) + qmin
    return (qi + 0.5) * (k * SIZE) + ORIGIN, col, views, srcm


def ramp(t):                                    # blue - teal - green - yellow - red
    stops = np.array([[0.19, 0.25, 0.58], [0.13, 0.57, 0.55], [0.55, 0.78, 0.25], [0.99, 0.86, 0.28], [0.85, 0.27, 0.18]])
    t = np.clip(t, 0, 1) * (len(stops) - 1); i = np.minimum(t.astype(int), len(stops) - 2); f = (t - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def build():
    s = state['times'] * SIZE; C, col, views, srcm = rebin(state['times'])
    if state['cut'] is not None:
        m = C[:, 2] <= state['cut']; C, col, views, srcm = C[m], col[m], views[m], srcm[m]
    state['centres'] = C; state['voxel'] = s
    if state['cmode'] == 1: col = ramp((C[:, 2] - ORIGIN[2]) / max(ztop - ORIGIN[2], 1e-6))
    elif state['cmode'] == 2: col = SRC_COL[srcm]
    elif state['cmode'] == 3: col = ramp(np.log1p(views) / np.log1p(np.percentile(VIEWS, 99) * state['times'] ** 2))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C)); pc.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, s)
    mode = ['photo RGB', 'height', 'source: green PatchMatch, blue plane, orange prior', 'panoramas that saw the voxel'][state['cmode']]
    print(f"voxel {s:.3f} {UNIT} | {len(C):,} voxels | colour: {mode}" + (f" | cut at {state['cut']:.2f}" if state['cut'] is not None else ''))
    return vg


def trajectory():
    if cams is None: return None
    s = 0.3 * SIZE / 0.1; pts, lines, cols = [], [], []; step = np.linalg.norm(np.diff(cams, axis=0), axis=1)
    for i in range(len(cams) - 1):
        if step[i] < 10 * np.median(step) + 1e-9:
            pts += [cams[i], cams[i + 1]]; lines.append([len(pts) - 2, len(pts) - 1]); cols.append([0.95, 0.35, 0.2])
    for i in range(0, len(cams), max(1, len(cams) // 40)):           # small frusta
        Rw = cam_R[i].T; w, h = cam_K.width / 2, cam_K.height / 2; f = cam_K.mean_focal_length()
        corners = [cams[i] + Rw @ np.array([sx * w / f, sy * h / f, 1.0]) * s for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        b = len(pts); pts += [cams[i]] + corners
        lines += [[b, b + k] for k in range(1, 5)] + [[b + k, b + 1 + k % 4] for k in range(1, 5)]
        cols += [[1.0, 0.85, 0.3]] * 8
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)), o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols)); return ls


# ---------------------------------------------------------------- viewer
W, H = map(int, a.size.split('x'))
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window('voxview - ' + str(src), W, H, visible=not a.screenshot)
geo = {'vox': build(), 'traj': trajectory() if state['traj'] else None}
for g in geo.values():
    if g is not None: vis.add_geometry(g)


def style():
    ro = vis.get_render_option()
    ro.background_color = np.array([0.11, 0.12, 0.14]) if state['dark'] else np.array([0.94, 0.95, 0.97])
    ro.light_on = state['light']; ro.line_width = 2.0


def home(v=vis):
    vc = v.get_view_control(); ctr = (IJK.min(0) + IJK.max(0) + 1) / 2 * SIZE + ORIGIN
    f = np.array([float(x) for x in a.view.split(',')]); f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) - f[2] * f; up /= np.linalg.norm(up)    # up must be orthogonal to front, or the ruler's ray is off
    vc.set_lookat(ctr); vc.set_up(up); vc.set_front(f); vc.set_zoom(a.zoom)
    return False


def rebuild(v):
    v.remove_geometry(geo['vox'], reset_bounding_box=False); geo['vox'] = build()
    v.add_geometry(geo['vox'], reset_bounding_box=False); return True


z0 = ORIGIN[2]
def finer(v): state['times'] = max(1, state['times'] - 1); return rebuild(v)
def coarser(v): state['times'] += 1; return rebuild(v)
def cmode(v): state['cmode'] = (state['cmode'] + 1) % 4; return rebuild(v)
def cut_toggle(v): state['cut'] = None if state['cut'] is not None else z0 + 0.7 * (ztop - z0); return rebuild(v)
def cut_down(v): state['cut'] = max(z0 + 0.25, (ztop if state['cut'] is None else state['cut']) - 0.25); return rebuild(v)
def cut_up(v): state['cut'] = None if state['cut'] is None or state['cut'] + 0.25 >= ztop else state['cut'] + 0.25; return rebuild(v)
def traj(v):
    if cams is None: return False
    if geo['traj'] is None: geo['traj'] = trajectory()
    state['traj'] = not state['traj']
    (v.add_geometry if state['traj'] else v.remove_geometry)(geo['traj'], reset_bounding_box=False); return True
def bg(v): state['dark'] = not state['dark']; style(); return True
def light(v): state['light'] = not state['light']; style(); return True


# ---------------------------------------------------------------- ruler
# Open3D drops its own mouse handling once a mouse callback is registered,
# so rotation (left drag) and panning (right / middle / Ctrl + left drag) are done here.
mouse = {'x': 0.0, 'y': 0.0, 'drag': None}; ruler = {'pts': [], 'geo': []}


def on_move(v, x, y):
    dx, dy = mouse['x'] - x, y - mouse['y']; mouse['x'], mouse['y'] = x, y
    if mouse['drag'] is None: return False
    vc = v.get_view_control()
    if mouse['drag'] == 'rotate': vc.rotate(-dx, dy)
    elif mouse['drag'] == 'pan': vc.translate(-dx, dy)
    else:                                  # roll: turn the up vector around the viewing direction
        E = vc.convert_to_pinhole_camera_parameters().extrinsic
        f, up, k = -E[2, :3], -E[1, :3], np.radians(0.3 * dx)
        f /= np.linalg.norm(f); vc.set_up(up * np.cos(k) + np.cross(f, up) * np.sin(k))
    v.update_renderer(); return False      # False: only redraw; True would re-upload all voxels to the GPU


def pick(v):
    """The voxel under the mouse: a ray from the camera through the pixel, the nearest voxel centre within one voxel of it."""
    cam = v.get_view_control().convert_to_pinhole_camera_parameters()
    K, E = cam.intrinsic.intrinsic_matrix, cam.extrinsic
    d = np.linalg.inv(K) @ np.array([mouse['x'], mouse['y'], 1.0])
    U, _, Vt = np.linalg.svd(E[:3, :3]); R, t = U @ Vt, E[:3, 3]
    o = -R.T @ t; d = R.T @ d; d /= np.linalg.norm(d)
    C = state['centres']; along = (C - o) @ d
    off = np.linalg.norm((C - o) - along[:, None] * d, axis=1)
    hit = np.flatnonzero((along > 0) & (off < state['voxel']))
    return C[hit[np.argmin(along[hit])]] if len(hit) else None


def show_ruler(v):
    for g in ruler['geo']: v.remove_geometry(g, reset_bounding_box=False)
    ruler['geo'] = []
    for q in ruler['pts']:
        s = o3d.geometry.TriangleMesh.create_sphere(state['voxel'] * 1.2); s.translate(q)
        s.paint_uniform_color([1.0, 0.1, 0.6]); ruler['geo'].append(s)
    if len(ruler['pts']) == 2:
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(ruler['pts'])), o3d.utility.Vector2iVector([[0, 1]]))
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.1, 0.6]]); ruler['geo'].append(ls)
    for g in ruler['geo']: v.add_geometry(g, reset_bounding_box=False)


def on_click(v, button, action, mods):
    if action == 0: mouse['drag'] = None; return False                # button released
    if not (button == 0 and mods & 1):                                 # not Shift + left: start a drag
        mouse['drag'] = ('pan' if button != 0 or mods & 2 else 'roll' if mods & 4 else 'rotate'); return False
    q = pick(v)
    if q is None: print('ruler: no voxel under the mouse'); return False
    ruler['pts'] = [q] if len(ruler['pts']) == 2 else ruler['pts'] + [q]     # a third click starts a new one
    if len(ruler['pts']) == 2:
        dv = ruler['pts'][1] - ruler['pts'][0]
        print(f"ruler: {fmt(np.linalg.norm(dv))}  (horizontal {fmt(np.hypot(dv[0], dv[1]))}, vertical {dv[2]:+.2f}; voxel {state['voxel']:.3f})")
    else:
        print(f"ruler: first point at height {fmt(q[2])}; Shift + click the second")
    show_ruler(v); return True


def clear_ruler(v): ruler['pts'] = []; show_ruler(v); return True
def helpk(v): print(__doc__.split('Keys:')[1]); return False


for k, f in {'[': finer, ']': coarser, 'C': cmode, 'K': cut_toggle, ',': cut_down, '.': cut_up, 'T': traj, 'B': bg, 'L': light, 'R': home,
             'H': helpk, 'X': clear_ruler}.items():
    vis.register_key_callback(ord(k), f)
vis.register_mouse_move_callback(on_move); vis.register_mouse_button_callback(on_click)
style(); home()

if a.screenshot:
    for _ in range(3): vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(a.screenshot, do_render=True); vis.destroy_window(); print('wrote', a.screenshot); sys.exit()
print('keys: [ ] voxel size | C colour | K cut the top off, , . move the cut | T path | B background | L light | R reset | H help\n'
      'ruler: Shift + left click on two voxels | X clear')
vis.run(); vis.destroy_window()
