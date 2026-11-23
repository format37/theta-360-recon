#!/usr/bin/env bash
# Dense cloud with COLMAP PatchMatch on a subset of the faces.
#   usage: scripts/run_dense.sh data/pass-0/rig.yaml
# dense.every in the YAML: every Nth panorama. The rover moves a few cm per shot; neighbours that close
# only add depth noise (drone-recon: more frames is not a better cloud).
# dense.max_sky: faces whose labels (split_faces.py, mask.semantic) are more sky than this get no depth map of
# their own: PatchMatch cannot be masked and would spend the same time on them for nothing.
# dense.keep_maps: false deletes the photometric depth maps and all normal maps once the cloud is fused (28 of the
# 32 MB of a 1024 px face: 7/8); depth_prior.py and mesh.py read only the geometric depth maps.
# Reads colmap/aligned and faces/, writes dense/: images/, masks/, stereo/depth_maps/, fused.ply.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=$(realpath "${1:?usage: run_dense.sh SESSION.yaml}")
eval "$(python3 "$T/project.py" "$Y")"     # FACES, MASKS, WS, DENSE, DENSE_SIZE, DENSE_SOURCES, DENSE_KEEP_MAPS
ts() { date +%H:%M:%S; }
rm -rf "$DENSE"; mkdir -p "$DENSE"

# 1. the subset as its own model: frames of the other panoramas are taken out
python3 -I - "$Y" "$T" <<'EOF'
import sys; sys.path.insert(0, sys.argv[2])
from pathlib import Path
import cv2, numpy as np, pycolmap
from project import load
p = load(sys.argv[1])
rec = pycolmap.Reconstruction(str(p.colmap / 'aligned'))
panos = sorted({Path(im.name).stem for im in rec.images.values() if im.has_pose})
keep = set(panos[::p.dense_every])
for fid in sorted({im.frame_id for im in rec.images.values() if im.has_pose and Path(im.name).stem not in keep}):
    rec.deregister_frame(fid)
out = p.dense / 'subset'; out.mkdir(parents=True); rec.write(str(out))
n, skip = 0, []
for im in rec.images.values():
    if not im.has_pose: continue
    m = cv2.imread(str(p.masks / (im.name + '.png')), cv2.IMREAD_GRAYSCALE)
    d = p.dense / 'masks' / (im.name + '.png'); d.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d), cv2.resize(m, (p.dense_size, p.dense_size), interpolation=cv2.INTER_NEAREST)); n += 1
    lab = p.labels / (str(Path(im.name).with_suffix('')) + '.png')
    if p.dense_max_sky < 1 and lab.exists() and (cv2.imread(str(lab), cv2.IMREAD_GRAYSCALE) == 1).mean() > p.dense_max_sky: skip.append(im.name)
(p.dense / 'skip.txt').write_text(''.join(s + '\n' for s in sorted(skip)))
print(f'dense subset: {len(keep)} of {len(panos)} panoramas, {n} faces; {len(skip)} of them more than {100 * p.dense_max_sky:.0f}% sky get no depth map')
EOF

# 2. workspace: the faces are already distortion-free pinholes, this only resizes them
echo "$(ts) workspace at $DENSE_SIZE px"
colmap image_undistorter --image_path "$FACES" --input_path "$DENSE/subset" --output_path "$DENSE" \
  --output_type COLMAP --max_image_size "$DENSE_SIZE" > "$DENSE/logs_undistort.txt" 2>&1
sed -i "s/^__auto__, [0-9]*$/__auto__, $DENSE_SOURCES/" "$DENSE/stereo/patch-match.cfg"
# COLMAP 3.11 resizes the images of a pinhole camera here but leaves the camera itself at the old size
# (UndistortReconstruction skips cameras that have no distortion). PatchMatch hides it by rescaling to
# max_image_size; fusion then reads colours at the wrong place. Make the cameras match the images.
python3 -I - "$DENSE" "$DENSE_SOURCES" <<'EOF2'
import sys, cv2, glob, pycolmap
d = sys.argv[1]; h, w = cv2.imread(sorted(glob.glob(d + '/images/*/*.jpg'))[0]).shape[:2]
rec = pycolmap.Reconstruction(d + '/sparse')
for c in rec.cameras.values():
    if (c.width, c.height) != (w, h): c.rescale(w, h)
rec.write(d + '/sparse'); print(f'workspace cameras: {w}x{h}')
# The skipped faces get no depth map, are not fused and are nobody's source. Sources are then listed by name,
# chosen as __auto__ does: the images sharing the most sparse points at a triangulation angle of 1 deg or more
# (75th percentile over the shared points), here among the faces that are not skipped.
skip = set(open(d + '/skip.txt').read().split())
if skip:
    import numpy as np
    n_src = int(sys.argv[2])
    ims = {i: im for i, im in rec.images.items() if im.has_pose}
    pts = {i: {q.point3D_id for q in im.points2D if q.has_point3D()} for i, im in ims.items()}
    C = {i: im.projection_center() for i, im in ims.items()}
    use = sorted((i for i in ims if ims[i].name not in skip), key=lambda i: ims[i].name)
    out, none = [], []
    for i in use:
        cand = []
        for j in use:
            sh = pts[i] & pts[j] if j != i else ()
            if len(sh) < 10: continue
            X = np.array([rec.points3D[k].xyz for k in sh]); a = X - C[i]; b = X - C[j]
            ang = np.degrees(np.arccos(np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1, 1)))
            if np.percentile(ang, 75) >= 1.0: cand.append((len(sh), ims[j].name))
        src = [nm for _, nm in sorted(cand, reverse=True)[:n_src]]
        if src: out += [ims[i].name, ', '.join(src)]
        else: none.append(ims[i].name)
    open(d + '/stereo/patch-match.cfg', 'w').write('\n'.join(out) + '\n')
    open(d + '/stereo/fusion.cfg', 'w').write('\n'.join(out[0::2]) + '\n')
    print(f'{len(out) // 2} faces get a depth map from up to {n_src} named sources; {len(skip)} skipped as sky, {len(none)} without a source')
EOF2

# 3. PatchMatch stereo, photometric then geometric pass
N=$(($(grep -c . "$DENSE/stereo/patch-match.cfg") / 2))
echo "$(ts) patch-match stereo, $N faces x 2 passes (grep -c 'Processing view' $DENSE/logs_patchmatch.txt)"
colmap patch_match_stereo --workspace_path "$DENSE" --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size "$DENSE_SIZE" > "$DENSE/logs_patchmatch.txt" 2>&1

# 4. fusion; masked pixels (rover, people, sky) are left out
echo "$(ts) fusion"
colmap stereo_fusion --workspace_path "$DENSE" --workspace_format COLMAP --input_type geometric \
  --output_path "$DENSE/fused.ply" --StereoFusion.min_num_pixels 3 --StereoFusion.mask_path "$DENSE/masks" \
  --StereoFusion.use_cache 1 --StereoFusion.cache_size 16 > "$DENSE/logs_fusion.txt" 2>&1
echo "$(ts) done: $(grep -a -m1 'element vertex' "$DENSE/fused.ply" | cut -d' ' -f3) points in $DENSE/fused.ply"
if [ "$DENSE_KEEP_MAPS" = 0 ]; then
  before=$(du -sh "$DENSE/stereo" | cut -f1)
  find "$DENSE/stereo/depth_maps" -name '*.photometric.bin' -delete; rm -rf "$DENSE/stereo/normal_maps"
  echo "$(ts) maps trimmed: stereo/ $before -> $(du -sh "$DENSE/stereo" | cut -f1) (geometric depth maps kept)"
fi
