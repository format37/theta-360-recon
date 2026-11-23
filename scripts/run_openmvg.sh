#!/usr/bin/env bash
# The comparison run: OpenMVG with its native spherical camera on the same panoramas, no faces.
#   usage: scripts/run_openmvg.sh data/pass-0/spherical.yaml
# Needs the docker image from tools/openmvg (docker build -t openmvg:local tools/openmvg).
# Reads the session's panoramas (photos: in the YAML) and the masks of the rig run: one mask per panorama
# (out/rig/masks_equirect/<shot>.png: people, sky, platform) when it made them, else the one rover mask
# (out/rig/product/mask_equirect.png). Writes out/<variant>/openmvg/{matches/, sfm/sfm_data.json, logs_*.txt}.
# PAIRS=rig: match only the panorama pairs the rig run matched (out/rig/product/pairs.json), not every pair.
#
# Steps: list images with camera model 7 (spherical) -> SIFT on the whole equirectangular picture ->
# pairs -> essential matrix on unit rays ("-g a", angular error) -> incremental SfM.
# A unit ray needs no intrinsics, so nothing about the camera is estimated.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=$(realpath "${1:?usage: run_openmvg.sh SESSION.yaml}")
eval "$(python3 "$T/project.py" "$Y")"     # RUN, SESSION, PHOTOS_GLOB
O="$RUN/openmvg"; rm -rf "$O"; mkdir -p "$O/images" "$O/matches" "$O/sfm"
python3 -I - "$Y" "$T" "$O/images" <<'EOF'
import os, shutil, sys; sys.path.insert(0, sys.argv[2])
from project import load
p = load(sys.argv[1]); dst = sys.argv[3]
for f in p.photos:
    try: os.link(f, os.path.join(dst, f.name))
    except OSError: shutil.copy(f, dst)
per_shot = p.session / 'out' / 'rig' / 'masks_equirect'; one = p.session / 'out' / 'rig' / 'product' / 'mask_equirect.png'
if per_shot.is_dir():                                    # OpenMVG: <image>_mask.png next to the image = a mask for that image
    for f in p.photos: shutil.copy(per_shot / f'{f.stem}.png', os.path.join(dst, f'{f.stem}_mask.png'))
    print(f'{len(p.photos)} panoramas, one mask each from {per_shot}')
elif one.exists():                                       # mask.png in the image folder = all images
    shutil.copy(one, os.path.join(dst, 'mask.png')); print(f'{len(p.photos)} panoramas, rover mask {one}')
EOF
ts() { date +%H:%M:%S; }
run() { docker run --rm -u "$(id -u):$(id -g)" -v "$O":/w openmvg:local "$@"; }

echo "$(ts) listing $(ls "$O/images"/*.JPG | wc -l) panoramas as spherical cameras"
run openMVG_main_SfMInit_ImageListing -i /w/images -o /w/matches -c 7 -f 1 > "$O/logs_list.txt" 2>&1
echo "$(ts) SIFT"
run openMVG_main_ComputeFeatures -i /w/matches/sfm_data.json -o /w/matches -m SIFT -p HIGH -n "$(nproc)" > "$O/logs_features.txt" 2>&1
if [ "${PAIRS:-all}" = rig ]; then
  python3 -I - "$RUN/../rig/product/pairs.json" "$O/matches/sfm_data.json" > "$O/matches/pairs.txt" <<'EOF'
import json, sys
sel = json.load(open(sys.argv[1]))['panorama_pairs']; views = json.load(open(sys.argv[2]))['views']
idx = {v['value']['ptr_wrapper']['data']['filename'].rsplit('.', 1)[0]: v['key'] for v in views}
for a, b in sel: print(min(idx[a], idx[b]), max(idx[a], idx[b]))
EOF
  PAIRFILE=/w/matches/pairs.txt; echo "$(ts) matching $(wc -l < "$O/matches/pairs.txt") pairs of the rig run"
else
  run openMVG_main_PairGenerator -i /w/matches/sfm_data.json -o /w/matches/pairs.bin > "$O/logs_match.txt" 2>&1
  PAIRFILE=/w/matches/pairs.bin; echo "$(ts) matching every pair"
fi
run openMVG_main_ComputeMatches -i /w/matches/sfm_data.json -p "$PAIRFILE" -o /w/matches/matches.putative.bin >> "$O/logs_match.txt" 2>&1
run openMVG_main_GeometricFilter -i /w/matches/sfm_data.json -m /w/matches/matches.putative.bin -g a -o /w/matches/matches.f.bin >> "$O/logs_match.txt" 2>&1
echo "$(ts) incremental SfM"
run openMVG_main_SfM -s INCREMENTAL -i /w/matches/sfm_data.json -m /w/matches -o /w/sfm > "$O/logs_sfm.txt" 2>&1
run openMVG_main_ConvertSfM_DataFormat -i /w/sfm/sfm_data.bin -o /w/sfm/sfm_data.json >> "$O/logs_sfm.txt" 2>&1
echo "$(ts) done"; grep -E "#Camera|#Poses|#Tracks|RMSE|views|poses|landmarks" "$O/logs_sfm.txt" | tail -12
