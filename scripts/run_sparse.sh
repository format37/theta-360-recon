#!/usr/bin/env bash
# Sparse SfM on the faces: features -> matching -> rig mapper.
#   usage: scripts/run_sparse.sh data/pass-0/rig.yaml
#   env:   SKIP_MATCH=1  keep the database and its matches, run the mapper only
# Reads faces/ and masks/, writes colmap/{database.db, database_rig.db, pairs.txt, sparse/N, logs_*.txt}.
#
# What our COLMAP can do with a 360 camera:
#   COLMAP 3.11 CLI   no spherical camera model; rigs only in rig_bundle_adjuster, after an ordinary mapper run
#   pycolmap 3.13     rigs and frames are part of the mapper (since 3.12), still no spherical model, no CUDA in the wheel
# So: SIFT and matching on the GPU with the 3.11 CLI, mapping with pycolmap 3.13 on the same database.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=$(realpath "${1:?usage: run_sparse.sh SESSION.yaml}")
eval "$(python3 "$T/project.py" "$Y")"     # FACES, MASKS, WS, FOCAL, SIZE, FEATURES
mkdir -p "$WS"; cd "$WS"; ts() { date +%T; }

if [ "${SKIP_MATCH:-0}" != 1 ]; then
  rm -rf sparse database.db database_rig.db
  echo "$(ts) $(find "$FACES" -name '*.jpg' | wc -l) faces ${SIZE}x${SIZE}, focal $FOCAL px (fixed)"

  # 1. SIFT on the GPU. One PINHOLE camera per face folder with the exact focal length; masked pixels give no keypoints.
  echo "$(ts) features"
  colmap feature_extractor --database_path database.db --image_path "$FACES" \
    --ImageReader.camera_model PINHOLE --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_params "$FOCAL,$FOCAL,$((SIZE/2)),$((SIZE/2))" --ImageReader.mask_path "$MASKS" \
    --SiftExtraction.use_gpu 1 --SiftExtraction.max_image_size "$SIZE" \
    --SiftExtraction.max_num_features "$FEATURES" > logs_extract.txt 2>&1

  # 2. Matching. Faces of one panorama share their centre: a match between them has no baseline and tells
  #    nothing about the geometry, so they are never paired.
  if python3 -I -c "import sys; sys.path.insert(0, '$T'); from project import load; sys.exit(0 if load('$Y').match_window else 1)"; then
    echo "$(ts) choosing pairs: sequential + retrieval (product/pairs.json)"
    python3 -I "$T/pairs.py" "$Y" > pairs.txt
  else                                                   # every face against every face of every OTHER panorama
    python3 -I - "$FACES" > pairs.txt <<'EOF'
import itertools, pathlib, sys
f = sorted(str(p.relative_to(sys.argv[1])) for p in pathlib.Path(sys.argv[1]).glob('v*/*.jpg'))
for a, b in itertools.combinations(f, 2):
    if pathlib.Path(a).stem != pathlib.Path(b).stem: print(a, b)
EOF
  fi
  echo "$(ts) matching $(wc -l < pairs.txt) face pairs"
  colmap matches_importer --database_path database.db --match_list_path pairs.txt --match_type pairs \
    --SiftMatching.use_gpu 1 --SiftMatching.guided_matching 1 > logs_match.txt 2>&1
  python3 -I "$T/pairs.py" "$Y" --report
fi

# 3. Mapper with the rig: the 8 faces of a panorama are one frame with one pose.
echo "$(ts) rig mapper (CPU; panoramas in the model: grep -c 'Registering image' $WS/logs_mapper.txt)"
python3 -I "$T/rig_mapper.py" "$Y" > logs_mapper.txt 2>&1
ls -d sparse/*/ > /dev/null 2>&1 || { echo "no model: mapper failed, see $WS/logs_mapper.txt"; exit 1; }
grep '^model' logs_mapper.txt
echo "$(ts) done"
