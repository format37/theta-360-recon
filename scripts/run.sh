#!/usr/bin/env bash
# The whole way from 360 photos to a mesh, a voxel model and a splat, for one session YAML.
#
#   scripts/run.sh data/pool/rig.yaml 2>&1 | tee data/pool/out/rig/run.log
#   FROM=dense scripts/run.sh data/pool/rig.yaml      # start at a later step: faces sparse align dense mesh voxels splat report
#   scripts/progress.sh data/pool/rig.yaml            # in a second terminal
#
# Every line below is a script that can also be run on its own with the same YAML.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd); Y=${1:?usage: run.sh SESSION.yaml}
STEPS=(faces sparse align dense mesh voxels splat report); FROM=${FROM:-faces}; go=0
on() { [ "$1" = "$FROM" ] && go=1; [ $go = 1 ]; }
[[ " ${STEPS[*]} " == *" $FROM "* ]] || { echo "FROM must be one of: ${STEPS[*]}"; exit 1; }

if on faces; then
  python3 -I "$T/split_faces.py" "$Y"       # 2, 3. masks (platform, people, sky), faces of the rig -> faces/, masks/, labels/, product/rig.json
  python3 -I "$T/inspect_panos.py" "$Y"     # 1. what the files are, overlap of neighbours -> product/inspect.json
fi
if on sparse; then
  "$T/run_sparse.sh" "$Y"                   # features, pairs, matching, rig mapper -> colmap/sparse/0, product/pairs.json, matches.json
fi
if on align; then
  python3 -I "$T/sfm_report.py" "$Y"        # registered, error in px and degrees, seam, path -> product/sfm.json
  python3 -I "$T/align_scale.py" "$Y"       # 5. up, ground (floor), axes, lens height -> colmap/aligned, product/align.json
fi
if on dense; then
  "$T/run_dense.sh" "$Y"                    # PatchMatch on every Nth panorama -> dense/fused.ply
fi
if on mesh; then
  python3 -I "$T/depth_prior.py" "$Y"       # 4. monocular depth tied to PatchMatch -> dense/prior/
  python3 -I "$T/mesh.py" "$Y"              # 4. planes, filled depth, coverage, TSDF -> product/mesh.ply, product/mesh.json
fi
if on voxels; then
  python3 -I "$T/voxels.py" "$Y"            # 6. the voxel model -> product/voxels.npz, voxels_cubes.ply   (look: scripts/voxview.py)
fi
if on splat; then
  python3 -I "$T/train_splat.py" "$Y"       # -> splat/splat.ply, splat/metrics.json
fi
if on report; then
  python3 -I "$T/export_view.py" "$Y"       # light copies for the web viewers -> splat/splat_view.ply, product/mesh_view.ply
  if python3 -I -c "import sys; sys.path.insert(0, '$T'); from project import load; sys.exit(0 if load('$Y').cfg.get('crop') else 1)"; then
    python3 -I "$T/crop.py" "$Y"            # the building alone (crop: in the YAML) -> product/*_building.ply, splat/splat_building.ply
  fi
  python3 -I "$T/render_views.py" "$Y"      # pictures -> product/renders/
  python3 -I "$T/measure.py" "$Y"           # 5. lengths to check with a tape, scale error -> product/measure.json, plan.png
  if python3 -I -c "import sys; sys.path.insert(0, '$T'); from project import load; sys.exit(0 if load('$Y').cfg.get('compare') else 1)"; then
    python3 -I "$T/compare.py" "$Y"         # this run next to the phone run -> product/compare.md (pass-0)
  fi
  python3 -I "$T/summary.py" "$Y"           # every number of the run on one page -> out/<variant>/summary.md
fi
