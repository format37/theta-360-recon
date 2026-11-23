#!/usr/bin/env bash
# Serve the project on this machine only and open the splat viewer. Ctrl+C stops the server.
# The pages cannot be opened as files (double click): a browser does not let a file:// page read the PLY.
#   scripts/view.sh                               # the rig run of pass-0
#   RUN=data/pool/out/rig scripts/view.sh         # another run
#   PORT=9000 scripts/view.sh
# The voxel model also has a native window: python3 scripts/voxview.py data/pool/rig.yaml
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
PORT=${PORT:-8766}; BIND=${BIND:-127.0.0.1}; RUN=${RUN:-data/pass-0/out/rig}
echo "splat: http://127.0.0.1:$PORT/viewer/splat.html?run=$RUN"
for d in "$RUN"/splat_*/; do [ -f "$d/splat_view.ply" ] && echo "       http://127.0.0.1:$PORT/viewer/splat.html?run=$RUN&splat=$(basename "$d")"; done
echo "mesh:  http://127.0.0.1:$PORT/viewer/mesh.html?run=$RUN"
echo "       http://127.0.0.1:$PORT/viewer/mesh.html?run=$RUN&mesh=mesh_mvs.ply      (PatchMatch only, no prior)"
[ -f "$RUN/product/voxels_cubes.ply" ] && echo "       http://127.0.0.1:$PORT/viewer/mesh.html?run=$RUN&mesh=voxels_cubes.ply  (the voxel model as cubes)"
[ -f "$RUN/product/mesh_building.ply" ] && echo "       http://127.0.0.1:$PORT/viewer/mesh.html?run=$RUN&mesh=mesh_building.ply (the building alone)"
[ -f "$RUN/product/voxels_cubes_building.ply" ] && echo "       http://127.0.0.1:$PORT/viewer/mesh.html?run=$RUN&mesh=voxels_cubes_building.ply (its voxels)"
[ -f "$RUN/splat/splat_building.ply" ] && echo "splat, the building alone: http://127.0.0.1:$PORT/viewer/splat.html?run=$RUN&ply=../$RUN/splat/splat_building.ply"
if [ "$BIND" = "127.0.0.1" ] && [ -z "$NO_OPEN" ]; then
  ( sleep 1; xdg-open "http://127.0.0.1:$PORT/viewer/splat.html?run=$RUN" >/dev/null 2>&1 ) &
fi
exec python3 -m http.server "$PORT" --bind "$BIND"
