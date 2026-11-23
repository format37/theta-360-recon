#!/usr/bin/env bash
# Progress of a run. In a second terminal: scripts/progress.sh data/pool/rig.yaml   (Ctrl+C to stop)
# It only reads the logs and folders of the run, so it cannot disturb it. From galaxy-s26-ultra-stereo.
eval "$(python3 "$(dirname "$0")/project.py" "${1:?usage: progress.sh SESSION.yaml}")"
count() { local n; n=$(grep -c "$1" "$2" 2>/dev/null); echo "${n:-0}"; }
while true; do
  F=$(find "$FACES" -name '*.jpg' 2>/dev/null | wc -l)
  f=$(count "Processed file" "$WS/logs_extract.txt")
  m=$(grep -a "Matching block" "$WS/logs_match.txt" 2>/dev/null | tail -1 | sed -E 's/.*\[([0-9]+\/[0-9]+)\].*/\1/')
  k=$(count "Registering image" "$WS/logs_mapper.txt")
  D=$(($(grep -c . "$DENSE/stereo/patch-match.cfg" 2>/dev/null || echo 0) / 2))
  d=$(count "Processing view" "$DENSE/logs_patchmatch.txt")
  s=$(grep -h "^step" "$RUN"/logs/*splat*.log 2>/dev/null | tail -1 | awk '{print $2}')
  printf '\r%s  faces %d   features %d/%d   matching %s   mapper %d/%d panoramas   dense %d/%d   splat step %s   ' \
    "$(date +%T)" "$F" "$f" "$F" "${m:--}" "$k" "$NPHOTOS" "$d" "$((2 * D))" "${s:--}"
  sleep 10
done
