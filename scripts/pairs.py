#!/usr/bin/env python3
"""Which faces to match, and afterwards: what the matching found.

    python3 -I scripts/pairs.py data/pool/rig.yaml > colmap/pairs.txt      # after feature extraction
    python3 -I scripts/pairs.py data/pool/rig.yaml --report               # after matches_importer

Selection (matching.window set in the YAML; without it run_sparse.sh matches every face pair, as on pass-0).
Every panorama is compared with every other one on the GPU: the masked SIFT descriptors of its faces, 2048 per
face at random, mutual nearest neighbours that pass the ratio test (0.8). No geometry yet: a count of
putative matches, enough to rank. Selected panorama pairs:
  sequential  each panorama with the next matching.window ones
  retrieval   each panorama with its matching.retrieval most similar ones anywhere in the walk (at least 40
              putative matches): where the walk comes back to a place, this closes the loop
Of the 64 face pairs of a selected panorama pair only those with matching.min_face_matches putative matches,
or the 12 best of the pair, go to COLMAP: two faces looking in opposite directions share nothing. Faces of one
panorama are never paired: they share their centre, a match between them has no baseline.
Writes product/pairs.json; the pairs go to stdout.

Report: from the two-view geometries COLMAP verified (15 inliers or more): verified panorama pairs by distance
in the walk, loop closures (verified pairs beyond the window), and weak panoramas, which have no verified
neighbour within 3 shots on either side: there the chain is thin and the model may break.
Writes product/matches.json.
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from project import load

p = load(sys.argv[1]); report = '--report' in sys.argv
con = sqlite3.connect(f'file:{p.colmap / "database.db"}?mode=ro', uri=True)
names = dict(con.execute('select image_id, name from images'))
panos = sorted({Path(n).stem for n in names.values()}); pidx = {n: i for i, n in enumerate(panos)}
face_of = {i: int(n.split('/')[0][1:]) for i, n in names.items()}; pano_of = {i: pidx[Path(n).stem] for i, n in names.items()}
N, V = len(panos), len(p.views)
log = lambda *a: print(*a, file=sys.stderr, flush=True)

if not report:
    import torch
    rng = np.random.default_rng(0); K = 2048; dev = 'cuda'
    desc = [[] for _ in range(N)]; fid = [[] for _ in range(N)]
    for iid, rows, cols, data in con.execute('select image_id, rows, cols, data from descriptors'):
        if not rows: continue
        d = np.frombuffer(data, np.uint8).reshape(rows, cols)
        d = d[rng.choice(rows, min(K, rows), replace=False)]
        desc[pano_of[iid]].append(d); fid[pano_of[iid]].append(np.full(len(d), face_of[iid], np.int64))
    D = [torch.nn.functional.normalize(torch.from_numpy(np.concatenate(d)).to(dev).float(), dim=1).half() for d in desc]
    F = [torch.from_numpy(np.concatenate(f)).to(dev) for f in fid]
    count = np.zeros((N, N), np.int32); faces = {}
    for i in range(N):
        for j in range(i + 1, N):
            S = (D[i] @ D[j].T).float()                                   # cosine similarity; distance^2 = 2 - 2 s
            top, arg = S.topk(2, dim=1)
            d1, d2 = (2 - 2 * top[:, 0]).clamp_min(0).sqrt(), (2 - 2 * top[:, 1]).clamp_min(0).sqrt()
            back = S.argmax(0)
            a = torch.arange(len(S), device=dev)
            ok = (back[arg[:, 0]] == a) & (d1 < 0.8 * d2)
            count[i, j] = count[j, i] = int(ok.sum())
            faces[i, j] = torch.bincount(F[i][ok] * V + F[j][arg[ok, 0]], minlength=V * V).view(V, V).cpu().numpy()
        if (i + 1) % 24 == 0: log(f'  {i + 1}/{N} panoramas compared')
    W, R = int(p.match_window), p.match_retrieval
    sel = {}
    for i in range(N):
        for j in range(i + 1, min(N, i + W + 1)): sel[i, j] = 'sequential'
        far = [j for j in np.argsort(-count[i]) if abs(int(j) - i) > W and count[i, j] >= 40][:R]
        for j in far: sel.setdefault((min(i, int(j)), max(i, int(j))), 'retrieval')
    out = []; kept = []
    for (i, j), why in sorted(sel.items()):
        c = faces[i, j]; order = np.argsort(-c, axis=None)
        take = {(int(a), int(b)) for a, b in zip(*np.unravel_index(order[:12], c.shape)) if c[a, b] >= 2}
        take |= {(int(a), int(b)) for a, b in zip(*np.nonzero(c >= p.match_min))}
        kept.append(len(take))
        for a, b in sorted(take): out.append(f'v{a:02d}/{panos[i]}.jpg v{b:02d}/{panos[j]}.jpg')
    print('\n'.join(out))
    nseq = sum(w == 'sequential' for w in sel.values()); nret = len(sel) - nseq
    seq1 = [count[i, i + 1] for i in range(N - 1)]
    J = dict(panoramas=N, faces_per_panorama=V, descriptors_per_face=K, window=W, retrieval=R, min_face_matches=p.match_min,
             panorama_pairs_sequential=nseq, panorama_pairs_retrieval=nret, face_pairs=len(out),
             face_pairs_per_panorama_pair_median=float(np.median(kept)), face_pairs_if_all=len(sel) * V * V,
             putative_matches_next_shot=dict(median=int(np.median(seq1)), min=int(np.min(seq1)), argmin=panos[int(np.argmin(seq1))]),
             panorama_pairs=[[panos[i], panos[j]] for i, j in sorted(sel)], retrieval_pairs=[[panos[i], panos[j], int(count[i, j])] for (i, j), w in sorted(sel.items()) if w == 'retrieval'],
             putative_matches=count.tolist())
    json.dump(J, open(p.product / 'pairs.json', 'w'))
    log(f'{N} panoramas: putative matches with the next shot median {J["putative_matches_next_shot"]["median"]} (min {J["putative_matches_next_shot"]["min"]} '
        f'at {J["putative_matches_next_shot"]["argmin"]}); selected {nseq} sequential + {nret} retrieval panorama pairs -> {len(out):,} face pairs '
        f'(median {J["face_pairs_per_panorama_pair_median"]:.0f} of {V * V} per panorama pair; all of them would be {len(sel) * V * V:,})')
    sys.exit()

# ---------------------------------------------------------------- report
sel = json.load(open(p.product / 'pairs.json')) if (p.product / 'pairs.json').exists() else {}
win = sel.get('window') or 0
tried = con.execute('select count(*) from matches').fetchone()[0]
inl = defaultdict(int); nfp = defaultdict(int)
for pid, rows in con.execute('select pair_id, rows from two_view_geometries'):
    if rows < 15: continue
    i2 = pid % 2147483647; i1 = (pid - i2) // 2147483647
    a, b = sorted((pano_of[i1], pano_of[i2]))
    inl[a, b] += rows; nfp[a, b] += 1
near = defaultdict(set)
for a, b in inl:
    if b - a <= 3: near[a].add(b); near[b].add(a)
by_gap = defaultdict(list)
for (a, b), r in inl.items(): by_gap[b - a].append(r)
loops = sorted(((a, b, r) for (a, b), r in inl.items() if b - a > max(win, 12)), key=lambda x: -x[2])
weak = [panos[i] for i in range(N) if not near[i]]
R = dict(face_pairs_tried=tried, panorama_pairs_verified=len(inl), face_pairs_verified=int(sum(nfp.values())),
         verified_by_shots_apart={int(d): dict(pairs=len(v), inliers_median=int(np.median(v))) for d, v in sorted(by_gap.items()) if d <= 12 or d % 10 == 0},
         loop_closures=len(loops), loop_closures_top=[[panos[a], panos[b], r] for a, b, r in loops[:30]],
         panoramas_in_a_loop_closure=len({x for a, b, _ in loops for x in (a, b)}), weak_panoramas=weak)
json.dump(R, open(p.product / 'matches.json', 'w'), indent=1)
g1 = by_gap.get(1, [0])
print(f"face pairs tried {tried:,}; verified (15+ inliers): {R['face_pairs_verified']:,} face pairs in {len(inl):,} panorama pairs; "
      f"next shot: {len(by_gap.get(1, []))}/{N - 1} verified, median {int(np.median(g1))} inliers summed over the face pairs")
print(f"loop closures (verified pairs over {max(win, 12)} shots apart): {len(loops)}, touching {R['panoramas_in_a_loop_closure']}/{N} panoramas; strongest: "
      + ', '.join(f'{panos[a][-4:]}-{panos[b][-4:]} ({r})' for a, b, r in loops[:8]))
print(f'{len(weak)} weak panoramas (no verified neighbour within 3 shots)' + (': ' + ' '.join(weak) if weak else ''))
