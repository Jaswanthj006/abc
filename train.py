#!/usr/bin/env python3
"""DriftLoc: find the 100x reference patch inside the 10x search image.

Three stages in one file. Run them in order; each stage prints its own
validation before you move on.

  stage1  PROPOSALS  multi-scale (8..12) x rotation (-3..+3 deg) ZNCC,
                     top-20 candidate windows per pair.
                     Validates: candidate recall on output/val.
  stage2  VERIFIER   small CNN that picks the true window among the ZNCC
                     candidates. Hard negatives = the other ZNCC peaks.
                     Validates: end-to-end pass@5 on val vs the ZNCC baseline.
  stage3  FULL RUN   proposals + verifier + sub-pixel on a held-out split.
                     Validates: pass@5/4/2/1 per noise bucket, PR curves,
                     runtime, failure panels.

Why this design (measured on output/val, 250 pairs, before writing this):
  - global ZNCC max: pass@5 ~55% (DRAM repeats every ~10 px -> wrong cell)
  - true window inside top-20 ZNCC peaks: ~85-90%   <- proposals are enough
  - ECC re-scoring 48%, multi-map consensus 27%     <- classical ranking fails
  - when the right cell is picked, sub-pixel error is already <1 px
  So the only open problem is choosing among ~20 look-alike windows.
  That is what the verifier learns, from manifest.csv labels only.

Commands (run manually, in order):

  python train.py stage1 --data output
  python train.py stage2 --data output --out model
  python train.py stage3 --data output --out model --split eval
  python train.py localize --reference REF.png --search SEARCH.png --out model

`localize` prints "x y" for one pair (sponsor interface). Without a trained
checkpoint it falls back to plain ZNCC + sub-pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Geometry / pipeline constants
# ---------------------------------------------------------------------------
SCALES = (8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0)  # ±20% around 10×
ANGLES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)              # template rotation (deg)
PEAKS_PER_MAP = 8                          # local maxima kept per ZNCC map
TOP_N = 20                                 # candidates kept per pair after NMS
NMS_RADIUS = 3.0                           # px between distinct candidates
POS_RADIUS = 5.0                           # candidate <= this from GT = positive
CROP = 128                                 # verifier input resolution
K_TRAIN = 16                               # positive + 15 hard negatives
SEARCH_CENTER = (500.0, 500.0)
TIE_PROB_GAP = 0.05                        # spec tie-break: near-ties only

PASS_THRESHOLDS = (1, 2, 4, 5)


# ---------------------------------------------------------------------------
# Stage 1: ZNCC proposal generator
# ---------------------------------------------------------------------------
def rotate_template(t: np.ndarray, ang: float) -> np.ndarray:
    if ang == 0.0:
        return t
    h, w = t.shape
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), ang, 1.0)
    return cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def local_peaks(R: np.ndarray, k: int, nsize: int = 5):
    """Top-k local maxima of a ZNCC map -> [(xi, yi, score)]."""
    ker = np.ones((nsize, nsize), np.uint8)
    dil = cv2.dilate(R, ker)
    ys, xs = np.where(R >= dil)
    if len(xs) == 0:
        return []
    ss = R[ys, xs]
    order = np.argsort(-ss)[:k]
    return [(int(xs[i]), int(ys[i]), float(ss[i])) for i in order]


def propose(ref: np.ndarray, sea: np.ndarray, top_n: int = TOP_N):
    """Multi-scale/rotation ZNCC. Returns (candidates, maps).

    candidate: dict(cx, cy, xi, yi, score, scale, angle, map_idx)
    maps[i]:   dict(R, tw, scale, angle, tmpl)  -- kept for sub-pixel + crops
    """
    maps = []
    raw = []
    for sc in SCALES:
        tw = int(round(1000.0 / sc))
        base = cv2.resize(ref, (tw, tw), interpolation=cv2.INTER_AREA)
        for ang in ANGLES:
            tmpl = rotate_template(base, ang)
            R = cv2.matchTemplate(sea, tmpl, cv2.TM_CCOEFF_NORMED)
            mi = len(maps)
            maps.append({"R": R, "tw": tw, "scale": sc, "angle": ang, "tmpl": tmpl})
            for xi, yi, s in local_peaks(R, PEAKS_PER_MAP):
                raw.append({"cx": xi + tw / 2.0, "cy": yi + tw / 2.0,
                            "xi": xi, "yi": yi, "score": s,
                            "scale": sc, "angle": ang, "map_idx": mi})
    raw.sort(key=lambda c: -c["score"])
    kept = []
    for c in raw:
        if all((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2 > NMS_RADIUS ** 2
               for k in kept):
            kept.append(c)
        if len(kept) >= top_n:
            break
    return kept, maps


def subpixel(R: np.ndarray, xi: int, yi: int):
    """1D parabolic fit around an integer ZNCC peak -> (dx, dy) in [-1, 1]."""
    dx = dy = 0.0
    if 0 < xi < R.shape[1] - 1:
        l, c, r = float(R[yi, xi - 1]), float(R[yi, xi]), float(R[yi, xi + 1])
        d = l - 2 * c + r
        if abs(d) > 1e-9:
            dx = float(np.clip(0.5 * (l - r) / d, -1, 1))
    if 0 < yi < R.shape[0] - 1:
        u, c, w = float(R[yi - 1, xi]), float(R[yi, xi]), float(R[yi + 1, xi])
        d = u - 2 * c + w
        if abs(d) > 1e-9:
            dy = float(np.clip(0.5 * (u - w) / d, -1, 1))
    return dx, dy


def candidate_patch(sea: np.ndarray, cand: dict, maps: list) -> np.ndarray:
    """2-channel verifier input: (search window, matched template), CROPxCROP."""
    m = maps[cand["map_idx"]]
    tw = m["tw"]
    win = sea[cand["yi"]:cand["yi"] + tw, cand["xi"]:cand["xi"] + tw]
    win = cv2.resize(win, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    tmp = cv2.resize(m["tmpl"], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    return np.stack([win, tmp])  # uint8 (2, CROP, CROP)


def cand_scalars(cand: dict) -> list[float]:
    return [cand["score"], cand["scale"] / 10.0 - 1.0, cand["angle"] / 2.0]


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------
def find_split_dir(data_root: str, name: str) -> str:
    aliases = {"val": ("val", "validation"), "train": ("train",),
               "test": ("test",), "eval": ("eval",)}
    for cand in aliases.get(name, (name,)):
        p = os.path.join(data_root, cand)
        if os.path.isfile(os.path.join(p, "manifest.csv")):
            return p
    raise FileNotFoundError(f"no split '{name}' with manifest.csv under {data_root}")


def load_manifest(split_dir: str) -> list[dict]:
    with open(os.path.join(split_dir, "manifest.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty manifest in {split_dir}")
    return rows


def resolve_image(split_dir: str, p: str, kind: str) -> str:
    for cand in (p, os.path.join(split_dir, p),
                 os.path.join(split_dir, kind, os.path.basename(p))):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"cannot resolve {kind} image: {p}")


def load_pair(split_dir: str, row: dict):
    ref = cv2.imread(resolve_image(split_dir, row["reference_path"], "reference"),
                     cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(resolve_image(split_dir, row["search_path"], "search"),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise FileNotFoundError(f"unreadable pair id={row.get('id')}")
    return ref, sea


def pass_table(errs) -> dict:
    e = np.asarray(errs, dtype=np.float64)
    out = {f"pass@{t}": float((e <= t).mean()) for t in PASS_THRESHOLDS}
    out.update(median=float(np.median(e)), mean=float(e.mean()),
               worst=float(e.max()), n=int(len(e)))
    return out


def print_bucket_table(per_bucket: dict, title: str):
    print(f"\n  {title}")
    hdr = f"  {'bucket':<14} {'n':>4} " + " ".join(f"p@{t:<2}" for t in PASS_THRESHOLDS) \
          + "  median   mean  worst"
    print(hdr)
    for b, m in per_bucket.items():
        cells = " ".join(f"{m[f'pass@{t}']:.2f}" for t in PASS_THRESHOLDS)
        print(f"  {b:<14} {m['n']:>4} {cells}  {m['median']:6.1f} {m['mean']:6.1f} {m['worst']:6.0f}")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1 command: validate proposal recall
# ---------------------------------------------------------------------------
def stage1(args):
    split_dir = find_split_dir(args.data, args.split)
    rows = load_manifest(split_dir)
    if args.limit:
        rows = rows[:args.limit]
    print(f"stage1: proposal recall on {args.split} ({len(rows)} pairs)")
    print(f"  grid: scales {SCALES} x angles {ANGLES}, top-{TOP_N} after {NMS_RADIUS}px NMS")

    recs, times = [], []
    for row in tqdm(rows):
        ref, sea = load_pair(split_dir, row)
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        t0 = time.time()
        cands, maps = propose(ref, sea)
        times.append(time.time() - t0)
        errs = [np.hypot(c["cx"] - gx, c["cy"] - gy) for c in cands]
        recs.append({"bucket": row.get("noise_bucket", "?"),
                     "best": min(errs), "top1": errs[0], "n": len(cands)})

    buckets = sorted({r["bucket"] for r in recs})
    print(f"\n  {'bucket':<14} {'n':>4}  recall@{TOP_N} (<=5px)  zncc-top1 p@5")
    for b in ["ALL"] + buckets:
        sub = recs if b == "ALL" else [r for r in recs if r["bucket"] == b]
        rec = np.mean([r["best"] <= POS_RADIUS for r in sub])
        top1 = np.mean([r["top1"] <= POS_RADIUS for r in sub])
        print(f"  {b:<14} {len(sub):>4}  {rec:16.3f}  {top1:13.3f}")
    print(f"\n  avg proposal time: {np.mean(times):.2f}s/pair")
    print("  PASS if recall@20 is ~0.85 overall: the answer is on the shortlist;")
    print("  the top1 column is the ceiling of any single-peak method (stage2 fixes it).")


# ---------------------------------------------------------------------------
# Stage 2: dump candidates, train verifier
# ---------------------------------------------------------------------------
def dump_candidates(data_root: str, split: str, out_dir: Path,
                    limit: int | None, redump: bool):
    npy_path = out_dir / f"cands_{split}.npy"
    csv_path = out_dir / f"cands_{split}.csv"
    if npy_path.is_file() and csv_path.is_file() and not redump:
        print(f"  reusing {npy_path.name} / {csv_path.name} (use --redump to rebuild)")
        return npy_path, csv_path

    split_dir = find_split_dir(data_root, split)
    rows = load_manifest(split_dir)
    if limit:
        rows = rows[:limit]
    print(f"  dumping candidates for {split} ({len(rows)} pairs)")

    crops = np.empty((len(rows) * TOP_N, 2, CROP, CROP), dtype=np.uint8)
    meta, count = [], 0
    for pi, row in enumerate(tqdm(rows)):
        ref, sea = load_pair(split_dir, row)
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        cands, maps = propose(ref, sea)
        errs = [float(np.hypot(c["cx"] - gx, c["cy"] - gy)) for c in cands]
        pos_i = int(np.argmin(errs)) if errs and min(errs) <= POS_RADIUS else -1
        for ci, (c, err) in enumerate(zip(cands, errs)):
            crops[count] = candidate_patch(sea, c, maps)
            meta.append({
                "row": count, "pair": pi, "id": row.get("id", pi),
                "bucket": row.get("noise_bucket", "?"),
                "cx": c["cx"], "cy": c["cy"], "score": c["score"],
                "scale": c["scale"], "angle": c["angle"], "err": err,
                "is_pos": int(ci == pos_i),
                # other candidates inside POS_RADIUS are neither pos nor neg
                "is_ignore": int(ci != pos_i and err <= POS_RADIUS),
                "gt_x": gx, "gt_y": gy,
            })
            count += 1

    np.save(npy_path, crops[:count])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    n_pos = sum(m["is_pos"] for m in meta)
    print(f"  saved {count} candidates ({n_pos}/{len(rows)} pairs have a positive)")
    return npy_path, csv_path


def read_cand_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("row", "pair", "is_pos", "is_ignore"):
            r[k] = int(r[k])
        for k in ("cx", "cy", "score", "scale", "angle", "err", "gt_x", "gt_y"):
            r[k] = float(r[k])
    return rows


def group_pairs(meta: list[dict]) -> list[dict]:
    pairs = {}
    for m in meta:
        pairs.setdefault(m["pair"], []).append(m)
    out = []
    for pi in sorted(pairs):
        cands = pairs[pi]
        pos = next((c for c in cands if c["is_pos"]), None)
        negs = [c for c in cands if not c["is_pos"] and not c["is_ignore"]]
        out.append({"pair": pi, "cands": cands, "pos": pos, "negs": negs,
                    "bucket": cands[0]["bucket"],
                    "gt": (cands[0]["gt_x"], cands[0]["gt_y"])})
    return out


class Verifier(nn.Module):
    """Scores one (search window, template) crop: is this the true location?"""

    def __init__(self):
        super().__init__()
        def block(ci, co, stride):
            return [nn.Conv2d(ci, co, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(
            *block(2, 32, 2),     # 64
            *block(32, 64, 2),    # 32
            *block(64, 64, 1),
            *block(64, 96, 2),    # 16
            *block(96, 128, 2),   # 8
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x, scalars):
        f = self.features(x).mean(dim=(2, 3))
        return self.head(torch.cat([f, scalars], dim=1)).squeeze(1)


def augment_channel(x: np.ndarray) -> np.ndarray:
    if random.random() < 0.7:
        x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0.0, 1.0)
    if random.random() < 0.5:
        x = np.clip(x + np.random.randn(*x.shape).astype(np.float32)
                    * random.uniform(0.005, 0.03), 0.0, 1.0)
    return x


class CandidateSet(Dataset):
    """One item = one training pair: positive crop first, then hard negatives."""

    def __init__(self, npy_path: Path, pairs: list[dict], train: bool):
        self.npy_path = npy_path
        self.pairs = [p for p in pairs if p["pos"] is not None and p["negs"]]
        self.train = train
        self._mm = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.load(self.npy_path, mmap_mode="r")
        p = self.pairs[i]
        n_neg = K_TRAIN - 1
        negs = (random.sample(p["negs"], n_neg) if len(p["negs"]) >= n_neg
                else p["negs"] + random.choices(p["negs"], k=n_neg - len(p["negs"])))
        rows = [p["pos"]] + negs
        crops = np.stack([self._mm[r["row"]] for r in rows]).astype(np.float32) / 255.0
        if self.train:
            for j in range(crops.shape[0]):
                for c in range(2):
                    crops[j, c] = augment_channel(crops[j, c])
        scal = np.array([[r["score"], r["scale"] / 10.0 - 1.0, r["angle"] / 2.0]
                         for r in rows], dtype=np.float32)
        return torch.from_numpy(crops), torch.from_numpy(scal)


@torch.no_grad()
def rank_validation(model, device, npy_path: Path, pairs: list[dict]):
    """Pick argmax-scored candidate per pair; compare against GT and ZNCC top1."""
    model.eval()
    mm = np.load(npy_path, mmap_mode="r")
    v_err, z_err, buckets = [], [], []
    for p in pairs:
        cands = p["cands"]
        crops = torch.from_numpy(
            np.stack([mm[c["row"]] for c in cands]).astype(np.float32) / 255.0)
        scal = torch.tensor([[c["score"], c["scale"] / 10.0 - 1.0, c["angle"] / 2.0]
                             for c in cands], dtype=torch.float32)
        logits = model(crops.to(device), scal.to(device)).cpu().numpy()
        pick = cands[int(np.argmax(logits))]
        ztop = max(cands, key=lambda c: c["score"])
        gx, gy = p["gt"]
        v_err.append(float(np.hypot(pick["cx"] - gx, pick["cy"] - gy)))
        z_err.append(float(np.hypot(ztop["cx"] - gx, ztop["cy"] - gy)))
        buckets.append(p["bucket"])
    return np.array(v_err), np.array(z_err), buckets


def stage2(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"stage2: verifier training  (device: {device.type}, out: {out_dir})")

    train_npy, train_csv = dump_candidates(args.data, "train", out_dir,
                                           args.limit, args.redump)
    val_npy, val_csv = dump_candidates(args.data, "val", out_dir, None, args.redump)

    train_pairs = group_pairs(read_cand_csv(train_csv))
    val_pairs = group_pairs(read_cand_csv(val_csv))
    train_ds = CandidateSet(train_npy, train_pairs, train=True)
    usable = len(train_ds)
    print(f"  train pairs: {len(train_pairs)} total, {usable} with a positive "
          f"({usable / max(len(train_pairs), 1):.0%}) -> "
          f"~{usable * (K_TRAIN - 1)} hard negatives per epoch")

    loader = DataLoader(train_ds, batch_size=args.batch_pairs, shuffle=True,
                        num_workers=0, drop_last=True)
    model = Verifier().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  verifier parameters: {n_par / 1e3:.0f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt_path = out_dir / "verifier.pt"
    hist_path = out_dir / "history.json"
    best_p5, history, since_best = -1.0, [], 0

    for epoch in range(args.epochs):
        model.train()
        t0, tot, nstep = time.time(), 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for crops, scal in pbar:
            B, K = crops.shape[:2]
            logits = model(crops.view(B * K, 2, CROP, CROP).to(device),
                           scal.view(B * K, 3).to(device)).view(B, K)
            # positive is always index 0 within its own candidate set
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long,
                                                       device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nstep += 1
            pbar.set_postfix(loss=f"{tot / nstep:.4f}")
        sched.step()

        v_err, z_err, buckets = rank_validation(model, device, val_npy, val_pairs)
        p5 = float((v_err <= 5).mean())
        zp5 = float((z_err <= 5).mean())
        print(f"  val end-to-end: verifier p@5={p5:.3f} med={np.median(v_err):.1f}px"
              f"  |  zncc-top1 p@5={zp5:.3f}  ({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "loss": tot / max(nstep, 1),
                        "val_pass@5": p5, "zncc_pass@5": zp5,
                        "val_median": float(np.median(v_err))})
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        if p5 > best_p5:
            best_p5, since_best = p5, 0
            torch.save({"model": model.state_dict(),
                        "config": {"crop": CROP, "scales": SCALES,
                                   "angles": ANGLES, "top_n": TOP_N}}, ckpt_path)
            print(f"  ** new best (val p@5 {p5:.3f}) -> {ckpt_path}")
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"  early stop: no val improvement for {args.patience} epochs")
                break

    # final per-bucket comparison with the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    v_err, z_err, buckets = rank_validation(model, device, val_npy, val_pairs)
    per_v = {"ALL": pass_table(v_err)}
    per_z = {"ALL": pass_table(z_err)}
    for b in sorted(set(buckets)):
        idx = [i for i, x in enumerate(buckets) if x == b]
        per_v[b] = pass_table(v_err[idx])
        per_z[b] = pass_table(z_err[idx])
    print_bucket_table(per_z, "val, ZNCC top-1 baseline (candidate-level)")
    print_bucket_table(per_v, "val, VERIFIER pick (candidate-level)")
    print(f"\n  best checkpoint: {ckpt_path}")
    print("  PASS if the verifier beats zncc-top1 clearly (target: p@5 ~0.75+).")
    print("  Then run stage3 for the held-out split with sub-pixel refinement.")


# ---------------------------------------------------------------------------
# Stage 3: full pipeline on a held-out split
# ---------------------------------------------------------------------------
def load_verifier(out_dir: Path, device):
    ckpt_path = out_dir / "verifier.pt"
    if not ckpt_path.is_file():
        return None
    model = Verifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model


@torch.no_grad()
def localize_pair(ref, sea, model, device):
    """Full pipeline on one pair -> dict with prediction + diagnostics."""
    t0 = time.time()
    cands, maps = propose(ref, sea)
    t_prop = time.time() - t0

    t0 = time.time()
    # No ZNCC peaks (flat/NaN maps): still print a coordinate. Spec tie-break
    # is the search-image center when nothing else ranks.
    if not cands:
        return {"x": SEARCH_CENTER[0], "y": SEARCH_CENTER[1],
                "conf": 0.0, "second": 0.0, "scale": 10.0, "angle": 0.0,
                "box_w": 100.0, "cands": cands, "t_prop": t_prop,
                "t_rank": time.time() - t0}
    if model is not None:
        crops = torch.from_numpy(np.stack(
            [candidate_patch(sea, c, maps) for c in cands]).astype(np.float32) / 255.0)
        scal = torch.tensor([cand_scalars(c) for c in cands], dtype=torch.float32)
        logits = model(crops.to(device), scal.to(device)).cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        order = np.argsort(-probs)
        top = order[0]
        # spec tie-break: among near-tied candidates, closest to image center
        tied = [i for i in order if probs[i] >= probs[top] - TIE_PROB_GAP]
        if len(tied) > 1:
            top = min(tied, key=lambda i: np.hypot(cands[i]["cx"] - SEARCH_CENTER[0],
                                                   cands[i]["cy"] - SEARCH_CENTER[1]))
        conf = float(probs[top])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
    else:  # classical fallback: ZNCC score only
        top = int(np.argmax([c["score"] for c in cands]))
        conf = float(cands[top]["score"])
        second = float(sorted((c["score"] for c in cands), reverse=True)[1]) \
            if len(cands) > 1 else 0.0
    t_rank = time.time() - t0

    c = cands[top]
    R = maps[c["map_idx"]]["R"]
    dx, dy = subpixel(R, c["xi"], c["yi"])
    tw = maps[c["map_idx"]]["tw"]
    px, py = c["xi"] + dx + tw / 2.0, c["yi"] + dy + tw / 2.0
    return {"x": px, "y": py, "conf": conf, "second": second,
            "scale": c["scale"], "angle": c["angle"], "box_w": tw,
            "cands": cands, "t_prop": t_prop, "t_rank": t_rank}


def save_failure_panel(sea, gt, pred, box_w, err, path: Path):
    img = cv2.cvtColor(sea, cv2.COLOR_GRAY2BGR)
    for (cx, cy), color in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        h = box_w / 2.0
        cv2.rectangle(img, (int(cx - h), int(cy - h)), (int(cx + h), int(cy + h)),
                      color, 2)
    cv2.putText(img, f"err={err:.1f}px  green=GT red=pred", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(path), img)


def pr_curve(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-conf)
    c = correct[order]
    tp = np.cumsum(c)
    fp = np.cumsum(~c)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(len(c), 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    ap = float(trapezoid(precision, recall))
    return precision, recall, ap


def stage3(args):
    out_dir = Path(args.out)
    device = pick_device()
    model = load_verifier(out_dir, device)
    mode = "verifier" if model is not None else "ZNCC fallback (no verifier.pt!)"
    split_dir = find_split_dir(args.data, args.split)
    rows = load_manifest(split_dir)
    if args.limit:
        rows = rows[:args.limit]
    print(f"stage3: full pipeline on {args.split} ({len(rows)} pairs, "
          f"ranking: {mode}, device: {device.type})")

    results = []
    for row in tqdm(rows):
        ref, sea = load_pair(split_dir, row)
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        out = localize_pair(ref, sea, model, device)
        err = float(np.hypot(out["x"] - gx, out["y"] - gy))
        best_cand = min((np.hypot(c["cx"] - gx, c["cy"] - gy) for c in out["cands"]),
                        default=float("inf"))
        results.append({
            "id": row.get("id", ""), "bucket": row.get("noise_bucket", "?"),
            "gt_x": gx, "gt_y": gy, "pred_x": out["x"], "pred_y": out["y"],
            "err": err, "conf": out["conf"], "margin": out["conf"] - out["second"],
            "scale": out["scale"], "angle": out["angle"], "box_w": out["box_w"],
            "best_cand_err": float(best_cand),
            "t_prop": out["t_prop"], "t_rank": out["t_rank"],
        })

    errs = np.array([r["err"] for r in results])
    buckets = [r["bucket"] for r in results]
    per = {"ALL": pass_table(errs)}
    for b in sorted(set(buckets)):
        per[b] = pass_table(errs[[i for i, x in enumerate(buckets) if x == b]])
    print_bucket_table(per, f"{args.split}, end-to-end with sub-pixel")

    rec = np.mean([r["best_cand_err"] <= POS_RADIUS for r in results])
    tp = np.mean([r["t_prop"] for r in results])
    tr = np.mean([r["t_rank"] for r in results])
    print(f"\n  proposal recall@{TOP_N}: {rec:.3f}   "
          f"time/pair: {tp + tr:.2f}s (propose {tp:.2f}s + rank {tr:.2f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # PR curves: confidence vs correctness (<=5px), overall + per bucket
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        conf = np.array([r["conf"] for r in results])
        correct = errs <= 5.0
        groups = [("ALL", np.ones(len(results), bool))]
        groups += [(b, np.array([x == b for x in buckets])) for b in sorted(set(buckets))]
        for name, mask in groups:
            if mask.sum() < 2:
                continue
            p, rcl, ap = pr_curve(conf[mask], correct[mask])
            ax.plot(rcl, p, marker="o", markersize=2.5, label=f"{name} (AP={ap:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"DriftLoc PR by noise bucket ({args.split}, tol=5px)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        pr_path = out_dir / f"pr_{args.split}.png"
        fig.savefig(pr_path, dpi=140)
        print(f"  PR curves -> {pr_path}")
    except ImportError:
        print("  matplotlib not available: skipped PR plot")

    # failure panels for the explainability slide
    fails = sorted((r for r in results if r["err"] > 5.0),
                   key=lambda r: -r["err"])[:args.failures]
    if fails:
        fdir = out_dir / f"failures_{args.split}"
        fdir.mkdir(exist_ok=True)
        by_id = {str(row.get("id", i)): row for i, row in enumerate(rows)}
        for r in fails:
            row = by_id[str(r["id"])]
            _, sea = load_pair(split_dir, row)
            save_failure_panel(sea, (r["gt_x"], r["gt_y"]),
                               (r["pred_x"], r["pred_y"]), r["box_w"], r["err"],
                               fdir / f"{r['id']}_{r['bucket']}.png")
        print(f"  {len(fails)} failure panels -> {fdir}")

    metrics = {"split": args.split, "mode": mode, "per_bucket": per,
               "proposal_recall": float(rec),
               "time_per_pair_s": float(tp + tr)}
    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics -> {out_dir / f'metrics_{args.split}.json'}")


# ---------------------------------------------------------------------------
# Sponsor interface: one pair -> "x y"
# ---------------------------------------------------------------------------
def localize(args):
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise SystemExit("could not read reference or search image")
    device = pick_device()
    model = None if args.plain else load_verifier(Path(args.out), device)
    out = localize_pair(ref, sea, model, device)
    print(f"{out['x']:.2f} {out['y']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1", help="validate ZNCC proposal recall")
    p1.add_argument("--data", default="output")
    p1.add_argument("--split", default="val")
    p1.add_argument("--limit", type=int, default=0)

    p2 = sub.add_parser("stage2", help="dump candidates + train verifier")
    p2.add_argument("--data", default="output")
    p2.add_argument("--out", default="model")
    p2.add_argument("--epochs", type=int, default=20)
    p2.add_argument("--batch-pairs", type=int, default=8)
    p2.add_argument("--lr", type=float, default=3e-4)
    p2.add_argument("--patience", type=int, default=5)
    p2.add_argument("--limit", type=int, default=0,
                    help="dump only the first N train pairs (smoke test)")
    p2.add_argument("--redump", action="store_true")

    p3 = sub.add_parser("stage3", help="full pipeline metrics on a split")
    p3.add_argument("--data", default="output")
    p3.add_argument("--out", default="model")
    p3.add_argument("--split", default="eval")
    p3.add_argument("--limit", type=int, default=0)
    p3.add_argument("--failures", type=int, default=5)

    pl = sub.add_parser("localize", help="one pair -> print 'x y'")
    pl.add_argument("--reference", required=True)
    pl.add_argument("--search", required=True)
    pl.add_argument("--out", default="model")
    pl.add_argument("--plain", action="store_true",
                    help="skip the verifier, use ZNCC + sub-pixel only")

    args = ap.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    {"stage1": stage1, "stage2": stage2, "stage3": stage3,
     "localize": localize}[args.cmd](args)


if __name__ == "__main__":
    main()
