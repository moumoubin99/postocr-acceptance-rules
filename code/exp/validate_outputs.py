#!/usr/bin/env python3
"""Validate job outputs before run_all_gpu.sh accepts (or skips) a job.
Usage: python exp/validate_outputs.py <path> [<path> ...]   (exit 0 = all valid)
  <model_dir>/config.json            -> config parses; every weight file (all index shards) is intact
  candidates_<corr>_<ds>_<split>.jsonl -> one parseable row per segments_<ds>_<split>.jsonl row,
                                        same doc_id/seg_idx order, cand str, finite score/min_lp
  qe_<cand>_<split>.jsonl            -> one finite qe per row of candidates_<cand>_<split>.jsonl
  qe_<cand>_metrics.json             -> parses
  verif_<cand>_<split>.jsonl         -> one finite v per row of candidates_<cand>_<split>.jsonl, att flag consistent
Env: DATA_DIR."""
import os, sys, json, math

DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

def rows(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def check_weights(p):
    """Truncation check without loading tensors: safetensors header offsets must end exactly at EOF;
    a torch .bin is a zip whose central directory (at the end) must parse and list a data.pkl."""
    import struct, zipfile
    size = os.path.getsize(p)
    if p.endswith(".safetensors"):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        assert size == 8 + n + end, f"{p}: truncated ({size} bytes, header expects {8 + n + end})"
    else:
        with zipfile.ZipFile(p) as z:
            assert any(x.endswith("data.pkl") for x in z.namelist()), f"{p}: not a torch checkpoint"

def check(path):
    b = os.path.basename(path)
    if b == "config.json":
        d = os.path.dirname(path); json.load(open(path))
        idx = [f for f in os.listdir(d) if f.endswith(".index.json")]
        if idx:   # sharded: every shard listed in the index must be present and intact
            shards = sorted(set(json.load(open(os.path.join(d, idx[0])))["weight_map"].values()))
        else:
            shards = [f for f in ("model.safetensors", "pytorch_model.bin") if os.path.exists(os.path.join(d, f))]
        assert shards, f"no weights in {d}"
        for f in shards: check_weights(os.path.join(d, f))
        return
    if b.startswith("candidates_") and b.endswith(".jsonl"):
        corr, rest = b[len("candidates_"):-len(".jsonl")].split("_", 1)
        ds, split = rest.rsplit("_", 1)
        seg = rows(os.path.join(DATA, f"segments_{ds}_{split}.jsonl")); got = rows(path)
        assert len(got) == len(seg), f"{b}: {len(got)} rows, segments have {len(seg)}"
        for k, (r, s) in enumerate(zip(got, seg)):
            assert r["doc_id"] == s["doc_id"] and r.get("seg_idx") == s.get("seg_idx"), f"{b}: row {k} misaligned"
            assert isinstance(r["cand"], str), f"{b}: row {k} cand"
            assert math.isfinite(r["score"]) and math.isfinite(r["min_lp"]), f"{b}: row {k} non-finite confidence"
        return
    if b.startswith("qe_") and b.endswith("_metrics.json"):
        json.load(open(path)); return
    if b.startswith("qe_") and b.endswith(".jsonl"):
        cand, split = b[len("qe_"):-len(".jsonl")].rsplit("_", 1)
        ref = rows(os.path.join(DATA, f"candidates_{cand}_{split}.jsonl")); got = rows(path)
        assert len(got) == len(ref), f"{b}: {len(got)} rows, candidates have {len(ref)}"
        for k, (r, c) in enumerate(zip(got, ref)):
            assert r["doc_id"] == c["doc_id"] and r.get("seg_idx") == c.get("seg_idx"), f"{b}: row {k} misaligned"
            assert math.isfinite(r["qe"]), f"{b}: row {k} non-finite qe"
        return
    if b.startswith("verif_") and b.endswith("_metrics.json"):
        json.load(open(path)); return
    if b.startswith("verif_") and b.endswith(".jsonl"):
        cand, split = b[len("verif_"):-len(".jsonl")].rsplit("_", 1)
        ref = rows(os.path.join(DATA, f"candidates_{cand}_{split}.jsonl")); got = rows(path)
        assert len(got) == len(ref), f"{b}: {len(got)} rows, candidates have {len(ref)}"
        for k, (r, c) in enumerate(zip(got, ref)):
            assert r["doc_id"] == c["doc_id"] and r.get("seg_idx") == c.get("seg_idx"), f"{b}: row {k} misaligned"
            assert r["att"] == (c["cand"] != c["ocr"]) and math.isfinite(r["v"]), f"{b}: row {k} att/v"
        return
    assert os.path.exists(path), f"missing {path}"

if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        try: check(p)
        except Exception as e: print(f"INVALID {p}: {e!r}"); bad += 1
    sys.exit(1 if bad else 0)
