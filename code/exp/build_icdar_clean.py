#!/usr/bin/env python3
"""A1: leakage-free ICDAR-2019 segments (fixes the train/dev/test content duplication
found in iter-1, where the old builders sampled training_18M, evaluation_4M AND full_22M
= their union at random).

  * official partition only: training_18M -> train/dev pool, evaluation_4M -> test
    (full_22M is ignored);
  * a page is used only if len(OCR_aligned) == len(GS_aligned) and OCR_aligned minus the
    '@' gaps reproduces OCR_toInput exactly (rejections are counted); literal '@' produced by
    Fraktur OCR is told apart from the gap symbol with OCR_toInput (resolve_literal_at);
  * content dedup: exact page hashes + MinHash-LSH near-duplicates on gold char 5-gram
    shingles (Jaccard >= JACC), merged with union-find; a group touching the evaluation
    partition goes to test; one representative page per group;
  * dev = DEV_FRAC of the training groups (seed 20260526);
  * residual audit (LSH has false negatives): every selected dev page is compared by brute
    force with every selected train page, every test page with train+dev, on gold 5-gram
    shingles; pages with Jaccard >= JACC or containment >= CONTAIN are dropped and counted;
  * OCR-only segmentation (exp/segutil.py): boundaries from the OCR string alone, exact
    spans + separators, no gold filter; gold attached through the alignment columns; both
    page texts reconstruct exactly (docs_* keep the full page for document CER);
  * residual exact-duplicate OCR segments vs train (boilerplate) are FLAGGED
    (dup_train=true), not dropped; disjointness of pages/groups across splits is asserted.

Env: RAW (dir containing training_18M_without_Finnish/ and evaluation_4M_without_Finnish/),
OUT (default data/), LANGS (EN,DE,FR,ES,NL), CAPS ("DE:300:80,FR:300:80" = max train+dev
groups : max test groups), TARGET (70), JACC (0.5), CONTAIN (0.8), DEV_FRAC (0.2).
Outputs segments_icdarc_{LANG}_{split}.jsonl, segments_icdarc_multi_{split}.jsonl,
docs_icdarc_{LANG}_{split}.jsonl, icdarc_overlap_report.json."""
import os, sys, json, glob, random, hashlib, re, collections
from datasketch import MinHash, MinHashLSH
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segutil import segment, map_from_aligned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("RAW", os.path.join(ROOT, "raw", "icdar2019"))
OUT = os.environ.get("OUT", os.path.join(ROOT, "data"))
LANGS = os.environ.get("LANGS", "EN,DE,FR,ES,NL").split(",")
CAPS = {k: (int(a), int(b)) for k, a, b in (c.split(":") for c in
        os.environ.get("CAPS", "DE:300:80,FR:300:80").split(",") if c)}
TARGET = int(os.environ.get("TARGET", 70))
JACC = float(os.environ.get("JACC", 0.5))
CONTAIN = float(os.environ.get("CONTAIN", 0.8))
DEV_FRAC = float(os.environ.get("DEV_FRAC", 0.2))
SEED = 20260526
PARTS = {"train": "training_18M_without_Finnish", "test": "evaluation_4M_without_Finnish"}
TAGS = {"[OCR_toInput]": "oi", "[OCR_aligned]": "oa", "[ GS_aligned]": "ga"}

def norm(s): return " ".join(s.split())

LIT = "\x00"

def resolve_literal_at(oa, oi):
    """Mark the '@' columns of OCR_aligned that are literal OCR '@' (LIT) and keep the rest as
    gaps. Greedy: an '@' column is literal iff the next unread OCR char is '@'. Exact: such a
    char must be consumed before the next non-'@' column, so any valid assignment differs only
    by permuting columns inside the same '@' run (identical text, gold map shifts within it)."""
    out = []; j = 0
    for c in oa:
        if c == "@":
            if j < len(oi) and oi[j] == "@": out.append(LIT); j += 1
            else: out.append("@")
        else:
            if j >= len(oi) or oi[j] != c: return None
            out.append(c); j += 1
    return "".join(out) if j == len(oi) else None

def parse(path):
    """Return (ocr, gold, oa, ga), or (None, reason) if the page does not validate.
    Exactly one space after the tag is removed, so alignment columns are preserved."""
    f = {}
    for line in open(path, encoding="utf-8", errors="replace").read().split("\n"):
        line = line.rstrip("\r")
        for tag, k in TAGS.items():
            if line.startswith(tag):
                body = line[len(tag):]
                f[k] = body[1:] if body.startswith(" ") else body
    oi, oa, ga = f.get("oi"), f.get("oa"), f.get("ga")
    if oi is None or oa is None or ga is None: return None, "missing_field"
    if len(oa) != len(ga): return None, "aligned_len_mismatch"
    if "@" in oi:   # literal '@' in the OCR (Fraktur) collides with the gap symbol: resolve it
        oa = resolve_literal_at(oa, oi)
        if oa is None: return None, "ocr_mismatch"
    if oa.replace("@", "").replace(LIT, "@") != oi: return None, "ocr_mismatch"
    oa = oa.replace(LIT, "\x01")    # any non-gap char: map_from_aligned only tests for '@'
    ocr, gold = oi, ga.replace("@", "")
    if not ocr.strip() or not gold.strip(): return None, "empty"
    return (ocr, gold, oa, ga), None

def shingles(s, k=5):
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i+k] for i in range(max(1, len(s) - k + 1))}

class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)

def build_lang(lang, report):
    pages = []  # dict(id, part, ocr_doc, gold_doc, oa, ga)
    rejected = collections.Counter()
    for part, d in PARTS.items():
        for p in sorted(glob.glob(os.path.join(RAW, d, lang, "**", "*.txt"), recursive=True)):
            r, why = parse(p)
            if r is None: rejected[f"{part}:{why}"] += 1; continue
            ocr, gold, oa, ga = r
            pages.append(dict(id=os.path.relpath(p, RAW), part=part, oa=oa, ga=ga, ocr_doc=ocr, gold_doc=gold))
    n = len(pages); uf = UF(n)
    # exact duplicates (gold or OCR page identical after whitespace/case normalisation)
    for key in ("gold_doc", "ocr_doc"):
        seen = {}
        for i, pg in enumerate(pages):
            h = hashlib.md5(norm(pg[key]).lower().encode()).hexdigest()
            if h in seen: uf.u(i, seen[h])
            else: seen[h] = i
    # near duplicates
    sh = [shingles(pg["gold_doc"]) for pg in pages]
    lsh = MinHashLSH(threshold=JACC, num_perm=128); mhs = []
    for i in range(n):
        m = MinHash(num_perm=128)
        for x in sh[i]: m.update(x.encode())
        mhs.append(m); lsh.insert(str(i), m)
    near = 0
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            j = int(j)
            if j != i and mhs[i].jaccard(mhs[j]) >= JACC and uf.f(i) != uf.f(j):
                uf.u(i, j); near += 1
    groups = collections.defaultdict(list)
    for i in range(n): groups[uf.f(i)].append(i)
    test_g, train_g, cross = [], [], 0
    for g, mem in groups.items():
        parts = {pages[i]["part"] for i in mem}
        if "test" in parts:
            if len(parts) > 1: cross += 1
            rep = sorted([i for i in mem if pages[i]["part"] == "test"], key=lambda i: pages[i]["id"])[0]
            test_g.append((g, rep))
        else:
            rep = sorted(mem, key=lambda i: pages[i]["id"])[0]
            train_g.append((g, rep))
    rng = random.Random(f"{SEED}-{lang}")
    test_g.sort(key=lambda t: pages[t[1]]["id"]); train_g.sort(key=lambda t: pages[t[1]]["id"])
    rng.shuffle(test_g); rng.shuffle(train_g)
    if lang in CAPS:
        train_g = train_g[:CAPS[lang][0]]; test_g = test_g[:CAPS[lang][1]]
    ndev = int(round(DEV_FRAC * len(train_g)))
    splits = {"train": train_g[ndev:], "dev": train_g[:ndev], "test": test_g}

    # independent brute-force residual audit on the selected pages
    def overlap(a, b):
        inter = len(sh[a] & sh[b])
        return inter / max(1, len(sh[a] | sh[b])), inter / max(1, min(len(sh[a]), len(sh[b])))
    def clashes(r, refs):
        for q in refs:
            jac, con = overlap(r, q)
            if jac >= JACC or con >= CONTAIN: return True
        return False
    audit = {}
    tr = [r for _, r in splits["train"]]
    keep = [t for t in splits["dev"] if not clashes(t[1], tr)]
    audit["dev_dropped_vs_train"] = len(splits["dev"]) - len(keep); splits["dev"] = keep
    ref = tr + [r for _, r in splits["dev"]]
    keep = [t for t in splits["test"] if not clashes(t[1], ref)]
    audit["test_dropped_vs_train_dev"] = len(splits["test"]) - len(keep); splits["test"] = keep
    # post-audit maxima (must be below thresholds)
    mx = {"jacc": 0.0, "contain": 0.0}
    for a, b in (("dev", "train"), ("test", "train"), ("test", "dev")):
        for _, r in splits[a]:
            for _, q in splits[b]:
                jac, con = overlap(r, q)
                mx["jacc"] = max(mx["jacc"], jac); mx["contain"] = max(mx["contain"], con)
    assert mx["jacc"] < JACC and mx["contain"] < CONTAIN, (lang, mx)
    audit["max_cross_split_jaccard"] = round(mx["jacc"], 4)
    audit["max_cross_split_containment"] = round(mx["contain"], 4)

    gs = {k: {g for g, _ in v} for k, v in splits.items()}
    ids = {k: {pages[r]["id"] for _, r in v} for k, v in splits.items()}
    for a in gs:
        for b in gs:
            if a < b: assert not (gs[a] & gs[b]) and not (ids[a] & ids[b]), (lang, a, b)
    out = {}; train_ocr = set()
    for split in ("train", "dev", "test"):
        segs, docs = [], []
        for g, r in splits[split]:
            pg = pages[r]
            docs.append(dict(doc_id=pg["id"], lang=lang, group=f"{lang}-{g}", ocr=pg["ocr_doc"], gold=pg["gold_doc"]))
            m = map_from_aligned(pg["oa"], pg["ga"])
            for k, s in enumerate(segment(pg["ocr_doc"], pg["gold_doc"], m, f"c{TARGET}")):
                segs.append(dict(doc_id=pg["id"], seg_idx=k, lang=lang, group=f"{lang}-{g}", **s))
        if split == "train": train_ocr = {norm(s["ocr"]) for s in segs if len(norm(s["ocr"])) >= 20}
        for s in segs: s["dup_train"] = split != "train" and norm(s["ocr"]) in train_ocr
        out[split] = (segs, docs)
    rep = dict(pages=n, rejected=dict(rejected), groups=len(groups), near_dup_merges=near,
               cross_partition_groups_to_test=cross, residual_audit=audit,
               pages_in_partition={p: sum(pg["part"] == p for pg in pages) for p in PARTS},
               cap=CAPS.get(lang))
    for split, (segs, docs) in out.items():
        rep[split] = dict(docs=len(docs), segs=len(segs), ocr_chars=sum(len(d["ocr"]) for d in docs),
                          gold_chars=sum(len(d["gold"]) for d in docs),
                          dup_train_segs=sum(s["dup_train"] for s in segs),
                          max_seg_ocr=max((len(s["ocr"]) for s in segs), default=0),
                          doc_cer=round(sum(Levenshtein.distance(d["ocr"], d["gold"]) for d in docs) /
                                        max(1, sum(len(d["gold"]) for d in docs)), 4))
    report[lang] = rep
    return out

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    report = dict(config=dict(RAW=RAW, LANGS=LANGS, CAPS=CAPS, TARGET=TARGET, JACC=JACC,
                              CONTAIN=CONTAIN, DEV_FRAC=DEV_FRAC, SEED=SEED))
    multi = collections.defaultdict(list)
    for lang in LANGS:
        out = build_lang(lang, report)
        for split, (segs, docs) in out.items():
            dump(os.path.join(OUT, f"segments_icdarc_{lang}_{split}.jsonl"), segs)
            dump(os.path.join(OUT, f"docs_icdarc_{lang}_{split}.jsonl"), docs)
            multi[split] += segs
        print(lang, json.dumps(report[lang]), flush=True)
    for split, segs in multi.items():
        dump(os.path.join(OUT, f"segments_icdarc_multi_{split}.jsonl"), segs)
    json.dump(report, open(os.path.join(OUT, "icdarc_overlap_report.json"), "w"), indent=1)
    print("wrote", OUT)
