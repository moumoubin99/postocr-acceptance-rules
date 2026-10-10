#!/usr/bin/env python3
"""iter-3: exact token -> character alignment of corrector confidence for the edit-level analysis.
Writes data/toklp_<corr>_<ds>_<split>.jsonl, one row per candidate row:
  {doc_id, seg_idx, spans: [[char_start, char_end, lp], ...] | null, end_lp, mode}
spans cover the STORED candidate string, one entry per model token (ByT5: one per UTF-8 byte, so a
multi-byte character has several entries with the same span); end_lp is the end-of-sequence token.

  byt5 : teacher-forced log-probs of the stored candidate bytes (+EOS) given the OCR input, from the
         checkpoint that generated it. Exact by construction: no dependence on the decode step that
         may have dropped invalid bytes, or on a generation that was truncated before EOS.
         JOBS="corr|dataset|model_dir|splits;..."  python exp/rescore_tokens.py byt5
  llm  : the stored lps are already the log-probs of the tokens of (candidate + end token); only the
         tokenizer offsets are needed. Rows whose token count differs from the stored lps, whose tokens do not
         decode back to the stored string, or whose offsets leave a character uncovered get spans=null.
         JOBS="corr|dataset|splits;..." LLM_MODEL=<tokenizer dir> python exp/rescore_tokens.py llm
Reads only ocr/cand/lps of the candidate files: no gold text is used."""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
CAND = os.environ.get("CAND_DIR", DATA)     # where the candidate files are read from
OUT_DIR = os.environ.get("OUT_DIR", DATA)   # where the toklp files go (smoke runs use a separate directory)
MAXLEN = 384

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def write(corr, ds, split, recs):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"toklp_{corr}_{ds}_{split}.jsonl")
    with open(out + ".part", "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out + ".part", out)
    return os.path.basename(out)

def byte_spans(c):
    sp = []
    for k, ch in enumerate(c): sp += [(k, k + 1)] * len(ch.encode("utf-8"))
    return sp

def run_byt5(jobs):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = int(os.environ.get("GEN_BATCH", 64)); cur = None
    for corr, ds, mdir, splits in jobs:
        if mdir != cur:
            tok = AutoTokenizer.from_pretrained(mdir)
            model = AutoModelForSeq2SeqLM.from_pretrained(mdir).to(dev).eval(); cur = mdir
            assert tok.eos_token_id == 1 and tok("a", add_special_tokens=False)["input_ids"] == [100], "not a ByT5 tokenizer"
        for split in splits.split(","):
            rows = load(os.path.join(CAND, f"candidates_{corr}_{ds}_{split}.jsonl")); t0 = time.time()
            if int(os.environ.get("LIMIT", 0)): rows = rows[:int(os.environ["LIMIT"])]
            recs = [None] * len(rows); dmax, ncmp = 0.0, 0
            order = sorted(range(len(rows)), key=lambda k: len(rows[k]["cand"].encode()))
            for i in range(0, len(order), B):
                idx = order[i:i + B]
                enc = tok([rows[k]["ocr"] for k in idx], return_tensors="pt", padding=True, truncation=True,
                          max_length=MAXLEN).to(dev)
                lab = [[b + 3 for b in rows[k]["cand"].encode("utf-8")] + [1] for k in idx]
                L = max(map(len, lab)); y = torch.full((len(idx), L), -100, dtype=torch.long)
                for j, l in enumerate(lab): y[j, :len(l)] = torch.tensor(l)
                with torch.no_grad():
                    lp = torch.log_softmax(model(**enc, labels=y.to(dev)).logits.float(), -1).cpu()
                for j, k in enumerate(idx):
                    l = lab[j]; v = lp[j, torch.arange(len(l)), torch.tensor(l)].tolist()
                    sp = byte_spans(rows[k]["cand"]); assert len(sp) == len(l) - 1
                    recs[k] = {"doc_id": rows[k]["doc_id"], "seg_idx": rows[k].get("seg_idx"), "mode": "byt5-tf",
                               "spans": [[a, b, round(x, 4)] for (a, b), x in zip(sp, v[:-1])], "end_lp": round(v[-1], 4)}
                    old = rows[k].get("lps")   # generation-time log-probs: equal when the byte mapping was already exact
                    if old and len(old) == len(v):
                        dmax = max(dmax, max(abs(a - b) for a, b in zip(old, v))); ncmp += 1
            name = write(corr, ds, split, recs)
            print(f"{name}: {len(rows)} rows, {ncmp} comparable with generation lps, max |diff| {dmax:.3f} ({time.time()-t0:.0f}s)", flush=True)

def run_llm(jobs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["LLM_MODEL"])
    end = os.environ.get("END_TOKEN") or next((t for t in ("<|im_end|>", "<|eot_id|>") if t in tok.get_vocab()), None) or tok.eos_token
    for corr, ds, splits in jobs:
        for split in splits.split(","):
            rows = load(os.path.join(CAND, f"candidates_{corr}_{ds}_{split}.jsonl")); recs = []; miss = 0
            for r in rows:
                c = r["cand"]; e = tok(c + end, add_special_tokens=False, return_offsets_mapping=True)
                off, lps = e["offset_mapping"], r.get("lps") or []
                rec = {"doc_id": r["doc_id"], "seg_idx": r.get("seg_idx"), "mode": "llm-offsets", "spans": None, "end_lp": None}
                cov = [False] * len(c)
                for a, b in off[:-1]:
                    for k in range(a, min(b, len(c))): cov[k] = True
                # usable only if the token count matches the stored lps, the tokens decode back to the stored string
                # (a normalizing tokenizer would shift offsets) and every character is covered by some token
                if (len(off) == len(lps) and len(off) >= 1 and off[-1][0] >= len(c) and all(cov)
                        and tok.decode(e["input_ids"][:-1]) == c):
                    rec["spans"] = [[min(a, len(c)), min(b, len(c)), x] for (a, b), x in zip(off[:-1], lps[:-1])]
                    rec["end_lp"] = lps[-1]
                else: miss += 1
                recs.append(rec)
            print(f"{write(corr, ds, split, recs)}: {len(rows)} rows, {miss} without spans", flush=True)

if __name__ == "__main__":
    jobs = [j.split("|") for j in os.environ["JOBS"].split(";") if j.strip()]
    {"byt5": run_byt5, "llm": run_llm}[sys.argv[1]](jobs)
