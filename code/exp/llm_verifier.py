#!/usr/bin/env python3
"""B6: second-pass LLM verifier (Qwen2.5-Instruct) over (OCR, candidate) pairs with document context.
For every attempted row (cand != ocr) of candidates_<cand>_<split>.jsonl the model is asked which of two
versions is closer to the original print; the gate score is the answer log-odds in favour of the candidate,
averaged over both presentation orders (A=OCR/B=candidate and A=candidate/B=OCR) to cancel position bias.
The threshold is calibrated on dev by the ledger, exactly like every other policy. Gold is never shown.

Output: verif_<cand>_<split>.jsonl  {doc_id, seg_idx, att, v, lo_ocr_first, lo_cand_first, missing}
        (one row per candidate row, same order; v = 0.0 and att = false for unattempted rows)
        verif_<cand>_metrics.json   latency (wall seconds, prompts, prompt tokens, ms per attempted segment)
Usage:  JOBS="byt5base8_blnocr|dev,test|1;qwen_blnocr|dev,test|1" python exp/llm_verifier.py
        (CAND|splits|CTX; CTX = neighbouring OCR segments shown on each side). One vLLM engine for all jobs.
A finished output whose <out>.meta.json fingerprint matches is skipped.
Env: LLM_MODEL, DATA_DIR, OUT_DIR, LIMIT, GPU_UTIL, RUN_TOKEN."""
import os, sys, json, time, math, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_corrector import DATA, OUT_DIR, MODEL, LIMIT, load, domain, vllm_engine, _model_rev, meta_matches
PROMPT_VERSION = "verif-v1-2026-10"
TOPK = 20; MAXTOK = 3900

def sys_prompt(dataset, row):
    return (f"You compare two versions of a short excerpt of {domain(dataset, row)}. One is raw OCR output, the other is "
            "an automatic correction of it. A good correction fixes OCR recognition errors while preserving the original "
            "wording, proper nouns and period spelling; a bad one changes words that were right, modernizes, paraphrases, "
            "or drops or adds text. Decide which version is closer to the original printed text.")

def user_msg(rows, i, ctx, first, second):
    r = rows[i]; lo, hi = i, i
    while lo > 0 and i - lo < ctx and rows[lo-1]["doc_id"] == r["doc_id"]: lo -= 1
    while hi < len(rows)-1 and hi - i < ctx and rows[hi+1]["doc_id"] == r["doc_id"]: hi += 1
    before = " ".join(x["ocr"] for x in rows[lo:i]); after = " ".join(x["ocr"] for x in rows[i+1:hi+1])
    head = (f"Context before (raw OCR):\n{before or '(none)'}\n\nContext after (raw OCR):\n{after or '(none)'}\n\n") if ctx > 0 else ""
    return (f"{head}Version A:\n{first}\n\nVersion B:\n{second}\n\n"
            "Which version is closer to the original printed text? Answer with a single letter, A or B.")

def letter_lps(tok, lp):
    """lp: {token_id: Logprob} for the first generated token -> (logP(A), logP(B), missing flag).
    Probability mass of every top-k token that decodes to the bare letter is summed; a letter absent from the
    top-k gets the smallest top-k log-prob (an upper bound on its true value) and is flagged."""
    acc = {"A": [], "B": []}; floor = min(x.logprob for x in lp.values())
    for tid, x in lp.items():
        s = (x.decoded_token if x.decoded_token is not None else tok.decode([tid])).strip()
        if s in acc: acc[s].append(x.logprob)
    def lse(v): m = max(v); return m + math.log(sum(math.exp(a - m) for a in v))
    return (lse(acc["A"]) if acc["A"] else floor, lse(acc["B"]) if acc["B"] else floor, int(not acc["A"] or not acc["B"]))

def fingerprint(inp, ctx, cand):
    st = os.stat(inp)
    return json.loads(json.dumps(dict(model=os.path.abspath(MODEL) if os.path.isdir(MODEL) else MODEL, model_rev=_model_rev(),
                code=hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest(), prompt=PROMPT_VERSION, ctx=ctx,
                limit=LIMIT, cand=cand, input=os.path.basename(inp), input_size=st.st_size, input_mtime=int(st.st_mtime),
                domain=os.environ.get("DOMAIN", ""))))

def complete(path, rows):  # same acceptance rule as exp/validate_outputs.py
    try:
        recs = load(path)
        return len(recs) == len(rows) and all(a["doc_id"] == b["doc_id"] and a.get("seg_idx") == b.get("seg_idx")
                                              and math.isfinite(a["v"]) for a, b in zip(recs, rows))
    except Exception:
        return False

def run_job(tok, cand, splits, ctx):
    from vllm import SamplingParams
    dataset = cand.split("_", 1)[1]; stats = {}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK)
    for split in splits.split(","):
        inp = os.path.join(DATA, f"candidates_{cand}_{split}.jsonl")
        out_path = os.path.join(OUT_DIR, f"verif_{cand}_{split}.jsonl")
        fp = fingerprint(inp, ctx, cand); rows = load(inp)
        if LIMIT: rows = rows[:LIMIT]
        if os.path.exists(out_path) and meta_matches(out_path + ".meta.json", fp) and complete(out_path, rows):
            print(f"skip {os.path.basename(out_path)} (fingerprint match)", flush=True); continue
        att = [i for i, r in enumerate(rows) if r["cand"] != r["ocr"]]
        def prompt(i, cand_first, c):
            r = rows[i]; a, b = (r["cand"], r["ocr"]) if cand_first else (r["ocr"], r["cand"])
            return tok.apply_chat_template([{"role": "system", "content": sys_prompt(dataset, r)},
                                            {"role": "user", "content": user_msg(rows, i, c, a, b)}],
                                           tokenize=False, add_generation_prompt=True)
        prompts, ntok, noctx = [], 0, 0
        for i in att:
            for cand_first in (False, True):
                p = prompt(i, cand_first, ctx); n = len(tok(p, add_special_tokens=False)["input_ids"])
                if n > MAXTOK and ctx > 0:      # very long segment: drop the context rather than truncate the pair
                    p = prompt(i, cand_first, 0); n = len(tok(p, add_special_tokens=False)["input_ids"]); noctx += 1
                assert n <= 4090, f"{cand} {split} row {i}: prompt of {n} tokens does not fit"
                prompts.append(p); ntok += n
        t0 = time.time(); outs = vllm_engine().generate(prompts, sp) if prompts else []; dt = time.time() - t0
        res = {}
        for k, i in enumerate(att):
            a1, b1, m1 = letter_lps(tok, outs[2*k].outputs[0].logprobs[0])       # A = OCR, B = candidate
            a2, b2, m2 = letter_lps(tok, outs[2*k+1].outputs[0].logprobs[0])     # A = candidate, B = OCR
            res[i] = dict(v=0.5 * ((b1 - a1) + (a2 - b2)), lo_ocr_first=b1 - a1, lo_cand_first=a2 - b2, missing=m1 + m2)
        with open(out_path + ".part", "w") as f:
            for i, r in enumerate(rows):
                rec = {"doc_id": r["doc_id"], "seg_idx": r.get("seg_idx"), "att": i in res}
                rec.update(res.get(i, dict(v=0.0, lo_ocr_first=0.0, lo_cand_first=0.0, missing=0)))
                f.write(json.dumps(rec) + "\n")
        if os.path.exists(out_path + ".meta.json"): os.remove(out_path + ".meta.json")
        os.replace(out_path + ".part", out_path)
        with open(out_path + ".meta.part", "w") as f: json.dump(fp, f)
        os.replace(out_path + ".meta.part", out_path + ".meta.json")
        stats[split] = dict(rows=len(rows), attempted=len(att), prompts=len(prompts), prompt_tokens=ntok, seconds=dt,
                            ms_per_attempted_segment=1000 * dt / max(1, len(att)), context_dropped=noctx // 2,
                            missing_letter=sum(x["missing"] > 0 for x in res.values()),
                            order_disagree=sum((x["lo_ocr_first"] > 0) != (x["lo_cand_first"] > 0) for x in res.values()))
        print(f"{split}: wrote {len(rows)} ({len(att)} attempted) -> {os.path.basename(out_path)} ({dt:.0f}s)", flush=True)
    if stats:
        mp = os.path.join(OUT_DIR, f"verif_{cand}_metrics.json")
        old = json.load(open(mp)) if os.path.exists(mp) else {}
        old.update(stats); old.update(model=os.path.basename(MODEL.rstrip("/")), ctx=ctx, prompt=PROMPT_VERSION,
                                      note="batched offline vLLM throughput on a GPU shared with other jobs")
        with open(mp + ".part", "w") as f: json.dump(old, f, indent=1)
        os.replace(mp + ".part", mp)

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [j.split("|") for j in os.environ["JOBS"].split(";") if j.strip()]
    vllm_engine(); print(f"ENGINE_READY {os.environ.get('RUN_TOKEN', '')}", flush=True)
    failed = []
    for cand, splits, ctx in jobs:
        print(f"=== verify {cand} {splits} ctx={ctx}", flush=True)
        try: run_job(tok, cand, splits, int(ctx))
        except Exception as e: print(f"FAILED job {cand}: {e!r}", flush=True); failed.append(cand)
    if failed: print("FAILED JOBS:", failed, flush=True); sys.exit(1)

if __name__ == "__main__":
    main()
