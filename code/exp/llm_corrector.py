#!/usr/bin/env python3
"""LLM post-OCR corrector (Qwen2.5-Instruct) — the 'No Free Lunches' setting.
Outputs candidates_<CORR_NAME>_<DATASET>_<split>.jsonl with
{doc_id, ocr, gold, cand, raw, score, min_lp, n_tok, lps, raw_score, raw_min_lp, ...pass-through}.
score/min_lp/lps are the log-probs of the FINAL stored candidate (after clean()), rescored
under the same prompt (+ the end-of-turn token); raw_* describe the raw generation.

Usage: python exp/llm_corrector.py dev[,test,...]           (one job from env)
       JOBS="blnocr|dev,test|qwen|0;blnocr|dev,test|qwenctx2|2" python exp/llm_corrector.py
         -> several jobs (DATASET|splits|CORR_NAME|CTX) on ONE model/engine load.
A finished output with a matching fingerprint (<out>.meta.json: model, prompt version, ctx,
limit, input size+mtime) is skipped, so a crashed run resumes where it stopped.
Env: DATASET, LLM_MODEL (default Qwen/Qwen2.5-7B-Instruct), CORR_NAME (default llm),
LIMIT (cap rows for smoke test), BATCH, DATA_DIR, OUT_DIR,
BACKEND (vllm | hf; vllm is the fast GPU path, hf works on CPU for smoke tests),
CTX (0 = correct the segment alone; k>0 = show k neighbouring segments of the same
     document on each side as read-only context and ask for the target only),
a JOBS entry may carry a 5th field, the prompt variant ("", p2, p3; see VARIANTS), END_TOKEN (end-of-turn token),
DOMAIN (prompt text; default chosen from DATASET), TP (vLLM tensor parallel size),
GPU_UTIL (vLLM gpu_memory_utilization, default 0.85)."""
import os, sys, json, time, re, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("OUT_DIR", DATA)
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LIMIT = int(os.environ.get("LIMIT", 0))
B = int(os.environ.get("BATCH", 16))
BACKEND = os.environ.get("BACKEND", "vllm")
PROMPT_VERSION = "v2-2026-10"
PASS = ("seg_idx", "ocr_sep", "gold_sep", "orig_id", "lang", "group", "dup_train")
LANGNAME = {"EN": "English", "DE": "German", "FR": "French", "ES": "Spanish", "NL": "Dutch"}
END = os.environ.get("END_TOKEN", "")   # end-of-turn token; resolved from the tokenizer in main() when unset
# prompt variants (iter-3 robustness): "" = the v2 prompt used for every earlier run; p2 = terse paraphrase;
# p3 = conservative ("leave unchanged when unsure"). A variant is the optional 5th field of a JOBS entry.
VARIANTS = ("", "p2", "p3")

def domain(dataset, row):
    if os.environ.get("DOMAIN"): return os.environ["DOMAIN"]
    if dataset.startswith(("bln", "hipe")): return "19th-century English newspaper text"
    if dataset.startswith("ext_"): return "historical English newspaper text"
    lang = LANGNAME.get(row.get("lang", ""), None)
    return f"historical {lang} documents" if lang else "historical documents"

def sys_prompt(dataset, row, variant=""):
    if variant == "p2":
        return (f"The user message contains text produced by OCR from {domain(dataset, row)}. Return the same text with "
                "the OCR errors corrected. Keep the historical spelling and every original word; make the fewest "
                "changes needed. Return only the corrected text.")
    if variant == "p3":
        return (f"You are a careful proofreader of OCR output from {domain(dataset, row)}. Correct a character or word "
                "only when you are confident it is an OCR recognition error; if you are unsure, leave it exactly as it "
                "is. Never rephrase, modernize, translate, complete or shorten the text. Output only the resulting "
                "text, with no quotes, labels, or commentary.")
    return (f"You correct OCR recognition errors in short excerpts of {domain(dataset, row)}. "
            "Fix obvious OCR mistakes (wrong/garbled characters, broken words, "
            "stray punctuation) but PRESERVE the original wording, proper nouns, and period spelling; "
            "do NOT modernize, paraphrase, translate, or add words. Output ONLY the corrected text, "
            "with no quotes, labels, or commentary.")

def user_msg(rows, i, ctx, variant=""):
    r = rows[i]
    if ctx <= 0 and variant == "p2":
        return f"Text:\n{r['ocr']}\n\nCorrection:"
    if ctx <= 0:
        return f"OCR text:\n{r['ocr']}\n\nCorrected text:"
    lo, hi = i, i
    while lo > 0 and i - lo < ctx and rows[lo-1]["doc_id"] == r["doc_id"]: lo -= 1
    while hi < len(rows)-1 and hi - i < ctx and rows[hi+1]["doc_id"] == r["doc_id"]: hi += 1
    before = " ".join(x["ocr"] for x in rows[lo:i]); after = " ".join(x["ocr"] for x in rows[i+1:hi+1])
    return (f"Context before (do not output):\n{before or '(none)'}\n\n"
            f"Context after (do not output):\n{after or '(none)'}\n\n"
            f"OCR text to correct:\n{r['ocr']}\n\n"
            "Corrected text (only the OCR text to correct, nothing from the context):")

def clean(out, ocr, variant=""):
    out = out.strip().strip('"').strip()
    pre = r"^(corrected text:|corrected:|here is the corrected text:?)\s*"
    if variant == "p2": pre = r"^(corrected text:|corrected:|correction:|here is the corrected text:?)\s*"   # p2 asks "Correction:"
    out = re.sub(pre, "", out, flags=re.I).strip()
    return out if out else ocr

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def max_new(rows):  # ~ chars/3 tokens for the output, generous head-room
    return min(1536, int(max(len(r["ocr"]) for r in rows) / 2.5) + 48)

def summarize(lps):
    fin = [x for x in lps if not math.isnan(x)]
    return sum(fin) / max(1, len(fin)), (min(fin) if fin else 0.0), fin

# ---------------------------------------------------------------- vLLM backend
def vllm_engine():
    if not hasattr(vllm_engine, "llm"):
        from vllm import LLM
        vllm_engine.llm = LLM(MODEL, dtype="bfloat16", tensor_parallel_size=int(os.environ.get("TP", 1)),
                              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.85)),
                              max_model_len=4096, seed=0)
    return vllm_engine.llm

def gen_vllm(tok, rows, prompts):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=max_new(rows), logprobs=1)
    res = []
    # the chat template already contains the special tokens: pass ids so no second BOS is added (Llama-3); for Qwen,
    # whose tokenizer adds no special tokens, this is the same input as the prompt string
    ids = [TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    for o in vllm_engine().generate(ids, sp):
        c = o.outputs[0]
        lps = [d[t].logprob if t in d else float("nan") for t, d in zip(c.token_ids, c.logprobs or [])]
        res.append((c.text, lps))
    return res

def score_vllm(tok, pairs):
    """pairs: [(prompt_ids, cont_ids)] -> per-pair log-probs of cont_ids given the prompt."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = vllm_engine().generate([TokensPrompt(prompt_token_ids=p + c) for p, c in pairs], sp)
    res = []
    for (p, c), o in zip(pairs, outs):
        pl = o.prompt_logprobs
        res.append([pl[len(p) + k][t].logprob if pl[len(p) + k] and t in pl[len(p) + k] else float("nan")
                    for k, t in enumerate(c)])
    return res

# ---------------------------------------------------------------- HF backend
def hf_model():
    import torch
    from transformers import AutoModelForCausalLM
    if not hasattr(hf_model, "m"):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model.m = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    return hf_model.m

def gen_hf(tok, rows, prompts):
    import torch
    model = hf_model(); res = []
    with torch.no_grad():
        for i in range(0, len(prompts), B):
            enc = tok(prompts[i:i+B], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=max_new(rows[i:i+B]), do_sample=False,
                               pad_token_id=tok.pad_token_id, output_scores=True,
                               return_dict_in_generate=True)
            tr = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
            newt = g.sequences[:, enc["input_ids"].shape[1]:]
            for j in range(newt.shape[0]):
                keep = [k for k, t in enumerate(newt[j].tolist()) if t != tok.pad_token_id]
                res.append((tok.decode(newt[j][keep], skip_special_tokens=True), [tr[j][k].item() for k in keep]))
    return res

def score_hf(tok, pairs):
    """Teacher-forced log-probs of cont_ids. lm_head is applied only at the continuation
    positions and batches are token-budgeted, so no [B, L, vocab] tensor is ever built."""
    import torch
    model = hf_model(); res = []; budget = int(os.environ.get("RESCORE_TOKENS", 8192))
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
    out = [None] * len(pairs); i = 0
    with torch.no_grad():
        while i < len(order):
            chunk = [order[i]]
            while i + len(chunk) < len(order) and \
                    (len(chunk) + 1) * max(len(pairs[k][0]) + len(pairs[k][1]) for k in chunk + [order[i + len(chunk)]]) <= budget:
                chunk.append(order[i + len(chunk)])
            L = max(len(pairs[k][0]) + len(pairs[k][1]) for k in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, k in enumerate(chunk):
                p, c = pairs[k]; ids[j, :len(p) + len(c)] = torch.tensor(p + c); att[j, :len(p) + len(c)] = 1
            hid = model.model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).last_hidden_state
            for j, k in enumerate(chunk):
                p, c = pairs[k]
                h = hid[j, len(p) - 1:len(p) + len(c) - 1]
                lp = torch.log_softmax(model.lm_head(h).float(), -1)
                out[k] = lp[torch.arange(len(c)), torch.tensor(c, device=lp.device)].tolist()
                del h, lp
            del hid, ids, att
            i += len(chunk)
    return out

# ---------------------------------------------------------------- jobs
def _code_md5():
    import hashlib
    return hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()

def _model_rev():  # sizes/mtimes of the local model files (a re-downloaded/changed model invalidates outputs)
    if not os.path.isdir(MODEL): return MODEL
    return sorted((f, os.path.getsize(os.path.join(MODEL, f)), int(os.path.getmtime(os.path.join(MODEL, f))))
                  for f in os.listdir(MODEL) if os.path.isfile(os.path.join(MODEL, f)))

def fingerprint(inp, ctx, dataset, variant=""):
    st = os.stat(inp)
    return json.loads(json.dumps(dict(model=os.path.abspath(MODEL) if os.path.isdir(MODEL) else MODEL, model_rev=_model_rev(),
                code=_code_md5(), prompt=PROMPT_VERSION,
                ctx=ctx, limit=LIMIT, dataset=dataset, input=os.path.basename(inp),
                input_size=st.st_size, input_mtime=int(st.st_mtime), domain=os.environ.get("DOMAIN", ""),
                variant=variant, end=END)))  # JSON-normalized (tuples -> lists)

def complete(path, rows):  # same acceptance rule as exp/validate_outputs.py
    try:
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        return len(recs) == len(rows) and all(
            r["doc_id"] == s["doc_id"] and r.get("seg_idx") == s.get("seg_idx") and isinstance(r["cand"], str)
            and math.isfinite(r["score"]) and math.isfinite(r["min_lp"]) for r, s in zip(recs, rows))
    except Exception:
        return False

def meta_matches(path, fp):
    try:
        return json.load(open(path)) == fp
    except Exception:   # truncated/garbled meta -> just regenerate
        return False

def run_job(tok, dataset, splits, corr, ctx, variant=""):
    assert variant in VARIANTS, f"unknown prompt variant {variant!r}"
    assert not (ctx > 0 and variant), "prompt variants are defined for ctx=0 only"
    gen = gen_vllm if BACKEND == "vllm" else gen_hf
    score = score_vllm if BACKEND == "vllm" else score_hf
    for split in splits.split(","):
        inp = os.path.join(DATA, f"segments_{dataset}_{split}.jsonl")
        out_path = os.path.join(OUT_DIR, f"candidates_{corr}_{dataset}_{split}.jsonl")
        fp = fingerprint(inp, ctx, dataset, variant)
        rows = load(inp)
        if LIMIT: rows = rows[:LIMIT]
        if os.path.exists(out_path) and meta_matches(out_path + ".meta.json", fp) and complete(out_path, rows):
            print(f"skip {os.path.basename(out_path)} (fingerprint match)", flush=True); continue
        t0 = time.time()
        prompts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt(dataset, r, variant)},
                                            {"role": "user", "content": user_msg(rows, i, ctx, variant)}],
                                           tokenize=False, add_generation_prompt=True)
                   for i, r in enumerate(rows)]
        res = gen(tok, rows, prompts)
        cands = [clean(txt, r["ocr"], variant) for r, (txt, _) in zip(rows, res)]
        pairs = [(tok(p, add_special_tokens=False)["input_ids"], tok(c + END, add_special_tokens=False)["input_ids"])
                 for p, c in zip(prompts, cands)]
        resc = score(tok, pairs)
        with open(out_path + ".part", "w") as f:
            for r, (txt, rlps), c, lps in zip(rows, res, cands, resc):
                sc, mn, fin = summarize(lps); rsc, rmn, _ = summarize(rlps)
                rec = {"doc_id": r["doc_id"], "ocr": r["ocr"], "gold": r["gold"], "cand": c, "raw": txt,
                       "score": sc, "min_lp": mn, "n_tok": len(lps), "lps": [round(x, 4) for x in fin],
                       "raw_score": rsc, "raw_min_lp": rmn, "raw_n_tok": len(rlps), "backend": BACKEND}
                rec.update({p: r[p] for p in PASS if p in r})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if os.path.exists(out_path + ".meta.json"): os.remove(out_path + ".meta.json")
        os.replace(out_path + ".part", out_path)
        with open(out_path + ".meta.part", "w") as f: json.dump(fp, f)
        os.replace(out_path + ".meta.part", out_path + ".meta.json")
        print(f"{split}: wrote {len(rows)} -> {os.path.basename(out_path)} ({time.time()-t0:.0f}s)", flush=True)

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    global END
    if not END:   # the token the chat template closes an assistant turn with (Qwen: <|im_end|>, Llama-3: <|eot_id|>)
        END = next((t for t in ("<|im_end|>", "<|eot_id|>") if t in tok.get_vocab()), None) or tok.eos_token
    assert len(tok(END, add_special_tokens=False)["input_ids"]) == 1, f"END token {END!r} is not a single token"
    print(f"END token {END!r}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.environ.get("JOBS"):
        jobs = [j.split("|") for j in os.environ["JOBS"].split(";") if j.strip()]
    else:
        jobs = [(os.environ.get("DATASET", "bln600"), sys.argv[1], os.environ.get("CORR_NAME", "llm"),
                 os.environ.get("CTX", "0"))]
    if BACKEND == "vllm":   # start the single engine up front (the launcher waits for this line)
        vllm_engine()
        # self-test: rescoring a greedy continuation must reproduce its generation log-probs,
        # otherwise fail now (the launcher then falls back to HF) instead of after a full pass
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        p = tok.apply_chat_template([{"role": "user", "content": "Say: the old mill"}], tokenize=False,
                                    add_generation_prompt=True)
        pid = tok(p, add_special_tokens=False)["input_ids"]
        o = vllm_engine().generate([TokensPrompt(prompt_token_ids=pid)], SamplingParams(temperature=0.0, max_tokens=8, logprobs=1))[0].outputs[0]
        g = [d[t].logprob for t, d in zip(o.token_ids, o.logprobs)]
        r = score_vllm(tok, [(pid, list(o.token_ids))])[0]
        diff = max(abs(a - b) for a, b in zip(g, r))
        assert len(r) == len(g) and diff < 0.05, f"vLLM rescoring self-test failed: {g} vs {r}"
        print(f"ENGINE_READY {os.environ.get('RUN_TOKEN', '')} (rescore self-test max diff {diff:.2e})", flush=True)
    failed = []
    for dataset, splits, corr, ctx, *var in jobs:
        print(f"=== job {dataset} {splits} {corr} ctx={ctx} variant={var[0] if var else ''}", flush=True)
        try:
            run_job(tok, dataset, splits, corr, int(ctx), var[0] if var else "")
        except Exception as e:  # keep the engine alive for the remaining jobs
            print(f"FAILED job {dataset} {corr}: {e!r}", flush=True); failed.append(f"{dataset}|{corr}")
    if failed:
        print("FAILED JOBS:", failed, flush=True); sys.exit(1)

if __name__ == "__main__":
    main()
