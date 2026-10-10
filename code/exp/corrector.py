#!/usr/bin/env python3
"""
M1: ByT5-small post-OCR corrector (neural corrector, stronger than SymSpell).
  train : fine-tune google/byt5-small on BLN600 train segments (ocr -> gold), save.
  gen   : load model, generate candidate corrections for a split, write JSONL.

Usage:
  python exp/corrector.py train
  python exp/corrector.py gen dev
  python exp/corrector.py gen test
  python exp/corrector.py gen dev,test          # several splits, one model load
iter-2 env additions: DATA_DIR (segments dir), OUT_DIR (candidates dir, default DATA_DIR),
MODEL_DIR (explicit model path), GEN_BATCH. gen now also records min_lp (min token
log-prob, Kim-2021-style confidence), n_tok and the per-token log-probs (lps), and passes
through seg_idx/lang/group/dup_train when present.
"""
import os, sys, json, time, random, shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("OUT_DIR", DATA)
DATASET = os.environ.get("DATASET", "bln600")              # data files to read
MODEL_DATASET = os.environ.get("MODEL_DATASET", DATASET)   # which trained model to load (gen)
CORR_NAME = os.environ.get("CORR_NAME", "byt5")            # candidate-file tag
BASE = os.environ.get("BASE_MODEL", "google/byt5-small")   # override for a stronger corrector
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "exp", f"{CORR_NAME}_corrector_{MODEL_DATASET}"))
PASS = ("seg_idx", "ocr_sep", "gold_sep", "orig_id", "lang", "group", "dup_train")
MAXLEN = 384  # ByT5 byte-level; raised from 192 to avoid truncating gold (CER contamination)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

class SegDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(r["ocr"], max_length=MAXLEN, truncation=True)
        y = self.tok(text_target=r["gold"], max_length=MAXLEN, truncation=True)
        x["labels"] = y["input_ids"]
        return x

def train():
    assert torch.cuda.is_available() or os.environ.get("ALLOW_CPU"), "CUDA not available"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model.config.use_cache = False  # required with gradient_checkpointing
    train_rows = load_jsonl(os.path.join(DATA, f"segments_{DATASET}_train.jsonl"))
    dev_rows = load_jsonl(os.path.join(DATA, f"segments_{DATASET}_dev.jsonl"))
    # length audit (bytes) to confirm MAXLEN is safe
    def tlen(s): return len(tok(s, add_special_tokens=True)["input_ids"])
    samp = random.Random(7).sample(train_rows, min(2000, len(train_rows)))
    for k in ("ocr", "gold"):
        p = np.percentile([tlen(r[k]) for r in samp], [95, 99, 100])
        print(f"  len[{k}] p95/p99/max = {p.astype(int).tolist()} (MAXLEN={MAXLEN})")
    tr = SegDS(train_rows, tok)
    dv = SegDS(random.Random(13).sample(dev_rows, min(int(os.environ.get("EVAL_N", 500)), len(dev_rows))), tok)
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR + "_ckpt",  # per-run dir: parallel jobs must not share it
        per_device_train_batch_size=int(os.environ.get("BATCH", 4)),
        per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH", 8)),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", 4)),
        learning_rate=float(os.environ.get("LR", 1e-4)),
        num_train_epochs=float(os.environ.get("EPOCHS", 3)), warmup_ratio=0.05, weight_decay=0.01,
        logging_steps=100, eval_strategy="epoch", save_strategy="no",
        bf16=bf16, fp16=use_cuda and not bf16, gradient_checkpointing=True,
        seed=int(os.environ.get("SEED", 42)),
        report_to=[], dataloader_num_workers=int(os.environ.get("WORKERS", 2)),
        predict_with_generate=False,
        max_steps=int(os.environ.get("MAX_STEPS", -1)))  # >0 only for smoke tests
    coll = DataCollatorForSeq2Seq(tok, model=model)
    tr_ = Seq2SeqTrainer(model=model, args=args, train_dataset=tr,
                         eval_dataset=dv, data_collator=coll)
    t0 = time.time(); tr_.train(); print(f"train time {time.time()-t0:.0f}s")
    tmp = MODEL_DIR + ".tmp"   # atomic: a crashed save never leaves a half-written MODEL_DIR
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp); tok.save_pretrained(tmp)
    old = MODEL_DIR + ".old"   # swap: at every instant a complete model exists at MODEL_DIR or .old
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(MODEL_DIR): os.replace(MODEL_DIR, old)
    os.replace(tmp, MODEL_DIR); shutil.rmtree(old, ignore_errors=True)
    print("saved ->", MODEL_DIR)

@torch.no_grad()
def gen(splits):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(dev).eval()
    model.config.use_cache = True; model.generation_config.use_cache = True  # training saved False
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in splits.split(","):
        gen_split(split, tok, model, dev)

def gen_split(split, tok, model, dev):
    rows = load_jsonl(os.path.join(DATA, f"segments_{DATASET}_{split}.jsonl"))
    if int(os.environ.get("LIMIT", 0)): rows = rows[:int(os.environ["LIMIT"])]
    out = os.path.join(OUT_DIR, f"candidates_{CORR_NAME}_{DATASET}_{split}.jsonl")
    B = int(os.environ.get("GEN_BATCH", 16)); t0 = time.time()
    with open(out + ".part", "w") as f:
        for i in range(0, len(rows), B):
            batch = rows[i:i+B]
            enc = tok([r["ocr"] for r in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAXLEN).to(dev)
            out_g = model.generate(**enc, max_length=MAXLEN, num_beams=1,
                                   output_scores=True, return_dict_in_generate=True)
            seqs = out_g.sequences
            cands = tok.batch_decode(seqs, skip_special_tokens=True)
            # sequence-level confidence = mean token log-prob over generated (non-pad) tokens
            trans = model.compute_transition_scores(seqs, out_g.scores, normalize_logits=True)
            gen_tok = seqs[:, 1:]
            mask = (gen_tok != tok.pad_token_id).float()
            conf = (trans * mask).sum(1) / mask.sum(1).clamp(min=1)
            mn = trans.masked_fill(mask == 0, 0.0).min(1).values
            for k, (r, c, sc) in enumerate(zip(batch, cands, conf.tolist())):
                m = mask[k].bool()
                lps = [round(x, 4) for x in trans[k][m].tolist()]
                rec = {"doc_id": r["doc_id"], "ocr": r["ocr"], "gold": r["gold"], "cand": c,
                       "score": sc, "min_lp": mn[k].item(), "n_tok": len(lps), "lps": lps}
                rec.update({p: r[p] for p in PASS if p in r})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % (B*20) == 0:
                print(f"  {split}: {i+len(batch)}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    os.replace(out + ".part", out)
    print(f"{split}: wrote {len(rows)} -> {os.path.basename(out)}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train": train()
    elif mode == "gen": gen(sys.argv[2])
    elif mode == "train_gen":  # train, then generate for the given splits in the same process
        train(); gen(sys.argv[2])
