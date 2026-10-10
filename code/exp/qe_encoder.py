#!/usr/bin/env python3
"""B4: contextual QE encoder for selective acceptance.
A ByT5 encoder reads the byte pair (OCR, candidate) and regresses the acceptance utility
    u = (lev(ocr, gold) - lev(cand, gold)) / max(1, len(gold))      (>0: accepting helps)
Trained ONLY on the router's development split, cross-fitted with GroupKFold(FOLDS) over
documents: dev rows get out-of-fold predictions, test rows get the mean of the fold models.
The predictions are then a feature/score for the router (iter1_claim_ledger stacking) and
a standalone acceptance score.

Usage: CAND=byt5base8_blnocr python exp/qe_encoder.py
  reads  candidates_{CAND}_dev.jsonl / _test.jsonl from DATA_DIR
  writes qe_{CAND}_dev.jsonl (oof) / qe_{CAND}_test.jsonl and qe_{CAND}_metrics.json to OUT_DIR
Env: BASE_MODEL (default google/byt5-small), EPOCHS 4, LR 1e-4, BATCH 32, FOLDS 5,
SEED 20260526, MAXLEN 768, LIMIT/MAX_STEPS (smoke tests only)."""
import os, json, time, random, math
import numpy as np, torch
from torch import nn
from rapidfuzz.distance import Levenshtein
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from transformers import AutoTokenizer, T5EncoderModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
OUT = os.environ.get("OUT_DIR", DATA)
CAND = os.environ["CAND"]
BASE = os.environ.get("BASE_MODEL", "google/byt5-small")
EPOCHS = float(os.environ.get("EPOCHS", 4)); LR = float(os.environ.get("LR", 1e-4))
BATCH = int(os.environ.get("BATCH", 32)); FOLDS = int(os.environ.get("FOLDS", 5))
SEED = int(os.environ.get("SEED", 20260526)); MAXLEN = int(os.environ.get("MAXLEN", 768))
LIMIT = int(os.environ.get("LIMIT", 0)); MAX_STEPS = int(os.environ.get("MAX_STEPS", -1))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows[:LIMIT] if LIMIT else rows

def util(r):
    return (Levenshtein.distance(r["ocr"], r["gold"]) - Levenshtein.distance(r["cand"], r["gold"])) / max(1, len(r["gold"]))

class QE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = T5EncoderModel.from_pretrained(BASE)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.d_model, 1))
    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

def batches(tok, rows, ys, shuffle, rng):
    idx = list(range(len(rows)))
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        b = idx[i:i+BATCH]
        enc = tok([rows[k]["ocr"] for k in b], [rows[k]["cand"] for k in b], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAXLEN)
        yield b, enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV), torch.tensor([ys[k] for k in b], dtype=torch.float32, device=DEV)

def fit(tok, rows, ys, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    model = QE().to(DEV); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = MAX_STEPS if MAX_STEPS > 0 else math.ceil(EPOCHS * math.ceil(len(rows) / BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1) / max(1, int(0.06*total))) * max(0.0, (total - s) / total))
    amp = DEV == "cuda" and torch.cuda.is_bf16_supported(); step = 0
    while step < total:
        for _, ids, att, y in batches(tok, rows, ys, True, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model(ids, att)
            loss = nn.functional.huber_loss(pred.float(), y, delta=0.05)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step >= total: break
    return model.eval()

@torch.no_grad()
def predict(model, tok, rows):
    out = np.zeros(len(rows)); amp = DEV == "cuda" and torch.cuda.is_bf16_supported()
    for b, ids, att, _ in batches(tok, rows, [0.0]*len(rows), False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out[b] = model(ids, att).float().cpu().numpy()
    return out

def main():
    t0 = time.time()
    dev = load(os.path.join(DATA, f"candidates_{CAND}_dev.jsonl"))
    test = load(os.path.join(DATA, f"candidates_{CAND}_test.jsonl"))
    tok = AutoTokenizer.from_pretrained(BASE)
    ys = [max(-0.5, min(0.5, util(r))) for r in dev]
    groups = [r["doc_id"] for r in dev]
    oof = np.zeros(len(dev)); tp = np.zeros(len(test))
    for f, (tr, va) in enumerate(GroupKFold(FOLDS).split(dev, groups=groups)):
        m = fit(tok, [dev[i] for i in tr], [ys[i] for i in tr], SEED + f)
        oof[va] = predict(m, tok, [dev[i] for i in va]); tp += predict(m, tok, test) / FOLDS
        print(f"fold {f}: n_tr={len(tr)} n_va={len(va)} rho={spearmanr(oof[va], [ys[i] for i in va])[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None
    os.makedirs(OUT, exist_ok=True)
    for name, rows, pr in (("dev", dev, oof), ("test", test, tp)):
        with open(os.path.join(OUT, f"qe_{CAND}_{name}.jsonl"), "w") as fh:
            for r, p in zip(rows, pr):
                fh.write(json.dumps({"doc_id": r["doc_id"], "seg_idx": r.get("seg_idx"), "qe": float(p)}) + "\n")
    yt = [util(r) for r in test]
    met = dict(cand=CAND, base=BASE, n_dev=len(dev), n_test=len(test), folds=FOLDS, epochs=EPOCHS, lr=LR,
               oof_spearman=float(spearmanr(oof, ys)[0]), test_spearman=float(spearmanr(tp, yt)[0]),
               seconds=round(time.time() - t0))
    json.dump(met, open(os.path.join(OUT, f"qe_{CAND}_metrics.json"), "w"), indent=1)
    print(json.dumps(met), flush=True)

if __name__ == "__main__":
    main()
