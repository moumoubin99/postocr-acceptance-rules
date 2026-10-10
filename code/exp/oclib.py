#!/usr/bin/env python3
"""Shared lib: SymSpell corrector, CER, text-only features (inference-safe)."""
import os, re, json, statistics, math
import importlib.resources as ir
from rapidfuzz.distance import Levenshtein
from symspellpy import SymSpell, Verbosity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

_sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
_sym.load_dictionary(str(ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt"), 0, 1)
DICT = _sym._words
_pre = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)

def in_dict(t): return t.lower() in DICT
def _rcase(o, r): return r.upper() if o.isupper() else (r[:1].upper()+r[1:] if o[:1].isupper() else r)

def symspell_correct(seg):
    out = []
    for tok in seg.split():
        m = _pre.match(tok); pre, core, suf = m.group(1), m.group(2), m.group(3)
        if len(core) < 2 or not core.isalpha() or in_dict(core):
            out.append(tok); continue
        s = _sym.lookup(core, Verbosity.TOP, max_edit_distance=2, include_unknown=False)
        if not s or s[0].term == core.lower():
            out.append(tok); continue
        out.append(pre + _rcase(core, s[0].term) + suf)
    return " ".join(out)

def cer(a, b):
    if not b: return 0.0 if not a else 1.0
    return Levenshtein.distance(a, b) / len(b)

def features(ocr, cand):
    """Text-only features from (ocr, candidate) — usable at inference, no gold."""
    n = max(1, len(ocr)); toks = ocr.split(); ct = cand.split()
    oov = sum(1 for t in toks if (m:=_pre.match(t).group(2)) and m.isalpha() and len(m) >= 2 and not in_dict(m))
    oov_cand = sum(1 for t in ct if (m:=_pre.match(t).group(2)) and m.isalpha() and len(m) >= 2 and not in_dict(m))
    # token-level change count via alignment-free zip on min length + length delta
    changed = sum(1 for a, b in zip(toks, ct) if a != b) + abs(len(toks) - len(ct))
    ed = Levenshtein.distance(ocr, cand)
    logfreqs = [math.log10(DICT[t.lower()] + 1) for t in toks if t.lower() in DICT]
    return [
        len(ocr), len(toks),
        statistics.mean(len(t) for t in toks) if toks else 0.0,
        oov / max(1, len(toks)),                       # oov_frac
        oov_cand / max(1, len(ct)),                    # oov_frac after correction
        changed / max(1, len(toks)),                   # frac tokens changed
        ed / n,                                        # char edit-density of correction
        (len(cand) - len(ocr)) / n,                    # length delta ratio
        sum(c.isdigit() for c in ocr) / n,
        sum((not c.isalnum() and not c.isspace()) for c in ocr) / n,
        sum(c.isupper() for c in ocr) / n,
        sum(ord(c) > 127 for c in ocr) / n,
        statistics.mean(logfreqs) if logfreqs else 0.0,
    ]

FEATNAMES = ["char_len","n_tokens","mean_tok_len","oov_frac","oov_frac_cand",
             "frac_changed","edit_density","len_delta","digit_ratio","punct_ratio",
             "upper_ratio","nonascii_ratio","mean_logfreq"]

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
