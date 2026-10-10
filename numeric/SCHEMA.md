# Numeric evidence schema

Each configuration directory contains `dev.npz`, `test.npz` and `evaluation.json`.

- SHA256 fields (`doc_sha256`, `segment_sha256`, `ocr_sha256`, `gold_sha256`, `candidate_sha256`) are uint8 matrices with 32 digest bytes per row. They are identifiers, never source text.
- `ocr_error`, `candidate_error`, `gold_length`, `ocr_length`, `candidate_length`: integer character counts. Micro CER is sum selected error / sum gold_length. Harm is the count of accepted attempted rows where candidate_error > ocr_error.
- `row_index`, `segment_index`, `doc_group_index`: alignment/grouping indices; `attempted`, `dup_train` and `accept_*`: binary flags. For ICDAR, `normalized_ocr_sha256` hashes OCR after whitespace collapsing, and `normalized_ocr_length` is its length; `language_code` uses 0=EN, 1=DE, 2=FR, 3=ES, 4=NL. A `dup_train` flag indicates that this normalized OCR appears among same-language training segments with normalized length >=20; it does not indicate a gold-string duplicate.
- `features_text`: 13 features in `code/exp/oclib.py::FEATNAMES`; `features_alignment`: 12 numeric alignment features from `iter2_ledger.py::align_feats`; `features_confidence` (when present): the seven features from `conf_feats`; optional `qe_encoder_score` is a supplied precomputed QE scalar.
- `damage_counts`: columns num_broken, num_fixed, cap_broken, cap_fixed, from evaluation-only damage proxies.
- `score_router_cpu_recheck`: newly recomputed main-router scores, development out-of-fold/test held out. These are explicitly distinguished from preserved original aggregate metrics. `score_edit_cap` is negative edit density; `score_conf_mean` and `score_conf_min` are available decoder log-probability scores.
- `accept_<rule>_<operating point>`: decisions using development-selected thresholds. `evaluation.json` records the threshold, the current CPU-check metrics, the preserved original aggregate metrics, and equality checks. Operating points are opt, 10pct, 25pct, 50pct; test harm is not guaranteed to equal its development target.

No policy output is presented as proof of statistical coverage or as original corrector inference. The optional router refit produces separate diagnostics and does not change the original result files.
