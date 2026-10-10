# Access and regeneration guide

The commands for `verify_numeric.py`, `reproduce_tables_figures.py`, `verify_splits.py` and `recheck_router.py` operate entirely on the supplied attachment. Obtaining source text or a GPU is unnecessary for these checks. Full candidate generation is a separate, more demanding workflow; the package does not claim that the historical corrector runs can be reproduced exactly from the attachment alone.

## Obtain the original corpora from their providers

The following provider links were checked on 10 October 2026. They are dataset landing/release pages, not a guarantee that an unauthenticated direct download remains available.

1. **BLN600:** the authors' paper identifies the Sheffield dataset DOI [10.15131/shef.data.25439023](https://doi.org/10.15131/shef.data.25439023). Follow this dataset landing page, obtain the provider's BLN600 ZIP and follow its README extraction instructions. Retain the provider README and CC BY-NC-ND 4.0 license. The input needed by this study is the matching `.txt` files in `Ground Truth/` and `OCR Text/`; images are not needed for the post-OCR evaluation. The package does not redistribute the original ZIP or either text directory.
2. **ICDAR-2019:** use the [official competition dataset page](https://sites.google.com/view/icdar2019-postcorrectionocr/dataset), which links to the [official Zenodo record 3515403](https://zenodo.org/records/3515403). The record supplies `ICDAR2019-POCR-ground-truth.zip` and describes training/evaluation/full partitions and the different source-specific licenses. Obtain the training and evaluation material with released reference text and preserve its notices. This study uses EN, DE, FR, ES and NL; it does not use Finnish. The scripts expect `training_18M_without_Finnish/` and `evaluation_4M_without_Finnish/` below the raw input directory; if the provider archive has an additional wrapper directory, point `RAW` at the directory containing those two folders. The `full_22M` union must not be substituted into both partitions. No single unrestricted redistribution license is asserted. If access or allowed use is unclear for a selected subcorpus, contact the original provider via the official record.
3. **HIPE-OCRepair-2026 external sets:** use the [provider repository](https://github.com/hipe-eval/HIPE-OCRepair-2026-data) and the [v0.9.5 final unmasked-data release](https://github.com/hipe-eval/HIPE-OCRepair-2026-data/releases/tag/v0.9.5). Keep the complete release directory layout and documentation. The study's builder reads the English JSONL files under `data/v0.9/impresso-snippets/en/` and `data/v0.9/overproof/en/`. The local source project contains that layout; the attachment does not establish an exact upstream commit for the historical experiments. A fresh provider release must therefore be checked against the recorded input hashes before assuming it is the identical input. See the dataset-specific notices, including the structured-format/annotation CC BY-NC-SA 4.0 terms.

## Isolated input layout for historical experiment code

Create a separate local workspace named `rebuild/`, copy `code/exp/` from this attachment into `rebuild/exp/`, and leave the attachment's `results/`, `numeric/` and `reference/` unchanged. The historical helpers infer their project root as the parent of `exp/`.

```text
rebuild/
  exp/                         # copy of this attachment's code/exp/
  data/
    Ground Truth/*.txt         # provider BLN600 reference files
    OCR Text/*.txt             # provider BLN600 OCR files
  raw/
    icdar2019/
      training_18M_without_Finnish/EN/...  # plus DE, FR, ES, NL
      evaluation_4M_without_Finnish/EN/... # plus DE, FR, ES, NL
    HIPE-OCRepair-2026-data/
      data/v0.9/impresso-snippets/en/*_train_en.jsonl
      data/v0.9/impresso-snippets/en/*_dev_en.jsonl
      data/v0.9/impresso-snippets/en/*_test_en.jsonl
      data/v0.9/overproof/en/*_train_en.jsonl
      data/v0.9/overproof/en/*_dev_en.jsonl
      data/v0.9/overproof/en/*_test_en.jsonl
  models/                      # independently obtained weights/checkpoints
```

From `rebuild/`, the builders' actual interfaces are:

```sh
python exp/build_bln600_ocrseg.py
RAW=raw/icdar2019 OUT=data python exp/build_icdar_clean.py
HIPE_DATA=raw/HIPE-OCRepair-2026-data OUT=data python exp/build_external.py
```

These commands are documented interfaces, not reruns performed in preparing this package. BLN600 variants use the document shuffle seed 20260526; the ICDAR builder's defaults use EN/DE/FR/ES/NL, development fraction 0.2, Jaccard 0.5, containment 0.8, and DE/FR caps 300 training/development groups and 80 test groups. Verify generated document hashes against `splits/document_split_map.jsonl.gz` before proceeding. Legacy BLN600 segmentation is a separate historical sensitivity condition, not a replacement for OCR-only segmentation.

## Models and candidate generation

The historical scripts contain the generation, cleaning and confidence-rescoring implementations. No model weights or original corrected candidate strings are included here.

| Family | Provider model identifiers / required artifact | Actual script interface |
|---|---|---|
| SymSpell | `symspellpy` and its bundled English frequency dictionary | `DATASET=blnocr DATA_DIR=data OUT_DIR=data python exp/gen_symspell.py dev,test` |
| ByT5-small/base | Google `google/byt5-small` / `google/byt5-base`, plus the study's fine-tuned checkpoint matching the configuration | `exp/corrector.py train` and `gen dev,test`; controls include DATASET, MODEL_DATASET, CORR_NAME, BASE_MODEL, MODEL_DIR, DATA_DIR and OUT_DIR |
| Qwen | Qwen `Qwen/Qwen2.5-7B-Instruct`; configuration-specific context and prompt settings | `DATASET=blnocr CORR_NAME=qwen LLM_MODEL=Qwen/Qwen2.5-7B-Instruct DATA_DIR=data OUT_DIR=data python exp/llm_corrector.py dev,test` |
| Llama / prompt variants | Meta `meta-llama/Llama-3.1-8B-Instruct`, subject to provider access/terms; this family and Qwen prompt B/C are secondary configurations | Same LLM helper with provider-approved weights and the appropriate JOBS/prompt variant settings |

ByT5 trained checkpoints, original weight revision identifiers, historical training environments and all original inference metadata are not released in this attachment. The eight-epoch base-model and multilingual configurations require their own original training settings/checkpoints; the script defaults alone do not recover those runs. An independently trained replacement may produce different candidates and cannot be labelled as the original experiment. Similarly, LLM decoding may vary with model revision, tokenizer, backend and hardware. Consult the manuscript and code, retain exact weight/tokenizer revisions, and compare any independently generated files to the configuration's `evaluation.json::input_sha256` and the released candidate digest arrays. Hash equality establishes identity of the compared content, not permission to redistribute it.

Candidate naming is `data/candidates_<corrector>_<dataset>_<split>.jsonl`. The CPU evaluator expects original `ocr`, `gold`, `cand`, `doc_id` and segment metadata; neural confidence analyses additionally need score/min_lp/n_tok/lps. Edit-level confidence needs `toklp_<corrector>_<dataset>_<split>.jsonl`, generated through `exp/rescore_tokens.py` from the checkpoint/tokenizer that produced that same candidate. QE/verifier secondary analyses have further predictions and model dependencies not replaced by the primary numeric archive.

The source experiment code additionally imports packages such as PyTorch, Transformers, vLLM, datasketch and SymSpell. `requirements-numeric-rebuild.txt` records the actual CPU verification environment only, not a complete historical GPU lockfile. Do not assume that installing it makes full model generation reproducible. Source-text rebuilding and model generation are not invoked by any of the attachment's lightweight verification commands.

## What the attachment can independently verify

- All file SHA256 digests and numeric schemas.
- Frozen OCR/candidate error aggregation for the 26 primary configurations.
- Supplied main segment-router, edit-cap and available mean/min-confidence scores, thresholds and decisions at four operating points; optional re-fit from numeric features only.
- Document partition separation and ICDAR normalized-OCR training-duplicate flags.
- All 14 reference tables and 1,475 number values from the original aggregate JSON, and eight numeric plots from their included inputs.

The attachment cannot independently re-run source-text near-duplicate clustering, corrector inference/training, document reassembly, edit-level text reconstruction or every secondary experiment. Obtain permitted source data and the matching private artifacts from the original providers/authors for those checks; this statement does not promise unrestricted release or provider permission.
