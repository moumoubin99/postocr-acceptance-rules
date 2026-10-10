# Post-OCR correction acceptance: code and numerical data

Research materials for **When Should a Post-OCR Correction Be Accepted? A Controlled Comparison of Acceptance Rules Across Correctors, Corpora and Operating Points** by Jibin Zhang and Hui Gui.

This release compares acceptance rules on frozen correction candidates; it does not introduce a new OCR corrector. It includes numerical evaluation evidence and analysis code, with no original OCR, reference or corrected-text strings, source images, model weights, manuscript drafts or author correspondence.

## Contents

| Location | Contents |
| --- | --- |
| `numeric/` | 52 development/test NumPy archives for 26 primary configurations, containing 102,578 segment records, numerical features, errors, decisions and SHA256 digests |
| `results/` | 339 original aggregate result JSON files |
| `splits/` | Hash-only document partition maps, training segment hashes and overlap audit reports |
| `code/exp/` | Historical experiment, candidate-generation and evaluation code |
| `code/paper/` | Table and figure generation code |
| `figure_data/`, `reference/` | Plot inputs, reference tables, numerical definitions and vector figures |
| `ACCESS_AND_REGENERATION.md` | Source corpus access, expected input layouts and reproducibility boundaries |
| `LICENSES_AND_DATA_SOURCES.md`, `LICENSES.md` | Source-specific rights and release licensing scope |
| `SHA256SUMS.json` | SHA256 digests of the files in this release |

The obsolete historical MD5 claim manifest and previously generated verification outputs are excluded. The supplied original aggregate results and numeric archives are preserved byte-for-byte. Generated outputs are written to `generated/` when a reader runs the scripts.

## Verify the supplied evidence

Install the dependencies appropriate to the desired check in an isolated Python environment. `requirements-numeric-rebuild.txt` records the CPU verification environment, not the historical corrector-training environment.

```sh
python verify_numeric.py
python verify_splits.py
python reproduce_tables_figures.py
```

Optional CPU re-fit of the small main segment router:

```sh
python recheck_router.py
```

The numeric evidence reproduces all 52 never/always CER baselines. In the earlier local CPU check, the re-fitted main router and edit-cap/available confidence decisions reproduced all 392 relevant CER/harm entries. Table regeneration reproduced 14 tables and 1,475 reference number values; eight numeric plots can be regenerated. These checks are distinct from model training or inference.

## Reproducibility limits

The attachment alone cannot rerun corrector training/inference, source-text near-duplicate clustering, edit-level text reconstruction, all bootstrap/verifier/risk-control analyses or every secondary configuration. Full regeneration needs independently obtained source corpora and the matching model and candidate-generation artifacts. Original trained ByT5 checkpoints and complete historical inference/training environment records are not supplied.

Source corpora are BLN600, ICDAR-2019 post-OCR correction data and the English impresso/Overproof sets of HIPE-OCRepair-2026. Obtain them from their providers under the applicable terms described in `LICENSES_AND_DATA_SOURCES.md` and `ACCESS_AND_REGENERATION.md`. This release does not grant permission to redistribute those texts or models.

## Versioned download

Use the `v1.0.1` release archive and its SHA256 checksum for the version referenced in the submission. The release archive contains the same research files as the release commit. Verification of archive integrity does not establish exact historical model reproducibility or journal acceptance.
