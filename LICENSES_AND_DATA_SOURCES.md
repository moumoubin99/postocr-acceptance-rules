# Source data and license notice

This package excludes source images, OCR strings, human reference strings, corrected candidate strings, and model weights. It supplies derived numeric evaluation evidence and hashes; source corpus licenses remain separate and are not overridden by this notice.

| Source | Provider/source | License evidence retained in the source project | Access |
|---|---|---|---|
| BLN600 | Callum Booth, Alan Thomas and Robert Gaizauskas; Gale/British Library newspaper excerpts | Original `data/readme.md` and `data/LICENSE` state CC BY-NC-ND 4.0; no permission to redistribute transformed source text is asserted here | Dataset: https://doi.org/10.15131/shef.data.25439023 ; paper: https://aclanthology.org/2024.lrec-main.219/ |
| ICDAR-2019 post-OCR | ICDAR post-OCR competition corpus; L3i / University of La Rochelle and BnF | The original attachment did not include a license. The official Zenodo description checked on 10 October 2026 gives separate source-specific licenses: EN1 NC-SA 3.0, DE sources BY/SA/NC-ND, FR sources CC0 or NC-SA, ES NC-SA, NL BY. These do not form a single unrestricted license. Confirm the applicable selected-source terms with the provider; no source text is included here | Provider: https://sites.google.com/view/icdar2019-postcorrectionocr/dataset ; dataset: https://zenodo.org/records/3515403 |
| HIPE-OCRepair-2026 / impresso snippets | HIPE-eval / Impresso project | Local upstream dataset documentation states CC BY-NC-SA 4.0 | https://github.com/hipe-eval/HIPE-OCRepair-2026-data |
| HIPE-OCRepair-2026 / Overproof | Overproof corpus, structured by HIPE-eval | Local upstream documentation states CC BY-NC-SA 4.0 for structured format/annotations. Its discussion of newspaper public-domain status is not a rights determination by the authors of this package | https://github.com/hipe-eval/HIPE-OCRepair-2026-data/blob/main/documentation/README-overproof.md |

No open-source licence is granted for the authors' code in this release. Copyright remains with the authors. Public availability of a file does not by itself grant an unrestricted reuse licence. No permission for third-party data is granted by this release. Numeric hashes make traceability possible after a reviewer independently obtains the source data; they do not provide the excluded texts or permit reconstructing them.

See `ACCESS_AND_REGENERATION.md` for obtaining the provider artifacts, release/version limits, placing them under the expected raw/data directories and regenerating candidates only when the necessary model artifacts and permissions are available.
