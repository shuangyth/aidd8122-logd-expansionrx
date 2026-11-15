# Data sources and licence

- Dataset: OpenADMET-ExpansionRx challenge data (Hugging Face), licence CC BY 4.0.
  https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data
- Files used: `expansion_data_train.csv` (SHA256 c5214ad8...f6e4), `expansion_data_test.csv` (SHA256 e21c0ef1...c85e).
  Full hashes, download time and Hugging Face revision are written to `results/data_provenance.json`.
- Challenge description (early-to-later-stage split): https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/
- Raw CSVs are not committed; `run_project.py` downloads them and verifies the SHA256.
- The time ordering of the split follows the publisher's description; the CSVs carry no per-record dates.
- Test labels are public, so this is a retrospective evaluation, not a prospective blind test.
