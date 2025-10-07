# NLI Transformer Fine-Tuning (ModernBERT)

Fine-tuning **ModernBERT** for **Natural Language Inference** on the All-NLI dataset (entailment / contradiction / neutral).  
Includes metric reporting (accuracy, precision/recall/F1), confusion matrices, and targeted error analysis.

## File
- `modernbert_nli_trainer.py` — end-to-end training, evaluation, confusion matrix visualization, and error analysis.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python modernbert_nli_trainer.py
```

## Notes
- Uses `answerdotai/ModernBERT-base` and a 1% split by default for fast runs.
- Swap to larger splits for full-scale results.
