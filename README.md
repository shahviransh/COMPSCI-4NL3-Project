# COMPSCI 4NL3 — Multi-Class Text Classification on 20 Newsgroups

**Group 37** · Abdul-Hadi Siddiqui, Eduardo Salvacion, Viransh Shah  
`{salvacie,sidda1,shahv47}@mcmaster.ca`

Course project for COMPSCI 4NL3 (Natural Language Processing): topic classification on the [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset, with a human annotation phase and three supervised models of increasing complexity.

## Overview

We classify Usenet posts into one of 20 topic categories after stripping headers, footers, and quotes (`remove=('headers','footers','quotes')`) so models cannot leak labels from metadata. The work has two parts:

1. **Annotation** — each teammate labeled 100 documents (plus overlap re-annotation) to calibrate the task and measure agreement.
2. **Modeling** — three systems plus baselines, evaluated on the held-out test set (7,532 documents).

| Model | Description | Test Acc | Macro F1 |
|-------|-------------|----------|----------|
| Random / Majority | Untrained baselines | ~5% | — |
| LR + TF-IDF | Trained baseline | 68.4% | 0.667 |
| **M1** Multi-TF-IDF + LinearSVC | Word + char *n*-grams | 70.5% | 0.693 |
| **M2** RoBERTa-large (best) | Fine-tuned encoder | **73.8%** | **0.726** |
| **M3** ModernBERT-large | Long-context encoder | 72.5% | 0.715 |

## Repository layout

```
├── src/                          # Annotation tooling
│   ├── main.py                   # Interactive annotation UI
│   ├── prepare_reannotation.py   # Build overlap assignments
│   ├── calculate_agreement.py    # Cohen / Fleiss / Krippendorff
│   ├── export_all_annotations.py # Export to Excel
│   ├── generate_annotation_report.py
│   ├── environment.yml
│   └── README.md
├── docs/
│   ├── Project Proposal.pdf
│   ├── report/                   # Annotation setup report (TeX + PDF)
│   └── final report/
│       ├── train_models.py       # Train & evaluate M1–M3 + baselines
│       └── Final Project TeX/    # Final report (ACL-style)
└── README.md
```

## Annotation (`src/`)

### Setup

```bash
cd src
conda env create -f environment.yml
conda activate annotation
```

Requires Python ≥3.7, scikit-learn, numpy, and `krippendorff`. Excel export additionally needs `pandas` (and an Excel engine such as `openpyxl`).

### Workflow

| Phase | What to do |
|-------|------------|
| 1. Initial labeling | `python main.py` → mode 1 · 100 docs per person |
| 2. Overlap setup | `python prepare_reannotation.py` (once) |
| 3. Re-annotation | `python main.py` → mode 2 · ~15 docs per person |
| 4. Agreement | `python calculate_agreement.py` |
| Optional | `python generate_annotation_report.py` · `python export_all_annotations.py` |

Guideline details and label definitions are in the annotation setup report under [`docs/report/`](docs/report/).

See [`src/README.md`](src/README.md) for the full annotation guide.

## Training models (`docs/final report/`)

```bash
cd "docs/final report"

# Classical only
pip install scikit-learn matplotlib seaborn numpy

# Transformer models (GPU recommended)
pip install torch transformers datasets accelerate

python train_models.py --models 1          # LinearSVC
python train_models.py --models 2          # RoBERTa-large
python train_models.py --models 3          # ModernBERT-large
python train_models.py --models 1 2 3      # all three
```

Metrics and confusion matrices are written under `docs/final report/Final Project TeX/figures/`.

## Documentation

| Document | Location |
|----------|----------|
| Project proposal | [`docs/Project Proposal.pdf`](docs/Project%20Proposal.pdf) |
| Annotation setup report | [`docs/report/Annotation Setup.pdf`](docs/report/Annotation%20Setup.pdf) |
| Final report (PDF) | [`docs/final%20report/Final%20Project%20TeX/main.pdf`](docs/final%20report/Final%20Project%20TeX/main.pdf) |
| Final report (source) | [`docs/final report/Final Project TeX/main.tex`](docs/final%20report/Final%20Project%20TeX/main.tex) |

To rebuild the final PDF:

```bash
cd "docs/final report/Final Project TeX"
latexmk -pdf main.tex
```

## License

This repository is released under the [GNU GPLv3](LICENSE).
