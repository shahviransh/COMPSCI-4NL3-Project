# 20 Newsgroups Annotation
## COMPSCI 4NL3 - Group 37

## Setup

```bash
conda env create -f annotation/environment.yml
conda activate annotation
```

## Annotating

### Phase 1: Initial Annotation (100 documents per person)

1. Read `annotation_guidelines.md`
2. Run `python main.py`
3. Enter your name, choose mode 1
4. Annotate documents (1-20 or q to quit)

### Phase 2: Re-annotation Setup (one person)

```bash
python prepare_reannotation.py
```

### Phase 3: Re-annotation (15 documents per person)

1. Run `python main.py`
2. Enter your name, choose mode 2
3. Annotate assigned documents

### Phase 4: Calculate Agreement (one person)

```bash
python calculate_agreement.py
```

## Output Files

- `annotations_<name>.json` - Your annotations
- `annotations_<name>_backup.json` - Auto backup

## Agreement Interpretation

| Kappa/Alpha | Interpretation |
|-------------|----------------|
| < 0.00 | Poor |
| 0.00-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |
