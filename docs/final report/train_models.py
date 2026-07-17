"""
Group 37 - COMPSCI 4NL3 (Winter 2026)
Multi-Class Text Classification on 20 Newsgroups

Usage
  # Specific model(s):
  python train_models.py --models NUM NUM NUM  (where NUM is 1, 2, or 3)

  # All three models sequentially:
  python train_models.py --models 1 2 3

Requirements
  Core:        pip install scikit-learn matplotlib seaborn numpy
  GPU models:  pip install torch transformers datasets accelerate
"""

import argparse
import json
import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# GPU auto-selection
def _auto_select_gpus(min_free_gib: float = 50.0) -> None:
    """
    Query nvidia-smi and restrict CUDA_VISIBLE_DEVICES to only the GPUs that
    have at least *min_free_gib* GiB of free VRAM.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        free_gpus = []
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                idx      = parts[0].strip()
                free_mib = float(parts[1].strip())
                if free_mib >= min_free_gib * 1024:
                    free_gpus.append(idx)

        if free_gpus:
            visible = ",".join(free_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            print(f"\n  [GPU] Auto-selected GPUs {visible} "
                  f"(each has ≥{min_free_gib:.0f} GiB free)")
        else:
            # Fall back: use all GPUs and hope for the best
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            print(f"\n  [GPU] WARNING — no GPU with ≥{min_free_gib:.0f} GiB free; "
                  f"falling back to all GPUs")
    except Exception as exc:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        print(f"\n  [GPU] nvidia-smi query failed ({exc}); using all GPUs")

# Paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.join(SCRIPT_DIR, "Final Project TeX", "figures")
CKPT_DIR      = os.path.join(SCRIPT_DIR, "model_checkpoints")
BERT_CLS_CKPT = os.path.join(CKPT_DIR, "bert_classifier")
DEBERTA_CKPT  = os.path.join(CKPT_DIR, "roberta_large_classifier")
for d in (FIGURES_DIR, CKPT_DIR, BERT_CLS_CKPT, DEBERTA_CKPT):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42


# DATA
def load_data():
    """Load 20 Newsgroups (no headers/footers/quotes), stratified val split."""
    print("=" * 65)
    print("Loading 20 Newsgroups  (remove=headers, footers, quotes)")
    print("=" * 65)
    train_raw = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    test_raw = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes"),
        random_state=RANDOM_STATE,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_raw.data, train_raw.target,
        test_size=0.15, random_state=RANDOM_STATE, stratify=train_raw.target,
    )
    X_te, y_te = test_raw.data, test_raw.target
    names = train_raw.target_names
    print(f"  train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}  classes={len(names)}\n")
    return X_tr, X_val, X_te, y_tr, y_val, y_te, names



# BASELINES
class RandomBaseline:
    def __init__(self, n_cls): self.n_cls = n_cls; np.random.seed(RANDOM_STATE)
    def fit(self, X, y): pass
    def predict(self, X): return np.random.randint(0, self.n_cls, size=len(X))

class MajorityBaseline:
    def fit(self, X, y): self.maj = Counter(y).most_common(1)[0][0]
    def predict(self, X): return np.full(len(X), self.maj)



# LR + TF-IDF BASELINE  (trained model baseline, no GPU)
def run_lr_baseline(X_tr, X_val, X_te, y_tr, y_val, y_te, names):
    """
    Logistic Regression with TF-IDF - the standard trained-model baseline.

    Uses word unigrams + bigrams (50 k features, sublinear TF) and the
    saga solver, which scales well to large sparse feature matrices and
    supports L2 regularisation with C=1.0.  Serves as the reference
    point against which all three main models are compared.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("\n" + "─" * 65)
    print("[Baseline - Trained]  Logistic Regression + TF-IDF")
    print("─" * 65)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])
    # Apply the same classical preprocessing for consistency
    X_tr_c  = [_preprocess_classical(t) for t in X_tr]
    X_val_c = [_preprocess_classical(t) for t in X_val]
    X_te_c  = [_preprocess_classical(t) for t in X_te]

    print("  Fitting...", flush=True)
    pipe.fit(X_tr_c, y_tr)

    val_preds = pipe.predict(X_val_c)
    print(f"\n  Validation  acc={accuracy_score(y_val, val_preds):.4f}  "
          f"F1={f1_score(y_val, val_preds, average='macro'):.4f}")

    preds = pipe.predict(X_te_c)
    result = evaluate("LR + TF-IDF (baseline)", y_te, preds)
    save_cm(y_te, preds, names, "LR + TF-IDF baseline", "confusion_lr_baseline.png")
    return result



# MODEL 1 - Multi-Granularity TF-IDF + LinearSVC  (Abdul-Hadi)
def build_model1():
    """
    Feature-engineering-first approach.

    Two complementary TF-IDF views are concatenated:
      - Word n-grams  (1-3): capture vocabulary, collocations, and short phrases.
        e.g. "hard drive", "gun control law", "solar system"
      - Character n-grams (3-6, char_wb): capture morphology, handles spelling
        variation, and is robust to out-of-vocabulary tokens that word models miss.
        char_wb wraps tokens in spaces, preventing cross-word boundary matches.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    word_feats = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),        # bigrams sufficient; trigrams add noise
        max_features=150_000,      # larger budget now that noise tokens are gone
        sublinear_tf=True,
        min_df=2,
        max_df=0.90,               # discard terms present in >90% of docs
        stop_words="english",
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]{1,}\b",  # skip pure-punct tokens
    )
    char_feats = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),        # 3-char min avoids high-noise 2-char patterns
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    # No external Normalizer: each TfidfVectorizer already produces L2-normalised
    # rows, and FeatureUnion's concatenation preserves that property well enough
    # for LinearSVC with C<1 to act as its own implicit feature selector.
    return Pipeline([
        ("features", FeatureUnion([("word", word_feats), ("char", char_feats)])),
        ("clf", LinearSVC(C=0.5, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def run_model1(X_tr, X_val, X_te, y_tr, y_val, y_te, names):
    print("\n" + "─" * 65)
    print("[Model 1]  Multi-granularity TF-IDF + LinearSVC  (Abdul-Hadi)")
    print("─" * 65)

    # Apply classical preprocessing before TF-IDF fitting
    print("  Preprocessing (noise removal, normalisation) ...", flush=True)
    X_tr_c  = [_preprocess_classical(t) for t in X_tr]
    X_val_c = [_preprocess_classical(t) for t in X_val]
    X_te_c  = [_preprocess_classical(t) for t in X_te]

    m = build_model1()
    print("  Fitting word + char TF-IDF union...", flush=True)
    m.fit(X_tr_c, y_tr)

    val_preds = m.predict(X_val_c)
    print(f"\n  Validation  acc={accuracy_score(y_val, val_preds):.4f}  "
          f"F1={f1_score(y_val, val_preds, average='macro'):.4f}")

    preds = m.predict(X_te_c)
    result = evaluate("Multi-granularity TF-IDF + LinearSVC", y_te, preds)
    save_cm(y_te, preds, names, "LinearSVC (word+char TF-IDF)", "confusion_model1.png")
    print("\n  Per-class report:")
    print(classification_report(y_te, preds, target_names=names, digits=3))
    top_confused(y_te, preds, names)
    return result


# MODEL 2 - Fine-tuned RoBERTa-large  (Eduardo)
import re as _re

# Pre-compiled regexes
_URL_RE    = _re.compile(r'https?://\S+|www\.\S+', _re.IGNORECASE)
_EMAIL_RE  = _re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
_NUM_RE    = _re.compile(r'\b\d+(?:[.,/]\d+)*\b')
# 20NG-specific junk: "ax", repeated-x spam, uuencode begin/end lines
_AX_RE     = _re.compile(r'\bax\b|\bm?ax+\b', _re.IGNORECASE)
_UUENC_RE  = _re.compile(r'^begin \d{3} \S+|^end$|^[M-Z][^\n]{59}$', _re.MULTILINE)


def _preprocess_classical(text):
    """
    Preprocessing pipeline applied to raw text before TF-IDF vectorisation.

    1. Noise lines
       Lines that are mostly punctuation, whitespace, or non-alphabetic
       characters (separators, UUencoded blobs, ASCII art, table borders).
       Dropped when line length > 8 and alpha_ratio < 0.30.

    2. UUencode artifacts
       Lines matching the UUencode "begin NNN filename / [M...] / end" pattern
       are stripped (they were missed by the header remover).

    3. Email addresses -> "EMAILADDR"
       Normalises the (large) variety of email strings to a single token,
       reducing sparsity and preventing overfitting to specific addresses.

    4. URLs -> "HTTPURL"
       Same rationale: http/https/www URLs collapse to one token.

    5. Numbers -> "NUM"
       Standalone numeric tokens (years, prices, statistics) are replaced.
       This prevents the model from memorising specific numbers seen in
       training documents while still preserving the semantic signal of
       "a number was here".

    6. 20NG "ax" spam
       A known artefact of the dataset: "ax", "max", "maxax", "maxaxax" etc.
       are meaningless repeated characters from a mail-mangling bug.

    7. Whitespace normalisation
       Collapses multiple spaces/newlines to a single space.
    """
    # Step 1 - UUencode artifact removal
    text = _UUENC_RE.sub(" ", text)

    # Step 2 - noise line removal
    lines = text.split("\n")
    kept = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 8:
            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
            if alpha_ratio < 0.30:
                continue
        kept.append(line)
    text = " ".join(kept)

    # Step 3 - normalize emails, URLs, numbers, ax-spam
    text = _EMAIL_RE.sub("EMAILADDR", text)
    text = _URL_RE.sub("HTTPURL", text)
    text = _NUM_RE.sub("NUM", text)
    text = _AX_RE.sub(" ", text)

    # Step 4 - collapse whitespace
    return _re.sub(r"\s+", " ", text).strip()


def _clean_text_for_bert(text):
    """
    Extra preprocessing pass applied only to transformer inputs (Model 2/3).

    Steps
    1. Split on newlines; discard lines where fewer than 30 % of characters
       are alphabetic AND the line is longer than 8 chars (catches tables,
       code, MIME artefacts, separator lines).
    2. Collapse consecutive whitespace / blank lines to a single space.
    3. Strip leading/trailing whitespace.
    """
    lines = text.split("\n")
    kept = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 8:
            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
            if alpha_ratio < 0.30:
                continue
        kept.append(line)
    return _re.sub(r"\s+", " ", " ".join(kept)).strip()


def run_model2(X_tr, X_val, X_te, y_tr, y_val, y_te, names):
    """
    Fine-tune roberta-large for 20-way classification.

    RoBERTa-large (355 M params, 24 layers, hidden 1024) is a robustly
    optimised BERT pre-training: dynamic masking, no next-sentence
    prediction, larger mini-batches (8 K sequences), more data (160 GB),
    and longer training.  These changes consistently improve downstream
    classification over BERT-large.

    Preprocessing
    - _clean_text_for_bert(): removes noise lines before tokenisation.
    - Head + tail: keeps first 256 and last 256 BPE tokens for documents
      longer than 512, preserving topic intro and conclusion.

    Hyperparameters
    - per_device_train_batch_size = 16  (bf16, H100)
    - learning_rate = 1e-5  with 10 % linear warm-up then linear decay
    - weight_decay = 0.01  (AdamW)
    - 5 epochs, patience-2 early stopping on val macro-F1
    - max_length = 512 tokens  (head 256 + tail 256)
    """
    print("\n" + "─" * 65)
    print("[Model 2]  Fine-tuning roberta-large (HF Hub)  (Eduardo)")
    print("─" * 65)
    try:
        import torch
        from datasets import Dataset as HFDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"\n  [SKIP] Missing dependency: {e}")
        print("  Install with:  pip install torch transformers datasets accelerate")
        return None

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = max(torch.cuda.device_count(), 1)
    print(f"  Device: {device_str}  |  GPUs available: {n_gpu}")

    MODEL_ID    = "roberta-large"
    NUM_CLASSES = len(names)
    MAX_LEN     = 512
    HEAD        = 256
    TAIL        = 256
    BATCH_SIZE  = 16
    NUM_EPOCHS  = 5

    print(f"  Loading tokeniser from HF Hub: {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("  Cleaning text (removing noise lines) ...", flush=True)
    X_tr_c  = [_clean_text_for_bert(t) for t in X_tr]
    X_val_c = [_clean_text_for_bert(t) for t in X_val]
    X_te_c  = [_clean_text_for_bert(t) for t in X_te]

    def tokenize_head_tail(batch):
        enc = tokenizer(batch["text"], truncation=False, padding=False,
                        add_special_tokens=True)
        out_ids, out_mask = [], []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            if len(ids) <= MAX_LEN:
                out_ids.append(ids)
                out_mask.append(mask)
            else:
                out_ids.append(ids[:HEAD] + ids[-(TAIL):])
                out_mask.append(mask[:HEAD] + mask[-(TAIL):])
        return {"input_ids": out_ids, "attention_mask": out_mask}

    def make_hf_dataset(texts, labels):
        ds = HFDataset.from_dict({"text": texts, "label": list(map(int, labels))})
        return ds.map(tokenize_head_tail, batched=True,
                      remove_columns=["text"], desc="Tokenising")

    train_ds = make_hf_dataset(X_tr_c,  y_tr)
    val_ds   = make_hf_dataset(X_val_c, y_val)
    test_ds  = make_hf_dataset(X_te_c,  [0] * len(X_te_c))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"  Loading model from HF Hub: {MODEL_ID} ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES,
    )

    def compute_metrics(ep):
        logits, labels = ep
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    steps_per_epoch = len(train_ds) // (BATCH_SIZE * n_gpu)
    warmup_steps    = max(1, int(0.10 * steps_per_epoch * NUM_EPOCHS))

    training_args = TrainingArguments(
        output_dir=BERT_CLS_CKPT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=1e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=(device_str == "cuda"),
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n  Fine-tuning {MODEL_ID}  (head+tail {HEAD}+{TAIL}, bf16) ...", flush=True)
    trainer.train()

    val_out   = trainer.predict(val_ds)
    val_preds = np.argmax(val_out.predictions, axis=-1)
    print(f"\n  Validation  acc={accuracy_score(y_val, val_preds):.4f}  "
          f"F1={f1_score(y_val, val_preds, average='macro'):.4f}")

    test_out   = trainer.predict(test_ds)
    test_preds = np.argmax(test_out.predictions, axis=-1)
    result = evaluate("Fine-tuned RoBERTa-large", y_te, test_preds)
    save_cm(y_te, test_preds, names, "Fine-tuned RoBERTa-large", "confusion_model2.png")
    print("\n  Per-class report:")
    print(classification_report(y_te, test_preds, target_names=names, digits=3))
    top_confused(y_te, test_preds, names)

    # Explicit GPU memory cleanup before Model 3 loads
    # Trainer, model, and DataParallel replicas all hold GPU memory.
    del trainer, model, train_ds, val_ds, test_ds, collator, tokenizer
    import gc as _gc2, torch as _t2
    _gc2.collect()
    if _t2.cuda.is_available():
        _t2.cuda.empty_cache()
        _t2.cuda.synchronize()

    return result



# MODEL 3 - ModernBERT-large  (Viransh)
def run_model3(X_tr, X_val, X_te, y_tr, y_val, y_te, names):
    """
    Fine-tune answerdotai/ModernBERT-large for 20-way classification.

    ModernBERT-large (395 M params, 28 layers, hidden 1024, released Dec 2024)
    scales up the ModernBERT architecture with:
      - Rotary Position Embeddings (RoPE) — supports up to 8192 tokens natively.
      - Flash Attention 2 — hardware-optimised kernel; 2-4* faster on H100.
      - Alternating global + local attention — every 3rd layer is global,
        others use sliding-window local attention (window 128).
      - Trained on 2 T tokens; the large variant additionally trains with
        a sequence length of 8192 for extended-context tasks.

    Hyperparameters
    - learning_rate = 2e-5
    - warmup 20%, weight_decay 0.01, 8 epochs, patience-3 early stopping
    """
    print("\n" + "─" * 65)
    print("[Model 3]  ModernBERT-large (Viransh)")
    print("─" * 65)
    try:
        import torch
        from datasets import Dataset as HFDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        print(f"\n  [SKIP] Missing dependency: {e}")
        print("  Install with:  pip install torch transformers datasets accelerate")
        return None

    # Hint the CUDA allocator to reuse fragmented segments; prevents OOM when
    # a large model (RoBERTa) was freed immediately before this one.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    n_gpu = max(torch.cuda.device_count(), 1)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device_str}  |  GPUs available: {n_gpu}")

    MODEL_ID    = "answerdotai/ModernBERT-large"
    MAX_LEN     = 4096   # covers >99 % of 20NG posts; avoids OOM at 8192 with DataParallel
    NUM_CLASSES = len(names)
    BATCH_SIZE  = 8      # safe per-device batch with 4096-token max
    GRAD_ACCUM  = 4      # 8 * n_gpu * 4 → effective batch 64 on 2 GPUs, 128 on 4
    NUM_EPOCHS  = 8      # more epochs; early stopping ends it if val plateaus

    print(f"  Loading tokeniser from HF Hub: {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("  Cleaning text (removing noise lines) ...", flush=True)
    X_tr_c  = [_clean_text_for_bert(t) for t in X_tr]
    X_val_c = [_clean_text_for_bert(t) for t in X_val]
    X_te_c  = [_clean_text_for_bert(t) for t in X_te]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)

    def make_hf_dataset(texts, labels):
        ds = HFDataset.from_dict({"text": texts, "label": list(map(int, labels))})
        return ds.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenising")

    train_ds = make_hf_dataset(X_tr_c,  y_tr)
    val_ds   = make_hf_dataset(X_val_c, y_val)
    test_ds  = make_hf_dataset(X_te_c,  [0] * len(X_te_c))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"  Loading model from HF Hub: {MODEL_ID} ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_CLASSES,
    )

    def compute_metrics(ep):
        logits, labels = ep
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    optim_steps_per_epoch = (len(train_ds) // (BATCH_SIZE * n_gpu)) // GRAD_ACCUM
    # 20% warmup (doubled from 10%) smooths out the noisy first epoch that was
    # observed with 4096-token sequences.
    warmup_steps = max(1, int(0.20 * optim_steps_per_epoch * NUM_EPOCHS))

    training_args = TrainingArguments(
        output_dir=DEBERTA_CKPT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        # gradient_checkpointing is required: ModernBERT-large fills all 79 GiB
        # on a single H100 at MAX_LEN=4096 without it (OOM at step 5). 
        # Explicit gradient_checkpointing_kwargs disables the reentrant
        # variant so it is compatible with Flash Attention 2.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    eff_batch = BATCH_SIZE * max(n_gpu, 1) * GRAD_ACCUM
    print(f"\n  Training {MODEL_ID}  "
          f"(effective batch={eff_batch}, max_len={MAX_LEN}, bf16=True) ...",
          flush=True)
    trainer.train()

    val_out = trainer.predict(val_ds)
    val_preds = np.argmax(val_out.predictions, axis=-1)
    print(f"\n  Validation  acc={accuracy_score(y_val, val_preds):.4f}  "
          f"F1={f1_score(y_val, val_preds, average='macro'):.4f}")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in (-1, 0):
        test_out = trainer.predict(test_ds)
        test_preds = np.argmax(test_out.predictions, axis=-1)
        result = evaluate("ModernBERT-large (4*H100)", y_te, test_preds)
        save_cm(y_te, test_preds, names, "ModernBERT-large", "confusion_model3.png")
        print("\n  Per-class report:")
        print(classification_report(y_te, test_preds, target_names=names, digits=3))
        top_confused(y_te, test_preds, names)
        return result
    return None



# EVALUATION HELPERS
def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true,    y_pred, average="macro", zero_division=0)
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  F1 (Macro):           {f1m:.4f}")
    print(f"  F1 (Weighted):        {f1w:.4f}")
    print(f"  Precision (Macro):    {pre:.4f}")
    print(f"  Recall (Macro):       {rec:.4f}")
    return dict(model=name, accuracy=float(acc), f1_macro=float(f1m),
                f1_weighted=float(f1w), precision_macro=float(pre),
                recall_macro=float(rec))


def save_cm(y_true, y_pred, names, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    # Strip the leading "comp." / "rec." / "talk." / etc. only when all
    # remaining suffixes are unique;
    suffixes = [n.split(".")[-1] for n in names]
    labels = suffixes if len(set(suffixes)) == len(suffixes) else names

    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.3, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 6})
    ax.set_title(f"Confusion Matrix - {title}", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
    ax.set_ylabel("True",      fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix -> {path}")


def top_confused(y_true, y_pred, names, n=10):
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    pairs = sorted(
        [(cm[i, j], names[i], names[j])
         for i in range(len(names)) for j in range(len(names)) if cm[i, j] > 0],
        reverse=True,
    )
    print(f"\n  Top {n} confused pairs:")
    for cnt, t, p in pairs[:n]:
        print(f"    {t:<35} -> {p:<35} ({cnt}*)")


def print_summary(results):
    valid = [r for r in results if r and r.get("accuracy") is not None]
    print("\n" + "=" * 75)
    print("  FINAL COMPARISON TABLE")
    print("=" * 75)
    print(f"  {'Model':<42} {'Acc':>7} {'F1 Macro':>9} {'Prec':>7} {'Rec':>7}")
    print("  " + "-" * 72)
    for r in valid:
        print(f"  {r['model']:<42} {r['accuracy']:>7.4f} {r['f1_macro']:>9.4f} "
              f"{r['precision_macro']:>7.4f} {r['recall_macro']:>7.4f}")
    print("=" * 75)



# MAIN
def main():
    parser = argparse.ArgumentParser(description="Group 37 - train 20NG models")
    parser.add_argument(
        "--models", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3],
        metavar="N",
        help="Which models to run: 1=TF-IDF+SVC  2=RoBERTa-large  3=ModernBERT-large",
    )
    args = parser.parse_args()

    # Auto-select GPUs *before* any CUDA context is created.
    # 50 GiB threshold skips any GPU that
    # has <50 GiB free (e.g. a GPU running a VLLM server with ~34 GB used).
    if any(m in args.models for m in (2, 3)):
        _auto_select_gpus(min_free_gib=50.0)

    # Under DDP (torchrun), only rank 0 should print headers / save figures
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank in (-1, 0):
        print("=" * 65)
        print("Group 37 - 20 Newsgroups Classification  |  COMPSCI 4NL3")
        print("Models to run:", args.models)
        print("=" * 65)

    X_tr, X_val, X_te, y_tr, y_val, y_te, names = load_data()
    n_cls = len(names)
    all_results = []

    # Baselines (always run, fast)
    if local_rank in (-1, 0):
        rb = RandomBaseline(n_cls); rb.fit(X_tr, y_tr)
        all_results.append(evaluate("Random Baseline", y_te, rb.predict(X_te)))

        mb = MajorityBaseline(); mb.fit(X_tr, y_tr)
        all_results.append(evaluate("Majority Class Baseline", y_te, mb.predict(X_te)))

        r = run_lr_baseline(X_tr, X_val, X_te, y_tr, y_val, y_te, names)
        if r: all_results.append(r)

    # Model 1
    if 1 in args.models and local_rank in (-1, 0):
        r = run_model1(X_tr, X_val, X_te, y_tr, y_val, y_te, names)
        if r: all_results.append(r)

    # Model 2
    if 2 in args.models:
        r = run_model2(X_tr, X_val, X_te, y_tr, y_val, y_te, names)
        if r and local_rank in (-1, 0): all_results.append(r)
        # Free GPU memory before loading the next large model
        import gc, torch as _torch
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()

    # Model 3
    if 3 in args.models:
        r = run_model3(X_tr, X_val, X_te, y_tr, y_val, y_te, names)
        if r and local_rank in (-1, 0): all_results.append(r)

    # Summary
    if local_rank in (-1, 0):
        print_summary(all_results)
        out_path = os.path.join(FIGURES_DIR, "all_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results   -> {out_path}")
        print(f"  Figures   -> {FIGURES_DIR}/")
        print(f"  Checkpoints -> {CKPT_DIR}/")
        print("\n  Done.")


if __name__ == "__main__":
    main()