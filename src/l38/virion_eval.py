"""L39 real eval: phage virion-protein classification (frozen embeddings + linear probe).

Reproduces the ESM-PVP (Li & Liang 2023) task shape -- classify whether a
phage protein is a virion/structural protein (capsid, tail, baseplate, etc.)
from sequence alone -- using UniProt's own KW-0946 (Virion) keyword as the
label, sourced directly via UniProt REST (no external repo). This is the
actual downstream benchmark the L39 perplexity result needs to translate
into before it counts as a real win; perplexity alone is not a publishable
result.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from src.l38.phage_data import clean_sequences, parse_fasta

DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "phage"
SEED = 0
MAX_LENGTH = 512
EMBED_BATCH_SIZE = 16


def load_labeled_dataset() -> Tuple[List[str], np.ndarray]:
    positive = clean_sequences(parse_fasta(DATA_DIR / "virion_positive.fasta"))
    negative = clean_sequences(parse_fasta(DATA_DIR / "virion_negative.fasta"))

    # De-duplicate: a sequence appearing in both sets (shouldn't per the KW-0946
    # filter, but cheap to guard against) is dropped from both to avoid a
    # contradictory label.
    pos_set, neg_set = set(positive), set(negative)
    overlap = pos_set & neg_set
    positive = [s for s in positive if s not in overlap]
    negative = [s for s in negative if s not in overlap]

    sequences = positive + negative
    labels = np.array([1] * len(positive) + [0] * len(negative))
    return sequences, labels


@torch.no_grad()
def embed_sequences(
    sequences: List[str], model_path: str, device: str,
    max_length: int = MAX_LENGTH, batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding per sequence, from a FROZEN
    encoder (base or fine-tuned MLM backbone -- the encoder body is shared
    between AutoModelForMaskedLM and AutoModel for ESM-2, so this loads
    cleanly from either a HF hub id or a local fine-tuned checkpoint dir)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 10000, seed: int = SEED) -> dict:
    """Bootstrap over held-out test-set INDICES (resample which test examples
    are scored, not the classifier itself) to get a 95% CI for a metric.
    Confirms a base-vs-finetuned gap survives resampling, i.e. isn't an
    artifact of exactly which examples landed in this particular test split.
    """
    n = len(y_true)
    rng = np.random.RandomState(seed)
    boot_values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
    return {
        "point_estimate": float(metric_fn(y_true, y_pred, y_prob)),
        "ci_lower": float(np.percentile(boot_values, 2.5)),
        "ci_upper": float(np.percentile(boot_values, 97.5)),
        "bootstrap_std": float(boot_values.std(ddof=1)),
    }


def _accuracy_metric(y_true, y_pred, y_prob):
    return accuracy_score(y_true, y_pred)


def _f1_metric(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred)


def _auc_metric(y_true, y_pred, y_prob):
    return roc_auc_score(y_true, y_prob)


def evaluate_probe(embeddings: np.ndarray, labels: np.ndarray, seed: int = SEED, n_boot: int = 10000) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy_ci": bootstrap_metric_ci(y_test, preds, probs, _accuracy_metric, n_boot, seed),
        "f1_ci": bootstrap_metric_ci(y_test, preds, probs, _f1_metric, n_boot, seed),
        "auc_ci": bootstrap_metric_ci(y_test, preds, probs, _auc_metric, n_boot, seed),
        "y_test": y_test.tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
    }


def paired_bootstrap_metric_diff(
    y_true: np.ndarray, pred_a: np.ndarray, prob_a: np.ndarray,
    pred_b: np.ndarray, prob_b: np.ndarray, metric_fn, n_boot: int = 10000, seed: int = SEED,
) -> dict:
    """Paired bootstrap on the SAME resampled indices for two models evaluated
    on the SAME held-out test set (guaranteed here since evaluate_probe uses
    an identical seed/labels/length -> identical train_test_split indices
    every call). This is a much more powerful/correct test than comparing two
    independently-computed CIs, because it directly asks "on this same set of
    examples, does model B beat model A" rather than "do B's and A's CIs
    happen to overlap" (which conflates shared per-example variance into two
    separate CIs and understates significance for a real paired effect).
    Returns the bootstrap distribution of (metric_b - metric_a); a 95% CI
    entirely above 0 means B is significantly better than A.
    """
    n = len(y_true)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        metric_a = metric_fn(y_true[idx], pred_a[idx], prob_a[idx])
        metric_b = metric_fn(y_true[idx], pred_b[idx], prob_b[idx])
        diffs[i] = metric_b - metric_a
    return {
        "point_estimate_diff": float(metric_fn(y_true, pred_b, prob_b) - metric_fn(y_true, pred_a, prob_a)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "significant_at_95pct": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def run_comparison(
    base_model_path: str, finetuned_model_path: str, device: str,
    load_data_fn=load_labeled_dataset, max_length: int = MAX_LENGTH, embed_batch_size: int = EMBED_BATCH_SIZE,
) -> dict:
    """load_data_fn must return (sequences, labels) -- defaults to the
    original UniProt-keyword dataset; pass a different loader (e.g.
    src.l38.phavip_real_data.load_pvp_labeled_dataset) to reuse this same
    bootstrap-tested comparison machinery on a different labeled dataset."""
    sequences, labels = load_data_fn()
    labels = np.asarray(labels)
    print(f"loaded {len(sequences)} sequences ({labels.sum()} positive, {len(labels)-labels.sum()} negative)", flush=True)

    print(f"embedding with base model: {base_model_path}", flush=True)
    base_embeddings = embed_sequences(sequences, base_model_path, device, max_length=max_length, batch_size=embed_batch_size)
    base_results = evaluate_probe(base_embeddings, labels)
    print(f"base results: {base_results}", flush=True)

    print(f"embedding with fine-tuned model: {finetuned_model_path}", flush=True)
    ft_embeddings = embed_sequences(sequences, finetuned_model_path, device, max_length=max_length, batch_size=embed_batch_size)
    ft_results = evaluate_probe(ft_embeddings, labels)
    print(f"fine-tuned results: {ft_results}", flush=True)

    y_test_base = np.array(base_results["y_test"])
    y_test_ft = np.array(ft_results["y_test"])
    assert np.array_equal(y_test_base, y_test_ft), (
        "base and finetuned evaluate_probe calls produced different test splits -- "
        "paired bootstrap requires identical held-out examples"
    )

    pred_base = np.array(base_results["preds"])
    prob_base = np.array(base_results["probs"])
    pred_ft = np.array(ft_results["preds"])
    prob_ft = np.array(ft_results["probs"])

    paired = {
        "accuracy": paired_bootstrap_metric_diff(y_test_base, pred_base, prob_base, pred_ft, prob_ft, _accuracy_metric),
        "f1": paired_bootstrap_metric_diff(y_test_base, pred_base, prob_base, pred_ft, prob_ft, _f1_metric),
        "auc": paired_bootstrap_metric_diff(y_test_base, pred_base, prob_base, pred_ft, prob_ft, _auc_metric),
    }
    print(f"paired bootstrap (finetuned - base): {paired}", flush=True)

    return {"base": base_results, "finetuned": ft_results, "paired_diff": paired}
