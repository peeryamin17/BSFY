import re

import numpy as np
from sacrebleu import CHRF, BLEU


def _tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _prf(pred, ref):
    p = _tokenize(pred)
    r = _tokenize(ref)
    if not p:
        return 0.0, 0.0, 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    precision = common / len(p)
    recall = common / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_all(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    refs = [[r] for r in references]
    chrf = CHRF(word_order=2).corpus_score(predictions, refs).score
    bleu = BLEU().corpus_score(predictions, refs).score
    geo = (chrf * bleu) ** 0.5

    exact = sum(p == r for p, r in zip(predictions, references)) / len(predictions)

    precisions, recalls, f1s = [], [], []
    pred_lens, ref_lens = [], []
    for p, r in zip(predictions, references):
        pr, rc, f = _prf(p, r)
        precisions.append(pr)
        recalls.append(rc)
        f1s.append(f)
        pred_lens.append(len(_tokenize(p)))
        ref_lens.append(len(_tokenize(r)))

    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    pred_lens = np.asarray(pred_lens, dtype=float)
    ref_lens = np.asarray(ref_lens, dtype=float)
    rmse = float(np.sqrt(np.mean((pred_lens - ref_lens) ** 2)))
    r = 0.0
    if len(pred_lens) > 1 and np.std(pred_lens) > 0 and np.std(ref_lens) > 0:
        r = float(np.corrcoef(pred_lens, ref_lens)[0, 1])
    ss_res = float(np.sum((ref_lens - pred_lens) ** 2))
    ss_tot = float(np.sum((ref_lens - np.mean(ref_lens)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "chrF++": round(chrf, 2),
        "BLEU": round(bleu, 2),
        "geo_mean": round(geo, 2),
        "accuracy": round(exact * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "RMSE": round(rmse, 2),
        "Pearson R": round(r, 3),
        "R-squared": round(r2, 3),
    }
