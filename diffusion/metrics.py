import numpy as np

def pr_auc(scores, y_true):
    idx = np.argsort(-scores)
    y = y_true[idx]
    P = float(y.sum())
    if P <= 0:
        return np.nan

    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    auc = 0.0

    for i in range(len(y)):
        if y[i] > 0:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / P
        prec = tp / (tp + fp)
        auc += (recall - prev_recall) * prec
        prev_recall = recall

    return float(auc)

def ranking_metrics(top, pos_eval, cand, topk):
    tp = sum(1 for n in top if n in pos_eval)
    P = max(1, int(len(pos_eval)))

    recall_at_k = float(tp) / P
    precision_at_k = float(tp) / max(1, len(top))
    base_rate = float(len(pos_eval)) / max(1, len(cand))
    enrich = (precision_at_k / base_rate) if base_rate > 0 else None

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "enrichment_factor_at_k": enrich,
    }