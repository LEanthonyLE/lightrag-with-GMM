from __future__ import annotations

import numpy as np
from typing import Any

from lightrag.utils import logger


def _normalize_scores(scores: np.ndarray, direction: str) -> np.ndarray:
    """Convert scores so higher always means more relevant.

    VDB returns cosine distance (0=identical, 2=opposite) where lower is better.
    Rerank returns relevance scores where higher is better.
    """
    if direction == "lower_better":
        return 1.0 - scores
    return scores


def _safe_gmm_fit(
    scores_2d: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fit 2-component GMM. Returns (labels, n_clusters).

    Falls back to 1 cluster if 2-component fit is degenerate.
    """
    from sklearn.mixture import GaussianMixture

    if len(scores_2d) < 3:
        return np.zeros(len(scores_2d), dtype=int), 1

    try:
        gmm = GaussianMixture(n_components=2, n_init=1, random_state=0)
        gmm.fit(scores_2d)
        labels = gmm.predict(scores_2d)
        n_clusters = len(np.unique(labels))
    except Exception:
        logger.debug("GMM fit failed, falling back to single cluster")
        labels = np.zeros(len(scores_2d), dtype=int)
        n_clusters = 1

    return labels, n_clusters


def gmm_select_k(
    scores: np.ndarray,
    min_k: int = 1,
    max_k: int = 5,
) -> tuple[list[int], int]:
    """Select k results via GMM clustering on similarity scores.

    Fits a 2-component GMM to the score distribution and keeps items
    in the high-mean cluster (more relevant), bounded by [min_k, max_k].

    Args:
        scores: 1D array of relevance scores (higher = more relevant).
        min_k: Minimum items to return (safety floor).
        max_k: Maximum items to return (safety ceiling).

    Returns:
        (kept_indices, selected_k): indices of items to keep, and the selected count.
    """
    n = len(scores)

    if n <= min_k:
        return list(range(n)), n

    scores_2d = scores.reshape(-1, 1)
    labels, n_clusters = _safe_gmm_fit(scores_2d)

    if n_clusters == 1:
        kept = min(n, max_k)
        return list(range(kept)), kept

    # Split indices by cluster
    cluster_scores: dict[int, list[float]] = {}
    cluster_indices: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        cluster_scores.setdefault(int(label), []).append(float(scores[i]))
        cluster_indices.setdefault(int(label), []).append(i)

    # Identify "good" cluster = higher mean
    cluster_means = {c: np.mean(v) for c, v in cluster_scores.items()}
    good_cluster = max(cluster_means, key=lambda c: cluster_means[c])
    bad_cluster = min(cluster_means, key=lambda c: cluster_means[c])

    good_indices = cluster_indices[good_cluster]
    # Sort good cluster by score descending
    good_indices.sort(key=lambda i: scores[i], reverse=True)

    if len(good_indices) > max_k:
        good_indices = good_indices[:max_k]
    elif len(good_indices) < min_k:
        # Supplement from bad cluster (best-scoring first)
        bad_indices = cluster_indices[bad_cluster]
        bad_indices.sort(key=lambda i: scores[i], reverse=True)
        need = min_k - len(good_indices)
        good_indices = good_indices + bad_indices[:need]

    return good_indices, len(good_indices)


def gmm_filter_results(
    results: list[dict[str, Any]],
    score_key: str = "distance",
    score_direction: str = "lower_better",
    min_k: int = 1,
    max_k: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Filter a list of result dicts using GMM on their score field.

    This is the main entry point for retrieval stages. It extracts scores
    from the result dicts, normalizes direction, runs GMM clustering, and
    returns only the results in the "good" cluster.

    Args:
        results: List of result dicts from VDB query or reranker.
        score_key: Key in each dict holding the score value (e.g. "distance").
        score_direction: "lower_better" for cosine distance, "higher_better" for rerank.
        min_k: Minimum items to return.
        max_k: Maximum items to return.

    Returns:
        (filtered_results, k): filtered list and the auto-selected k.
    """
    if len(results) <= min_k:
        return results, len(results)

    raw_scores = np.array(
        [float(r.get(score_key, 0.0)) for r in results], dtype=np.float64
    )
    normalized = _normalize_scores(raw_scores, score_direction)

    kept_indices, selected_k = gmm_select_k(normalized, min_k=min_k, max_k=max_k)

    filtered = [results[i] for i in kept_indices]
    return filtered, selected_k
