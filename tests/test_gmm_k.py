"""Unit tests for lightrag.gmm_k — GMM-based auto top-k selection."""

import numpy as np
import pytest

from lightrag.gmm_k import (
    _normalize_scores,
    _safe_gmm_fit,
    gmm_select_k,
    gmm_filter_results,
)


class TestNormalizeScores:
    def test_lower_better_converts_to_similarity(self):
        scores = np.array([0.05, 0.50, 0.95, 2.00], dtype=np.float64)
        result = _normalize_scores(scores, "lower_better")
        expected = np.array([0.95, 0.50, 0.05, -1.00], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_better_is_identity(self):
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        result = _normalize_scores(scores, "higher_better")
        np.testing.assert_array_equal(result, scores)


class TestSafeGMMFit:
    def test_clear_bimodal_separation(self):
        # Two well-separated clusters
        scores = np.array(
            [[0.95], [0.92], [0.88], [0.15], [0.12], [0.08]], dtype=np.float64
        )
        labels, n_clusters = _safe_gmm_fit(scores)
        assert n_clusters == 2
        # First 3 (high scores) should share same label
        assert labels[0] == labels[1] == labels[2]
        # Last 3 (low scores) should share same label
        assert labels[3] == labels[4] == labels[5]
        # Two groups should differ
        assert labels[0] != labels[3]

    def test_too_few_results_returns_single_cluster(self):
        scores = np.array([[0.5], [0.6]], dtype=np.float64)
        labels, n_clusters = _safe_gmm_fit(scores)
        assert n_clusters == 1
        np.testing.assert_array_equal(labels, np.array([0, 0]))

    def test_handles_all_identical_scores(self):
        scores = np.array([[0.5], [0.5], [0.5], [0.5]], dtype=np.float64)
        labels, n_clusters = _safe_gmm_fit(scores)
        assert n_clusters >= 1


class TestGMMSelectK:
    def test_bimodal_keeps_high_mean_cluster(self):
        # High cluster: [0.88, 0.85, 0.82, 0.80] (mean ~0.84)
        # Low cluster:  [0.20, 0.18, 0.15] (mean ~0.18)
        scores = np.array(
            [0.88, 0.20, 0.85, 0.18, 0.82, 0.15, 0.80, 0.19],
            dtype=np.float64,
        )
        indices, k = gmm_select_k(scores, min_k=3, max_k=10)
        assert 3 <= k <= 7  # Should be around 4-5 (high cluster size)
        # Kept indices should have high scores
        for i in indices:
            assert scores[i] > 0.5

    def test_unimodal_bounded_to_max_k(self):
        # All high quality — should keep all but bounded by max_k
        scores = np.array([0.7, 0.6, 0.5] * 20, dtype=np.float64)
        indices, k = gmm_select_k(scores, min_k=3, max_k=10)
        assert k <= 10
        assert k >= 3

    def test_small_n_returns_all(self):
        scores = np.array([0.9, 0.1], dtype=np.float64)
        indices, k = gmm_select_k(scores, min_k=3, max_k=10)
        assert k == 2
        assert indices == [0, 1]

    def test_respects_max_k_bound(self):
        scores = np.array(
            [0.92 + i * 0.001 for i in range(50)], dtype=np.float64
        )
        indices, k = gmm_select_k(scores, min_k=5, max_k=15)
        assert k <= 15

    def test_supplements_from_bad_cluster_when_below_min_k(self):
        # Make good cluster very small (only 2 items)
        scores = np.array(
            [0.95, 0.93, 0.11, 0.10, 0.09, 0.08], dtype=np.float64
        )
        indices, k = gmm_select_k(scores, min_k=4, max_k=10)
        assert k >= 4
        # Should include the 2 high items + top items from low cluster
        assert 0 in indices  # 0.95
        assert 1 in indices  # 0.93


class TestGMMFilterResults:
    def test_filter_distance_scores(self):
        results = [
            {"id": "a", "distance": 0.05, "content": "relevant1"},
            {"id": "b", "distance": 0.08, "content": "relevant2"},
            {"id": "c", "distance": 0.12, "content": "relevant3"},
            {"id": "d", "distance": 0.88, "content": "noise1"},
            {"id": "e", "distance": 0.92, "content": "noise2"},
            {"id": "f", "distance": 0.95, "content": "noise3"},
        ]
        filtered, k = gmm_filter_results(
            results,
            score_key="distance",
            score_direction="lower_better",
            min_k=3,
            max_k=10,
        )
        assert 3 <= k <= 5
        # Relevant items (low distance) should be kept
        kept_ids = {r["id"] for r in filtered}
        assert "a" in kept_ids
        assert "b" in kept_ids
        assert "c" in kept_ids

    def test_higher_better_direction(self):
        results = [
            {"id": "a", "rerank_score": 0.92},
            {"id": "b", "rerank_score": 0.88},
            {"id": "c", "rerank_score": 0.15},
            {"id": "d", "rerank_score": 0.10},
        ]
        filtered, k = gmm_filter_results(
            results,
            score_key="rerank_score",
            score_direction="higher_better",
            min_k=2,
            max_k=10,
        )
        kept_ids = {r["id"] for r in filtered}
        assert "a" in kept_ids
        assert "b" in kept_ids

    def test_small_input_passthrough(self):
        results = [
            {"id": "a", "distance": 0.5},
            {"id": "b", "distance": 0.6},
        ]
        filtered, k = gmm_filter_results(
            results,
            score_key="distance",
            score_direction="lower_better",
            min_k=3,
        )
        assert k == 2
        assert len(filtered) == 2

    def test_missing_score_key_defaults_to_zero(self):
        results = [
            {"id": "a"},  # no score key
            {"id": "b", "distance": 0.1},
        ]
        filtered, k = gmm_filter_results(
            results,
            score_key="distance",
            score_direction="lower_better",
            min_k=1,
        )
        assert len(filtered) == k
