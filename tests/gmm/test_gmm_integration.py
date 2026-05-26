"""End-to-end tests for GMM auto top-k in the full query pipeline.

Tests the complete retrieval flow: query embedding -> VDB retrieval -> GMM
filtering -> chunk processing -> result assembly. Uses naive mode to avoid
keyword-extraction LLM calls, keeping tests self-contained and offline.
"""

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, Tokenizer


# ---------------------------------------------------------------------------
# simple tokenizer (avoids tiktoken dependency in offline tests)
# ---------------------------------------------------------------------------

class _SimpleTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


# ---------------------------------------------------------------------------
# keyword-based mock embedding — produces bimodal cosine-distance distributions
# ---------------------------------------------------------------------------

_ML_KW = [
    "machine", "learning", "neural", "network", "gradient", "classification",
    "regression", "training", "deep", "supervised", "unsupervised", "embedding",
    "backpropagation", "optimization", "loss",
]
_COOKING_KW = [
    "cooking", "recipe", "ingredient", "bake", "boil", "kitchen", "flavor",
    "dish", "cuisine", "food", "chef", "roast", "grill", "sauce", "spice",
]


def _topic_score(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


async def _topic_embedding(texts: list[str]) -> np.ndarray:
    """Return topic-specific vectors with bimodal cluster separation.

    Both topics live in the same low-dimensional subspace so cosine distance
    stays within the default 0.2 threshold, but ML-topic vectors are closer
    to the ML query direction than cooking-topic vectors.

    Per-document index-based jitter ensures variation within each cluster
    so the GMM can detect two distinct modes.
    """
    dim = 128
    result = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        ml = _topic_score(text, _ML_KW)
        ck = _topic_score(text, _COOKING_KW)
        # centroid direction: dim-0 heavy, dim-1 light
        if ml >= ck:
            result[i, 0] = 1.0
            result[i, 1] = 0.6
        else:
            result[i, 0] = 0.6
            result[i, 1] = 1.0
        # per-document jitter for within-cluster variance
        seed = hash(text) % 1000
        result[i, 2] = (seed - 500) * 0.0001
    return result


async def _mock_llm(*_args, **_kwargs) -> str:
    return "Mock LLM response for testing."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_rag(tmp_path, llm_func=None, embedding_func=None):
    """Create a LightRAG instance wired with mock LLM + topic embedding."""
    return LightRAG(
        working_dir=str(tmp_path / "gmm_integration"),
        graph_storage="NetworkXStorage",
        llm_model_func=llm_func or _mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=128,
            max_token_size=8192,
            func=embedding_func or _topic_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizer()),
    )


def _build_chunks(
    ml_count: int = 12,
    cooking_count: int = 8,
) -> dict[str, dict]:
    """Build a dict of chunk-id -> {content} with bimodal topics."""
    chunks: dict[str, dict] = {}
    for i in range(ml_count):
        chunks[f"chunk-ml-{i}"] = {
            "content": f"Machine learning neural network training data classification "
            f"regression deep supervised optimization gradient {i}"
        }
    for i in range(cooking_count):
        chunks[f"chunk-cooking-{i}"] = {
            "content": f"Cooking recipe ingredients bake boil kitchen flavor dish "
            f"cuisine chef roast sauce {i}"
        }
    return chunks


# ---------------------------------------------------------------------------
# offline integration tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.offline


class TestGMMNaiveQueryPipeline:
    """End-to-end GMM tests using naive query mode (no keyword extraction)."""

    @pytest.mark.asyncio
    async def test_bimodal_gmm_filters_out_irrelevant_chunks(self, tmp_path):
        """ML query should GMM-filter away cooking chunks, keeping ML ones."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        chunks = _build_chunks(ml_count=12, cooking_count=8)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        result = await rag.aquery_llm(
            "What is machine learning and gradient descent?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,  # skip LLM generation
                auto_top_k=True,
                top_k=20,
                gmm_min_k=3,
                gmm_max_k=60,
            ),
        )

        assert result["status"] == "success"
        gmm = result["metadata"]["processing_info"]["gmm"]
        assert gmm["enabled"] is True
        assert gmm["raw_count"] >= 12  # at least the ML chunks retrieved
        assert gmm["min_k"] <= gmm["selected_k"] <= gmm["max_k"]
        # GMM should drop the irrelevant cooking chunks
        assert gmm["selected_k"] < gmm["raw_count"]

    @pytest.mark.asyncio
    async def test_auto_top_k_disabled_no_gmm_info(self, tmp_path):
        """When auto_top_k=False, gmm_info must NOT appear in processing_info."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        chunks = _build_chunks(ml_count=6, cooking_count=6)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        result = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=False,  # disabled
                top_k=20,
            ),
        )

        assert result["status"] == "success"
        assert "gmm" not in result["metadata"]["processing_info"]

    @pytest.mark.asyncio
    async def test_bimodal_auto_vs_static_comparison(self, tmp_path):
        """GMM-enabled query should return fewer chunks than static top_k."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        chunks = _build_chunks(ml_count=12, cooking_count=8)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        # Query WITH GMM
        result_gmm = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=True,
                top_k=20,
                gmm_min_k=3,
                gmm_max_k=60,
            ),
        )

        # Query WITHOUT GMM (static top_k)
        result_static = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=False,
                top_k=20,
                chunk_top_k=20,
            ),
        )

        gmm_final = result_gmm["metadata"]["processing_info"]["final_chunks_count"]
        static_final = result_static["metadata"]["processing_info"]["final_chunks_count"]
        gmm_info = result_gmm["metadata"]["processing_info"]["gmm"]

        # GMM should select fewer chunks than the full pool
        assert gmm_info["selected_k"] <= static_final
        # At minimum, min_k chunks must be present
        assert gmm_final >= 3

    @pytest.mark.asyncio
    async def test_small_result_below_min_k_returns_all(self, tmp_path):
        """When VDB returns fewer results than gmm_min_k, all are kept."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        # Only 4 ML chunks, no cooking — small result set
        chunks = _build_chunks(ml_count=4, cooking_count=0)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        result = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=True,
                top_k=10,
                gmm_min_k=5,  # higher than the 4 available
                gmm_max_k=60,
            ),
        )

        gmm = result["metadata"]["processing_info"]["gmm"]
        assert gmm["enabled"] is True
        # raw_count < min_k, so selected_k == raw_count
        assert gmm["selected_k"] == gmm["raw_count"]
        assert gmm["selected_k"] <= 4

    @pytest.mark.asyncio
    async def test_unimodal_respects_max_k(self, tmp_path):
        """All-similar documents (single topic) should be capped at gmm_max_k."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        # 30 ML-only chunks = unimodal distribution
        chunks = _build_chunks(ml_count=30, cooking_count=0)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        result = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=True,
                top_k=40,
                gmm_min_k=3,
                gmm_max_k=15,
            ),
        )

        gmm = result["metadata"]["processing_info"]["gmm"]
        assert gmm["enabled"] is True
        assert gmm["selected_k"] <= 15  # capped by max_k
        assert gmm["selected_k"] >= 3  # at least min_k

    @pytest.mark.asyncio
    async def test_gmm_fields_match_query_param_settings(self, tmp_path):
        """GMM info reports the same min_k/max_k that were configured."""
        rag = _make_rag(tmp_path)
        await rag.initialize_storages()

        chunks = _build_chunks(ml_count=12, cooking_count=8)
        await rag.chunks_vdb.upsert(chunks)
        await rag.text_chunks.upsert(chunks)

        result = await rag.aquery_llm(
            "What is machine learning?",
            param=QueryParam(
                mode="naive",
                only_need_context=True,
                auto_top_k=True,
                top_k=20,
                gmm_min_k=5,
                gmm_max_k=25,
            ),
        )

        gmm = result["metadata"]["processing_info"]["gmm"]
        assert gmm["min_k"] == 5
        assert gmm["max_k"] == 25
        assert 5 <= gmm["selected_k"] <= 25
        assert gmm["raw_count"] >= gmm["selected_k"]
