import logging
from typing import Dict, Any

from llm_query.retrievers import (
    RetrievalStrategy,
    OllamaLLMRetriever,
    OpenRouterLLMRetriever,
    EmbeddingRetriever,
)

# Public symbols
__all__ = ["query_scene"]

# -----------------------------------------------------------------------------
# Two-stage retrieve & rerank parameters
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -----------------------------------------------------------------------------
# Main public API
# -----------------------------------------------------------------------------

def query_scene(
    scene_name: str,
    query: str,
    data_dir: str = "vlm_caption/outputs",
    k: int = 5,
    per_field_k: int = 10,
    k_retrieval: int = 50,
    score_threshold: float = 0.0,
    embedder_name: str = DEFAULT_EMBEDDER,
    cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
    ce_only: bool = False,
    retrieval_strategy: str = "embedding",
    **strategy_kwargs: Any,
) -> Dict:
    """Rank scene objects according to *retrieval_strategy*.

    By default the classic two-stage embedding → cross-encoder pipeline is used.
    When *retrieval_strategy* is set to ``"llm_ollama"`` the ranking is
    delegated to a local LLM served by Ollama (see
    :class:`llm_query.retrievers.OllamaLLMRetriever`).
    """

    retrieval_strategy_lc = str(retrieval_strategy).lower()

    if retrieval_strategy_lc == RetrievalStrategy.LLM_OLLAMA.value:
        retriever = OllamaLLMRetriever(
            model_name=strategy_kwargs.get("model_name", "gemma3:4b-it-qat"),
            ollama_host=strategy_kwargs.get("ollama_host"),
            temperature=strategy_kwargs.get("temperature", 0.7),
            xml_token_buffer=strategy_kwargs.get("xml_token_buffer", 5000),
        )
        return retriever.rank(scene_name, query, k=k, data_dir=data_dir)

    # ------------------------------------------------------------------
    # LLM strategy via OpenRouter
    # ------------------------------------------------------------------

    if retrieval_strategy_lc == RetrievalStrategy.LLM_OPENROUTER.value:
        retriever = OpenRouterLLMRetriever(
            model_name=strategy_kwargs.get(
                "model_name", "mistralai/mistral-small-3.2-24b-instruct:free"
            ),
            api_key=strategy_kwargs.get("api_key"),
            temperature=strategy_kwargs.get("temperature", 0.7),
            xml_token_buffer=strategy_kwargs.get("xml_token_buffer", 5000),
        )
        return retriever.rank(scene_name, query, k=k, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Embedding (default) strategy
    # ------------------------------------------------------------------

    retriever = EmbeddingRetriever(
        embedder_name=embedder_name,
        cross_encoder_name=cross_encoder_name,
        per_field_k=per_field_k,
        k_retrieval=k_retrieval,
        score_threshold=score_threshold,
        ce_only=ce_only or (retrieval_strategy_lc == RetrievalStrategy.CE_ONLY.value),
    )

    return retriever.rank(scene_name, query, k=k, data_dir=data_dir)