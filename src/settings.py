from llama_index.core.bridge.pydantic import Field, BaseModel

from src.constants import (
    TOP_N,
    QDRANT_URL,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    EMBEDDING_MODEL,
    CONTEXTUAL_MODEL,
    EMBEDDING_SERVICE,
    ELASTIC_SEARCH_URL,
    CONTEXTUAL_SERVICE,
    NUM_CHUNKS_TO_RECALL,
    CONTEXTUAL_CHUNK_SIZE,
    ELASTIC_SEARCH_INDEX_NAME,
    ORIGINAL_RAG_COLLECTION_NAME,
    CONTEXTUAL_RAG_COLLECTION_NAME,
)


class Settings(BaseModel):
    """
    Settings for the contextual RAG.

    Attributes:
        chunk_size (int): Default chunk size
        service (str): The LLM service, e.g., "openai"
        model (str): The LLM model name, e.g., "gpt-4o-mini"
        original_rag_collection_name (str): The original RAG collection name
        contextual_rag_collection_name (str): The contextual RAG collection name
        qdrant_url (str): The QdrantVectorDB URL
        elastic_search_url (str): The ElasticSearch URL
        elastic_search_index_name (str): The ElasticSearch index name
        num_chunks_to_recall (int): The number of chunks to recall
        semantic_weight (float): The semantic weight
        bm25_weight (float): The BM25 weight
        top_n (int): Top n documents after reranking
    """

    chunk_size: int = Field(description="The chunk size", default=CONTEXTUAL_CHUNK_SIZE)

    service: str = Field(description="The LLM service", default=CONTEXTUAL_SERVICE)

    model: str = Field(description="The LLM model", default=CONTEXTUAL_MODEL)

    embed_service: str = Field(
        description="The embedding service", default=EMBEDDING_SERVICE
    )

    embed_model: str = Field(description="The embedding model", default=EMBEDDING_MODEL)

    original_rag_collection_name: str = Field(
        description="The original RAG collection name",
        default=ORIGINAL_RAG_COLLECTION_NAME,
    )

    contextual_rag_collection_name: str = Field(
        description="The contextual RAG collection name",
        default=CONTEXTUAL_RAG_COLLECTION_NAME,
    )

    qdrant_url: str = Field(description="The QdrantVectorDB URL", default=QDRANT_URL)

    elastic_search_url: str = Field(
        description="The Elastic URL", default=ELASTIC_SEARCH_URL
    )
    elastic_search_index_name: str = Field(
        description="The Elastic index name", default=ELASTIC_SEARCH_INDEX_NAME
    )
    num_chunks_to_recall: int = Field(
        description="The number of chunks to recall", default=NUM_CHUNKS_TO_RECALL
    )

    # Reference: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
    semantic_weight: float = Field(
        description="The semantic weight", default=SEMANTIC_WEIGHT
    )
    bm25_weight: float = Field(description="The BM25 weight", default=BM25_WEIGHT)

    top_n: int = Field(description="Top n documents after reranking", default=TOP_N)


setting = Settings()
