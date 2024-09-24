from llama_index.core.bridge.pydantic import BaseModel


class RAGType:
    """
    RAG type schema.

    Attributes:
        ORIGIN (str): Origin RAG type.
        CONTEXTUAL (str): Contextual RAG type.
        BOTH (str): Both Origin and Contextual RAG type.
    """

    ORIGIN = "origin"
    CONTEXTUAL = "contextual"
    BOTH = "both"


class DocumentMetadata(BaseModel):
    """
    Document metadata schema.

    Attributes:
        doc_id (str): Document ID.
        original_content (str): Original content of the document.
        contextualized_content (str): Contextualized content of the document which will be prepend to the original content.
    """

    doc_id: str
    original_content: str
    contextualized_content: str


class ElasticSearchResponse(BaseModel):
    """
    ElasticSearch response schema.

    Attributes:
        doc_id (str): Document ID.
        content (str): Content of the document.
        contextualized_content (str): Contextualized content of the document.
        score (float): Score of the document.
    """

    doc_id: str
    content: str
    contextualized_content: str
    score: float
