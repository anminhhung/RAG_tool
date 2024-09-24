from llama_index.core.bridge.pydantic import BaseModel


class RAGType:
    ORIGIN = "origin"
    CONTEXTUAL = "contextual"
    BOTH = "both"


class DocumentMetadata(BaseModel):
    doc_id: str
    original_content: str
    contextualized_content: str
