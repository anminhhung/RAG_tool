import os
import sys
import uuid
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

sys.path.append(str(Path(Path(__file__)).parent.parent.parent))

import qdrant_client
from llama_index.core import QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, Node
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings

from src.constants import CONTEXTUAL_PROMPT
from src.db.elastic_search import ElasticSearch
from src.schemas import RAGType, DocumentMetadata
from src.settings import Settings as ConfigSettings, setting
from src.readers.paper_reader import llama_parse_read_paper

load_dotenv()
logger = logging.getLogger(__name__)

Settings.chunk_size = setting.chunk_size


class RAG:
    def __init__(self, setting: ConfigSettings):
        self.setting = setting
        embed_model = OpenAIEmbedding()
        Settings.embed_model = embed_model

        self.llm = OpenAI(model=setting.model)
        Settings.llm = self.llm

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )

        self.qdrant_client = qdrant_client.QdrantClient(
            host=setting.qdrant_host, port=setting.qdrant_port
        )

    def split_document(
        self,
        document: Document | list[Document],
        show_progress: bool = True,
    ) -> list[Document]:
        if isinstance(document, Document):
            document = [document]

        nodes = self.splitter.get_nodes_from_documents(
            document, show_progress=show_progress
        )

        return [Document(text=node.get_content()) for node in nodes]

    def add_contextual_content(
        self,
        single_document: Document,
    ) -> list[Document]:
        whole_document = single_document.text
        splited_documents = self.split_document(single_document)
        documents: list[Document] = []
        documents_metadata: list[dict] = []

        for chunk in splited_documents:
            messages = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                ChatMessage(
                    role="user",
                    content=CONTEXTUAL_PROMPT.format(
                        WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                    ),
                ),
            ]
            response = self.llm.chat(messages)
            contextualized_content = response.message.content

            # Prepend the contextualized content to the chunk
            new_chunk = contextualized_content + "\n\n" + chunk.text

            doc_id = str(uuid.uuid4())
            documents.append(
                Document(
                    text=new_chunk,
                    metadata=dict(
                        doc_id=doc_id,
                    ),
                ),
            )
            documents_metadata.append(
                DocumentMetadata(
                    doc_id=doc_id,
                    original_content=whole_document,
                    contextualized_content=contextualized_content,
                ),
            )

        return documents, documents_metadata

    def get_contextual_documents(self, raw_documents: str | Path) -> list[Document]:

        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        for raw_document in tqdm(raw_documents):
            documents.extend(self.add_contextual_content(raw_document)[0])
            documents_metadata.extend(self.add_contextual_content(raw_document)[1])

        return documents, documents_metadata

    def get_origin_documents(self, raw_documents: list[Document]) -> list[Document]:

        documents = self.split_document(raw_documents)

        return documents

    def ingest_data(
        self,
        documents: list[Document],
        show_progress: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):
        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name
        logger.info("Collection name: %s", collection_name)

        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=show_progress
        )

        return index

    def get_qdrant_vector_store_index(
        self, client: qdrant_client.QdrantClient, collection_name: str
    ):
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )

    def get_query_engine(self, type: Literal["origin", "contextual", "both"]):
        if type == RAGType.ORIGIN:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.original_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.CONTEXTUAL:
            return self.get_qdrant_vector_store_index(
                client=self.qdrant_client,
                collection_name=self.setting.contextual_rag_collection_name,
            ).as_query_engine()

        elif type == RAGType.BOTH:
            return {
                "origin": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.original_rag_collection_name,
                ).as_query_engine(),
                "contextual": self.get_qdrant_vector_store_index(
                    client=self.qdrant_client,
                    collection_name=self.setting.contextual_rag_collection_name,
                ).as_query_engine(),
            }

    def run_ingest(
        self,
        folder_dir: str | Path,
        type: Literal["origin", "contextual", "both"] = "contextual",
    ):
        raw_documents = llama_parse_read_paper(folder_dir)

        if type == RAGType.BOTH:
            origin_documents = self.get_origin_documents(raw_documents=raw_documents)
            contextual_documents, documents_metadata = self.get_contextual_documents(
                raw_documents=raw_documents
            )

            self.ingest_data(origin_documents, type=RAGType.ORIGIN)
            self.ingest_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.es.index_documents(documents_metadata)

            logger.info("Ingested data for both origin and contextual")
        else:
            if type == RAGType.ORIGIN:
                documents = self.get_origin_documents(raw_documents=raw_documents)
            else:
                documents, documents_metadata = self.get_contextual_documents(
                    raw_documents=raw_documents
                )

            self.ingest_data(documents, type=type)

            if type == RAGType.CONTEXTUAL:
                # Elastic search indexing
                self.es.index_documents(documents_metadata)

            logger.info(f"Ingested data for {type}")

    def origin_rag_search(self, query: str):
        index = self.get_query_engine(RAGType.ORIGIN)
        return index.query(query)

    def contextual_rag_search(self, query: str, k: int = 150):
        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight

        index = self.get_qdrant_vector_store_index(
            self.qdrant_client, self.setting.contextual_rag_collection_name
        )

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=k,
        )

        query_engine = RetrieverQueryEngine(retriever=retriever)

        semantic_results: Response = query_engine.query(query)

        nodes = semantic_results.source_nodes

        semantic_doc_id = [node.metadata["doc_id"] for node in nodes]

        def get_content_by_doc_id(doc_id: str):
            for node in nodes:
                if node.metadata["doc_id"] == doc_id:
                    return node.text
            return ""

        bm25_results = self.es.search(query, k=k)

        bm25_doc_id = [result["doc_id"] for result in bm25_results]

        combined_nodes: list[NodeWithScore] = []

        combined_ids = list(set(semantic_doc_id + bm25_doc_id))

        # Compute score according to: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
        for id in combined_ids:
            score = 0
            content = ""
            if id in semantic_doc_id:
                index = semantic_doc_id.index(id)
                score += semantic_weight * (1 / (index + 1))
                content = get_content_by_doc_id(id)
            if id in bm25_doc_id:
                index = bm25_doc_id.index(id)
                score += bm25_weight * (1 / (index + 1))

                if content == "":
                    content = bm25_results[index]["content"]

            combined_nodes.append(
                NodeWithScore(
                    node=Node(
                        text=content,
                    ),
                    score=score,
                )
            )

        reranker = CohereRerank(
            top_n=self.setting.top_n,
            api_key=os.getenv("COHERE_API_KEY"),
        )

        query_bundle = QueryBundle(query_str=query)

        retrieved_nodes = reranker.postprocess_nodes(combined_nodes, query_bundle)

        text_nodes = [Node(text=node.node.text) for node in retrieved_nodes]

        vector_store = VectorStoreIndex(
            nodes=text_nodes,
        ).as_query_engine()

        return vector_store.query(query)
