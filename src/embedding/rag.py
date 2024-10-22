import os
import sys
import json
import uuid
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal, Sequence

sys.path.append(str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.schema import NodeWithScore, Node
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)

from src.constants import (
    CONTEXTUAL_PROMPT,
    QA_PROMPT,
    CONTEXTUAL_SERVICE,
    CONTEXTUAL_MODEL,
)
from src.schemas import RAGType, DocumentMetadata
from src.readers.file_reader import parse_multiple_files
from src.embedding.elastic_search import ElasticSearch
from src.settings import Settings as ConfigSettings, setting as config_setting


def time_format():
    now = datetime.now()
    return f'{now.strftime("%H:%M:%S")} - DEBUG - '


load_dotenv()
ic.configureOutput(includeContext=True, prefix=time_format)

Settings.chunk_size = config_setting.chunk_size


class RAG:
    """
    Retrieval-Augmented Generation (RAG) class to handle the indexing and searching of both Origin and Contextual RAG.
    """

    setting: ConfigSettings
    llm: OpenAI
    splitter: SemanticSplitterNodeParser
    es: ElasticSearch
    qdrant_client: QdrantClient

    def __init__(self, setting: ConfigSettings):
        """
        Initialize the RAG class with the provided settings.

        Args:
            setting (ConfigSettings): The settings for the RAG.
        """
        self.setting = setting
        ic(setting)

        embed_model = OpenAIEmbedding()
        Settings.embed_model = embed_model

        self.llm = self.load_llm(
            service=CONTEXTUAL_SERVICE, model_name=CONTEXTUAL_MODEL
        )
        Settings.llm = self.llm

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )

        self.qdrant_client = QdrantClient(
            url=setting.qdrant_url,
        )

    def load_llm(self, service: str, model_name: str) -> FunctionCallingLLM:
        """
        Load the LLM model based on the contextual service.

        Args:
            service (str): The contextual service.
            model_name (str): The model name.

        Returns:
            FunctionCallingLLM: The LLM model.
        """
        self.llm_service = service
        self.llm_model_name = model_name

        if service == "openai":
            return OpenAI(model=model_name)

        elif service == "huggingface":
            return LlamaCPP(
                model_url=model_name,
                model_path=None,
                temperature=0.1,
                max_new_tokens=2048,
                # context_window=3900,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": 1},
                verbose=True,
            )

    def split_document(
        self,
        document: Document | list[Document],
        show_progress: bool = True,
    ) -> list[list[Document]]:
        """
        Split the document into chunks.

        Args:
            document (Document | list[Document]): The document to split.
            show_progress (bool): Show the progress bar.

        Returns:
            list[list[Document]]: List of documents after splitting.
        """
        if isinstance(document, Document):
            document = [document]

        assert isinstance(document, list)

        documents: list[list[Document]] = []

        document = tqdm(document, desc="Splitting...") if show_progress else document

        for doc in document:
            nodes = self.splitter.get_nodes_from_documents([doc])
            documents.append([Document(text=node.get_content()) for node in nodes])

        return documents

    def preprocess_message(self, messages: Sequence[ChatMessage]):
        """
        Preprocess the message for the LLM.

        Args:
            message (Sequence[ChatMessage]): The message to preprocess.

        Returns:
            Sequence[ChatMessage]: The preprocessed message.
        """
        if self.llm_service == "huggingface":
            if len(messages) % 2 == 0:
                return messages[1:]
        return messages

    def add_contextual_content(
        self,
        origin_document: Document,
        splited_documents: list[Document],
    ) -> list[Document]:
        """
        Add contextual content to the splited documents.

        Args:
            origin_document (Document): The original document.
            splited_documents (list[Document]): The splited documents from the original document.

        Returns:
            list[Document]: List of documents with contextual content.
        """

        whole_document = origin_document.text
        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

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

            response = self.llm.chat(self.preprocess_message(messages))
            contextualized_content = response.message.content

            # Prepend the contextualized content to the chunk
            new_chunk = contextualized_content + "\n\n" + chunk.text

            # Manually generate a doc_id for indexing in elastic search
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

    def get_contextual_documents(
        self, raw_documents: list[Document], splited_documents: list[list[Document]]
    ) -> tuple[list[Document], list[DocumentMetadata]]:
        """
        Get the contextual documents from the raw and splited documents.

        Args:
            raw_documents (list[Document]): List of raw documents.
            splited_documents (list[list[Document]]): List of splited documents from the raw documents one by one.

        Returns:
            (tuple[list[Document], list[DocumentMetadata]]): Tuple of contextual documents and its metadata.
        """

        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        assert len(raw_documents) == len(splited_documents)

        for raw_document, splited_document in tqdm(
            zip(raw_documents, splited_documents),
            desc="Adding contextual content ...",
            total=len(raw_documents),
        ):
            document, metadata = self.add_contextual_content(
                raw_document, splited_document
            )
            documents.extend(document)
            documents_metadata.extend(metadata)

        return documents, documents_metadata

    def ingest_data(
        self,
        documents: list[Document],
        show_progress: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):
        """
        Ingest the data to the QdrantVectorStore.

        Args:
            documents (list[Document]): List of documents to ingest.
            show_progress (bool): Show the progress bar.
            type (Literal["origin", "contextual"]): The type of RAG to ingest.
        """

        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name

        ic(type, collection_name)

        vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=show_progress
        )

        return index  # noqa

    def insert_data(
        self,
        documents: list[Document],
        show_progress: bool = True,
        type: Literal["origin", "contextual"] = "contextual",
    ):

        if type == "origin":
            collection_name = self.setting.original_rag_collection_name
        else:
            collection_name = self.setting.contextual_rag_collection_name

        ic(type, collection_name)

        vector_store_index = self.get_qdrant_vector_store_index(
            client=self.qdrant_client,
            collection_name=collection_name,
        )

        documents = (
            tqdm(documents, desc=f"Adding more data to {type} ...")
            if show_progress
            else documents
        )
        for document in documents:
            vector_store_index.insert(document)

    def get_qdrant_vector_store_index(
        self, client: QdrantClient, collection_name: str
    ) -> VectorStoreIndex:
        """
        Get the QdrantVectorStoreIndex from the QdrantVectorStore.

        Args:
            client (QdrantClient): The Qdrant client.
            collection_name (str): The collection name.

        Returns:
            VectorStoreIndex: The VectorStoreIndex from the QdrantVectorStore.
        """
        ic(collection_name)

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )

    def get_query_engine(
        self, type: Literal["origin", "contextual", "both"]
    ) -> BaseQueryEngine | dict[str, BaseQueryEngine]:
        """
        Get the query engine for the RAG.

        Args:
            type (Literal["origin", "contextual", "both"]): The type of RAG.

        Returns:
            BaseQueryEngine | dict[str, BaseQueryEngine]: The query engine.
        """
        ic(type)

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
    ) -> None:
        """
        Run the ingest process for the RAG.

        Args:
            folder_dir (str | Path): The folder directory containing the papers.
            type (Literal["origin", "contextual", "both"]): The type to ingest. Default to `contextual`.
        """
        ic(folder_dir, type)

        raw_documents = parse_multiple_files(folder_dir)
        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        if type == RAGType.ORIGIN:
            self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.ingest_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.ingest_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.es.index_documents(contextual_documents_metadata)

            ic(f"Ingested data for {type}")

    def run_add_files(
        self, files_or_folders: list[str], type: Literal["origin", "contextual", "both"]
    ):
        """
        Add files to the database.

        Args:
            files_or_folders (list[str]): List of file paths or folder to be ingested.
            type (Literal["origin", "contextual", "both"]): Type of RAG type to ingest.
        """
        ic(files_or_folders, type)

        raw_documents = parse_multiple_files(files_or_folders)
        splited_documents = self.split_document(raw_documents)

        ingest_documents: list[Document] = []
        if type == RAGType.BOTH or type == RAGType.ORIGIN:
            for each_splited in splited_documents:
                ingest_documents.extend(each_splited)

        if type == RAGType.ORIGIN:
            self.insert_data(ingest_documents, type=RAGType.ORIGIN)

        else:
            if type == RAGType.BOTH:
                self.insert_data(ingest_documents, type=RAGType.ORIGIN)

            contextual_documents, contextual_documents_metadata = (
                self.get_contextual_documents(
                    raw_documents=raw_documents, splited_documents=splited_documents
                )
            )

            assert len(contextual_documents) == len(contextual_documents_metadata)

            self.insert_data(contextual_documents, type=RAGType.CONTEXTUAL)

            self.es.index_documents(contextual_documents_metadata)

            ic(f"Added data for {type}")

    def origin_rag_search(self, query: str) -> str:
        """
        Search the query in the Origin RAG.

        Args:
            query (str): The query to search.

        Returns:
            str: The search results.
        """
        ic(query)

        index = self.get_query_engine(RAGType.ORIGIN)
        return index.query(query)

    def contextual_rag_search(
        self, query: str, k: int = 150, debug: bool = False
    ) -> str:
        """
        Search the query with the Contextual RAG.

        Args:
            query (str): The query to search.
            k (int): The number of documents to return. Default to `150`.
            debug (bool): debug mode.

        Returns:
            str: The search results.
        """
        ic(query, k, debug)

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

        semantic_doc_id = [
            node.metadata["doc_id"] for node in semantic_results.source_nodes
        ]

        def get_content_by_doc_id(doc_id: str):
            for node in semantic_results.source_nodes:
                if node.metadata["doc_id"] == doc_id:
                    return node.text
            return ""

        bm25_results = self.es.search(query, k=k)
        bm25_doc_id = [result.doc_id for result in bm25_results]

        combined_nodes: list[NodeWithScore] = []
        combined_ids = list(set(semantic_doc_id + bm25_doc_id))

        # Compute score according to: https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
        semantic_count = 0
        bm25_count = 0
        both_count = 0
        for id in combined_ids:
            score = 0
            content = ""
            if id in semantic_doc_id:
                index = semantic_doc_id.index(id)
                score += semantic_weight * (1 / (index + 1))
                content = get_content_by_doc_id(id)
                semantic_count += 1

            if id in bm25_doc_id:
                index = bm25_doc_id.index(id)
                score += bm25_weight * (1 / (index + 1))

                if content == "":
                    content = (
                        bm25_results[index].contextualized_content
                        + "\n\n"
                        + bm25_results[index].content
                    )
                bm25_count += 1
            if id in semantic_doc_id and id in bm25_doc_id:
                both_count += 1

            combined_nodes.append(
                NodeWithScore(
                    node=Node(
                        text=content,
                    ),
                    score=score,
                )
            )

        if debug:
            ic(semantic_count, bm25_count, both_count)

        reranker = CohereRerank(
            top_n=self.setting.top_n,
            api_key=os.getenv("COHERE_API_KEY"),
        )

        query_bundle = QueryBundle(query_str=query)

        retrieved_nodes = reranker.postprocess_nodes(combined_nodes, query_bundle)

        contexts = [n.node.text for n in retrieved_nodes]

        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            ChatMessage(
                role="user",
                content=QA_PROMPT.format(
                    context_str=json.dumps(contexts),
                    query_str=query,
                ),
            ),
        ]

        response = self.llm.chat(self.preprocess_message(messages)).message.content

        return response
