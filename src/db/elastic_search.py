from icecream import ic
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from llama_index.core.bridge.pydantic import Field

from src.schemas import DocumentMetadata, ElasticSearchResponse


class ElasticSearch:
    """
    ElasticSearch client to index and search documents for contextual RAG.
    """

    url: str = Field(..., description="Elastic Search URL")

    def __init__(self, url: str, index_name: str):
        """
        Initialize the ElasticSearch client.

        Args:
            url (str): URL of the ElasticSearch server
            index_name (str): Name of the index used to be created for contextual RAG
        """
        self.es_client = Elasticsearch(url)
        self.index_name = index_name
        self.create_index()

        ic("ElasticSearch client initialized !")

    def create_index(self):
        """
        Create the index for contextual RAG from provided index name.
        """
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "text", "index": False},
                }
            },
        }

        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            ic(f"Created index: {self.index_name}")

    def index_documents(self, documents_metadata: list[DocumentMetadata]) -> bool:
        """
        Index the documents to the ElasticSearch index.

        Args:
            documents_metadata (list[DocumentMetadata]): List of documents metadata to index.
        """
        ic("Indexing documents...")

        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "doc_id": metadata.doc_id,
                    "content": metadata.original_content,
                    "contextualized_content": metadata.contextualized_content,
                },
            }
            for metadata in documents_metadata
        ]

        success, _ = bulk(self.es_client, actions)
        if success:
            ic("Indexed documents successfully !")

        self.es_client.indices.refresh(index=self.index_name)

        return success

    def search(self, query: str, k: int = 20) -> list[ElasticSearchResponse]:
        """
        Search the documents relevant to the query.

        Args:
            query (str): Query to search
            k (int): Number of documents to return

        Returns:
            list[ElasticSearchResponse]: List of ElasticSearch response objects.
        """
        ic(query)

        self.es_client.indices.refresh(
            index=self.index_name
        )  # Force refresh before each search
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)

        return [
            ElasticSearchResponse(
                doc_id=hit["_source"]["doc_id"],
                content=hit["_source"]["content"],
                contextualized_content=hit["_source"]["contextualized_content"],
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]
