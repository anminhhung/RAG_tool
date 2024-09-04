import os
import torch
import chromadb
import urllib.parse
import dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.core.tools import FunctionTool
from src.constants import DOCUMENT_EMBEDDING_MODEL_NAME, DOCUMENT_EMBEDDING_SERVICE

dotenv.load_dotenv(override=True)

simple_content_template = """
Link: {paper_link}
Document: {paper_content}
"""

simple_web_search_template = """
Title: {title}
Link: {search_link}
Content: {search_content}
"""

def load_document_search_tool():
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if DOCUMENT_EMBEDDING_SERVICE == "ollama":
        embed_model = OllamaEmbedding(model_name=DOCUMENT_EMBEDDING_MODEL_NAME)
    elif DOCUMENT_EMBEDDING_SERVICE == "hf":
        embed_model = HuggingFaceEmbedding(
            model_name=DOCUMENT_EMBEDDING_MODEL_NAME, 
            cache_folder="./models", 
            device=device_type, 
            embed_batch_size=64)
    elif DOCUMENT_EMBEDDING_SERVICE == "openai":
        embed_model = OpenAIEmbedding(
            model=DOCUMENT_EMBEDDING_MODEL_NAME, 
            api_key=os.environ["OPENAI_API_KEY"])
    else:
        raise NotImplementedError() 
    chroma_client = chromadb.PersistentClient(path="./DB/docs")
    chroma_collection = chroma_client.get_or_create_collection("gemma_assistant_aio_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)    
    # load the vectorstore
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    paper_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model)

    concept_retriever = paper_index.as_retriever(
        similarity_top_k=10,
    )
    
    # rerank_postprocessor = SentenceTransformerRerank(
    #     model='mixedbread-ai/mxbai-rerank-xsmall-v1',
    #     top_n=5, # number of nodes after re-ranking, 
    #     keep_retrieval_score=True
    # )
    
    def retrieve_ai_concepts(query_str: str):
        
        retriever_response =  concept_retriever.retrieve(query_str)
        
        retriever_result = []
        for n in retriever_response:
            file_name = n.node.metadata["file_name"]
            # paper_id = list(n.node.relationships.items())[0][1].node_id
            paper_content = n.node.get_content(metadata_mode=MetadataMode.LLM)
            
            document_link = f"https://github.com/BachNgoH/AIO_Documents/tree/main/Documents/{file_name}"
            n.node.text = simple_content_template.format(
                    paper_link=urllib.parse.quote(document_link), 
                    paper_content=paper_content
            )
            
            retriever_result.append(n.node.text)

        return retriever_result
            
        
    # paper_search_tool = QueryEngineTool.from_defaults(
    #     query_engine=paper_query_engine,
    #     description="Useful for answering questions related to scientific papers",
    # )
    return FunctionTool.from_defaults(retrieve_ai_concepts, description="Useful for answering about AI and Python concepts")