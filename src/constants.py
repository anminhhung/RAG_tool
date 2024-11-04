from config.config import get_config

cfg = get_config("config/config.yaml")
cfg.merge_from_file("config/postgres.yaml")


STREAM = cfg.MODEL.STREAM
SERVICE = cfg.MODEL.SERVICE
TEMPERATURE = cfg.MODEL.TEMPERATURE
MODEL_ID = cfg.MODEL.MODEL_ID

EMBEDDING_SERVICE = cfg.MODEL.EMBEDDING_SERVICE
EMBEDDING_MODEL = cfg.MODEL.EMBEDDING_MODEL


CONTEXTUAL_PROMPT = """<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

QA_PROMPT = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

# Contextual RAG
CONTEXTUAL_CHUNK_SIZE = cfg.CONTEXTUAL_RAG.CHUNK_SIZE
CONTEXTUAL_SERVICE = cfg.CONTEXTUAL_RAG.SERVICE
CONTEXTUAL_MODEL = cfg.CONTEXTUAL_RAG.MODEL

ORIGINAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.ORIGIN_RAG_COLLECTION_NAME
CONTEXTUAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME

QDRANT_URL = cfg.CONTEXTUAL_RAG.QDRANT_URL
ELASTIC_SEARCH_URL = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_URL
ELASTIC_SEARCH_INDEX_NAME = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME

NUM_CHUNKS_TO_RECALL = cfg.CONTEXTUAL_RAG.NUM_CHUNKS_TO_RECALL
SEMANTIC_WEIGHT = cfg.CONTEXTUAL_RAG.SEMANTIC_WEIGHT
BM25_WEIGHT = cfg.CONTEXTUAL_RAG.BM25_WEIGHT
TOP_N = cfg.CONTEXTUAL_RAG.TOP_N

SUPPORTED_FILE_EXTENSIONS = [".pdf", ".docx", ".csv", ".html", ".xlsx", ".json", ".txt"]

AGENT_TYPE = cfg.AGENT.TYPE
