import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.tools import FunctionTool

from src.embedding import RAG
from src.settings import setting


def load_contextual_rag_tool():
    rag = RAG(setting)

    def answer_query(query_str: str) -> str:
        """
        A helpfull function to answer a query.

        Args:
            query_str (str): The query string to search for.

        Returns:
            str: The answer to the query.
        """
        return rag.contextual_rag_search(query_str)

    return FunctionTool.from_defaults(
        fn=answer_query,
        description="A useful tool to answer queries of user using RAG.",
    )
