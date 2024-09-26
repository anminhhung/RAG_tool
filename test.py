import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.readers.llama_parse import LlamaParse

load_dotenv()

loader = LlamaParse(result_type="markdown", api_key=os.getenv("LLAMA_PARSE_API_KEY"))

documents = loader.load_data(Path("sample/2409.13588v1.pdf"))
