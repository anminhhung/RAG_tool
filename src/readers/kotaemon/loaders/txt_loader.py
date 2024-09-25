import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Optional

from src.readers.kotaemon.base import Document, split_text

from llama_index.core.readers.base import BaseReader


class TxtReader(BaseReader):
    def __init__(self, max_words_per_page: int = 2048, *args, **kwargs):
        self.max_words_per_page = max_words_per_page

    def run(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> list[Document]:
        return self.load_data(Path(file_path), extra_info=extra_info, **kwargs)

    def load_data(
        self, file_path: Path, extra_info: Optional[dict] = None, **kwargs
    ) -> list[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        texts = split_text(text, max_tokens=self.max_words_per_page)

        metadata = extra_info or {}
        return [Document(text=t, metadata=metadata) for t in texts]
