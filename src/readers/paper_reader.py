import os
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
import google.generativeai as genai
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader

load_dotenv()


def gemini_read_paper_content(paper_dir: Path | str, save_dir: Path | str = "output"):
    paper_dir = Path(paper_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    paper_file = [str(file) for file in paper_dir.glob("*.pdf")]
    model = genai.GenerativeModel("gemini-1.5-flash")

    documents = []

    for file in tqdm(paper_file):
        assert isinstance(file, str)

        file_name = Path(file).stem + ".txt"

        ic(file)

        pdf_file = genai.upload_file(file)
        response = model.generate_content(
            [
                r"""Extract all content from this paper, must be in human readable order. Each paper content is put in seperate <page></page> tag""",
                pdf_file,
            ]
        )
        documents.append(Document(text=response.text))

        with open(save_dir / file_name, "w") as f:
            f.write(response.text)

    return documents


def gemini_read_paper_content_single_file(file_path: Path | str):
    model = genai.GenerativeModel("gemini-1.5-flash")

    assert isinstance(file_path, str)

    pdf_file = genai.upload_file(file_path)
    response = model.generate_content(
        [
            r"""Extract all content from this paper, must be in human readable order. Each paper content is put in seperate <page></page> tag""",
            pdf_file,
        ]
    )

    return Document(text=response.text)


def llama_parse_read_paper(paper_dir: Path | str, save_dir: Path | str = "output"):
    paper_dir = Path(paper_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    parser = LlamaParse(
        result_type="markdown", api_key=os.getenv("LLAMA_PARSE_API_KEY")
    )

    documents = []
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_dir=paper_dir, file_extractor=file_extractor, exclude=[".keep"]
    ).load_data(show_progress=True)

    return documents


def llama_parse_single_file(file_path: Path | str) -> Document:
    parser = LlamaParse(
        result_type="markdown", api_key=os.getenv("LLAMA_PARSE_API_KEY")
    )

    file_path = Path(file_path)

    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return documents
