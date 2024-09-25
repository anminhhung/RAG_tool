import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from icecream import ic
from dotenv import load_dotenv
from llama_parse import LlamaParse
import google.generativeai as genai
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from .utils import get_files_from_folder_or_file_paths, get_extractor

load_dotenv()


def gemini_read_paper_content(
    paper_dir: Path | str, save_dir: Path | str = "output"
) -> list[Document]:
    """
    Read the content of the paper using the Gemini.

    Args:
        paper_dir (str | Path): Path to the directory containing the papers
        save_dir (str | Path): Path to the directory to save the extracted content
    Returns:
        list[Document]: List of documents from all papers.
    """
    paper_dir = Path(paper_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    paper_file = [str(file) for file in paper_dir.glob("*.pdf")]
    model = genai.GenerativeModel("gemini-1.5-flash")

    documents: list[Document] = []

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


def gemini_read_paper_content_single_file(file_path: Path | str) -> Document:
    """
    Read the content of one paper using the Gemini.

    Args:
        file_path (str | Path): Path to the paper file
    Returns:
        Document: Document object from the paper.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    pdf_file = genai.upload_file(file_path)
    response = model.generate_content(
        [
            r"""Extract all content from this paper, must be in human readable order. Each paper content is put in seperate <page></page> tag""",
            pdf_file,
        ]
    )

    return Document(text=response.text)


def llama_parse_read_paper(paper_dir: Path | str) -> list[Document]:
    """
    Read the content of the paper using  LlamaParse.

    Args:
        paper_dir (str | Path): Path to the directory all containing the papers.
    Returns:
        list[Document]: List of documents from all papers.
    """
    ic(paper_dir)

    paper_dir = Path(paper_dir)

    parser = LlamaParse(
        result_type="markdown", api_key=os.getenv("LLAMA_PARSE_API_KEY")
    )

    documents = []
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_dir=paper_dir, file_extractor=file_extractor, exclude=[".keep"]
    ).load_data(show_progress=True)

    ic(len(documents))

    return documents


def llama_parse_single_file(file_path: Path | str) -> Document:
    """
    Read the content of one paper using LlamaParse.

    Args:
        file_path (str | Path): Path to the paper file.
    Returns:
        Document: Document object from the paper.
    """
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


def parse_multiple_file(files_or_folder: list[str] | str) -> list[Document]:
    """
    Read the content of multiple papers.

    Args:
        files_or_folder (list[str] | str): List of file paths or folder paths containing the papers.
    Returns:
        list[Document]: List of documents from all papers.
    """
    if isinstance(files_or_folder, str):
        files_or_folder = [files_or_folder]

    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    if len(valid_files) == 0:
        raise ValueError("No valid files found.")

    ic(valid_files)

    file_extractor = get_extractor()

    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_extractor=file_extractor,
    ).load_data(show_progress=True)

    return documents
