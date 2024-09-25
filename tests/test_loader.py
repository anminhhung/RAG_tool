import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# We import here to make sure it is successfully installed.
from llama_index.readers.llama_parse import LlamaParse  # noqa

from src.readers import (
    DocxReader,
    JSONReader,
    UnstructuredReader,
    PandasCSVReader,
    PandasExcelReader,
    TxtReader,
)


def test_load_html():
    path = Path("sample/test.html")
    _ = UnstructuredReader().load_data(path)
    assert True


def test_load_docx():
    path = Path("sample/test.docx")
    path2 = Path("sample/dummy.docx")
    _ = DocxReader().load_data(path)
    _ = DocxReader().load_data(path2)
    assert True


def test_load_json():
    path = Path("sample/test.json")
    _ = JSONReader().load_data(path)
    assert True


def test_load_csv():
    path = Path("sample/dummy.csv")
    _ = PandasCSVReader(pandas_config=dict(on_bad_lines="skip")).load_data(path)
    assert True


def test_load_xlsx():
    path = Path("sample/dummy.xlsx")
    _ = PandasExcelReader().load_data(path)
    assert True


def test_load_txt():
    path = Path("sample/test.txt")
    _ = TxtReader().load_data(path)
    assert True
