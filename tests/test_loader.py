import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.readers import parse_multiple_file


def test_load_html_file():
    file = Path("./sample/test.html")
    _ = parse_multiple_file(str(file))
    assert True


def test_load_docx_file():
    file = Path("./sample/test.docx")
    _ = parse_multiple_file(str(file))
    assert True


def test_load_csv_file():
    file = Path("./sample/dummy.csv")
    _ = parse_multiple_file(str(file))
    assert True


def test_load_xlsx_file():
    file = Path("./sample/dummy.xlsx")
    _ = parse_multiple_file(str(file))
    assert True


def test_load_json_file():
    file = Path("./sample/test.json")
    _ = parse_multiple_file(str(file))
    assert True
