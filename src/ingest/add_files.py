import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
from src.embedding import RAG
from src.settings import Settings


def load_parser():
    parser = argparse.ArgumentParser(description="Add papers to the database")
    parser.add_argument(
        "--type",
        type=str,
        choices=["origin", "contextual", "both"],
        help="Type of RAG type to ingest",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="List of file paths or file folders to be ingested",
    )
    return parser.parse_args()


def main():
    args = load_parser()

    setting = Settings()

    rag = RAG(setting)

    rag.run_add_files(files_or_folders=args.files, type=args.type)


if __name__ == "__main__":
    main()
