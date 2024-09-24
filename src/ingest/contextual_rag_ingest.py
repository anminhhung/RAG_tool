import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embedding import RAG
from src.settings import setting


def load_parser():
    parser = argparse.ArgumentParser(description="Ingest data")
    parser.add_argument(
        "--folder_dir",
        type=str,
        help="Path to the folder containing the documents",
    )
    parser.add_argument(
        "--type",
        choices=["origin", "contextual", "both"],
        required=True,
    )
    return parser.parse_args()


def main():
    args = load_parser()

    rag = RAG(setting=setting)

    rag.run_ingest(folder_dir=args.folder_dir, type=args.type)


if __name__ == "__main__":
    main()
