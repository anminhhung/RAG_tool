#!/bin/bash

# TYPE must be: ["origin", "contextual", "both"]
TYPE=${1:-both}

FOLDER_DIR=$2

python src/ingest/contextual_rag_ingest.py --type "$TYPE" --folder_dir "$FOLDER_DIR"
