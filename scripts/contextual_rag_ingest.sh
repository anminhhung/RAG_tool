#!/bin/bash

# TYPE must be: ["origin", "contextual", "both"]
TYPE=${1:-both}

python src/ingest/contextual_rag_ingest.py --folder_dir papers --type "$TYPE"
