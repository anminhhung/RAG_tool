#!/bin/bash

# TYPE must be in ["origin", "contextual", "both"]
TYPE=$1

python src/ingest/add_files.py --type "$TYPE" --files "${@:2}"
