#!/bin/bash

# Usage: ./batch_eval.sh /path/to/main_folder

main_folder="$1"

if [ -z "$main_folder" ]; then
    echo "Usage: $0 /path/to/main_folder"
    exit 1
fi

# Find all leaf directories (directories with no subdirectories)
find "$main_folder" -type d | while read -r dir; do
    # Check if the directory has no subdirectories
    if [ "$(find "$dir" -mindepth 1 -type d | wc -l)" -eq 0 ]; then
        echo "Evaluating: $dir"
        python eval.py -f -d "$dir" -j 48 -df gemini_api
    fi
done