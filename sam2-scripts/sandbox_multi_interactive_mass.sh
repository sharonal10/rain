#!/bin/bash

# Edit this
main_dir="renders_no_bg"

for subdir in "$main_dir"/*/; do
    subdir_name=$(basename "$subdir")

    echo "$subdir_name"

    input_dir="renders_no_bg/$subdir_name"
    if [ ! -d "$input_dir" ]; then
        echo "Directory $input_dir does not exist, skipping..."
        continue
    fi

    python sandbox_multi_interactive.py -i "$input_dir" -o "mass_part_masks/$subdir_name"
done

