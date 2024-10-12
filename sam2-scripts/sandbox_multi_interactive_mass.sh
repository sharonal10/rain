#!/bin/bash

# Path to the main directory containing the subdirectories
main_dir="renders_no_bg"

# Iterate through all subdirectories in the main directory
for subdir in "$main_dir"/*/; do
    # Extract the name of the subdirectory
    subdir_name=$(basename "$subdir")

    echo "$subdir_name"

    # Check if the input directory exists before running the command
    input_dir="renders_no_bg/$subdir_name"
    if [ ! -d "$input_dir" ]; then
        echo "Directory $input_dir does not exist, skipping..."
        continue
    fi

    # Run the python command for each subdirectory
    python sandbox_multi_interactive.py -i "$input_dir" -o "mass_part_masks/$subdir_name"
done

