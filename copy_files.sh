#!/bin/bash

# Check for correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Define source and destination directories from command-line arguments
SRC_DIR=$1
DST_DIR=$2

# Check if the source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Source directory does not exist."
    exit 1
fi

# Create the destination directory if it doesn't exist
if [ ! -d "$DST_DIR" ]; then
    mkdir -p "$DST_DIR"
fi

# Initialize counter for new filenames
counter=0

# Loop through the files in the source directory
# Assuming files are named like '0.png', '100.png', ..., '9900.png', etc.
for i in $(seq 0 100 9999); do
    FILE_NAME="${i}.png"
    SRC_PATH="${SRC_DIR}/${FILE_NAME}"

    # Check if the file exists before attempting to copy
    if [ -f "$SRC_PATH" ]; then
        NEW_FILE_NAME="${counter}.png"
        DST_PATH="${DST_DIR}/${NEW_FILE_NAME}"
        cp "$SRC_PATH" "$DST_PATH"
        echo "Copied $SRC_PATH to $DST_PATH as $NEW_FILE_NAME"
        ((counter++))  # Increment the counter for the next file
    fi
done

echo "Operation completed."
