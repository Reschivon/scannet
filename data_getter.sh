#!/bin/bash

# Base directory where ScanNet data is stored
BASE_DIR="/mnt/e/scannet/scans"
DEST_DIR='.'

# Download ScanNet data
# python scannetlib/download_scannetv2.py -o /mnt/e/scannet/

# Define the start scene
START_SCENE="scene0012_00"

# Flag to check if the current scene is after or same as the start scene
START_PROCESSING=false

# Loop through each scene directory in the ScanNet scans
for SCENE_DIR in $(ls -d ${BASE_DIR}/scene* | sort); do

    # Extract scene name from the directory path
    SCENE_NAME=$(basename $SCENE_DIR)
    
    echo $SCENE_NAME
    
    # Check if current scene is the start scene or later
    if [[ "$SCENE_NAME" == "$START_SCENE" ]]; then
        START_PROCESSING=true
    fi

    # If the current scene is the start scene or a later scene, process it
    if $START_PROCESSING; then
        echo "Processing $SCENE_NAME..."
    
        # Convert .sens file to image sequences after 0009_00
        python scannetlib/reader.py --filename "${SCENE_DIR}/${SCENE_NAME}.sens" --output_path "${SCENE_NAME}_data"

        # # Unzip the label-filt.zip to a new directory
        # unzip -u "${SCENE_DIR}/${SCENE_NAME}_2d-label-filt.zip" -d "${SCENE_DIR}/${SCENE_NAME}_2d-label-filt"

        # # Copy necessary files
        # ./copy_files.sh "${SCENE_DIR}/${SCENE_NAME}_2d-label-filt/label-filt" "${SCENE_NAME}_data/labels"
    fi
done

echo "Processing complete."
