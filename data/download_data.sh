#!/usr/bin/env bash
# Download the VNAT dataset from MIT Lincoln Laboratory's data portal.
# Dataset page: https://www.ll.mit.edu/r-d/datasets/vnat-dataset
#
# The files must be placed in this directory (data/) before running the notebooks.
# Expected files after download:
#   data/VNAT_Dataframe_release_1.h5
#   data/VNAT_Feature_Dataframe_release_1.h5
#
# Lincoln Lab requires registration at the link above before download.
# Once you have the direct URLs, replace the placeholders below and run:
#   bash data/download_data.sh

set -e

DEST="$(dirname "$0")"

echo "Downloading VNAT_Dataframe_release_1.h5 ..."
# curl -L "<YOUR_URL_HERE>" -o "$DEST/VNAT_Dataframe_release_1.h5"

echo "Downloading VNAT_Feature_Dataframe_release_1.h5 ..."
# curl -L "<YOUR_URL_HERE>" -o "$DEST/VNAT_Feature_Dataframe_release_1.h5"

echo ""
echo "To obtain download URLs, register at:"
echo "  https://www.ll.mit.edu/r-d/datasets/vnat-dataset"
echo ""
echo "After placing the .h5 files in data/, run 01_eda.ipynb to generate data/features.csv"
