#! /bin/bash

# Usage: ./colmap_sparse.sh ./lego_known_poses

WORKSPACE=$1

COLMAP_CMD=/Applications/COLMAP.app/Contents/MacOS/colmap
COLMAP_CMD=colmap
COLMAP_CMD=/data/gg921/bin/colmap

# Recompute features from the images of the known camera poses
$COLMAP_CMD feature_extractor \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images

# Feature matching
$COLMAP_CMD exhaustive_matcher \
    --database_path $WORKSPACE/database.db

# Create sparse map
mkdir -p $WORKSPACE/sparse

$COLMAP_CMD mapper \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --output_path $WORKSPACE/sparse

#/usr/local/opt/qt5/lib/cmake/Qt5
