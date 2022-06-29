#! /bin/bash

# Usage: ./colmap_sparse.sh ./lego_known_poses

WORKSPACE=$1

# Recompute features from the images of the known camera poses
/data/gg921/bin/colmap feature_extractor \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images

# Feature matching
/data/gg921/bin/colmap exhaustive_matcher \
    --database_path $WORKSPACE/database.db

# Create sparse map
mkdir -p $WORKSPACE/sparse

/data/gg921/bin/colmap mapper \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --output_path $WORKSPACE/sparse
