#! /bin/bash

# Usage: ./colmap_all.sh ./lego_custom

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

/data/gg921/bin/colmap model_converter \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/0 \
    --output_type TXT

mkdir -p $WORKSPACE/dense

/data/gg921/bin/colmap image_undistorter \
    --image_path $WORKSPACE/images \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/dense \
    --output_type COLMAP \
    --max_image_size 2000

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

/data/gg921/bin/colmap patch_match_stereo \
    --workspace_path $WORKSPACE/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

/data/gg921/bin/colmap stereo_fusion \
    --workspace_path $WORKSPACE/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $WORKSPACE/dense/fused.ply

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi
