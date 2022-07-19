#! /bin/bash

# Usage: ./reconstruct_known_poses.sh ./lego_known_poses transforms.json
WORKSPACE=$1
POSES=$2

COLMAP_CMD=/Applications/COLMAP.app/Contents/MacOS/colmap
COLMAP_CMD=colmap
COLMAP_CMD=/data/gg921/bin/colmap

for dir in $WORKSPACE/*; do
    [ "$dir" = $WORKSPACE"/images" ] && continue
    [ "$dir" = $WORKSPACE"/transforms.json" ] && continue 
    rm -rf "$dir"
    echo "Deleting "$dir

    retval=$?
    if [ $retval -ne 0 ]; then
        echo "Return code was not zero but $retval"
        exit $retval
    fi
done

./colmap_sparse.sh $WORKSPACE

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

$COLMAP_CMD model_converter \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/0 \
    --output_type TXT

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

python3 sparse_known_poses.py $WORKSPACE $POSES

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

# Recompute features from the images of the known camera poses
$COLMAP_CMD feature_extractor \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

# Feature matching
$COLMAP_CMD exhaustive_matcher \
    --database_path $WORKSPACE/database.db

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

$COLMAP_CMD point_triangulator \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --input_path $WORKSPACE/sparse_known_poses_manual \
    --output_path $WORKSPACE/sparse_known_poses_triangulated

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

Compute a dense model
mkdir -p $WORKSPACE/dense

$COLMAP_CMD image_undistorter \
    --image_path $WORKSPACE/images \
    --input_path $WORKSPACE/sparse_known_poses_triangulated \
    --output_path $WORKSPACE/dense \
    --output_type COLMAP \
    --max_image_size 2000

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

$COLMAP_CMD patch_match_stereo \
    --workspace_path $WORKSPACE/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

$COLMAP_CMD stereo_fusion \
    --workspace_path $WORKSPACE/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $WORKSPACE/dense/fused.ply

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

$COLMAP_CMD poisson_mesher \
    --input_path $WORKSPACE/dense/fused.ply \
    --output_path $WORKSPACE/dense/meshed-poisson.ply

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

# /data/gg921/bin/colmap delaunay_mesher \
#     --input_path $WORKSPACE/dense \
#     --output_path $WORKSPACE/dense/meshed-delaunay.ply
# 
# retval=$?
# if [ $retval -ne 0 ]; then
#     echo "Return code was not zero but $retval"
#     exit $retval
# fi