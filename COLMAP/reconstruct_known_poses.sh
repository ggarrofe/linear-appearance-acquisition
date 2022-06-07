#! /bin/bash

# Usage: ./reconstruct_known_poses.sh ./lego_known_poses transforms.json
WORKSPACE=$1
POSES=$2

for dir in $WORKSPACE/*; do
    [ "$dir" = $WORKSPACE"/images" ] && continue 
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

/data/gg921/bin/colmap model_converter \
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
/data/gg921/bin/colmap feature_extractor \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

# Feature matching
/data/gg921/bin/colmap exhaustive_matcher \
    --database_path $WORKSPACE/database.db

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

/data/gg921/bin/colmap point_triangulator \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --input_path $WORKSPACE/sparse_known_poses_manual \
    --output_path $WORKSPACE/sparse_known_poses_triangulated

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

# Compute a dense model
mkdir -p $WORKSPACE/dense

/data/gg921/bin/colmap image_undistorter \
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

/data/gg921/bin/colmap poisson_mesher \
    --input_path $WORKSPACE/dense/fused.ply \
    --output_path $WORKSPACE/dense/meshed-poisson.ply

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

/data/gg921/bin/colmap delaunay_mesher \
    --input_path $WORKSPACE/dense \
    --output_path $WORKSPACE/dense/meshed-delaunay.ply

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi