#! /bin/bash

# Usage: ./colmap_dense.sh ./lego_nvidia

WORKSPACE=$1

rm -rf $WORKSPACE/dense
mkdir $WORKSPACE/dense

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
