#! /bin/bash

# Usage: ./build_mesh.sh ./path_to_scenedir

SCENE_DIR=$1

python3 imgs2poses_llff.py --match_type exhaustive_matcher --scenedir $SCENE_DIR

retval=$?
if [ $retval -ne 0 ]; then
    echo "Return code was not zero but $retval"
    exit $retval
fi

python3 poisson_mesher.py "$SCENE_DIR/dense/fused.ply" "$SCENE_DIR/dense/meshed-poisson.ply"
