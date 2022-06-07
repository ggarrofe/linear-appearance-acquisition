#! /bin/bash

# Usage: ./reconstruct.sh ./lego_known_poses
WORKSPACE=$1

/data/gg921/bin/colmap automatic_reconstructor \
    --workspace_path $WORKSPACE \
    --image_path $WORKSPACE/images

/data/gg921/bin/colmap model_converter \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/ \
    --output_type TXT
