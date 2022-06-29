ssh gg921@gpu17.doc.ic.ac.uk  

nvtop or nvidia-smi


cd Documents/Individual_Project/NeRF
/homes/gg921/.local/bin/jupyter notebook --no-browser --port=8081

ssh -L 8080:localhost:8081 gg921@gpu14.doc.ic.ac.uk



python3 scripts/python/build.py \
    --build_path "./build" \
    --colmap_path "./" \
    --boost_path "/usr/include/boost" \
    --qt_path "/usr/lib/qt5/bin" \
    --cuda_path "/usr/local/cuda/bin" \
    --cgal_path "C:/dev/CGAL-4.11.2/build"



python3 scripts/python/build.py \
    --build_path "./build" \
    --colmap_path "./" \
    --boost_path "/usr/lib/x86_64-linux-gnu/" \
    --qt_path "/usr/lib/qt5/bin" \
    --cuda_path "/usr/local/cuda/bin" \