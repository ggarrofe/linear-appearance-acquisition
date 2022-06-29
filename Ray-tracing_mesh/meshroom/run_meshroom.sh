cd /data/meshroom
export ALICEVISION_SENSOR_DB=/homes/gg921/Documents/Individual_Project/MeshRoom/Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/aliceVision/share/aliceVision/cameraSensors.db
export ALICEVISION_VOCTREE=/homes/gg921/Documents/Individual_Project/MeshRoom/Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/aliceVision/share/aliceVision/vlfeat_K80L3.SIFT.tree
LD_LIBRARY_PATH=/usr/lib/libnvidia-gtk3.so.470.57.01 PYTHONPATH=$PWD python3 meshroom/ui
