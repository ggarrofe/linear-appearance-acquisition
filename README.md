# A Linear Model for efficient Appearance Acquisition

The different methods presented in the report can be found in the [```Rendering```](Rendering) folder:
* Chapter 3 refers to [```Rendering/surface_rendering.py```](Rendering/surface_rendering.py).
* Chapter 4 includes the Voxelised Linear Mappings [(```Rendering/voxelised_radiance_mapping.py```)](Rendering/voxelised_radiance_mapping.py) and the Clusterised Linear Mappings [(```Rendering/clusterised_radiance_mapping.py```)](Rendering/clusterised_radiance_mapping.py).
* Chapter 5 described different Reflectance Mappings methods: 
  * using positional encoding [(```Rendering/reflectance_mapping.py```)](Rendering/reflectance_mapping.py).
  * using spherical harmonics [(```Rendering/reflectance_mapping_sphharm.py```)](Rendering/reflectance_mapping_sphharm.py).
* Chapter 6 presented 2 methods to leverage the linear mappings using Deep Learning techniques:
  * extending the linear mappings with more layers [(```Rendering/enhanced_reflectance_network.py```)](Rendering/enhanced_reflectance_network.py).
  * using an autodecoder approach [(```Rendering/reflectance_autodecoder.py```)](Rendering/reflectance_autodecoder.py).
  
All these scripts can be run using the configuration files in the [```Rendering/configs```](Rendering/configs) folder. 
You can see an example on how to run each of this scripts in the [```jupyter_notebooks/run_rendering.ipynb```](jupyter_notebooks/run_rendering.ipynb) notebook.
