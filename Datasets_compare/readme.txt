Notes about Replica Datasets:

(1) The depth image are stored in millimetre unit so we need to scale the depth value into meter by a coefficient of 1/1000.

(2) The provided poses are camera to world poses and the camera coordinates follows OpenCV convention (x-right, y-down,z-towards screen) instead of OpenGL convention used in NeRF's blender dataset.

(3) The camera trajectories are generated purely randomly by sampling camera positions and their look-at points.

(4) We provide two sets of random trajectories of each scene with different random seeds during generation.
