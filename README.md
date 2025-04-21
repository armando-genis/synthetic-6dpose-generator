# synthetic-6dpose-generator
A BlenderProc-powered tool for generating synthetic 6DoF pose estimation datasets. This pipeline renders scenes from a 3D model and outputs four dataset formats simultaneously: BOP, DOPE, EfficientPose, and YOLO56DOP. Easily configurable and optimized for object pose training pipelines.

docker compose build

docker compose up -d

docker exec -it 6dpose_generator_container /bin/zsh


pip install blenderproc

to install evething related to blenderproc 
blenderproc quickstart

/root/blender/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 -m pip install pyquaternion
/root/blender/blender-4.2.1-linux-x64/4.2/python/bin/python3.11 -m pip install tqdm

blenderproc run syntetic_data_dope.py 


pip3 install albumentations
pip3 install torch
pip3 install torchvision

python3 debug.py --data /workspace/generator/datasets/dope_data
