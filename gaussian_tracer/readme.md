# RayTracer (with Optix)

A CUDA Mesh RayTracer with BVH acceleration (with Optix). This repo is based on https://github.com/ashawkey/raytracing .

### Install

Download [Optix](https://developer.nvidia.com/designworks/optix/download) and extract it. Set the environment variable OptiX_INSTALL_DIR to wherever you installed the SDK `OptiX_INSTALL_DIR=<wherever you installed OptiX 7 SDK>`

You may need to install the previous version of Optix according to the version of your driver. See [here](https://developer.nvidia.com/designworks/optix/downloads/legacy) for details.
```bash
# clone the repo
git clone https://github.com/SuLvXiangXin/GassuainRaytracer.git
cd GassuainRaytracer

# use cmake to build the project for ptx file (for Optix)
mkdir build
cd build
cmake ..
make
cd ..

# (Optional) install the package. The CUDA extension could also be built just-in-time.
pip install .
```

### Usage

Example for a mesh normal renderer:

```bash
python renderer.py -p point_cloud.ply
```

### Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp)!
* Credits to [ashawkey](https://me.kiui.moe/)'s amazing [raytracing](https://github.com/ashawkey/raytracing)!
