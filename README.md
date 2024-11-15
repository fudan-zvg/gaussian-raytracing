# 3D Gaussian Ray Tracing
An implementation of 3D Gaussian Ray Tracing, inspired by the work [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/). This repository is based on our [OptiX-based differentiable 3D Gaussian Ray Tracer](https://github.com/fudan-zvg/gtracer)


## Installation

```bash
git clone https://github.com/fudan-zvg/gaussian-raytracing.git --recursive

# This step is same as 3DGS
conda env create --file environment.yml
conda activate gaussian_raytracing

# Install 3DGS's rasterizer
pip install submodules/diff-gaussian-rasterization

# Install 3DGS's simple-knn
pip install submodules/simple-knn

# Install 3D Gaussian Ray Tracer
cd submodules/gtracer && rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../ && cd ../../
pip install submodules/gtracer
```

## Dataset
Please follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for dataset preparation.


## Training

```bash
# E.g. train a lego
python train.py -s data/nerf_synthetic/lego -m outputs/lego --eval
# E.g. train a lego with GUI
python train.py -s data/nerf_synthetic/lego -m outputs/lego --eval --gui
```

## Evaluating

```bash
# Render images
python render.py -m outputs/lego
# Metrics
python metrics.py -m outputs/lego
```

## Interactive Viewer
Use a GUI to view the results.
```bash
python gui.py -m outputs/lego
```

## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{gu2024gaussian,
    title = {3D Gaussian Ray Tracing},
    author = {Gu, Chun and Zhang, Li},
    howpublished = {\url{https://github.com/fudan-zvg/gaussian-raytracing}},
    year = {2024}
}
```
