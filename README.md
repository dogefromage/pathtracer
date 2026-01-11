# pathtracer from scratch

Started out as a small project over the break and has spiraled out of control over time.

## Installation

Requirements: conda or miniconda, CUDA (tested on 12.8)

```bash
cd pathtracer/

# python setup
conda env create -f environment.yml

# compile project
make
```

## Usage

```bash
# cwd must be project root
python client/main.py config=client/configs/base.yml
```

## Example output

```
bin/raytracer
  --path-gltf assets/many_lights.gltf
  --dir-output output/20250808_004357/many_lights
  --output-resolution-x 1024
  --output-resolution-y 1024
  --sampling-samples 200
  --sampling-samples-every-update 50
  --sampling-seed 42
  --logger-log-stdout 1
  --logger-log-level 3
  --world-clear-color "0 0 0"

[INFO] Parsing .gltf... 
[INFO] Done parsing .gltf
[INFO] Building BVH...  
[INFO] Done building BVH
[INFO] Building LST... 
[INFO] Done building LST
[INFO] Copying scene to device... 
[INFO] Done [15MB]
[INFO] Copying bvh_t to device... 
[INFO] Done [27MB]
[INFO] Copying lst_t to device... 
[INFO] Done [48B]
[INFO] Launching kernel... 
[INFO] Rendering 200 samples in batches of 50, img size (1024, 1024)
[INFO] Kernel params <<<(64,64), (16,16)>>>
[INFO] Rendered 50 out of 200 S/px in 8.4s - 5.93 S/px/s - 6.22 MS/s
[INFO] Rendered 100 out of 200 S/px in 16.9s - 5.92 S/px/s - 6.21 MS/s
[INFO] Rendered 150 out of 200 S/px in 25.3s - 5.92 S/px/s - 6.21 MS/s
[INFO] Rendered 200 out of 200 S/px in 33.8s - 5.91 S/px/s - 6.20 MS/s
```

![Latest render](/example_output/20260111_192720/many_lights/render.png)

## Currently implements:
* basic path tracing on gpu using CUDA or optionally using CPU
* handles large scenes thanks to BVH spatial acceleration structure
* BSDF global illumination and transmission
* rendering to .png image
* BVH construction on CPU with surface area heuristic
* basic light source sampling

## TODO:
* more materials / principled bsdf

## Used sources and useful stuff

* [Computer Graphics at TU Wien](https://www.youtube.com/@cgtuwien)
* [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [Rendering an image, using C](https://stackoverflow.com/questions/27613601/rendering-an-image-using-c)
* [Möller Trumbore Ray Triangle Intersection Explained](https://www.youtube.com/watch?v=fK1RPmF_zjQ)
