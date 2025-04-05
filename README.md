# pathtracer from scratch

Started out as a small project over the break and has spiraled out of control over time. Runs on cuda-12.5.

Usage:
- Build the program using ```make``` (requires cuda compiler)
- The output will be placed in the ```./bin``` folder
- Test by writing ```./bin/raytracer -c pathtracer.yaml gltf/bust.gltf ```
- The output will be rendered to ```./output.png```

```sh
# usage:
Usage: ./bin/raytracer [options] <path_to_gltf>
Expects a gltf 2.0 model. Further settings can be applied in pathtracer.yaml.
  -c <pathtracer.yaml>  Pathtracer render settings file.
  -o <output.png>       Path to output image.
  -v                    Enable verbose printing.

# pathtracer.yaml layout:
TODO

# example output:
Parsing assets/spheres.gltf... Done 
Building bvh_t... [Done]
Copying scene to device... Done [105kB]
Copying bvh_t to device... Done [86kB]
Launching kernel... 
Rendering 300 samples in batches of 10, img size (1600, 1600)
Kernel params <<<(100,100), (16,16)>>>
Rendered 10 / 300 samples in 0.3s - 35.59 samples/s - 12.81 MPS/s
Rendered 20 / 300 samples in 0.4s - 45.25 samples/s - 16.29 MPS/s
Rendered 30 / 300 samples in 0.6s - 52.17 samples/s - 18.78 MPS/s
Rendered 40 / 300 samples in 0.7s - 56.58 samples/s - 20.37 MPS/s
Rendered 50 / 300 samples in 0.8s - 59.74 samples/s - 21.51 MPS/s
Rendered 60 / 300 samples in 1.0s - 61.98 samples/s - 22.31 MPS/s
Rendered 70 / 300 samples in 1.1s - 63.64 samples/s - 22.91 MPS/s
Rendered 80 / 300 samples in 1.2s - 64.99 samples/s - 23.40 MPS/s
...
```

![Latest render](/output.png)

## Currently implements:
* basic path tracing on gpu using CUDA or optionally using CPU
* handles large scenes thanks to BVH spacial acceleration structure
* BSDF global illumination and transmission
* rendering to .png image
* BVH construction on CPU with surface area heuristic

## TODO:
* more materials / principled bsdf
* light source sampling

## Used sources and useful stuff

* [Computer Graphics at TU Wien](https://www.youtube.com/@cgtuwien)
* [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [Rendering an image, using C](https://stackoverflow.com/questions/27613601/rendering-an-image-using-c)
* [Möller Trumbore Ray Triangle Intersection Explained](https://www.youtube.com/watch?v=fK1RPmF_zjQ)
