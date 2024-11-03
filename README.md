# pathtracer

Started out as a small project over the break and has spiraled out of control over time. Runs on cuda-12.5. Build the project using ```make```.

```sh
Place scene .obj and .mtl into assets/

Usage: ./bin/raytracer [options] <path_to_obj>
Expects an .obj file with right handed coordinate system.
  -w <width>         Sets width
  -h <height>        Sets height
  -s <size>          Sets both width and height
  -S <samples>       Sets image samples count (default 100)
  -r <samples/call>  Sets image samples per kernel call (default 10)
  -d <seed>          Sets Seed (default 42)
  -f <focal_length>  Focal length (default 0.40)
  -c <sensor_height> Sensor height (default 0.20)
  -v                 Do verbose printing (default false)

# example output:
Parsing assets/spheres.obj... Done 
Building bvh_t... [Done]
Copying scene to device... Done [105kB]
Copying bvh_t to device... Done [86kB]
Launching kernel... 
Rendering 300 samples in batches of 10, img size (1600, 1600)
Kernel params <<<(100,100), (16,16)>>>
Rendered 10 / 300 samples in 1.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 20 / 300 samples in 2.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 30 / 300 samples in 3.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 40 / 300 samples in 4.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 50 / 300 samples in 5.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 60 / 300 samples in 6.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 70 / 300 samples in 7.0s - 10.00 samples/s - 25.60 MPS/s
Rendered 80 / 300 samples in 8.0s - 10.00 samples/s - 25.60 MPS/s
...
```

![Latest render](/render.bmp)

## Currently implements:
* basic path tracing on gpu using CUDA or optionally using CPU
* handles large .obj / .mtl scenes
* rendering to .bmp image
* BVH construction on CPU with surface area heuristic

## TODO:
* better materials (already in .mtl file but only partially used)
* importance sampling
* improve performance of BVH construction

## Used sources and useful stuff

* [Computer Graphics at TU Wien](https://www.youtube.com/@cgtuwien)
* [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [Rendering an image, using C](https://stackoverflow.com/questions/27613601/rendering-an-image-using-c)
* [Möller Trumbore Ray Triangle Intersection Explained](https://www.youtube.com/watch?v=fK1RPmF_zjQ)
* [Wavefront .obj file](https://en.wikipedia.org/wiki/Wavefront_.obj_file)