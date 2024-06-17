# pathtracer

Started out as a small project over the break and has spiraled out of control over time. 

![Latest render](/render.bmp)

## Currently implements:
* basic path tracing on gpu using CUDA or optionally using CPU
* handles large .obj / .mtl scenes
* rendering to .bmp image
* BVH construction on CPU with surface area heuristic

## TODO:
* materials (already in .mtl file but not used)
* importance sampling

## Used sources and useful stuff

* [Computer Graphics at TU Wien](https://www.youtube.com/@cgtuwien)
* [How to build a BVH](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [Rendering an image, using C](https://stackoverflow.com/questions/27613601/rendering-an-image-using-c)
* [Möller Trumbore Ray Triangle Intersection Explained](https://www.youtube.com/watch?v=fK1RPmF_zjQ)
* [Wavefront .obj file](https://en.wikipedia.org/wiki/Wavefront_.obj_file)