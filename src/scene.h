#pragma once

#include "list.h"
#include "mathc.h"

#define OBJ_FILENAME_LENGTH 256
#define MATERIAL_NAME_SIZE 256
#define OBJ_LINE_SIZE 256
#define MAX_NGON_SIZE 4  // can only handle quads or triangles

typedef struct {
    int position;
    int normal;
    int texture;
} obj_face_vertex;

typedef struct {
    size_t vertex_count;
    obj_face_vertex vertices[MAX_NGON_SIZE];
    int material_index;
} obj_face;

typedef struct {
    char name[MATERIAL_NAME_SIZE];
    mfloat_t amb[VEC3_SIZE], diff[VEC3_SIZE], spec[VEC3_SIZE], emit[VEC3_SIZE];
    mfloat_t spec_exp, dissolved, refract_index;
    int model;
} obj_material;

typedef struct {
    int position, target, updir;
} obj_camera;

typedef struct {
    List vertex_list;
    List vertex_normal_list;
    List vertex_texture_list;
    List face_list;
    List material_list;

    obj_camera *camera;
} obj_growable_scene_data;



typedef struct {
    // keep vectors aligned in memory
    struct vec3 *vertex_list;
    struct vec3 *vertex_normal_list;
    struct vec3 *vertex_texture_list;
    obj_face *face_list;
    obj_material *material_list;

    size_t vertex_count;
    size_t vertex_normal_count;
    size_t vertex_texture_count;
    size_t face_count;
    size_t material_count;

    obj_camera camera;

} obj_scene_data; 

int parse_obj_scene(obj_scene_data *data_out, char *filename);
void delete_obj_data(obj_scene_data *data_out);

int scene_copy_to_device(obj_scene_data** dev_scene, obj_scene_data* host_scene);
int free_device_scene(obj_scene_data* dev_scene);