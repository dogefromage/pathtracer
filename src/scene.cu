#include <assert.h>
#include <cuda.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "list.h"
#include "scene.h"
#include "utils.h"

#define WHITESPACE " \t\n\r"

// #define PRINT_DEBUG(s)
#define PRINT_DEBUG(s) printf(s)

static char *filename = NULL;
static size_t linenumber = 0;

#define PARSE_ERR(s) (fprintf(stderr, "[%s:%lu] %s", filename, linenumber, s))

void obj_set_material_defaults(obj_material *mtl) {
    vec3_const(mtl->amb, 0.0);
    vec3_const(mtl->diff, 0.8);
    vec3_const(mtl->spec, 1.0);
    vec3_const(mtl->emit, 0.0);

    mtl->spec_exp = 0;
    mtl->dissolved = 1;
    mtl->refract_index = 1;

    mtl->model = 2;
}

void obj_fix_numbering(int *index, int curr_list_size) {
    if (*index == 0) {
        // 0 becomes -1 which symbols missing value
        *index = -1;
        return;
    }

    if (*index < 0) {
        // relative ordering
        *index += curr_list_size;
    } else {
        // normal ordering
        *index -= 1;
    }

    if (*index < 0 || *index >= curr_list_size) {
        PARSE_ERR("Out of bounds index.\n");
    }
}

void obj_parse_simple_index(char *token, int *index, int curr_list_size) {
    if (sscanf(token, "%d", index) < 1) {
        PARSE_ERR("Expected list index.\n");
    }
    obj_fix_numbering(index, curr_list_size);
}

void obj_parse_face_vertex(obj_growable_scene_data *data, char *token, obj_face_vertex *vert) {
    // face vertex:  [pos]/[tex]/[normal]
    vert->position = vert->texture = vert->normal = 0;
    sscanf(token, "%d/%d/%d", &vert->position, &vert->texture, &vert->normal);  // will only take values if provided
    obj_fix_numbering(&vert->position, data->vertex_list.item_count);           // fix numbering
    obj_fix_numbering(&vert->texture, data->vertex_texture_list.item_count);
    obj_fix_numbering(&vert->normal, data->vertex_normal_list.item_count);
}
int obj_parse_face_vertex_list(obj_growable_scene_data *data, obj_face_vertex *vertices, int max_verts) {
    char *token;
    int vertex_count = 0;
    while ((token = strtok(NULL, WHITESPACE)) != NULL) {
        if (vertex_count >= max_verts) {
            PARSE_ERR("Encountered too many verts in list.\n");
        }
        obj_parse_face_vertex(data, token, &vertices[vertex_count]);
        vertex_count++;
    }
    return vertex_count;
}
obj_face *obj_parse_face(obj_growable_scene_data *scene) {
    obj_face *face = (obj_face *)malloc(sizeof(obj_face));
    face->vertex_count = obj_parse_face_vertex_list(scene, face->vertices, MAX_NGON_SIZE);
    return face;
}

struct vec3 *obj_parse_vector() {
    struct vec3 *vec = (struct vec3 *)calloc(1, sizeof(struct vec3));

    char *tok_ref = NULL;
    char *line = strtok(NULL, "\n");
    char *token = strtok_r(line, WHITESPACE, &tok_ref);

    for (int i = 0; i < 3; i++) {
        if (token == NULL) {
            break;
        }
        vec->v[i] = atof(token);
        token = strtok_r(NULL, WHITESPACE, &tok_ref);
        // printf("e[%u] = %f\n", i, v->e[i]);
    }
    return vec;
}

void obj_parse_camera(obj_growable_scene_data *scene, obj_camera *camera) {
    char *pos = strtok(NULL, WHITESPACE);
    char *look = strtok(NULL, WHITESPACE);
    char *up = strtok(NULL, WHITESPACE);
    if (pos == NULL || look == NULL || up == NULL) {
        PARSE_ERR("Expected 3 indices for camera. 'c [position v] [target v] [updir vn]'.\n");
        return;
    }
    obj_parse_simple_index(pos, &camera->position, scene->vertex_list.item_count);
    obj_parse_simple_index(look, &camera->target, scene->vertex_list.item_count);
    obj_parse_simple_index(up, &camera->updir, scene->vertex_normal_list.item_count);
}

int obj_parse_mtl_file(obj_growable_scene_data *scene) {
    char *obj_filename = filename;
    size_t obj_linenumber = linenumber;

    // find mtl path relative to cwd
    char *filename_cpy = strdup(obj_filename);
    assert(filename_cpy);
    // get mtl filename relative to obj
    char *original_mtl_filename = strtok(NULL, WHITESPACE);
    if (original_mtl_filename == NULL) {
        PARSE_ERR("Expected .mtl filename.");
        return 1;
    }
    // construct combined filename
    char mtl_filename[OBJ_FILENAME_LENGTH];
    snprintf(mtl_filename, sizeof(mtl_filename), "%s/%s", dirname(filename_cpy), original_mtl_filename);

    free(filename_cpy);
    filename_cpy = NULL;

    // printf("MTL: %s\n", mtl_filename);

    filename = mtl_filename;
    linenumber = 0;

    // open scene
    FILE *mtl_file_stream = fopen(filename, "r");
    if (!mtl_file_stream) {
        PARSE_ERR("Error opening .mtl file.\n");
        return 1;
    }

    char *current_token;
    char current_line[OBJ_LINE_SIZE];
    obj_material *current_mtl = NULL;

    while (fgets(current_line, OBJ_LINE_SIZE, mtl_file_stream)) {
        current_token = strtok(current_line, WHITESPACE);
        linenumber++;

        // skip comments
        if (current_token == NULL || !strcmp(current_token, "//") || !strcmp(current_token, "#")) {
            continue;
        }
        // start material
        else if (!strcmp(current_token, "newmtl")) {
            current_mtl = (obj_material *)malloc(sizeof(obj_material));
            obj_set_material_defaults(current_mtl);
            // get the name
            strncpy(current_mtl->name, strtok(NULL, WHITESPACE), MATERIAL_NAME_SIZE - 1);
            List_insert(&scene->material_list, current_mtl);
            // printf("Created mtl %s, %d\n", current_mtl->name);
        }
        // ambient
        else if (!strcmp(current_token, "Ka") && current_mtl != NULL) {
            current_mtl->amb[0] = atof(strtok(NULL, " \t"));
            current_mtl->amb[1] = atof(strtok(NULL, " \t"));
            current_mtl->amb[2] = atof(strtok(NULL, " \t"));
        }
        // diff
        else if (!strcmp(current_token, "Kd") && current_mtl != NULL) {
            current_mtl->diff[0] = atof(strtok(NULL, " \t"));
            current_mtl->diff[1] = atof(strtok(NULL, " \t"));
            current_mtl->diff[2] = atof(strtok(NULL, " \t"));
            // printf("diff[%f,%f,%f]\n", current_mtl->diff[0], current_mtl->diff[1], current_mtl->diff[2]);
        }
        // specular
        else if (!strcmp(current_token, "Ke") && current_mtl != NULL) {
            current_mtl->emit[0] = atof(strtok(NULL, " \t"));
            current_mtl->emit[1] = atof(strtok(NULL, " \t"));
            current_mtl->emit[2] = atof(strtok(NULL, " \t"));
        }
        // specular
        else if (!strcmp(current_token, "Ks") && current_mtl != NULL) {
            current_mtl->spec[0] = atof(strtok(NULL, " \t"));
            current_mtl->spec[1] = atof(strtok(NULL, " \t"));
            current_mtl->spec[2] = atof(strtok(NULL, " \t"));
        }
        // shiny
        else if (!strcmp(current_token, "Ns") && current_mtl != NULL) {
            current_mtl->spec_exp = atof(strtok(NULL, " \t"));
        }
        // transparent
        else if (!strcmp(current_token, "d") && current_mtl != NULL) {
            current_mtl->dissolved = atof(strtok(NULL, " \t"));
        }
        // // reflection
        // else if (!strcmp(current_token, "r") && current_mtl != NULL) {
        //     current_mtl->reflect = atof(strtok(NULL, " \t"));
        // }
        // // glossy
        // else if (!strcmp(current_token, "sharpness") && current_mtl != NULL) {
        //     current_mtl->glossy = atof(strtok(NULL, " \t"));
        // }
        // refract index
        else if (!strcmp(current_token, "Ni") && current_mtl != NULL) {
            current_mtl->refract_index = atof(strtok(NULL, " \t"));
        }
        // illumination type
        else if (!strcmp(current_token, "illum") && current_mtl != NULL) {
            current_mtl->model = atoi(strtok(NULL, " \t"));
        }
        // // texture map
        // else if (!strcmp(current_token, "map_Ka") && current_mtl != NULL) {
        //     strncpy(current_mtl->texture_filename, strtok(NULL, " \t"), OBJ_FILENAME_LENGTH - 1);
        // }
        else {
            PARSE_ERR("Unknown command in .mtl file\n");
        }
    }

    fclose(mtl_file_stream);

    filename = obj_filename;
    linenumber = obj_linenumber;

    return 0;
}

int usemtl_find_material(List *material_list) {
    char *name;
    if (!(name = strtok(NULL, WHITESPACE))) {
        PARSE_ERR("Expected material name.\n");
        return -1;
    }
    for (size_t i = 0; i < material_list->item_count; i++) {
        obj_material *mat = (obj_material *)List_at(material_list, i);
        // printf("%s / %s\n", mat->name, name);
        if (!strcmp(mat->name, name)) {
            return i;
        }
    }
    PARSE_ERR("Unknown material.\n");
    return -1;
}

int obj_parse_obj_file(obj_growable_scene_data *growable_data, char *param_filename) {
    filename = param_filename;
    linenumber = 0;

    FILE *obj_file_stream = fopen(filename, "r");
    // open scene
    if (!obj_file_stream) {
        PARSE_ERR("Error opening .obj file.\n");
        return 1;
    }

    // ADD DEFAULT MATERIAL
    obj_material *missing_material = (obj_material *)malloc(sizeof(obj_material));
    obj_set_material_defaults(missing_material);
    missing_material->diff[0] = 1.0;  // bright pink
    missing_material->diff[1] = 0.0;
    missing_material->diff[2] = 1.0;
    // get the name
    strcpy(missing_material->name, "missing");
    int missing_material_index = List_insert(&growable_data->material_list, missing_material);

    int current_material_index = missing_material_index;
    char *current_token = NULL;
    char current_line[OBJ_LINE_SIZE];

    // parser loop
    while (fgets(current_line, OBJ_LINE_SIZE, obj_file_stream)) {
        current_token = strtok(current_line, WHITESPACE);
        linenumber++;

        // skip comments
        if (current_token == NULL || current_token[0] == '#') {
            continue;
        }

        // parse objects
        // process vertex
        else if (!strcmp(current_token, "v")) {
            List_insert(&growable_data->vertex_list, obj_parse_vector());
        }
        // process vertex normal
        else if (!strcmp(current_token, "vn")) {
            List_insert(&growable_data->vertex_normal_list, obj_parse_vector());
        }
        // process vertex texture
        else if (!strcmp(current_token, "vt")) {
            List_insert(&growable_data->vertex_texture_list, obj_parse_vector());
        }
        // process face
        else if (!strcmp(current_token, "f")) {
            obj_face *face = obj_parse_face(growable_data);
            face->material_index = current_material_index;
            List_insert(&growable_data->face_list, face);
        }
        // camera
        else if (!strcmp(current_token, "c")) {
            growable_data->camera = (obj_camera *)malloc(sizeof(obj_camera));
            obj_parse_camera(growable_data, growable_data->camera);
        }
        // usemtl
        else if (!strcmp(current_token, "usemtl")) {
            current_material_index = usemtl_find_material(&growable_data->material_list);
            if (current_material_index < 0) {
                current_material_index = missing_material_index;
            }
        }
        // mtllib
        else if (!strcmp(current_token, "mtllib")) {
            obj_parse_mtl_file(growable_data);
        }

        // object name
        else if (!strcmp(current_token, "o")) {
        }
        // smoothing
        else if (!strcmp(current_token, "s")) {
        }
        // group
        else if (!strcmp(current_token, "g")) {
        } else {
            PARSE_ERR("Unknown command.\n");
        }
    }
    fclose(obj_file_stream);
    return 0;
}

void obj_init_temp_storage(obj_growable_scene_data *growable_data) {
    List_init(&growable_data->vertex_list);
    List_init(&growable_data->vertex_normal_list);
    List_init(&growable_data->vertex_texture_list);

    List_init(&growable_data->face_list);

    // List_init(&growable_data->light_point_list);
    // List_init(&growable_data->light_quad_list);

    List_init(&growable_data->material_list);

    growable_data->camera = NULL;
}

void obj_copy_and_align_indirect_data(void **target, void **data, size_t count, size_t element_size) {
    // align sequental needed memory
    *target = malloc(count * element_size);
    char *target_block = (char *)(*target);
    for (size_t i = 0; i < count; i++) {
        memcpy(target_block, data[i], element_size);
        target_block += element_size;
    }
}

void obj_transfer_and_free(obj_scene_data *data_out, obj_growable_scene_data *growable_data) {
    // HARD COPY DATA INTO ALIGNED ARRAY OF VECTORS
    data_out->vertex_count = growable_data->vertex_list.item_count;
    data_out->vertex_normal_count = growable_data->vertex_normal_list.item_count;
    data_out->vertex_texture_count = growable_data->vertex_texture_list.item_count;
    data_out->face_count = growable_data->face_list.item_count;
    // data_out->light_point_count = growable_data->light_point_list.item_count;
    // data_out->light_quad_count = growable_data->light_quad_list.item_count;
    data_out->material_count = growable_data->material_list.item_count;

    obj_copy_and_align_indirect_data((void **)&data_out->vertex_list,
                                     growable_data->vertex_list.data,
                                     growable_data->vertex_list.item_count,
                                     sizeof(struct vec3));
    obj_copy_and_align_indirect_data((void **)&data_out->vertex_normal_list,
                                     growable_data->vertex_normal_list.data,
                                     growable_data->vertex_normal_list.item_count,
                                     sizeof(struct vec3));
    obj_copy_and_align_indirect_data((void **)&data_out->vertex_texture_list,
                                     growable_data->vertex_texture_list.data,
                                     growable_data->vertex_texture_list.item_count,
                                     sizeof(struct vec3));
    obj_copy_and_align_indirect_data((void **)&data_out->face_list,
                                     growable_data->face_list.data,
                                     growable_data->face_list.item_count,
                                     sizeof(obj_face));
    // obj_copy_and_align_indirect_data((void **)&data_out->light_point_list,
    //                                  growable_data->light_point_list.data,
    //                                  growable_data->light_point_list.item_count,
    //                                  sizeof(obj_light_point));
    // obj_copy_and_align_indirect_data((void **)&data_out->light_quad_list,
    //                                  growable_data->light_quad_list.data,
    //                                  growable_data->light_quad_list.item_count,
    //                                  sizeof(obj_light_quad));
    obj_copy_and_align_indirect_data((void **)&data_out->material_list,
                                     growable_data->material_list.data,
                                     growable_data->material_list.item_count,
                                     sizeof(obj_material));

    assert(growable_data->camera);
    data_out->camera = *growable_data->camera;

    // FREE

    List_free_full(&growable_data->vertex_list);
    List_free_full(&growable_data->vertex_normal_list);
    List_free_full(&growable_data->vertex_texture_list);
    List_free_full(&growable_data->face_list);
    // List_free_full(&growable_data->light_point_list);
    // List_free_full(&growable_data->light_quad_list);
    List_free_full(&growable_data->material_list);

    free(growable_data->camera);
}

int obj_finalize_data(obj_growable_scene_data *data) {
    // normalize normals, wikipedia says they're not necessarily unit length
    for (size_t i = 0; i < data->vertex_normal_list.item_count; i++) {
        struct vec3 *p = (struct vec3 *)data->vertex_normal_list.data[i];
        vec3_normalize(p->v, p->v);
    }

    int zero_vertex_tex_index = List_insert(
        &data->vertex_texture_list,
        calloc(1, sizeof(struct vec3)));

    // add missing normals
    for (size_t i = 0; i < data->face_list.item_count; i++) {
        obj_face *face = (obj_face *)data->face_list.data[i];

        // calculate face normal based on whole face
        struct vec3 normal = {0, 0, 0};
        for (size_t j = 0; j < face->vertex_count; j++) {
            obj_face_vertex *vert = &face->vertices[j];
            if (vert->position < 0) {
                fprintf(stderr, "Missing position vertex %lu on face %lu.\n", j, i);
                return 1;
            }

            if (j >= 2) {
                // calculate normal based on fan triangulation
                struct vec3 *V0 = (struct vec3 *)data->vertex_list.data[face->vertices[0].position];
                struct vec3 *Vl = (struct vec3 *)data->vertex_list.data[face->vertices[j - 1].position];
                struct vec3 *Vn = (struct vec3 *)data->vertex_list.data[face->vertices[j].position];
                struct vec3 edge1, edge2, temp_n;
                vec3_subtract(edge1.v, Vn->v, Vl->v);
                vec3_subtract(edge2.v, V0->v, Vl->v);
                vec3_cross(temp_n.v, edge1.v, edge2.v);
                vec3_add(normal.v, normal.v, temp_n.v);
            }
        }

        // normalize
        vec3_normalize(normal.v, normal.v);

        // set normals if missing
        int next_normal = -1;
        for (size_t j = 0; j < face->vertex_count; j++) {
            obj_face_vertex *vert = &face->vertices[j];
            if (vert->normal < 0) {
                if (next_normal < 0) {
                    // create new normal
                    struct vec3 *heap_normal = (struct vec3 *)malloc(sizeof(struct vec3));
                    vec3_assign(heap_normal->v, normal.v);
                    next_normal = List_insert(&data->vertex_normal_list, heap_normal);
                }
                vert->normal = next_normal;
            }
            if (vert->texture < 0) {
                vert->texture = zero_vertex_tex_index;  // default value
            }
        }
    }

    if (data->camera == NULL) {
        PARSE_ERR(".obj does not declare a camera.\n");
        return 1;
    }

    return 0;
}

void obj_growable_fprint(FILE *stream, obj_growable_scene_data *data) {
    for (size_t i = 0; i < data->vertex_list.item_count; i++) {
        struct vec3 *v = (struct vec3 *)(data->vertex_list.data[i]);
        fprintf(stream, "v %f %f %f\n", v->x, v->y, v->z);
    }
    for (size_t i = 0; i < data->vertex_normal_list.item_count; i++) {
        struct vec3 *v = (struct vec3 *)(data->vertex_normal_list.data[i]);
        fprintf(stream, "vn %f %f %f\n", v->x, v->y, v->z);
    }
    for (size_t i = 0; i < data->vertex_texture_list.item_count; i++) {
        struct vec3 *v = (struct vec3 *)(data->vertex_texture_list.data[i]);
        fprintf(stream, "vt %f %f %f\n", v->x, v->y, v->z);
    }
    // for (size_t i = 0; i < data->light_point_list.item_count; i++) {
    //     obj_light_point *lp = (obj_light_point *)data->light_point_list.data[i];
    //     fprintf(stream, "lp %d\n", lp->pos_index + 1);
    // }
    // for (size_t i = 0; i < data->light_quad_list.item_count; i++) {
    //     obj_light_quad *lq = (obj_light_quad *)data->light_quad_list.data[i];
    //     fprintf(stream, "lq");
    //     for (size_t j = 0; j < lq->vertex_count; j++) {
    //         fprintf(stream, " %d", lq->vertices[j] + 1);
    //     }
    //     fprintf(stream, "\n");
    // }
    for (size_t i = 0; i < data->face_list.item_count; i++) {
        obj_face *f = (obj_face *)data->face_list.data[i];
        fprintf(stream, "f");
        for (size_t j = 0; j < f->vertex_count; j++) {
            fprintf(stream, " %d/%d/%d",
                    f->vertices[j].position + 1,
                    f->vertices[j].texture + 1,
                    f->vertices[j].normal + 1);
        }
        fprintf(stream, "\n");
    }
    if (data->camera != NULL) {
        fprintf(stream, "c %d %d %d\n", data->camera->position + 1,
                data->camera->target + 1,
                data->camera->updir + 1);
    }
}

void obj_scene_data_fprint(FILE *stream, obj_scene_data *data) {
    for (size_t i = 0; i < data->vertex_count; i++) {
        struct vec3 *v = &data->vertex_list[i];
        fprintf(stream, "v %f %f %f\n", v->x, v->y, v->z);
    }
    for (size_t i = 0; i < data->vertex_normal_count; i++) {
        struct vec3 *v = &data->vertex_normal_list[i];
        fprintf(stream, "vn %f %f %f\n", v->x, v->y, v->z);
    }
    for (size_t i = 0; i < data->vertex_texture_count; i++) {
        struct vec3 *v = &data->vertex_texture_list[i];
        fprintf(stream, "vt %f %f %f\n", v->x, v->y, v->z);
    }
    for (size_t i = 0; i < data->face_count; i++) {
        obj_face *f = &data->face_list[i];
        fprintf(stream, "f");
        for (size_t j = 0; j < f->vertex_count; j++) {
            fprintf(stream, " %d/%d/%d",
                    f->vertices[j].position + 1,
                    f->vertices[j].texture + 1,
                    f->vertices[j].normal + 1);
        }
        fprintf(stream, "\n");
    }

    fprintf(stream, "c %d %d %d\n", data->camera.position + 1,
            data->camera.target + 1,
            data->camera.updir + 1);
}

int parse_obj_scene(obj_scene_data *data_out, char *filename) {
    printf("Parsing %s... ", filename);

    obj_growable_scene_data growable_data;
    obj_init_temp_storage(&growable_data);
    // PARSE
    if (obj_parse_obj_file(&growable_data, filename)) {
        return 1;
    }

    if (obj_finalize_data(&growable_data)) {
        return 1;
    }

    // // TEST GROWABLE
    // obj_growable_fprint(stdout, &growable_data);

    // TRANSFER
    obj_transfer_and_free(data_out, &growable_data);

    // // TEST SCENE
    // obj_scene_data_fprint(stdout, data_out);

    printf("Done \n");

    return 0;
}

void delete_obj_data(obj_scene_data *data) {
    free(data->vertex_list);
    free(data->vertex_normal_list);
    free(data->vertex_texture_list);
    free(data->face_list);

    data->vertex_list = NULL;
    data->vertex_normal_list = NULL;
    data->vertex_texture_list = NULL;
    data->face_list = NULL;
}

int scene_copy_to_device(obj_scene_data **dev_scene, obj_scene_data *host_scene) {
    printf("Copying scene to device... ");

    obj_scene_data temp_mirror = *host_scene;

    cudaError_t err;
    size_t curr_bytes, total_bytes;
    total_bytes = 0;

    curr_bytes = temp_mirror.vertex_count * sizeof(struct vec3);
    total_bytes += curr_bytes;
    err = cudaMalloc(&temp_mirror.vertex_list, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(temp_mirror.vertex_list, host_scene->vertex_list,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    curr_bytes = temp_mirror.vertex_normal_count * sizeof(struct vec3);
    total_bytes += curr_bytes;
    err = cudaMalloc(&temp_mirror.vertex_normal_list, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(temp_mirror.vertex_normal_list, host_scene->vertex_normal_list,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    curr_bytes = temp_mirror.vertex_texture_count * sizeof(struct vec3);
    total_bytes += curr_bytes;
    err = cudaMalloc(&temp_mirror.vertex_texture_list, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(temp_mirror.vertex_texture_list, host_scene->vertex_texture_list,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    curr_bytes = temp_mirror.face_count * sizeof(obj_face);
    total_bytes += curr_bytes;
    err = cudaMalloc(&temp_mirror.face_list, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(temp_mirror.face_list, host_scene->face_list,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    curr_bytes = temp_mirror.material_count * sizeof(obj_material);
    total_bytes += curr_bytes;
    err = cudaMalloc(&temp_mirror.material_list, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(temp_mirror.material_list, host_scene->material_list,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    // temp_mirror holds the correct device pointers to nested structures
    total_bytes += sizeof(obj_scene_data);
    err = cudaMalloc(dev_scene, sizeof(obj_scene_data));
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(*dev_scene, &temp_mirror,
                     sizeof(obj_scene_data), cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    printf("Done [%ldkB]\n", total_bytes / 1000);

    return 0;
}

int free_device_scene(obj_scene_data *dev_scene) {
    obj_scene_data temp_mirror;
    cudaMemcpy(&temp_mirror, dev_scene,
               sizeof(obj_scene_data), cudaMemcpyDeviceToHost);

    cudaFree(dev_scene);

    cudaFree(temp_mirror.vertex_list);
    cudaFree(temp_mirror.vertex_normal_list);
    cudaFree(temp_mirror.vertex_texture_list);
    cudaFree(temp_mirror.face_list);
    cudaFree(temp_mirror.material_list);

    return 0;
}
