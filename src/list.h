#pragma once

#include <stdlib.h>

typedef struct {
    size_t item_count;
    size_t reserved_count;
    void** data;
} List;

void List_init(List* list);
void List_reserve(List* list, size_t reserve_slots);
int List_insert(List* list, void* item);
void* List_at(List* list, size_t index);
void* List_begin(List* list);
void* List_end(List* list);

void List_free_self(List* list);
void List_free_full(List* list);