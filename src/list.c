#include "list.h"

#include <assert.h>

#define LIST_INITIAL_SIZE 8

void List_init(List* list) {
    list->item_count = 0;
    list->reserved_count = 0;
    list->data = NULL;
    List_reserve(list, LIST_INITIAL_SIZE);
}

void List_reserve(List* list, size_t reserve_slots) {
    if (list->reserved_count < reserve_slots) {
        list->data = realloc(list->data, sizeof(void*) * reserve_slots);
        assert(list->data);
        list->reserved_count = reserve_slots;
    }
}

int List_insert(List* list, void* item) {
    assert(list->data);
    if ((list->item_count) >= list->reserved_count) {
        List_reserve(list, 2 * list->reserved_count);
    }
    list->data[list->item_count] = item;
    return list->item_count++;
}

void List_free_self(List* list) {
    free(list->data);
    list->data = NULL;
    list->reserved_count = list->item_count = 0;
}
void List_free_full(List* list) {
    for (size_t i = 0; i < list->item_count; i++) {
        free(list->data[i]);
        list->data[i] = NULL;
    }
    List_free_self(list);
}

void* List_at(List* list, size_t index) {
    assert(list->data);
    assert(index < list->item_count);
    return list->data[index];
}

void* List_begin(List* list) {
    assert(list->data);
    return list->data;
}

void* List_end(List* list) {
    assert(list->data);
    return list->data[list->item_count];
}
