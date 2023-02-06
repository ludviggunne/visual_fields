
#ifndef SH_VECTOR_H
#define SH_VECTOR_H

/*
    Statically allocated, generic unordered vector.
*/

#ifndef SH_VECTOR_SIZE_T
#   include <stdlib.h>
#   define SH_VECTOR_SIZE_T size_t
#endif

#define SH_VECTOR_DECL(type, capacity, postfix)\
typedef struct { type data[capacity]; SH_VECTOR_SIZE_T size; } sh_vector_##postfix;\
\
int sh_vector_##postfix##_push(sh_vector_##postfix *vec, type data) {\
    if (vec->size == capacity) return 0;\
    vec->data[vec->size++] = data; return 1;}\
\
int sh_vector_##postfix##_remove(sh_vector_##postfix *vec, SH_VECTOR_SIZE_T index) {\
    if (index < 0 || index >= vec->size) return 0;\
    if (index < vec->size - 1) vec->data[index] = vec->data[vec->size - 1]; vec->size--; return 1;}\
\
void sh_vector_##postfix##_clear(sh_vector_##postfix *vec) { vec->size = 0; }

#define SH_VECTOR_INIT(postfix) ((sh_vector_##postfix) {0})

#endif /* SH_VECTOR_H */