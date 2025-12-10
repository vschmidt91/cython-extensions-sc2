# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY
from libc.math cimport sqrt, round, M_SQRT2
from libc.stdlib cimport malloc, free, realloc
from libc.stdint cimport uint8_t

# -----------------------------------------------------------------------------
# Types & Constants
# -----------------------------------------------------------------------------

ctypedef cnp.float32_t DTYPE_t
ctypedef Py_ssize_t INDEX_t
ctypedef uint8_t DIR_t

cdef INDEX_t INITIAL_CAPACITY = 1024
cdef DIR_t NO_DIRECTION = 255
cdef INDEX_t NO_INDEX = -1
cdef INDEX_t MEMORY_ERROR = -1

# -----------------------------------------------------------------------------
# Heap Operations
# -----------------------------------------------------------------------------

cdef inline void bubble_up(
    INDEX_t* index,
    DTYPE_t* priorities,
    INDEX_t* indirection,
    INDEX_t i
) noexcept nogil:
    cdef INDEX_t parent
    cdef INDEX_t move_index = index[i]
    cdef DTYPE_t move_priority = priorities[i]
    while i > 0:
        parent = (i - 1) >> 2
        if move_priority < priorities[parent]:
            index[i] = index[parent]
            priorities[i] = priorities[parent]
            indirection[index[i]] = i
            i = parent
        else:
            break
    index[i] = move_index
    priorities[i] = move_priority
    indirection[move_index] = i

cdef inline void bubble_down(
    INDEX_t* index,
    DTYPE_t* priorities,
    INDEX_t* indirection,
    INDEX_t i,
    INDEX_t size,
) noexcept nogil:
    cdef INDEX_t child, child0, min_child
    cdef INDEX_t move_index = index[i]
    cdef DTYPE_t move_priority = priorities[i]
    while True:
        child0 = (i << 2) + 1
        if child0 >= size:
            break
        min_child = child0
        child = child0 + 1
        if child < size and priorities[child] < priorities[min_child]:
            min_child = child
        child = child0 + 2
        if child < size and priorities[child] < priorities[min_child]:
            min_child = child
        child = child0 + 3
        if child < size and priorities[child] < priorities[min_child]:
            min_child = child
        if priorities[min_child] < move_priority:
            index[i] = index[min_child]
            priorities[i] = priorities[min_child]
            indirection[index[i]] = i
            i = min_child
        else:
            break
    index[i] = move_index
    priorities[i] = move_priority
    indirection[move_index] = i

cdef inline bint ensure_capacity(
    INDEX_t** index,
    DTYPE_t** priorities,
    INDEX_t* capacity
) noexcept nogil:
    cdef INDEX_t new_capacity = capacity[0] * 2
    cdef INDEX_t* new_index = <INDEX_t*>realloc(index[0], new_capacity * sizeof(INDEX_t))
    cdef DTYPE_t* new_priorities = <DTYPE_t*>realloc(priorities[0], new_capacity * sizeof(DTYPE_t))
    if not new_index or not new_priorities:
        return False
    index[0] = new_index
    priorities[0] = new_priorities
    capacity[0] = new_capacity
    return True

# -----------------------------------------------------------------------------
# Core Algorithm
# -----------------------------------------------------------------------------

cdef void dijkstra_core(
    INDEX_t** index_ptr,
    DTYPE_t** priority_ptr,
    INDEX_t* capacity_ptr,
    INDEX_t* indirection,
    INDEX_t* size_ptr,
    INDEX_t start,
    DTYPE_t* distance,
    DTYPE_t* cost,
    DIR_t* direction,
    INDEX_t stride
):

    cdef:
        INDEX_t i, neighbor, k
        DTYPE_t d, alternative
        INDEX_t* index = index_ptr[0]
        DTYPE_t* priorities = priority_ptr[0]
        INDEX_t size = size_ptr[0]
        INDEX_t[8] offsets = [-stride, stride, -1, 1, -stride - 1, -stride + 1, stride - 1, stride + 1]
        DTYPE_t[8] step_costs = [1.0, 1.0, 1.0, 1.0, M_SQRT2, M_SQRT2, M_SQRT2, M_SQRT2]

    while size > 0 and priorities[0] < distance[start]:

        # pop minimum
        i = index[0]
        d = priorities[0]
        indirection[i] = NO_INDEX
        size -= 1
        if size > 0:
            index[0] = index[size]
            priorities[0] = priorities[size]
            indirection[index[0]] = 0
            bubble_down(index, priorities, indirection, 0, size)

        # iterate neighbours
        for k in range(8):
            neighbor = i + offsets[k]
            alternative = d + step_costs[k] * cost[neighbor]
            if alternative < distance[neighbor]:
                distance[neighbor] = alternative
                direction[neighbor] = <DIR_t>k
                if indirection[neighbor] != NO_INDEX:
                    # decrease key
                    priorities[indirection[neighbor]] = alternative
                    bubble_up(index, priorities, indirection, indirection[neighbor])
                else:
                    # dynamic resize
                    if size >= capacity_ptr[0]:
                        if ensure_capacity(index_ptr, priority_ptr, capacity_ptr):
                            index = index_ptr[0]
                            priorities = priority_ptr[0]
                        else:
                            size_ptr[0] = MEMORY_ERROR
                            return
                    # enqueue
                    index[size] = neighbor
                    priorities[size] = alternative
                    indirection[neighbor] = size
                    bubble_up(index, priorities, indirection, size)
                    size += 1

    size_ptr[0] = size

# -----------------------------------------------------------------------------
# Python Interface
# -----------------------------------------------------------------------------

cdef class DijkstraOutput:
    cdef public DIR_t[:, ::1] direction
    cdef public DTYPE_t[:, ::1] distance

    cdef INDEX_t* index
    cdef DTYPE_t* priority
    cdef INDEX_t capacity

    cdef DTYPE_t[:, ::1] cost
    cdef INDEX_t[:, ::1] indirection
    cdef INDEX_t size
    cdef INDEX_t width

    def __cinit__(self,
                  DTYPE_t[:, :] cost,
                  INDEX_t[:, :] targets):
        cdef:
            INDEX_t x, y, flat_idx
            DTYPE_t c
            INDEX_t n_targets = targets.shape[0]

        self.cost = np.pad(cost, 1, "constant", constant_values=INFINITY)
        self.width = self.cost.shape[1]

        self.direction = np.full_like(self.cost, NO_DIRECTION, dtype=np.uint8)
        self.indirection = np.full_like(self.cost, NO_INDEX, dtype=np.intp)
        self.distance = np.full_like(self.cost, INFINITY)

        self.capacity = INITIAL_CAPACITY
        if n_targets * 2 > self.capacity:
            self.capacity = n_targets * 2

        self.index = <INDEX_t*>malloc(self.capacity * sizeof(INDEX_t))
        self.priority = <DTYPE_t*>malloc(self.capacity * sizeof(DTYPE_t))

        if not self.index or not self.priority:
            raise MemoryError("Could not allocate heap memory")

        self.size = 0

        cdef INDEX_t* indirection = &self.indirection[0,0]
        cdef DTYPE_t* distance = &self.distance[0,0]
        cdef DTYPE_t* cost_flat = &self.cost[0,0]

        for i in range(n_targets):
            x = targets[i, 0] + 1
            y = targets[i, 1] + 1
            flat_idx = x * self.width + y
            c = cost_flat[flat_idx]
            if c == INFINITY:
                continue
            self.index[self.size] = flat_idx
            self.priority[self.size] = c
            indirection[flat_idx] = self.size
            distance[flat_idx] = c
            self.size += 1
            bubble_up(self.index, self.priority, indirection, self.size - 1)

    def __dealloc__(self):
        if self.index is not NULL:
            free(self.index)
        if self.priority is not NULL:
            free(self.priority)

    cpdef get_path(self, (float, float) source, int limit=0, int max_distance=1):
        cdef INDEX_t i, j, x0, y0, flat_idx
        cdef DIR_t* direction = &self.direction[0, 0]

        i, j = self.get_closest_reachable_point(source, max_distance=max_distance)
        x0 = i + 1
        y0 = j + 1

        if x0 < 0 or y0 < 0 or x0 >= self.distance.shape[0] or y0 >= self.distance.shape[1]:
            return [(i, j)]

        if self.cost[x0, y0] == INFINITY:
            return [(i, j)]

        flat_idx = x0 * self.width + y0

        # Run Dijkstra Core
        dijkstra_core(
            &self.index,
            &self.priority,
            &self.capacity,
            &self.indirection[0, 0],
            &self.size,
            flat_idx,
            &self.distance[0, 0],
            &self.cost[0, 0],
            direction,
            self.width
        )

        if self.size == MEMORY_ERROR:
            raise MemoryError("Heap allocation failed during pathfinding")

        if limit == 0:
            limit = self.distance.size

        # Reconstruct path
        path = []
        cdef INDEX_t curr_flat = flat_idx
        cdef INDEX_t px = i
        cdef INDEX_t py = j
        cdef DIR_t move_direction

        # Local lookup to decode direction (must match loop order in core)
        cdef INDEX_t[8] offsets = [-self.width, self.width, -1, 1, -self.width - 1, -self.width + 1, self.width - 1, self.width + 1]
        cdef INDEX_t[8] offsets_x = [-1, 1, 0, 0, -1, -1, 1, 1]
        cdef INDEX_t[8] offsets_y = [0, 0, -1, 1, -1, 1, -1, 1]

        while len(path) < limit:
            path.append((px, py))
            move_direction = direction[curr_flat]
            if move_direction == NO_DIRECTION:
                break
            px -= offsets_x[move_direction]
            py -= offsets_y[move_direction]
            curr_flat -= offsets[move_direction]
        return path

    cpdef (INDEX_t, INDEX_t) get_closest_reachable_point(self, (float, float) source, int max_distance):
        cdef INDEX_t x0 = <INDEX_t>round(source[0])
        cdef INDEX_t y0 = <INDEX_t>round(source[1])
        cdef INDEX_t x_min = x0
        cdef INDEX_t y_min = y0
        cdef DTYPE_t min_distance_squared = INFINITY
        cdef DTYPE_t distance_squared
        cdef INDEX_t x, y
        cdef INDEX_t x_start = max(0, x0 - max_distance)
        cdef INDEX_t x_end = min(self.distance.shape[0] - 2, x0 + max_distance + 1)
        cdef INDEX_t y_start = max(0, y0 - max_distance)
        cdef INDEX_t y_end = min(self.distance.shape[1] - 2, y0 + max_distance + 1)
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if self.cost[x+1, y+1] != INFINITY:
                    distance_squared = (x - source[0])**2 + (y - source[1])**2
                    if distance_squared < min_distance_squared:
                        min_distance_squared = distance_squared
                        x_min = x
                        y_min = y
        return x_min, y_min

cpdef DijkstraOutput cy_dijkstra(
    object cost,
    object targets,
    bint checks_enabled = True,
):
    cdef DTYPE_t[:, ::1] cost_array = np.ascontiguousarray(cost, dtype=np.float32)
    cdef INDEX_t[:, ::1] target_array = np.ascontiguousarray(targets, dtype=np.intp)
    if checks_enabled:
        if np.any(np.less_equal(cost_array, 0.0)):
            raise Exception("invalid cost: entries must be strictly positive")
    return DijkstraOutput(cost_array, target_array)