# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from numpy.math cimport INFINITY
from libc.math cimport sqrt, round
# switched to libc.stdlib for nogil-safe realloc
from libc.stdlib cimport malloc, free, realloc

ctypedef cnp.float64_t DTYPE_t
ctypedef Py_ssize_t INDEX_t

# Constants
cdef DTYPE_t SQRT2 = 1.4142135623730951
cdef INDEX_t INITIAL_CAPACITY = 1024

# -----------------------------------------------------------------------------
# Heap Operations (Unchanged)
# -----------------------------------------------------------------------------

cdef inline void bubble_up(
    INDEX_t* heap_indices,
    DTYPE_t* heap_distances,
    INDEX_t* indices_ptr,
    INDEX_t index
) noexcept nogil:
    cdef INDEX_t parent
    cdef INDEX_t moving_node_idx = heap_indices[index]
    cdef DTYPE_t moving_node_dist = heap_distances[index]

    while index > 0:
        parent = (index - 1) >> 2
        if moving_node_dist < heap_distances[parent]:
            heap_indices[index] = heap_indices[parent]
            heap_distances[index] = heap_distances[parent]
            indices_ptr[heap_indices[index]] = index
            index = parent
        else:
            break

    heap_indices[index] = moving_node_idx
    heap_distances[index] = moving_node_dist
    indices_ptr[moving_node_idx] = index

cdef inline void bubble_down(
    INDEX_t* heap_indices,
    DTYPE_t* heap_distances,
    INDEX_t* indices_ptr,
    INDEX_t index,
    INDEX_t size,
) noexcept nogil:
    cdef INDEX_t child, child0, min_child
    cdef INDEX_t moving_node_idx = heap_indices[index]
    cdef DTYPE_t moving_node_dist = heap_distances[index]

    while True:
        child0 = (index << 2) + 1
        if child0 >= size:
            break

        min_child = child0
        child = child0 + 1
        if child < size and heap_distances[child] < heap_distances[min_child]:
            min_child = child
        child = child0 + 2
        if child < size and heap_distances[child] < heap_distances[min_child]:
            min_child = child
        child = child0 + 3
        if child < size and heap_distances[child] < heap_distances[min_child]:
            min_child = child

        if heap_distances[min_child] < moving_node_dist:
            heap_indices[index] = heap_indices[min_child]
            heap_distances[index] = heap_distances[min_child]
            indices_ptr[heap_indices[index]] = index
            index = min_child
        else:
            break

    heap_indices[index] = moving_node_idx
    heap_distances[index] = moving_node_dist
    indices_ptr[moving_node_idx] = index

# -----------------------------------------------------------------------------
# Core Algorithm with Dynamic Allocation
# -----------------------------------------------------------------------------

cdef INDEX_t dijkstra_core(
    # Pass pointers to pointers so we can update them on realloc
    INDEX_t** heap_indices_ptr,
    DTYPE_t** heap_distances_ptr,
    INDEX_t* capacity_ptr,
    INDEX_t* indices_ptr,
    INDEX_t size,
    INDEX_t start_flat_index,
    DTYPE_t* dist_ptr,
    DTYPE_t* cost_ptr,
    INDEX_t* pred_ptr,
    INDEX_t stride
) noexcept nogil:

    cdef:
        INDEX_t curr_idx, neighbor_idx, k, new_cap
        DTYPE_t d, alternative

        # Local cached pointers for speed in the loop
        INDEX_t* h_indices = heap_indices_ptr[0]
        DTYPE_t* h_dists = heap_distances_ptr[0]

        # Realloc temp pointers
        INDEX_t* tmp_indices
        DTYPE_t* tmp_dists

        INDEX_t[8] offsets
        DTYPE_t[8] step_costs

    offsets[0] = -stride;    step_costs[0] = 1.0
    offsets[1] = stride;     step_costs[1] = 1.0
    offsets[2] = -1;         step_costs[2] = 1.0
    offsets[3] = 1;          step_costs[3] = 1.0
    offsets[4] = -stride - 1; step_costs[4] = SQRT2
    offsets[5] = -stride + 1; step_costs[5] = SQRT2
    offsets[6] = stride - 1;  step_costs[6] = SQRT2
    offsets[7] = stride + 1;  step_costs[7] = SQRT2

    while size > 0 and h_dists[0] < dist_ptr[start_flat_index]:
        # 1. Pop Min
        curr_idx = h_indices[0]
        d = h_dists[0]

        indices_ptr[curr_idx] = -1
        size -= 1

        if size > 0:
            h_indices[0] = h_indices[size]
            h_dists[0] = h_dists[size]
            indices_ptr[h_indices[0]] = 0
            bubble_down(h_indices, h_dists, indices_ptr, 0, size)

        if d > dist_ptr[curr_idx]:
            continue

        # 2. Iterate Neighbors
        for k in range(8):
            neighbor_idx = curr_idx + offsets[k]
            alternative = d + step_costs[k] * cost_ptr[neighbor_idx]

            if alternative < dist_ptr[neighbor_idx]:
                dist_ptr[neighbor_idx] = alternative
                pred_ptr[neighbor_idx] = curr_idx

                if indices_ptr[neighbor_idx] != -1:
                    # Decrease Key
                    h_dists[indices_ptr[neighbor_idx]] = alternative
                    bubble_up(h_indices, h_dists, indices_ptr, indices_ptr[neighbor_idx])
                else:
                    # Insert

                    # --- Dynamic Memory Check ---
                    if size >= capacity_ptr[0]:
                        new_cap = capacity_ptr[0] * 2

                        tmp_indices = <INDEX_t*>realloc(h_indices, new_cap * sizeof(INDEX_t))
                        if tmp_indices == NULL:
                            return -1 # Allocation Error
                        h_indices = tmp_indices

                        tmp_dists = <DTYPE_t*>realloc(h_dists, new_cap * sizeof(DTYPE_t))
                        if tmp_dists == NULL:
                            # In a real scenario, you might want to free h_indices or handle partial failure
                            return -1
                        h_dists = tmp_dists

                        # Update the caller's pointers and capacity
                        heap_indices_ptr[0] = h_indices
                        heap_distances_ptr[0] = h_dists
                        capacity_ptr[0] = new_cap
                    # ----------------------------

                    h_indices[size] = neighbor_idx
                    h_dists[size] = alternative
                    indices_ptr[neighbor_idx] = size
                    size += 1
                    bubble_up(h_indices, h_dists, indices_ptr, size - 1)

    return size

# -----------------------------------------------------------------------------
# Python Interface
# -----------------------------------------------------------------------------

cdef class DijkstraOutput:
    cdef public INDEX_t[:, ::1] predecessors
    cdef public DTYPE_t[:, ::1] distance

    # Internal raw pointers for parallel arrays
    cdef INDEX_t* heap_indices
    cdef DTYPE_t* heap_distances

    # Track capacity
    cdef INDEX_t capacity

    cdef DTYPE_t[:, ::1] cost
    cdef INDEX_t[:, ::1] indices
    cdef INDEX_t size
    cdef INDEX_t width

    def __cinit__(self,
                  DTYPE_t[:, :] cost,
                  INDEX_t[:, :] targets):
        cdef:
            INDEX_t x, y, flat_idx
            DTYPE_t c
            INDEX_t n_targets = targets.shape[0]

        # Pad the cost array with infinity
        self.cost = np.ascontiguousarray(np.pad(cost, 1, "constant", constant_values=INFINITY))
        self.width = self.cost.shape[1]

        # Allocate grids
        self.predecessors = np.full_like(self.cost, -1, dtype=np.intp)
        self.indices = np.full_like(self.cost, -1, dtype=np.intp)
        self.distance = np.full_like(self.cost, INFINITY, dtype=np.float64)

        # Dynamic Heap Allocation
        # Start with max(INITIAL_CAPACITY, n_targets * 2) to avoid immediate reallocs
        self.capacity = INITIAL_CAPACITY
        if n_targets * 2 > self.capacity:
            self.capacity = n_targets * 2

        self.heap_indices = <INDEX_t*>malloc(self.capacity * sizeof(INDEX_t))
        self.heap_distances = <DTYPE_t*>malloc(self.capacity * sizeof(DTYPE_t))

        if not self.heap_indices or not self.heap_distances:
            raise MemoryError("Could not allocate heap memory")

        self.size = 0

        # Get raw pointers for initialization
        cdef INDEX_t* indices_ptr = &self.indices[0,0]
        cdef DTYPE_t* dist_ptr = &self.distance[0,0]
        cdef DTYPE_t* cost_ptr = &self.cost[0,0]

        # Initialize queue with targets
        for i in range(n_targets):
            x = targets[i, 0] + 1
            y = targets[i, 1] + 1
            flat_idx = x * self.width + y

            c = cost_ptr[flat_idx]
            if c == INFINITY:
                continue

            # Resize check (unlikely during init unless huge n_targets vs capacity)
            if self.size >= self.capacity:
                self.capacity *= 2
                self.heap_indices = <INDEX_t*>realloc(self.heap_indices, self.capacity * sizeof(INDEX_t))
                self.heap_distances = <DTYPE_t*>realloc(self.heap_distances, self.capacity * sizeof(DTYPE_t))
                if not self.heap_indices or not self.heap_distances:
                    raise MemoryError()

            self.heap_indices[self.size] = flat_idx
            self.heap_distances[self.size] = c

            indices_ptr[flat_idx] = self.size
            dist_ptr[flat_idx] = c
            self.size += 1
            bubble_up(self.heap_indices, self.heap_distances, indices_ptr, self.size - 1)

    def __dealloc__(self):
        # Use free() for libc allocations
        if self.heap_indices is not NULL:
            free(self.heap_indices)
        if self.heap_distances is not NULL:
            free(self.heap_distances)

    cpdef get_path(self, (float, float) source, int limit=0, int max_distance=1):
        cdef INDEX_t i, j, x0, y0, flat_idx
        cdef INDEX_t* pred_ptr = &self.predecessors[0, 0]

        # Find closest reachable point
        i, j = self.get_closest_reachable_point(source, max_distance=max_distance)

        x0 = i + 1
        y0 = j + 1

        if x0 < 0 or y0 < 0 or x0 >= self.distance.shape[0] or y0 >= self.distance.shape[1]:
            return [(i, j)]

        if self.cost[x0, y0] == INFINITY:
            return [(i, j)]

        flat_idx = x0 * self.width + y0

        # Run Dijkstra Core
        # Pass ADDRESSES of the pointers (&self.heap_indices) so realloc updates propagate
        self.size = dijkstra_core(
            &self.heap_indices,
            &self.heap_distances,
            &self.capacity,
            &self.indices[0, 0],
            self.size,
            flat_idx,
            &self.distance[0, 0],
            &self.cost[0, 0],
            pred_ptr,
            self.width
        )

        if self.size == -1:
            raise MemoryError("Heap allocation failed during pathfinding")

        if limit == 0:
            limit = self.distance.size

        # Reconstruct path
        path = []
        cdef INDEX_t curr_flat = flat_idx
        cdef INDEX_t next_flat
        cdef INDEX_t px, py

        while len(path) < limit:
            px = (curr_flat // self.width) - 1
            py = (curr_flat % self.width) - 1

            if px < 0 or py < 0:
                 break

            path.append((px, py))

            next_flat = pred_ptr[curr_flat]
            if next_flat == -1:
                break
            curr_flat = next_flat

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
    DTYPE_t[:, :] cost,
    INDEX_t[:, :] targets,
    bint checks_enabled = True,
):
    if checks_enabled:
        if np.any(np.less_equal(cost, 0.0)):
            raise Exception("invalid cost: entries must be strictly positive")
    return DijkstraOutput(cost, targets)