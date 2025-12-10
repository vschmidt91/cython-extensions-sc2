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
from cpython.mem cimport PyMem_Malloc, PyMem_Free

ctypedef cnp.float64_t DTYPE_t
ctypedef Py_ssize_t INDEX_t

# Constants
cdef DTYPE_t SQRT2 = 1.4142135623730951

# -----------------------------------------------------------------------------
# Heap Operations (Split Arrays / Structure of Arrays)
# -----------------------------------------------------------------------------

cdef inline void swap(
    INDEX_t* heap_indices,
    DTYPE_t* heap_distances,
    INDEX_t* indices_ptr,
    INDEX_t i,
    INDEX_t j,
) noexcept nogil:
    """Swaps items in the parallel heap arrays and updates lookup indices."""
    cdef INDEX_t idx_i = heap_indices[i]
    cdef DTYPE_t dist_i = heap_distances[i]

    cdef INDEX_t idx_j = heap_indices[j]
    cdef DTYPE_t dist_j = heap_distances[j]

    heap_indices[i] = idx_j
    heap_distances[i] = dist_j

    heap_indices[j] = idx_i
    heap_distances[j] = dist_i

    # Update the lookup table: flat_index -> heap_position
    indices_ptr[idx_i] = j
    indices_ptr[idx_j] = i

cdef inline void bubble_up(
    INDEX_t* heap_indices,
    DTYPE_t* heap_distances,
    INDEX_t* indices_ptr,
    INDEX_t index
) noexcept nogil:
    cdef INDEX_t parent
    # Capture the moving node (the "hole" is now at 'index')
    cdef INDEX_t moving_node_idx = heap_indices[index]
    cdef DTYPE_t moving_node_dist = heap_distances[index]

    while index > 0:
        # Bitwise shift for division by 4 (arity)
        parent = (index - 1) >> 2

        if moving_node_dist < heap_distances[parent]:
            # Move parent down into the hole
            heap_indices[index] = heap_indices[parent]
            heap_distances[index] = heap_distances[parent]

            # Update parent's lookup immediately
            indices_ptr[heap_indices[index]] = index

            # The hole moves up to the parent
            index = parent
        else:
            break

    # Place the moving node into the final hole
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
    # Capture the moving node (the "hole" is now at 'index')
    cdef INDEX_t moving_node_idx = heap_indices[index]
    cdef DTYPE_t moving_node_dist = heap_distances[index]

    while True:
        # Bitwise shift for multiplication by 4
        child0 = (index << 2) + 1

        if child0 >= size:
            break

        # Find smallest child (Arity 4)
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

        # Check if we need to swap with the smallest child
        if heap_distances[min_child] < moving_node_dist:
            # Move child up into the hole
            heap_indices[index] = heap_indices[min_child]
            heap_distances[index] = heap_distances[min_child]

            # Update child's lookup immediately
            indices_ptr[heap_indices[index]] = index

            # The hole moves down to the child
            index = min_child
        else:
            break

    # Place the moving node into the final hole
    heap_indices[index] = moving_node_idx
    heap_distances[index] = moving_node_dist
    indices_ptr[moving_node_idx] = index

# -----------------------------------------------------------------------------
# Core Algorithm
# -----------------------------------------------------------------------------

cdef INDEX_t dijkstra_core(
    INDEX_t* heap_indices,    # Parallel heap array 1
    DTYPE_t* heap_distances,  # Parallel heap array 2
    INDEX_t* indices_ptr,     # Flat pointer to indices lookup array
    INDEX_t size,
    INDEX_t start_flat_index,
    DTYPE_t* dist_ptr,
    DTYPE_t* cost_ptr,
    INDEX_t* pred_ptr,
    INDEX_t total_pixels,
    INDEX_t stride
) noexcept nogil:

    cdef:
        INDEX_t curr_idx, neighbor_idx, k
        DTYPE_t d, alternative

        # Precompute neighbor offsets and costs on stack
        INDEX_t[8] offsets
        DTYPE_t[8] step_costs

    # Setup neighbor lookup tables based on image width (stride)
    offsets[0] = -stride;    step_costs[0] = 1.0
    offsets[1] = stride;     step_costs[1] = 1.0
    offsets[2] = -1;         step_costs[2] = 1.0
    offsets[3] = 1;          step_costs[3] = 1.0
    offsets[4] = -stride - 1; step_costs[4] = SQRT2
    offsets[5] = -stride + 1; step_costs[5] = SQRT2
    offsets[6] = stride - 1;  step_costs[6] = SQRT2
    offsets[7] = stride + 1;  step_costs[7] = SQRT2

    while size > 0 and heap_distances[0] < dist_ptr[start_flat_index]:
        # 1. Pop Min
        curr_idx = heap_indices[0]
        d = heap_distances[0]

        # Remove from heap: replace root with last element
        indices_ptr[curr_idx] = -1
        size -= 1

        if size > 0:
            heap_indices[0] = heap_indices[size]
            heap_distances[0] = heap_distances[size]
            indices_ptr[heap_indices[0]] = 0
            bubble_down(heap_indices, heap_distances, indices_ptr, 0, size)

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
                    heap_distances[indices_ptr[neighbor_idx]] = alternative
                    bubble_up(heap_indices, heap_distances, indices_ptr, indices_ptr[neighbor_idx])
                else:
                    # Insert
                    if size >= total_pixels:
                        return size # Should not happen

                    heap_indices[size] = neighbor_idx
                    heap_distances[size] = alternative
                    indices_ptr[neighbor_idx] = size
                    size += 1
                    bubble_up(heap_indices, heap_distances, indices_ptr, size - 1)

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

        # Heap allocation (Structure of Arrays)
        cdef INDEX_t total_pixels = self.cost.shape[0] * self.cost.shape[1]
        self.heap_indices = <INDEX_t*>PyMem_Malloc(total_pixels * sizeof(INDEX_t))
        self.heap_distances = <DTYPE_t*>PyMem_Malloc(total_pixels * sizeof(DTYPE_t))

        if not self.heap_indices or not self.heap_distances:
            raise MemoryError()

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

            self.heap_indices[self.size] = flat_idx
            self.heap_distances[self.size] = c

            indices_ptr[flat_idx] = self.size
            dist_ptr[flat_idx] = c
            self.size += 1
            bubble_up(self.heap_indices, self.heap_distances, indices_ptr, self.size - 1)

    def __dealloc__(self):
        if self.heap_indices is not NULL:
            PyMem_Free(self.heap_indices)
        if self.heap_distances is not NULL:
            PyMem_Free(self.heap_distances)

    cpdef get_path(self, (float, float) source, int limit=0, int max_distance=1):
        cdef INDEX_t i, j, x0, y0, flat_idx
        cdef INDEX_t* pred_ptr = &self.predecessors[0, 0]

        # Find closest reachable point
        i, j = self.get_closest_reachable_point(source, max_distance=max_distance)

        x0 = i + 1
        y0 = j + 1

        # Basic bounds check
        if x0 < 0 or y0 < 0 or x0 >= self.distance.shape[0] or y0 >= self.distance.shape[1]:
            return [(i, j)]

        if self.cost[x0, y0] == INFINITY:
            return [(i, j)]

        flat_idx = x0 * self.width + y0

        # Run Dijkstra Core
        self.size = dijkstra_core(
            self.heap_indices,
            self.heap_distances,
            &self.indices[0, 0],
            self.size,
            flat_idx,
            &self.distance[0, 0],
            &self.cost[0, 0],
            pred_ptr,
            self.distance.size,
            self.width
        )

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