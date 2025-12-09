# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free
from libc.math cimport sqrt, round, fabs

from numpy.math cimport INFINITY

# -----------------------------------------------------------------------------
# CONSTANTS & STRUCTS
# -----------------------------------------------------------------------------

DEF HEAP_ARITY = 4

ctypedef cnp.float64_t DTYPE_t
ctypedef Py_ssize_t INDEX_t

cdef INDEX_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
cdef INDEX_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
cdef DTYPE_t SQRT2 = 1.41421356237
cdef DTYPE_t[8] NEIGHBOURS_D = [1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2]

cdef struct PriorityQueueItem:
    INDEX_t x, y
    DTYPE_t distance

# -----------------------------------------------------------------------------
# HEAP OPERATIONS (Inlined & Nogil)
# -----------------------------------------------------------------------------

cdef inline void bubble_up(PriorityQueueItem* heap, INDEX_t[:, :] indices, INDEX_t index) noexcept nogil:
    cdef INDEX_t parent
    while index != 0:
        parent = (index - 1) // HEAP_ARITY
        if heap[index].distance < heap[parent].distance:
            heap[index], heap[parent] = heap[parent], heap[index]
            indices[heap[index].x, heap[index].y] = index
            indices[heap[parent].x, heap[parent].y] = parent
            index = parent
        else:
            break

cdef inline void bubble_down(PriorityQueueItem* heap, INDEX_t[:, :] indices, INDEX_t size, INDEX_t index) noexcept nogil:
    cdef INDEX_t swap, child, i
    while True:
        swap = index
        i = HEAP_ARITY * index + 1
        for child in range(i, min(i + HEAP_ARITY, size)):
            if heap[child].distance < heap[swap].distance:
                swap = child
        if swap != index:
            heap[index], heap[swap] = heap[swap], heap[index]
            indices[heap[index].x, heap[index].y] = index
            indices[heap[swap].x, heap[swap].y] = swap
            index = swap
        else:
            break

# -----------------------------------------------------------------------------
# CORE ALGORITHM (NOGIL)
# -----------------------------------------------------------------------------

cdef void _dijkstra_core(
    DTYPE_t[:, :] cost,        # Padded cost grid
    INDEX_t[:, :] targets,     # Target list (unpadded coords)
    DTYPE_t[:, :] distance,    # Output: Padded distance grid
    INDEX_t[:, :] forward_x,   # Output: Padded forward pointers X
    INDEX_t[:, :] forward_y,   # Output: Padded forward pointers Y
    INDEX_t[:, :] indices      # Temporary: Heap indices map
) noexcept nogil:

    cdef:
        INDEX_t i, k, x, y, x2, y2
        DTYPE_t d, alternative

        # Local Heap Management
        INDEX_t size = 0
        INDEX_t capacity = max(targets.shape[0] * 2, 1024)
        PriorityQueueItem* heap = <PriorityQueueItem*>malloc(capacity * sizeof(PriorityQueueItem))
        PriorityQueueItem* new_heap

    if not heap:
        return # Memory allocation failure

    # 1. Initialize Heap with Targets
    for i in range(targets.shape[0]):
        x = targets[i, 0] + 1  # Shift to padded coordinates
        y = targets[i, 1] + 1

        if cost[x, y] == INFINITY:
            continue

        heap[size].x = x
        heap[size].y = y
        heap[size].distance = cost[x, y]

        distance[x, y] = cost[x, y]
        indices[x, y] = size
        size += 1
        bubble_up(heap, indices, size - 1)

    # 2. Main Loop
    while size > 0:
        # Pop Min
        x = heap[0].x
        y = heap[0].y
        d = heap[0].distance

        indices[x, y] = -1
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            indices[heap[0].x, heap[0].y] = 0
            bubble_down(heap, indices, size, 0)

        # Stale Node Check
        if d > distance[x, y]:
            continue

        # Expand Neighbors
        for k in range(8):
            # No boundary checks needed due to padding!
            x2 = x + NEIGHBOURS_X[k]
            y2 = y + NEIGHBOURS_Y[k]

            if cost[x2, y2] == INFINITY:
                continue

            alternative = d + NEIGHBOURS_D[k] * cost[x2, y2]

            if alternative < distance[x2, y2]:
                distance[x2, y2] = alternative
                forward_x[x2, y2] = x
                forward_y[x2, y2] = y

                if indices[x2, y2] != -1:
                    # Decrease Key
                    heap[indices[x2, y2]].distance = alternative
                    bubble_up(heap, indices, indices[x2, y2])
                else:
                    # Insert
                    if size == capacity:
                        capacity *= 2
                        new_heap = <PriorityQueueItem*>realloc(heap, capacity * sizeof(PriorityQueueItem))
                        if not new_heap:
                            free(heap)
                            return # Hard crash prevention
                        heap = new_heap

                    heap[size].x = x2
                    heap[size].y = y2
                    heap[size].distance = alternative
                    indices[x2, y2] = size
                    size += 1
                    bubble_up(heap, indices, size - 1)

    free(heap)

# -----------------------------------------------------------------------------
# PATH RETRIEVAL (NOGIL)
# -----------------------------------------------------------------------------

cdef void _trace_path_grid(
    INDEX_t ix, INDEX_t iy,
    DTYPE_t[:, :] distance,
    INDEX_t[:, :] fwd_x,
    INDEX_t[:, :] fwd_y,
    INDEX_t limit,
    INDEX_t[:, :] out_path, # Pre-allocated output buffer
    INDEX_t* out_len        # Pointer to write result length
) noexcept nogil:
    cdef INDEX_t k, nx, ny

    # We work in padded coordinates internally, write unpadded to output
    for k in range(limit):
        out_path[k, 0] = ix - 1
        out_path[k, 1] = iy - 1
        out_len[0] = k + 1

        # Target reached? (Distance is close to 0 or cost of cell)
        if distance[ix, iy] <= 1.0:
            break

        nx = fwd_x[ix, iy]
        ny = fwd_y[ix, iy]

        # No forward pointer or cycle to self
        if nx == -1 or (nx == ix and ny == iy):
            break

        ix = nx
        iy = ny

cdef void _trace_path_smooth(
    INDEX_t ix, INDEX_t iy,
    DTYPE_t[:, :] distance,
    INDEX_t limit,
    INDEX_t[:, :] out_path,
    INDEX_t* out_len
) noexcept nogil:
    cdef:
        double fx = <double>ix
        double fy = <double>iy
        double step_size = 0.5
        double d_dx, d_dy, length
        INDEX_t k
        INDEX_t cx, cy # Current integer cell

    for k in range(limit):
        cx = <INDEX_t>round(fx)
        cy = <INDEX_t>round(fy)

        out_path[k, 0] = cx - 1
        out_path[k, 1] = cy - 1
        out_len[0] = k + 1

        # 1. Stop if we are very close to target (small distance value)
        if distance[cx, cy] < step_size:
            break

        # 2. Compute Gradient (Central Difference)
        # Safe because we have padding
        d_dx = (distance[cx + 1, cy] - distance[cx - 1, cy]) * 0.5
        d_dy = (distance[cx, cy + 1] - distance[cx, cy - 1]) * 0.5

        length = sqrt(d_dx*d_dx + d_dy*d_dy)

        if length < 1e-6:
            break # Minima

        # 3. Move Downhill
        fx -= (d_dx / length) * step_size
        fy -= (d_dy / length) * step_size


# -----------------------------------------------------------------------------
# WRAPPER CLASS
# -----------------------------------------------------------------------------

cdef class DijkstraGrid:
    cdef public DTYPE_t[:, :] distance
    cdef public INDEX_t[:, :] forward_x
    cdef public INDEX_t[:, :] forward_y
    cdef public INDEX_t h, w

    def __cinit__(self, DTYPE_t[:, :] cost, INDEX_t[:, :] targets):
        """
        Calculates the distance field immediately upon initialization.
        Input cost should be strictly positive.
        """
        # Validate inputs
        if np.any(np.less_equal(cost, 0.0)):
            raise ValueError("Cost entries must be strictly positive")

        self.h = cost.shape[0]
        self.w = cost.shape[1]

        # 1. Allocate Padded Internal Grids
        # We pad inputs by 1 to completely remove boundary checks in C code
        cdef DTYPE_t[:, :] cost_padded = np.pad(cost, 1, "constant", constant_values=INFINITY)
        self.distance = np.full_like(cost_padded, INFINITY)
        self.forward_x = np.full_like(cost_padded, -1, dtype=np.intp)
        self.forward_y = np.full_like(cost_padded, -1, dtype=np.intp)

        # Temporary array for heap indices
        cdef INDEX_t[:, :] indices = np.full_like(cost_padded, -1, dtype=np.intp)

        # 2. Run Core (Release GIL)
        with nogil:
            _dijkstra_core(cost_padded, targets, self.distance,
                           self.forward_x, self.forward_y, indices)

    cpdef get_path(self, tuple source, int limit=0, int max_search=1, bint smooth=False):
        """
        Retrieves a path from the pre-calculated grid.
        """
        cdef:
            INDEX_t x0, y0
            INDEX_t len_out = 0
            INDEX_t[:, :] path_buffer

        if limit <= 0:
            limit = self.h * self.w

        # 1. Find valid start point (search unpadded coords)
        x0, y0 = self._find_start(source, max_search)

        # Convert to padded coords for internal use
        x0 += 1
        y0 += 1

        # If start is invalid
        if self.distance[x0, y0] == INFINITY:
            return []

        # 2. Allocate output buffer
        # (We use a numpy array buffer to pass to C)
        path_arr = np.empty((limit, 2), dtype=np.intp)
        path_buffer = path_arr

        # 3. Trace Path (Release GIL)
        with nogil:
            if smooth:
                _trace_path_smooth(x0, y0, self.distance, limit, path_buffer, &len_out)
            else:
                _trace_path_grid(x0, y0, self.distance, self.forward_x, self.forward_y,
                                 limit, path_buffer, &len_out)

        # 4. Return slice
        return path_arr[:len_out]

    cdef tuple _find_start(self, tuple source, int r):
        # Quick helper to find nearest valid cell in unpadded coords
        cdef INDEX_t sx = <INDEX_t>round(source[0])
        cdef INDEX_t sy = <INDEX_t>round(source[1])
        cdef INDEX_t x, y, bx, by
        cdef double min_d = 1e9, d

        bx, by = sx, sy

        # If the exact point is valid, return it immediately
        # (Check padded distance)
        if 0 <= sx < self.h and 0 <= sy < self.w:
            if self.distance[sx+1, sy+1] != INFINITY:
                return (sx, sy)

        # Spiral/Box search
        cdef INDEX_t x_min = max(0, sx - r)
        cdef INDEX_t x_max = min(self.h, sx + r + 1)
        cdef INDEX_t y_min = max(0, sy - r)
        cdef INDEX_t y_max = min(self.w, sy + r + 1)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if self.distance[x+1, y+1] != INFINITY:
                    d = (x - source[0])**2 + (y - source[1])**2
                    if d < min_d:
                        min_d = d
                        bx = x
                        by = y
        return (bx, by)

cpdef DijkstraGrid cy_dijkstra(DTYPE_t[:, :] cost, INDEX_t[:, :] targets, bint checks_enabled):
    return DijkstraGrid(cost, targets)