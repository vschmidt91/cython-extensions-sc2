# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
from cython import boundscheck, wraparound

cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sqrt

DEF HEAP_ARITY = 4

ctypedef cnp.float64_t DTYPE_t
ctypedef Py_ssize_t INDEX_t

cdef INDEX_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
cdef INDEX_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
cdef DTYPE_t SQRT2 = np.sqrt(2)
cdef DTYPE_t[8] NEIGHBOURS_D = [1, 1, 1, 1, SQRT2, SQRT2, SQRT2, SQRT2]


cdef struct PriorityQueueItem:
    INDEX_t x, y
    DTYPE_t distance

cdef inline void swap(
    PriorityQueueItem* heap,
    INDEX_t[:, :] indices,
    INDEX_t i,
    INDEX_t j,
) noexcept nogil:
    cdef PriorityQueueItem item_i = heap[j]
    cdef PriorityQueueItem item_j = heap[i]
    heap[i] = item_i
    heap[j] = item_j
    indices[item_i.x, item_i.y] = i
    indices[item_j.x, item_j.y] = j


cdef inline void bubble_up(
    PriorityQueueItem* heap,
    INDEX_t[:, :] indices,
    INDEX_t arity,
    INDEX_t index
) noexcept nogil:
    cdef INDEX_t parent
    while index != 0:
        parent = (index - 1) // arity
        if heap[index].distance < heap[parent].distance:
            swap(heap, indices, index, parent)
            index = parent
        else:
            break

cdef inline void bubble_down(
    PriorityQueueItem* heap,
    INDEX_t[:, :] indices,
    INDEX_t arity,
    INDEX_t index,
    INDEX_t size,
) noexcept nogil:
    cdef INDEX_t child, child0, next
    while True:
        next = index
        child0 = arity * index + 1
        for child in range(child0, min(size, child0 + arity)):
            if heap[child].distance < heap[next].distance:
                next = child
        if next != index:
            swap(heap, indices, index, next)
            index = next
        else:
            break

cdef INDEX_t dijkstra_core(
    PriorityQueueItem* heap,
    INDEX_t[:, :] indices,
    INDEX_t arity,
    INDEX_t size,
    INDEX_t x0,
    INDEX_t y0,
    DTYPE_t[:, :] distance,
    DTYPE_t[:, :] cost,
    INDEX_t[:, :] forward_x,
    INDEX_t[:, :] forward_y,
) noexcept nogil:

    cdef:
        INDEX_t x, y, x2, y2
        DTYPE_t d, alternative

    while size != 0 and heap[0].distance < distance[x0, y0]:
        # dequeue
        x = heap[0].x
        y = heap[0].y
        d = heap[0].distance
        indices[x, y] = -1
        size -= 1
        heap[0] = heap[size]
        indices[heap[0].x, heap[0].y] = 0
        bubble_down(heap, indices, arity, 0, size)
        if d > distance[x, y]:
            continue
        for k in range(8):
            x2 = x + NEIGHBOURS_X[k]
            y2 = y + NEIGHBOURS_Y[k]
            alternative = distance[x, y] + NEIGHBOURS_D[k] * cost[x2, y2]
            if alternative < distance[x2, y2]:
                distance[x2, y2] = alternative
                forward_x[x2, y2] = x
                forward_y[x2, y2] = y
                if indices[x2, y2] != -1:
                    # decrease_key
                    heap[indices[x2, y2]].distance = alternative
                    bubble_up(heap, indices, arity, indices[x2, y2])
                else:
                    # enqueue
                    if size >= distance.shape[0] * distance.shape[1]:
                        return size
                    size += 1
                    heap[size - 1] = PriorityQueueItem(x2, y2, alternative)
                    indices[x2, y2] = size - 1
                    bubble_up(heap, indices, arity, size - 1)
    return size



cdef class DijkstraOutput:
    cdef public INDEX_t[:, :] forward_x
    """Forward pointer grid (x-coordinates)."""
    cdef public INDEX_t[:, :] forward_y
    """Forward pointer grid (y-coordinates)."""
    cdef public DTYPE_t[:, :] distance
    """Distance grid."""
    cdef PriorityQueueItem* heap
    cdef DTYPE_t[:, :] cost
    cdef INDEX_t[:, :] targets
    cdef INDEX_t[:, :] indices
    cdef INDEX_t arity
    cdef INDEX_t size
    def __cinit__(self,
                  DTYPE_t[:, :] cost,
                  INDEX_t[:, :] targets):
        cdef:
            INDEX_t x, y
            DTYPE_t c
        self.cost = np.pad(cost, 1, "constant", constant_values=np.inf)
        self.targets = targets
        self.forward_x = np.full_like(self.cost, -1, np.intp)
        self.forward_y = np.full_like(self.cost, -1, np.intp)
        self.indices = np.full_like(self.cost, -1, np.intp)
        self.distance = np.full_like(self.cost, np.inf)

        self.arity = HEAP_ARITY
        self.heap = <PriorityQueueItem*>PyMem_Malloc(self.cost.size * sizeof(PriorityQueueItem))
        self.size = 0

        # initialize queue with targets
        for i in range(targets.shape[0]):
            # add to heap
            x = targets[i, 0] + 1
            y = targets[i, 1] + 1
            c = self.cost[x, y]
            if self.cost[x, y] == np.inf:
                continue
            self.heap[self.size] = PriorityQueueItem(x, y, c)
            self.indices[x, y] = self.size
            self.distance[x, y] = c
            self.size += 1
            bubble_up(self.heap, self.indices, self.arity, self.size - 1)


    def __dealloc__(self):
        PyMem_Free(self.heap)

    cpdef get_path(self, (float, float) source, int limit=0, int max_distance=1):
        """

        Follow the path from a given source using the forward pointer grids.

        Parameters
        ----------
        source :
            Start point.
        limit :
            Maximum length of the returned path. Defaults to 0 indicating no limit.
        max_distance :
            Size of the search region for a valid starting point. Defaults to 1.

        Returns
        -------
        list[tuple[int, int]] :
            The lowest cost path from source to any of the targets.

        """
        cdef INDEX_t x, y, x0, y0, i, j
        path = list[tuple[int, int]]()
        i, j = self.get_closest_reachable_point(source, max_distance=max_distance)

        x0 = i + 1
        y0 = j + 1
        if x0 < 0 or y0 < 0 or x0 >= self.distance.shape[0] or y0 >= self.distance.shape[1]:
            return [(i, j)]
        if self.cost[x0, y0] == np.inf:
            return [(i, j)]

        self.size = dijkstra_core(self.heap, self.indices, self.arity, self.size, x0, y0, self.distance, self.cost, self.forward_x, self.forward_y)

        if limit == 0:
            # set a fallback limit to be safe
            # a path longer than this must contain a cycle, so it should never be hit anyway
            limit = self.distance.shape[0] * self.distance.shape[1]

        x = x0
        y = y0
        while len(path) < limit:
            if x < 0 or y < 0:
                # pointer value of -1 marks no pointer
                break
            path.append((x - 1, y - 1))
            x, y = self.forward_x[x, y], self.forward_y[x, y]
        return path

    cpdef (INDEX_t, INDEX_t) get_closest_reachable_point(self, (float, float) source, int max_distance):
        """

        Search the region for a point that can reach a target.

        Parameters
        ----------
        source :
            Start point as float coordinates.
        limit :
            Maximum distance between the source and returned point.

        Returns
        -------
        tuple[int, int] :
            The closest integer coordinates to the source with finite distance.

        """
        cdef INDEX_t x0 = <INDEX_t>round(source[0])
        cdef INDEX_t y0 = <INDEX_t>round(source[1])
        cdef INDEX_t x_min = x0
        cdef INDEX_t y_min = y0
        cdef DTYPE_t min_distance_squared = np.inf
        cdef DTYPE_t distance_squared = np.inf
        cdef INDEX_t x
        cdef INDEX_t y

        for x in range(max(0, x0 - max_distance), min(self.distance.shape[0] - 2, x0 + max_distance + 1)):
            for y in range(max(0, y0 - max_distance), min(self.distance.shape[1] - 2, y0 + max_distance + 1)):
                if self.cost[x+1, y+1] != np.inf:
                    distance_squared = (x - source[0]) ** 2 + (y - source[1]) ** 2
                    if distance_squared < min_distance_squared:
                        x_min = x
                        y_min = y
                        min_distance_squared = distance_squared

        return x_min, y_min


cpdef DijkstraOutput cy_dijkstra(
    DTYPE_t[:, :] cost,
    INDEX_t[:, :] targets,
    bint checks_enabled = True,
):
    """

    Run Dijkstras algorithm on a grid, yielding many-target-shortest paths for each position.

    Parameters
    ----------
    cost :
        Cost grid. Entries must be positive. Set unpathable cells to infinity.
    targets :
        Target array of shape (*, 2) containing x and y coordinates of the target points.
    checks_enabled :
        Pass False to deactivate grid value and target coordinates checks. Defaults to True.

    Returns
    -------
    DijkstraOutput :
        Pathfinding object containing distance and forward pointer grids.

    """

    if checks_enabled:
        if np.any(np.less_equal(cost, 0.0)):
            raise Exception("invalid cost: entries must be strictly positive")

        if any((
            np.less(targets, 0).any(),
            np.greater_equal(targets[:, 0], cost.shape[0]).any(),
            np.greater_equal(targets[:, 1], cost.shape[1]).any(),
        )):
            raise Exception(f"Target out of bounds")

    return DijkstraOutput(cost, targets)