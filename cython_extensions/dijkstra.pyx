import numpy as np
from cython import boundscheck, wraparound

cimport numpy as cnp
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sqrt

DEF HEAP_ARITY = 4

ctypedef cnp.float64_t DTYPE_t

cdef Py_ssize_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
cdef Py_ssize_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
cdef DTYPE_t SQRT2 = np.sqrt(2)
cdef DTYPE_t[8] NEIGHBOURS_D = [1, 1, 1, 1, SQRT2, SQRT2, SQRT2, SQRT2]


cdef struct PriorityQueueItem:
    Py_ssize_t x, y
    DTYPE_t distance


@boundscheck(False)
@wraparound(False)
cdef inline void bubble_up(PriorityQueueItem* heap, Py_ssize_t[:, :] indices, Py_ssize_t index):
    cdef Py_ssize_t parent
    while index != 0:
        parent = (index - 1) // HEAP_ARITY
        if heap[index].distance < heap[parent].distance:
            heap[index], heap[parent] = heap[parent], heap[index]
            indices[heap[index].x, heap[index].y] = index
            indices[heap[parent].x, heap[parent].y] = parent
            index = parent
        else:
            break

@boundscheck(False)
@wraparound(False)
cdef inline void bubble_down(PriorityQueueItem* heap, Py_ssize_t[:, :] indices, Py_ssize_t size, Py_ssize_t index):
    cdef Py_ssize_t swap, child, i
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


cdef class DijkstraOutput:
    cdef public Py_ssize_t[:, :] forward_x
    """Forward pointer grid (x-coordinates)."""
    cdef public Py_ssize_t[:, :] forward_y
    """Forward pointer grid (y-coordinates)."""
    cdef public DTYPE_t[:, :] distance
    """Distance grid."""
    cdef PriorityQueueItem* heap
    cdef DTYPE_t[:, :] cost
    cdef Py_ssize_t[:, :] targets
    cdef Py_ssize_t[:, :] indices
    cdef Py_ssize_t capacity
    cdef Py_ssize_t size
    def __cinit__(self,
                  DTYPE_t[:, :] cost,
                  Py_ssize_t[:, :] targets):
        cdef:
            Py_ssize_t x, y
            DTYPE_t c
        self.cost = np.pad(cost, 1, "constant", constant_values=np.inf)
        self.targets = targets
        self.forward_x = np.full_like(cost, -1, np.intp)
        self.forward_y = np.full_like(cost, -1, np.intp)
        self.indices = np.full_like(cost, -1, np.intp)
        self.distance = np.full_like(cost, np.inf)

        self.capacity = targets.shape[0]
        self.heap = <PriorityQueueItem*>PyMem_Malloc(self.capacity * sizeof(PriorityQueueItem))
        if not self.heap:
            raise MemoryError()
        self.size = self.capacity

        # initialize queue with targets
        for i in range(targets.shape[0]):
            # add to heap
            x = targets[i, 0]
            y = targets[i, 1]
            c = cost[x, y]
            self.heap[i] = PriorityQueueItem(x, y, c)
            self.indices[x, y] = i
            self.distance[x, y] = c
            bubble_up(self.heap, self.indices, i)


    def __dealloc__(self):
        PyMem_Free(self.heap)

    @boundscheck(False)
    @wraparound(False)
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
        cdef Py_ssize_t x, y, x0, y0
        path = list[tuple[int, int]]()
        x0, y0 = self.get_closest_reachable_point(source, max_distance=max_distance)

        # check that source is within bounds
        if x0 < 0 or y0 < 0 or x0 >= self.distance.shape[0] or y0 >= self.distance.shape[1] or self.cost[x0+1, y0+1] == np.inf:
            return [(x0, y0)]

        while self.size != 0 and self.heap[0].distance < self.distance[x0, y0]:
            # dequeue
            x = self.heap[0].x
            y = self.heap[0].y
            d = self.heap[0].distance
            self.indices[x, y] = -1
            self.size -= 1
            self.heap[0] = self.heap[self.size]
            self.indices[self.heap[0].x, self.heap[0].y] = 0
            bubble_down(self.heap, self.indices, self.size, 0)
            if d > self.distance[x, y]:
                continue
            for k in range(8):
                x2 = x + NEIGHBOURS_X[k]
                y2 = y + NEIGHBOURS_Y[k]
                if self.cost[x2+1, y2+1] == np.inf:
                    continue
                alternative = self.distance[x, y] + NEIGHBOURS_D[k] * self.cost[x2+1, y2+1]
                if alternative < self.distance[x2, y2]:
                    self.distance[x2, y2] = alternative
                    self.forward_x[x2, y2] = x
                    self.forward_y[x2, y2] = y
                    if self.indices[x2, y2] != -1:
                        # decrease_key
                        self.heap[self.indices[x2, y2]].distance = alternative
                        bubble_up(self.heap, self.indices, self.indices[x2, y2])
                    else:
                        # enqueue
                        self.size += 1
                        if self.size > self.capacity:
                            self.capacity *= HEAP_ARITY
                            new_heap = <PriorityQueueItem*>PyMem_Realloc(self.heap, self.capacity * sizeof(PriorityQueueItem))
                            if not new_heap:
                                raise MemoryError()
                            self.heap = new_heap
                        self.heap[self.size - 1] = PriorityQueueItem(x2, y2, alternative)
                        self.indices[x2, y2] = self.size - 1
                        bubble_up(self.heap, self.indices, self.size - 1)

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
            path.append((x, y))
            x, y = self.forward_x[x, y], self.forward_y[x, y]
        return path

    @boundscheck(False)
    @wraparound(False)
    cpdef (Py_ssize_t, Py_ssize_t) get_closest_reachable_point(self, (float, float) source, int max_distance):
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
        cdef Py_ssize_t x0 = <Py_ssize_t>(source[0] + 0.5)
        cdef Py_ssize_t y0 = <Py_ssize_t>(source[1] + 0.5)
        cdef Py_ssize_t x_min = x0
        cdef Py_ssize_t y_min = y0
        cdef DTYPE_t min_distance_squared = np.inf
        cdef DTYPE_t distance_squared = np.inf
        cdef Py_ssize_t x
        cdef Py_ssize_t y

        for x in range(max(0, x0 - max_distance), min(self.distance.shape[0], x0 + max_distance + 1)):
            for y in range(max(0, y0 - max_distance), min(self.distance.shape[1], y0 + max_distance + 1)):
                if self.cost[x+1, y+1] != np.inf:
                    distance_squared = (x - source[0]) ** 2 + (y - source[1]) ** 2
                    if distance_squared < min_distance_squared:
                        x_min = x
                        y_min = y
                        min_distance_squared = distance_squared

        return x_min, y_min


@boundscheck(False)
@wraparound(False)
cpdef DijkstraOutput cy_dijkstra(
    DTYPE_t[:, :] cost,
    Py_ssize_t[:, :] targets,
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