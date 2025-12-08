import numpy as np
from cython import boundscheck, wraparound

cimport numpy as cnp
from libc.math cimport sqrt
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

DEF DEFAULT_HEAP_ARITY = 4

ctypedef cnp.float64_t DTYPE_t

cdef Py_ssize_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
cdef Py_ssize_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
cdef DTYPE_t SQRT2 = np.sqrt(2)
cdef DTYPE_t SQRT2_MINUS_1 = SQRT2 - 1.0
cdef DTYPE_t[8] NEIGHBOURS_D = [1, 1, 1, 1, SQRT2, SQRT2, SQRT2, SQRT2]

cdef inline DTYPE_t min_c(DTYPE_t a, DTYPE_t b):
    return a if a < b else b

cdef struct PriorityQueueItem:
    Py_ssize_t x
    Py_ssize_t y
    DTYPE_t distance
    DTYPE_t distance_lookahead

cdef bint compare(PriorityQueueItem a, PriorityQueueItem b):
    return a.distance < b.distance or (a.distance == b.distance and a.distance_lookahead < b.distance_lookahead)

@boundscheck(False)
@wraparound(False)
cdef class DijkstraPathing:
    cdef public DTYPE_t[:, :] cost
    cdef public Py_ssize_t[:, :] targets
    cdef public Py_ssize_t[:, :] forward_x
    cdef public Py_ssize_t[:, :] forward_y
    cdef public DTYPE_t[:, :] distance
    cdef PriorityQueueItem* heap
    cdef Py_ssize_t arity
    cdef Py_ssize_t capacity
    cdef Py_ssize_t size
    cdef Py_ssize_t[:, :] indirection
    cdef public DTYPE_t[:, :] distance_lookahead
    cdef Py_ssize_t neighbour_count
    """Priority queue."""
    @boundscheck(False)
    @wraparound(False)
    def __cinit__(self,
                  DTYPE_t[:, :] cost,
                  Py_ssize_t[:, :] targets,
                  Py_ssize_t arity = DEFAULT_HEAP_ARITY,
                  bint include_diagonals = True):
        cdef:
            Py_ssize_t i
            Py_ssize_t x
            Py_ssize_t y

        self.cost = np.pad(cost, 1, "constant", constant_values=np.inf)
        self.targets = targets

        self.capacity = 128
        self.heap = <PriorityQueueItem*>PyMem_Malloc(self.capacity * sizeof(PriorityQueueItem))
        if not self.heap:
            raise MemoryError()
        self.size = 0
        self.arity = arity
        self.indirection = np.full_like(cost, -1, np.intp)
        self.distance = np.full_like(cost, np.inf, np.float64)
        self.distance_lookahead = np.full_like(cost, np.inf, np.float64)
        self.forward_x = np.full_like(cost, -1, np.intp)
        self.forward_y = np.full_like(cost, -1, np.intp)
        self.neighbour_count = 8 if include_diagonals else 4

        for i in range(self.targets.shape[0]):
            x = self.targets[i, 0]
            y = self.targets[i, 1]
            self.distance_lookahead[x, y] = 0.0
            self.enqueue(PriorityQueueItem(x, y, 0.0, 0.0))

    def __dealloc__(self):
        PyMem_Free(self.heap)

    @boundscheck(False)
    @wraparound(False)
    cdef DTYPE_t distance_key(self, Py_ssize_t x, Py_ssize_t y):
        return min_c(self.distance[x, y], self.distance_lookahead[x, y])

    @boundscheck(False)
    @wraparound(False)
    cdef DTYPE_t heuristic_euclidean(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t x0, Py_ssize_t y0):
        return sqrt((x - x0) ** 2 + (y - y0) ** 2)

    @boundscheck(False)
    @wraparound(False)
    cdef DTYPE_t heuristic_octile(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t x0, Py_ssize_t y0):
        # Euclidean is slow. Use Octile distance for 8-connected grids.
        cdef DTYPE_t dx = abs(x - x0)
        cdef DTYPE_t dy = abs(y - y0)
        if dx > dy:
            return SQRT2_MINUS_1 * dy + dx
        else:
            return SQRT2_MINUS_1 * dx + dy

    @boundscheck(False)
    @wraparound(False)
    cdef update(self, Py_ssize_t x, Py_ssize_t y, Py_ssize_t x0, Py_ssize_t y0):
        cdef:
            DTYPE_t alternative
            DTYPE_t best
            DTYPE_t d
            DTYPE_t h
            Py_ssize_t x2
            Py_ssize_t y2
            Py_ssize_t xb
            Py_ssize_t yb
            cdef bint go_up
        if self.cost[x+1, y+1] == np.inf:
            return
        if self.distance_lookahead[x, y] != 0.0:
            best = np.inf
            for k in range(self.neighbour_count):
                x2 = x + NEIGHBOURS_X[k]
                y2 = y + NEIGHBOURS_Y[k]
                if self.cost[x2+1, y2+1] == np.inf:
                    continue
                alternative = self.distance[x2, y2] + NEIGHBOURS_D[k] * self.cost[x2+1, y2+1]
                if alternative < best:
                    best = alternative
                    xb = x2
                    yb = y2
            self.distance_lookahead[x, y] = best
            self.forward_x[x, y] = xb
            self.forward_y[x, y] = yb


        # Check consistency
        d = self.distance_key(x, y)
        h = self.heuristic_octile(x, y, x0, y0)
        if self.distance[x, y] != self.distance_lookahead[x, y]:
            # Node is inconsistent, should be in heap
            index = self.indirection[x, y]
            item = PriorityQueueItem(x, y, d + h, d)
            if index != -1:
                go_up = compare(item, self.heap[index])
                self.set_item(index, item)
                if go_up:
                    self.bubble_up(index)
                else:
                    self.bubble_down(index)
            else:
                self.enqueue(item)
        else:
            # Node is consistent, should NOT be in heap
            self.delete_item(x, y)

    @boundscheck(False)
    @wraparound(False)
    cdef compute_shortest_path(self, Py_ssize_t x, Py_ssize_t y):
        cdef:
            PriorityQueueItem item
            DTYPE_t d
            DTYPE_t h
        while self.size > 0:
            d = self.distance_key(x, y)
            item = self.heap[0]
            if not (item.distance < d or (item.distance == d and item.distance_lookahead < d)) and self.distance[x, y] == self.distance_lookahead[x, y]:
                break
            item = self.dequeue()
            if self.distance[item.x, item.y] > self.distance_lookahead[item.x, item.y]:
                self.distance[item.x, item.y] = self.distance_lookahead[item.x, item.y]
                for k in range(self.neighbour_count):
                    self.update(item.x + NEIGHBOURS_X[k], item.y + NEIGHBOURS_Y[k], x, y)
            else:
                self.distance[item.x, item.y] = np.inf
                self.update(item.x, item.y, x, y)
                for k in range(self.neighbour_count):
                    self.update(item.x + NEIGHBOURS_X[k], item.y + NEIGHBOURS_Y[k], x, y)

    @boundscheck(False)
    @wraparound(False)
    cdef swap(self, Py_ssize_t i, Py_ssize_t j):
        self.indirection[self.heap[i].x, self.heap[i].y] = j
        self.indirection[self.heap[j].x, self.heap[j].y] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    @boundscheck(False)
    @wraparound(False)
    cdef void delete_item(self, Py_ssize_t x, Py_ssize_t y):
        cdef index = self.indirection[x, y]
        if index != -1:
            self.heap[index].distance = -np.inf
            self.heap[index].distance_lookahead = -np.inf
            self.bubble_up(index)
            self.dequeue()

    @boundscheck(False)
    @wraparound(False)
    cdef void set_item(self, Py_ssize_t index, PriorityQueueItem item):
        self.heap[index] = item
        self.indirection[item.x, item.y] = index

    @boundscheck(False)
    @wraparound(False)
    cdef void bubble_up(self, Py_ssize_t index):
        cdef Py_ssize_t parent
        while index != 0:
            parent = (index - 1) // self.arity
            if compare(self.heap[index], self.heap[parent]):
                self.swap(index, parent)
                index = parent
            else:
                break

    @boundscheck(False)
    @wraparound(False)
    cdef void bubble_down(self, Py_ssize_t index):
        cdef Py_ssize_t child
        cdef Py_ssize_t swap_idx
        cdef Py_ssize_t i

        while True:
            swap_idx = index
            i = self.arity * index + 1
            # Optimized loop bounds calculation
            if i >= self.size:
                break

            for child in range(i, min(i + self.arity, self.size)):
                if compare(self.heap[child], self.heap[swap_idx]):
                    swap_idx = child

            if swap_idx != index:
                self.swap(index, swap_idx)
                index = swap_idx
            else:
                break

    @boundscheck(False)
    @wraparound(False)
    cdef PriorityQueueItem dequeue(self):
        cdef PriorityQueueItem item = self.heap[0]
        self.size -= 1
        self.set_item(0, self.heap[self.size])
        self.indirection[item.x, item.y] = -1
        self.bubble_down(0)
        return item

    @boundscheck(False)
    @wraparound(False)
    cdef enqueue(self, PriorityQueueItem item):
        cdef:
            Py_ssize_t index = self.size
            PriorityQueueItem* new_heap
        self.size += 1
        if self.size > self.capacity:
            self.capacity *= self.arity
            new_heap = <PriorityQueueItem*>PyMem_Realloc(self.heap, self.capacity * sizeof(PriorityQueueItem))
            if new_heap:
                self.heap = new_heap
            else:
                raise MemoryError()
        self.set_item(index, item)
        self.bubble_up(index)

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

        Returns
        -------
        list[tuple[int, int]] :
            The lowest cost path from source to any of the targets.

        """
        cdef:
            DTYPE_t cost
            DTYPE_t alternative
            Py_ssize_t x2
            Py_ssize_t y2
        path = list[tuple[int, int]]()
        x, y = self.get_closest_reachable_point(source, max_distance=max_distance)

        # check that source is within bounds
        if x < 0 or y < 0 or x >= self.distance.shape[0] or y >= self.distance.shape[1]:
            return [(x, y)]

        if self.cost[x+1, y+1] == np.inf:
            return [(x, y)]

        self.compute_shortest_path(x, y)
        if limit == 0:
            # set a fallback limit to be safe
            # a path longer than this must contain a cycle, so it should never be hit anyway
            limit = self.distance.shape[0] * self.distance.shape[1]

        while len(path) < limit:
            if x < 0 or y < 0:
                break
            path.append((x, y))
            x, y = self.forward_x[x, y], self.forward_y[x, y]
        return path

    @boundscheck(False)
    @wraparound(False)
    cpdef (int, int) get_closest_reachable_point(self, (float, float) source, int max_distance):
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
        x0 = int(source[0])
        y0 = int(source[1])

        x_min = x0
        y_min = y0
        min_distance_squared = np.inf

        for x in range(max(0, x0 - max_distance), min(self.distance.shape[0], x0 + max_distance + 1)):
            for y in range(max(0, y0 - max_distance), min(self.distance.shape[1], y0 + max_distance + 1)):
                if self.cost[x+1, y+1] < np.inf:
                    distance_squared = (x - source[0]) ** 2 + (y - source[1]) ** 2
                    if distance_squared < min_distance_squared:
                        x_min = x
                        y_min = y
                        min_distance_squared = distance_squared

        return x_min, y_min


@boundscheck(False)
@wraparound(False)
cpdef DijkstraPathing cy_dijkstra(
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
        Pathfinding object containing containing distance and forward pointer grids.

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

    return DijkstraPathing(cost, targets)