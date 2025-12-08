from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_equal
from sc2.bot_ai import BotAI

from cython_extensions import cy_dijkstra

pytest_plugins = ("pytest_asyncio",)

MAPS: list[Path] = [
    map_path
    for map_path in (Path(__file__).parent / "combat_data").iterdir()
    if map_path.suffix == ".xz"
]


class TestDijkstraGeneric:

    def test_raises_on_nonpositive_entries(self):
        targets = np.array([[0, 0]])
        cost = np.ones((3, 3))
        cost[1, 1] = -1
        with pytest.raises(Exception):
            cy_dijkstra(cost, targets)

    def test_raises_on_target_out_of_bounds(self):
        cost = np.ones((3, 3))
        with pytest.raises(Exception):
            cy_dijkstra(cost, np.array([[1, -1]]))
        with pytest.raises(Exception):
            cy_dijkstra(cost, np.array([[-1, 1]]))
        with pytest.raises(Exception):
            cy_dijkstra(cost, np.array([[1, 3]]))
        with pytest.raises(Exception):
            cy_dijkstra(cost, np.array([[3, 1]]))


class TestDijkstraBasic:
    def test_maze(self):
        cost = np.array([
            [1, 1, np.inf, 1, 1],
            [1, 1, np.inf, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, np.inf, 1, 1],
            [1, 1, np.inf, 1, 1],
        ]).T
        targets = np.array([[4, 0]])
        pathing = cy_dijkstra(cost, targets)
        print("#1")
        assert_equal(
            pathing.get_path((0, 0)),
            [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)],
        )
        print("#2")
        assert_equal(
            pathing.get_path((0, 2)),
            [(0, 2), (1, 2), (2, 2), (3, 1), (4, 0)],
        )
        print("#3")
        assert_equal(
            pathing.get_path((0, 4)),
            [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
        )

    def test_heuristic(self):
        cost = np.array([
            [1, 1, 1, 1, 1],
        ]).astype(np.float64).T
        targets = np.array([[4, 0]])
        pathing = cy_dijkstra(cost, targets)
        print("#1")
        assert_equal(
            pathing.get_path((3, 0)),
            [(3, 0), (4, 0)],
        )
        print("#2")
        assert_equal(
            pathing.get_path((0, 0)),
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        )

    def test_find_valid_start(self):
        # test that the algorithm searches for valid starting point near the given coordinates
        # setup a rectangular map surrounded by unpathable border
        cost = np.pad(np.ones((3, 3)), 1, constant_values=np.inf)
        # find paths towards center
        pathing = cy_dijkstra(cost, np.array([[2, 2]]))
        # test that starting point will be rounded
        assert_equal(pathing.get_path((1.1, 1.3)), [(1, 1), (2, 2)])
        assert_equal(pathing.get_path((0.8, 0.7)), [(1, 1), (2, 2)])
        # test that invalid start snaps to the closest valid one
        assert_equal(pathing.get_path((0.1, 2)), [(1, 2), (2, 2)])
        assert_equal(pathing.get_path((-2.2, 6.9), max_distance=5), [(1, 3), (2, 2)])


@pytest.mark.parametrize("bot", MAPS, indirect=True)
class TestDijkstra:
    scenarios = [(map_path.name, {"map_path": map_path}) for map_path in MAPS]

    def test_pathing(self, bot: BotAI, event_loop):
        cost = np.where(
            bot.game_info.pathing_grid.data_numpy.T == 1, 1.0, np.inf
        ).astype(np.float64)
        targets = np.array([u.position.rounded for u in bot.enemy_units if bot.in_pathing_grid(u)], np.intp)
        pathing = cy_dijkstra(cost, targets)

        for unit in bot.units:
            p = unit.position.rounded
            if cost[p] == np.inf:
                continue
            path = pathing.get_path(p)
            assert 1 <= len(path)
            if len(path) == 1:
                assert pathing.distance[path[0]] == np.inf
            else:
                # test that the distance was calculated correctly along the path
                path_backwards = path[::-1]
                dist_expected = 0.0
                for i, (p, q) in enumerate(zip(path_backwards[1:], path_backwards)):
                    dist_factor = np.linalg.norm(np.array(p) - np.array(q))
                    dist_expected += cost[p] * dist_factor
                    assert pathing.distance[p] == dist_expected
