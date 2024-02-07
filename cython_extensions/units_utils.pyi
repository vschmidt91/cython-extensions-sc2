from typing import Union

from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

def cy_center(units: Union[Units, list[Unit]]) -> tuple[float, float]:
    """
    54.2 µs ± 137 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    `python-sc2`'s `units.center` alternative:
    107 µs ± 255 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    Example:
    ```py
    from ares.cython_functions.units_utils import cy_center

    centroid: Tuple[float, float] = cy_center(self.workers)

    # centroid_point2 = Point2(centroid)
    ```

    Parameters
    ----------
    units :

    Returns
    -------
    tuple[float, float] :
        Centroid of all units positions
    """
    ...

def cy_closest_to(
    position: Union[Point2, tuple[float, float]], units: Union[Units, list[Unit]]
) -> Unit:
    """Iterate through `units` to find closest to `position`.

    Example:
    ```py
    from ares.cython_functions.units_utils import cy_closest_to

    closest_unit = cy_closest_to(self.start_location, self.workers)
    ```

    14.3 µs ± 135 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

    python-sc2's `units.closest_to()` alternative:
    98.9 µs ± 240 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    If using `units.closest_to(Point2):
    200 µs ± 1.02 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

    Parameters
    ----------
    position :
        Position to measure distance from.
    units :
        Collection of units we want to check.

    Returns
    -------
    Unit :
        Unit closest to `position`.

    """
    ...

def cy_in_attack_range(
    unit: Unit, units: Union[Units, list[Unit]], bonus_distance: float = 0.0
) -> list[Unit]:
    """Find all units that unit can shoot at.

    Doesn't check if the unit weapon is ready. See:
    `ares.cython_functions.combat_utils.attack_ready`

    Example:
    ```py
    from ares.cython_functions.units_utils import cy_in_attack_range

    in_attack_range = cy_in_attack_range(self.workers[0], self.enemy_units)
    ```

    7.28 µs ± 26.3 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

    python-sc2's `units.in_attack_range_of(unit)` alternative:
    30.4 µs ± 271 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    Parameters
    ----------
    unit :
        Position to measure distance from.
    units :
        Collection of units we want to check.
    bonus_distance :

    Returns
    -------
    list[Unit] :
        Units that are in attack range of `unit`.

    """
    ...

def cy_sorted_by_distance_to(
    units: Union[Units, list[Unit]], position: Point2, reverse: bool = False
) -> list[Unit]:
    """Sort units by distance to `position`

    Example:
    ```py
    from ares.cython_functions.units_utils import cy_sorted_by_distance_to

    sorted_by_distance = cy_sorted_by_distance_to(self.workers, self.start_location)
    ```

    33.7 µs ± 190 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    python-sc2's `units.sorted_by_distance_to(position)` alternative:
    246 µs ± 830 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

    Parameters
    ----------
    units :
        Units we want to sort.
    position :
        Sort by distance to this position.
    reverse :
        Not currently used.
    Returns
    -------
    list[Unit] :
        Units sorted by distance to position.

    """
    ...
