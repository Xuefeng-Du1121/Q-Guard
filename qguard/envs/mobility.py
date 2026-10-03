from __future__ import annotations

import numpy as np


def random_walk_positions(
    positions: np.ndarray,
    speed: float,
    grid_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if positions.size == 0:
        return positions

    angles = rng.uniform(0.0, 2.0 * np.pi, size=positions.shape[0])
    delta = np.column_stack((np.cos(angles), np.sin(angles))) * speed
    moved = positions + delta
    return np.clip(moved, 0.0, grid_size)


def manhattan_grid_positions(
    count: int,
    grid_size: float,
    rng: np.random.Generator,
    roads_per_axis: int = 8,
) -> np.ndarray:
    roads = np.linspace(0.0, grid_size, roads_per_axis)
    positions = np.zeros((count, 2), dtype=float)
    for idx in range(count):
        if rng.random() < 0.5:
            positions[idx, 0] = rng.choice(roads)
            positions[idx, 1] = rng.uniform(0.0, grid_size)
        else:
            positions[idx, 0] = rng.uniform(0.0, grid_size)
            positions[idx, 1] = rng.choice(roads)
    return positions
