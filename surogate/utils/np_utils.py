import numpy as np


def get_seed(random_state: np.random.RandomState | None = None) -> int:
    if random_state is None:
        random_state = np.random.RandomState()
    seed_max = np.iinfo(np.int32).max
    seed = random_state.randint(0, seed_max)
    return seed
