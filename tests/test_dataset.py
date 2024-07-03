import numpy as np
import pytest
from scdc import dataset


@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("t0", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("y0", [0.1, 0.2, 0.3])
def test_create_decay_curve(
    tau: float,
    t0: float,
    y0: float,
) -> None:
    data = dataset.create_decay_curve(tau, t0, y0)
    assert data.tau == round(tau, 4)
    assert data.t0 == round(t0, 4)
    assert data.y0 == round(y0, 4)
    assert data.y.max() == 1.0
    assert data.t[data.y.argmax()] > t0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("tau", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("t0", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("y0", [0.1, 0.2, 0.3])
def test_add_noise(
    seed: int,
    tau: float,
    t0: float,
    y0: float,
) -> None:
    random_state = np.random.RandomState(seed)
    data = dataset.create_decay_curve(tau, t0, y0)
    data_with_noise = dataset.add_noise(data, random_state)
    assert data.y.size == data_with_noise.y.size
    assert not (data.y == data_with_noise.y).all()
    assert (data.t == data_with_noise.t).all()
    assert data.tau == data_with_noise.tau
    assert data.t0 == data_with_noise.t0
    assert data.y0 == data_with_noise.y0
