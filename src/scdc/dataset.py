from __future__ import annotations

import functools
import typing as t

import numpy as np
import numpy.typing as npt
from scipy import integrate

NBit = t.TypeVar("NBit", bound=npt.NBitBase)


class DecayCurve(t.NamedTuple):
    t: npt.NDArray[np.floating[NBit]]
    y: npt.NDArray[np.floating[NBit]]
    tau: float
    t0: float
    y0: float


def create_decay_curve(
    tau: float,
    t0: float,
    y0: float,
    *,
    size: int = 500,
    sigma: float = 1e-2,
    k: int = 2,
) -> DecayCurve:
    time = np.linspace(-1.0, 2.0, (size - 1) * 3 + 1)
    mu = 0.2
    input_fn = functools.partial(_pulse, mu=mu, sigma=sigma)
    sol = integrate.solve_ivp(
        lambda t, y: -y / tau + input_fn(t),
        t_span=(0.0, time.max()),
        y0=[0.0],
        method="Radau",
        t_eval=time[time >= 0.0],
    )
    y = np.hstack((np.zeros(time.size - sol.y[0].size), sol.y[0]))
    y = (1.0 - y0) * y / max(y) + y0
    y /= max(y)
    time += t0 - (mu - k * sigma)
    data = DecayCurve(
        t=np.linspace(0.0, 1.0, size),
        y=y[time >= 0.0][:size],
        tau=round(tau, 4),
        t0=round(t0, 4),
        y0=round(y0, 4),
    )
    return data


def _pulse(t: float, mu: float, sigma: float) -> float:
    return np.exp(-(((t - mu) / sigma) ** 2))


def add_noise(
    data: DecayCurve,
    random_state: np.random.RandomState,
    noise_ratio: float = 0.05,
) -> DecayCurve:
    noise = random_state.normal(scale=noise_ratio, size=data.y.size)
    return DecayCurve(
        t=data.t.copy(),
        y=data.y + noise,
        tau=data.tau,
        t0=data.t0,
        y0=data.y0,
    )
