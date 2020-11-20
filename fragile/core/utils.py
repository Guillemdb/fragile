import judo
import numpy


NUMPY_IGNORE_WARNINGS_PARAMS = {
    "divide": "ignore",
    "over": "ignore",
    "under": "ignore",
    "invalid": "ignore",
}


def get_plangym_env(swarm: "Swarm") -> "plangym.BaseEnvironment":  # noqa: F821
    """Return the :class:`plangym.Environment` of the target Swarm."""
    from plangym import (
        BaseEnvironment as PlangymEnv,
        ParallelEnvironment as PlangymParallelEnv,
    )
    from fragile import core
    from fragile.distributed import ParallelEnv as FragileParallelEnv, RayEnv

    fragile_env = swarm.env
    if isinstance(fragile_env, (FragileParallelEnv, RayEnv)):
        fragile_env = fragile_env._local_env
    if isinstance(fragile_env, core.DiscreteEnv):
        if not isinstance(fragile_env._env, PlangymEnv):
            raise TypeError("swarm.env needs to represent a `plangym.Environment`.")
    elif not isinstance(fragile_env, core.DiscreteEnv):
        raise TypeError("swarm.env needs to represent a `plangym.Environment`")
    plangym_env = fragile_env._env
    if isinstance(plangym_env, PlangymParallelEnv):
        return plangym_env.plangym_env
    else:
        return plangym_env


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return (
            judo.to_numpy(x).mean(),
            judo.to_numpy(x).std(),
            judo.to_numpy(x).max(),
            judo.to_numpy(x).min(),
        )
    except AttributeError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan
