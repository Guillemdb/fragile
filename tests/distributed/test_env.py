import sys
from typing import Callable, Tuple

import judo
from judo import dtype
import pytest

from fragile.core.env import Environment as CoreEnv
from fragile.core.states import StatesModel
from fragile.distributed.env import ParallelEnv, RayEnv
from fragile.optimize.benchmarks import Rastrigin
from tests.core.test_env import discrete_atari_env, classic_control_env, TestEnvironment
from tests.distributed.ray import init_ray, ray
from tests.optimize.test_env import local_minimizer  # TestFunction, local_minimizer

N_WALKERS = 10


@pytest.fixture()
def batch_size():
    return N_WALKERS


def ray_env():
    env = RayEnv(classic_control_env, n_workers=1)
    return env


def ray_function():
    return RayEnv(local_minimizer, n_workers=2)


def parallel_environment():
    return ParallelEnv(discrete_atari_env, n_workers=2, blocking=True)


def parallel_function():
    return ParallelEnv(env_callable=lambda: Rastrigin(dims=2), n_workers=2)


def create_env_and_model_states(name="classic") -> Callable:
    def _ray_env():
        init_ray()
        env = ray_env()
        params = {"actions": {"dtype": dtype.int64}, "critic": {"dtype": dtype.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _ray_function():
        init_ray()
        env = ray_function()
        params = {"actions": {"dtype": dtype.int64}, "critic": {"dtype": dtype.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _parallel_function():
        env = parallel_function()
        params = {
            "actions": {"dtype": dtype.float32, "size": (2,)},
            "critic": {"dtype": dtype.float32},
        }
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _parallel_environment():
        env = parallel_environment()
        params = {"actions": {"dtype": dtype.int64}, "critic": {"dtype": dtype.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=judo.ones(N_WALKERS), critic=judo.ones(N_WALKERS))
        return env, states

    if name.lower() == "ray_env":
        return _ray_env
    elif name.lower() == "ray_function":
        return _ray_function
    elif name.lower() == "parallel_function":
        return _parallel_function
    elif name.lower() == "parallel_environment":
        return _parallel_environment


env_fixture_params = (
    ["ray_env"] if judo.Backend.can_use_cuda() else ["ray_env", "parallel_environment"]
)


@pytest.fixture(params=env_fixture_params, scope="class")
def env_data(request) -> Tuple[CoreEnv, StatesModel]:
    if request.param in env_fixture_params:
        env, model_states = create_env_and_model_states(request.param)()
    else:
        raise ValueError("Environment not well defined: %s" % request.param)
    yield env, model_states
    if "ray" in request.param:

        def kill_ray_env():
            try:
                for e in env.envs:
                    e.__ray_terminate__.remote()
            except AttributeError:
                pass
            ray.shutdown()

        kill_ray_env()

    elif "parallel" in request.param:

        def kill_parallel_env():
            env.close()

        kill_parallel_env()


ray_function_fixture_params = ["ray_function"]


@pytest.fixture(params=ray_function_fixture_params)
def function_env(request):
    return create_env_and_model_states(request.param)()[0]


@pytest.fixture(scope="class")
def states_dim():
    return 2
