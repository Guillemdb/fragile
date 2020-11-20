from typing import Callable

import judo
from judo import Backend, Bounds, functions, tensor
import pytest

from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.benchmarks import sphere
from fragile.optimize.env import Function, MinimizerWrapper
from tests.core.test_env import TestEnvironment

N_WALKERS = 50

# Backend.set_backend("numpy")
def function() -> Function:
    return Function.from_bounds_params(
        function=sphere, shape=(2,), low=tensor([-10, -5]), high=tensor([10, 5]),
    )


def local_minimizer():
    bounds = Bounds(shape=(2,), high=10, low=-5, dtype=judo.float)
    env = Function(function=sphere, bounds=bounds)
    return MinimizerWrapper(env)


def custom_domain_function():
    bounds = Bounds(shape=(2,), high=10, low=-5, dtype=judo.float)
    env = Function(
        function=sphere, bounds=bounds, custom_domain_check=lambda x, *args: judo.norm(x, 1) < 5.0
    )
    return env


def create_env_and_model_states(name="classic") -> Callable:
    def _function():
        env = function()
        params = {"actions": {"dtype": judo.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _local_minimizer():
        env = local_minimizer()
        params = {"actions": {"dtype": judo.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _custom_domain_function():
        env = custom_domain_function()
        params = {"actions": {"dtype": judo.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    if name.lower() == "function":
        return _function
    elif name.lower() == "custom_domain_function":
        return _custom_domain_function
    elif name.lower() == "local_minimizer":
        return _local_minimizer


env_fixture_params = ["function", "custom_domain_function", "local_minimizer"]


@pytest.fixture(params=env_fixture_params, scope="class")
def env_data(request):
    if request.param in env_fixture_params:
        env, model_states = create_env_and_model_states(request.param)()

    else:
        raise ValueError("Environment not well defined: %s" % request.param)
    return env, model_states


@pytest.fixture(scope="class")
def function_env() -> Function:
    return Function.from_bounds_params(
        function=lambda x: judo.ones(len(x)),
        shape=(2,),
        low=tensor([-10, -5]),
        high=tensor([10, 5]),
    )


@pytest.fixture(scope="class")
def batch_size():
    return N_WALKERS


@pytest.fixture(scope="class")
def states_dim():
    return 2


class TestFunction:
    def test_init_error(self):
        with pytest.raises(TypeError):
            Function(function=sphere, bounds=(True, False))

    def test_from_bounds_params_error(self):
        with pytest.raises(TypeError):
            Function.from_bounds_params(function=sphere)

    @pytest.mark.parametrize("batch_s", [1, 10])
    def test_reset_batch_size(self, function_env, batch_s):
        new_states: StatesEnv = function_env.reset(batch_size=batch_s)
        assert isinstance(new_states, StatesEnv)
        """assert not (new_states.observs == 0).all().item()
        assert (new_states.rewards == 1).all().item(), (
            new_states.rewards,
            new_states.rewards.shape,
        )"""
        assert (new_states.oobs == 0).all().item()
        assert len(new_states.rewards.shape) == 1
        assert new_states.rewards.shape[0] == batch_s
        assert new_states.oobs.shape[0] == batch_s
        assert new_states.observs.shape[0] == batch_s
        assert new_states.observs.shape[1] == 2

    def test_step(self, function_env, batch_size):
        states = function_env.reset(batch_size=batch_size)
        actions = StatesModel(
            actions=judo.zeros(states.observs.shape), batch_size=batch_size, dt=judo.ones((1, 2)),
        )
        new_states: StatesEnv = function_env.step(actions, states)
        assert isinstance(new_states, StatesEnv)
        assert new_states.oobs[0].item() == 0

    def test_minimizer_getattr(self):
        bounds = Bounds(shape=(2,), high=10, low=-5, dtype=float)
        env = Function(function=sphere, bounds=bounds)
        minim = MinimizerWrapper(env)
        assert minim.shape == env.shape

    @pytest.mark.skipif(not Backend.is_numpy(), reason="only in numpy for now")
    def test_minimizer_step(self):
        minim = local_minimizer()
        params = {"actions": {"dtype": judo.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        assert minim.shape == minim.shape
        states = minim.step(model_states=states, env_states=minim.reset(N_WALKERS))
        assert judo.allclose(states.rewards.min(), 0)
