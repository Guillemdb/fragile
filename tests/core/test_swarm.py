import judo
from judo import Backend, random_state, tensor
from plangym import AtariEnvironment, ClassicControl
import pytest

from fragile.core.dt_samplers import GaussianDt
from fragile.core.env import BaseEnvironment, DiscreteEnv
from fragile.core.models import BaseModel, DiscreteUniform, NormalContinuous
from fragile.core.states import OneWalker
from fragile.core.swarm import Swarm
from fragile.core.walkers import BaseWalkers, Walkers
from fragile.optimize.benchmarks import Rastrigin
from fragile.optimize.swarm import FunctionMapper


def create_cartpole_swarm():
    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x),
        walkers=Walkers,
        env=lambda: DiscreteEnv(ClassicControl("CartPole-v0")),
        reward_limit=121,
        n_walkers=150,
        max_epochs=300,
        reward_scale=2,
    )
    return swarm


def create_atari_swarm():
    env = AtariEnvironment(name="MsPacman-ram-v0", clone_seeds=True, autoreset=True)
    dt = GaussianDt(min_dt=3, max_dt=100, loc_dt=5, scale_dt=2)
    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x, critic=dt),
        walkers=Walkers,
        env=lambda: DiscreteEnv(env),
        n_walkers=67,
        max_epochs=500,
        reward_scale=2,
        reward_limit=751,
    )
    return swarm


def create_function_swarm():
    env = Rastrigin(dims=2)
    swarm = FunctionMapper(
        model=lambda x: NormalContinuous(bounds=env.bounds),
        env=lambda: env,
        n_walkers=5,
        max_epochs=5,
        prune_tree=True,
        reward_scale=2,
        minimize=False,
    )
    return swarm


swarm_dict = {
    "cartpole": create_cartpole_swarm,
    "atari": create_atari_swarm,
    "function": create_function_swarm,
}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "cartpole": 120,
    "atari": 750,
    "function": 10,
}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    return swarm_dict.get(request.param, create_cartpole_swarm)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
    score = test_scores[request.param]
    return swarm, score


class TestSwarm:
    def test_repr(self, swarm):
        assert isinstance(swarm.__repr__(), str)

    def test_init_not_crashes(self, swarm):
        assert swarm is not None

    def test_env_init(self, swarm):
        assert hasattr(swarm.walkers.states, "will_clone")

    def test_attributes(self, swarm):
        assert isinstance(swarm.env, BaseEnvironment)
        assert isinstance(swarm.model, BaseModel)
        assert isinstance(swarm.walkers, BaseWalkers)

    def test_reset_no_params(self, swarm):
        swarm.reset()

    def test_reset_with_root_walker(self, swarm):
        swarm.reset()
        param_dict = swarm.walkers.env_states.get_params_dict()
        obs_dict = param_dict["observs"]
        state_dict = param_dict["states"]
        obs_size = obs_dict.get("size", obs_dict["shape"][1:])
        state_size = state_dict.get("size", state_dict["shape"][1:])
        obs = judo.astype(random_state.random(obs_size), obs_dict["dtype"])
        state = judo.astype(random_state.random(state_size), state_dict["dtype"])
        reward = 160290
        root_walker = OneWalker(observ=obs, reward=reward, state=state)
        swarm.reset(root_walker=root_walker)
        swarm_best_id = swarm.best_id
        root_walker_id = root_walker.id_walkers
        assert (swarm.best_state == state).all()
        assert (swarm.best_obs == obs).all(), (obs, tensor(swarm.best_obs))
        assert swarm.best_reward == reward
        assert (swarm.walkers.env_states.observs == obs).all()
        assert (swarm.walkers.env_states.states == state).all()
        assert (swarm.walkers.env_states.rewards == reward).all()
        if Backend.is_numpy():
            assert (swarm.walkers.states.id_walkers == root_walker.id_walkers).all()
            assert swarm_best_id == root_walker_id[0]

    def test_step_does_not_crashes(self, swarm):
        swarm.reset()
        swarm.step_walkers()

    def test_score_gets_higher(self, swarm_with_score):
        swarm, target_score = swarm_with_score
        swarm.walkers.seed(160290)
        swarm.reset()
        swarm.run()
        reward = (
            swarm.get("cum_rewards").min()
            if swarm.walkers.minimize
            else swarm.get("cum_rewards").max()
        )
        assert (
            reward <= target_score if swarm.walkers.minimize else reward >= target_score
        ), "Iters: {}, rewards: {}".format(swarm.walkers.epoch, swarm.walkers.states.cum_rewards)
