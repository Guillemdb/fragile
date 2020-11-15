import judo
from judo import Bounds
import pytest

from fragile.optimize import FunctionMapper
from fragile.optimize.models import NormalContinuous


@pytest.fixture(scope="class")
def mapper():
    def potential_well(x):
        return judo.sum((x - 1) ** 2, 1) - 1

    bounds = Bounds.from_tuples([(-10, 10), (-5, 5)])

    def model(x):
        return NormalContinuous(bounds=bounds, env=x)

    return FunctionMapper.from_function(
        n_vectors=5,
        function=potential_well,
        bounds=bounds,
        model=model,
        shape=(2,),
        n_walkers=10,
        reward_scale=1,
        accumulate_rewards=False,
    )


@pytest.fixture(scope="class")
def finished_swarm(mapper):
    mapper.walkers.reset()
    mapper.reset()
    mapper.walkers.max_epochs = 500
    mapper.run()
    return mapper


class TestFunctionMapper:
    def test_from_function(self, mapper):
        pass

    def test_score_gets_higher(self, finished_swarm):
        reward = finished_swarm.walkers.states.cum_rewards.max().item()
        assert reward <= 60, "Iters: {}, rewards: {}".format(
            finished_swarm.walkers.epoch, finished_swarm.walkers.cum_rewards
        )

    def test_start_same_pos(self, mapper):
        mapper.start_same_pos = True
        mapper.reset()
        assert (mapper.walkers.env_states.observs == mapper.walkers.env_states.observs[0]).all()
