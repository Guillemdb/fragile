import pytest

from fragile.distributed.distributed_export import BestWalker, DistributedExport
from tests.distributed.ray import init_ray, ray


def create_cartpole_swarm():
    from fragile.core import DiscreteEnv, DiscreteUniform, Swarm
    from plangym import ClassicControl

    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: DiscreteEnv(ClassicControl("CartPole-v0")),
        reward_limit=51,
        n_walkers=50,
        max_epochs=100,
        reward_scale=2,
    )
    return swarm


swarm_types = [create_cartpole_swarm]


def create_distributed_export():
    return DistributedExport(create_cartpole_swarm, n_swarms=2)


swarm_dict = {"export": create_distributed_export}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "export": 50,
}


def kill_swarm(swarm):
    try:
        for e in swarm.swarms:
            e.__ray_terminate__.remote()
        swarm.param_server.__ray_terminate__.remote()
    except AttributeError:
        pass
    ray.shutdown()


class TestExportInterface:
    @pytest.fixture(params=swarm_names, scope="class")
    def swarm_with_score(self, request):
        init_ray()
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        score = test_scores[request.param]
        request.addfinalizer(lambda: kill_swarm(swarm))
        return swarm, score

    def test_get_best(self, swarm_with_score):
        swarm, _ = swarm_with_score
        best_walker = swarm.get_best()
        assert isinstance(best_walker, BestWalker)

    def test_reset_does_not_crash(self, swarm_with_score):
        swarm, _ = swarm_with_score
        swarm.reset()
        assert swarm.epoch == 0

    def test_score_gets_higher(self, swarm_with_score):
        swarm, target_score = swarm_with_score
        swarm.reset()
        swarm.run(report_interval=50)
        reward = swarm.get_best().rewards
        assert reward > target_score, "Iters: {}, rewards: {}".format(
            swarm.walkers.epoch, swarm.walkers.states.cum_rewards
        )
