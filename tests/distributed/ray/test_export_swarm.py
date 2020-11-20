import sys

from judo import dtype, tensor
import pytest

from fragile.distributed.ray.export_swarm import ExportedWalkers, ExportSwarm
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
    return ExportSwarm.remote(create_cartpole_swarm)


swarm_dict = {"export": create_distributed_export}
swarm_names = list(swarm_dict.keys())


def kill_swarm(swarm):
    try:
        swarm.__ray_terminate__.remote()
    except AttributeError:
        pass
    ray.shutdown()


@pytest.fixture(params=swarm_names, scope="class")
def export_swarm(request):
    init_ray()
    swarm = swarm_dict.get(request.param)()
    yield swarm
    kill_swarm(swarm)


# @pytest.mark.skipif(sys.version_info >= (3, 8), reason="Requires python3.7 or lower")
class TestExportInterface:
    def test_reset(self, export_swarm):
        reset = ray.get(export_swarm.reset.remote())
        assert reset is None

    def test_get_data(self, export_swarm):
        states_attr = ray.get(export_swarm.get.remote("cum_rewards"))

        assert dtype.is_tensor(states_attr), (type(states_attr), states_attr)
        env_attr = ray.get(export_swarm.get.remote("observs"))
        assert dtype.is_tensor(env_attr)
        model_attr = ray.get(export_swarm.get.remote("actions"))
        assert dtype.is_tensor(model_attr)
        walkers_attr = ray.get(export_swarm.get.remote("minimize"))
        assert dtype.is_bool(walkers_attr)
        swarm_attr = ray.get(export_swarm.get.remote("n_import"))
        assert dtype.is_int(swarm_attr)

    def test_get_empty_walkers(self, export_swarm):
        walkers = ray.get(export_swarm.get_empty_export_walkers.remote())
        assert isinstance(walkers, ExportedWalkers)
        assert len(walkers) == 0

    def test_run_exchange_step(self, export_swarm):
        empty_walkers = ray.get(export_swarm.get_empty_export_walkers.remote())
        ray.get(export_swarm.run_exchange_step.remote(empty_walkers))

        walkers = ExportedWalkers(3)
        walkers.rewards = tensor([999, 777, 333], dtype=dtype.float)
        walkers.states = tensor(
            [[999, 999, 999, 999], [777, 777, 777, 777], [333, 333, 333, 333]], dtype=dtype.float
        )
        walkers.id_walkers = tensor([999, 777, 333], dtype=dtype.float)
        walkers.observs = tensor(
            [[999, 999, 999, 999], [777, 777, 777, 777], [333, 333, 333, 333]], dtype=dtype.float
        )
        ray.get(export_swarm.reset.remote())
        exported = ray.get(export_swarm.run_exchange_step.remote(walkers))
        best_found = ray.get(export_swarm.get.remote("best_reward"))
        assert len(exported) == ray.get(export_swarm.get.remote("n_export"))
        assert best_found == 999
