import judo
from judo import dtype, tensor
import numpy
import pytest

from fragile.distributed.export_swarm import ExportSwarm, ExportedWalkers, ParamServer

from tests.core.test_swarm import create_cartpole_swarm, TestSwarm


class ExportDummy(ExportSwarm):
    def __init__(self, swarm, max_len: int = 20, add_global_best: bool = True, *args, **kwargs):
        super(ExportDummy, self).__init__(swarm=swarm, *args, **kwargs)
        self.param_server = ParamServer(
            max_len=max_len, add_global_best=add_global_best, minimize=self.swarm.walkers.minimize
        )
        self._exchange_next = ExportedWalkers(0)

    def reset(self, *args, **kwargs):
        self.swarm.reset(*args, **kwargs)
        self.param_server.best.update(
            states=self.swarm.walkers.states.best_state,
            id_walkers=self.swarm.walkers.states.best_id,
            rewards=self.swarm.walkers.states.best_reward,
            observs=self.swarm.walkers.states.best_obs,
        )
        self._exchange_next = ExportedWalkers(0)

    def run_step(self):
        walkers = self.run_exchange_step(self._exchange_next)
        self._exchange_next = self.param_server.exchange_walkers(walkers)


@pytest.fixture()
def export_swarm(request=None) -> ExportSwarm:
    from tests.core.test_swarm import create_cartpole_swarm

    params = request.param if request is not None else {}
    return ExportSwarm(swarm=create_cartpole_swarm(), **params)


def create_export_swarm():
    swarm = create_cartpole_swarm()
    return ExportDummy(swarm)


swarm_dict = {
    "export": create_export_swarm,
}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "export": 120,
}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    return swarm_dict.get(request.param)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    swarm = swarm_dict.get(request.param)()
    score = test_scores[request.param]
    return swarm, score


# @pytest.mark.skipif(True, reason="debug")
class TestExportedSwarm:
    swarm_params = (
        {"export_best": False},
        {"import_best": False},
        {"n_import": 3},
        {"n_export": 5},
    )

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_init(self, export_swarm):
        pass

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_create_export_walkers(self, export_swarm):
        indexes = judo.arange(5)
        walkers = export_swarm._create_export_walkers(indexes)
        assert isinstance(walkers, ExportedWalkers)
        assert len(walkers) == 5
        assert (walkers.observs == export_swarm.walkers.env_states.observs[indexes]).all()
        assert (walkers.rewards == export_swarm.walkers.states.cum_rewards[indexes]).all()
        assert (walkers.states == export_swarm.walkers.env_states.states[indexes]).all()
        assert (walkers.id_walkers == export_swarm.walkers.states.id_walkers[indexes]).all()

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_get_export_index(self, export_swarm):
        export_swarm.reset()
        export_swarm.run_step()
        index = export_swarm._get_export_index()
        assert len(index) == export_swarm.n_export
        if export_swarm._export_best:
            best_index = export_swarm.walkers.get_best_index()
            assert best_index in index, (best_index, index)

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_imported_best_is_better(self, export_swarm):
        export_swarm.reset()
        export_swarm.run_step()
        walkers = ExportedWalkers(1)
        walkers.rewards = tensor([1]) * numpy.inf
        new_is_better = export_swarm._imported_best_is_better(walkers)
        assert new_is_better, export_swarm.best_reward
        walkers = ExportedWalkers(1)
        export_swarm.walkers.minimize = True
        walkers.rewards = tensor([1]) * numpy.NINF
        new_is_better = export_swarm._imported_best_is_better(walkers)
        assert new_is_better, export_swarm.best_reward
        export_swarm.walkers.minimize = False

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_get_merge_indexes(self, export_swarm):
        walkers = ExportedWalkers(2)
        local_ix, import_ix = export_swarm._get_merge_indexes(walkers)
        assert len(local_ix) == len(import_ix)
        assert len(local_ix) == export_swarm.n_import

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_cross_fai_iteration(self, export_swarm):
        walkers = ExportedWalkers(export_swarm.n_import)
        local_ix, import_ix = export_swarm._get_merge_indexes(walkers)
        compas_ix, will_clone = export_swarm._cross_fai_iteration(
            local_ix=local_ix, import_ix=import_ix, walkers=walkers
        )
        assert len(compas_ix) == export_swarm.n_import

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_import_best(self, export_swarm):
        walkers = ExportedWalkers(2)
        walkers.rewards = tensor([999.0, 2.0])
        walkers.states = tensor([0.0, 1.0])
        walkers.id_walkers = tensor([10.0, 11.0])
        walkers.observs = tensor([[0, 0, 0, 0], [2, 3, 1, 2]], dtype=dtype.float)
        export_swarm.import_best(walkers)
        assert export_swarm.best_reward == 999
        assert export_swarm.walkers.states.best_state == walkers.states[0]
        assert (export_swarm.walkers.states.best_obs == walkers.observs[0]).all()
        assert export_swarm.walkers.states.best_id == walkers.id_walkers[0]

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_clone_to_imported(self, export_swarm):
        walkers = ExportedWalkers(3)
        walkers.rewards = tensor([999, 777, 333], dtype=dtype.float)
        walkers.states = tensor([999, 777, 333], dtype=dtype.float)
        walkers.id_walkers = tensor([999, 777, 333], dtype=dtype.float)
        walkers.observs = tensor(
            [[999, 999, 999, 999], [777, 777, 777, 777], [333, 333, 333, 333]], dtype=dtype.float
        )

        compas_ix = tensor([0, 1])
        will_clone = tensor([True, False])
        local_ix = tensor([0, 1])
        import_ix = tensor([0, 1])

        export_swarm._clone_to_imported(
            compas_ix=compas_ix,
            will_clone=will_clone,
            local_ix=local_ix,
            import_ix=import_ix,
            walkers=walkers,
        )
        assert export_swarm.walkers.states.cum_rewards[0] == 999.0
        assert export_swarm.walkers.env_states.states[0] == 999.0
        assert (export_swarm.walkers.env_states.observs[0] == judo.ones(4) * 999).all()

    @pytest.mark.parametrize("export_swarm", swarm_params, indirect=True)
    def test_run_exchange_step(self, export_swarm):
        export_swarm.reset()

        walkers_0 = ExportedWalkers(0)
        exported = export_swarm.run_exchange_step(walkers_0)
        assert len(exported) == export_swarm.n_export
        walkers = ExportedWalkers(3)
        walkers.rewards = tensor([999, 777, 333], dtype=dtype.float)
        walkers.states = tensor(
            [[999, 999, 999, 999], [777, 777, 777, 777], [333, 333, 333, 333]], dtype=dtype.float
        )
        walkers.id_walkers = tensor([999, 777, 333], dtype=dtype.float)
        walkers.observs = tensor(
            [[999, 999, 999, 999], [777, 777, 777, 777], [333, 333, 333, 333]], dtype=dtype.float
        )
        export_swarm.reset()
        exported = export_swarm.run_exchange_step(walkers)
        assert len(exported) == export_swarm.n_export
        assert export_swarm.best_reward == 999.0
