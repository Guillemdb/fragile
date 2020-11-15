from judo import Backend, Bounds
import numpy as np
import pytest

from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.models import CMAES, ESModel

from tests.core.test_model import TestModel

BATCH_SIZE = 10


def create_model(name="es_model"):
    if name == "es_model":
        bs = Bounds(low=-10, high=10, shape=(BATCH_SIZE,))
        return lambda: ESModel(bounds=bs)
    raise ValueError("Invalid param `name`.")


model_fixture_params = ["es_model"]


@pytest.fixture(scope="class", params=model_fixture_params)
def model(request):
    prev_backend = Backend.get_current_backend()
    Backend.set_backend("numpy")
    yield create_model(request.param)()
    Backend.set_backend(prev_backend)


@pytest.fixture(scope="class")
def batch_size():
    return BATCH_SIZE


@pytest.fixture()
def cmaes():
    bs = Bounds(low=-10, high=10, shape=(5,))
    return CMAES(bounds=bs, sigma=0.3)


class DeterministicCMAES(CMAES):
    def reset(self, init_xmean, batch_size, model_states, noise, *args, **kwargs):
        self.pop_size = batch_size
        self._init_algorithm_params(batch_size)
        # Take the first sample from a random uniform distribution
        model_states = super(CMAES, self).reset(
            batch_size=batch_size, model_states=model_states, *args, **kwargs
        )
        self.x_mean = init_xmean
        actions = self._sample_actions(noise)
        model_states.update(actions=actions)
        return model_states

    def _sample_actions(self, noise):
        actions = np.empty((self.pop_size, self.n_dims))
        for i in range(self.pop_size):
            action = self.x_mean + self.sigma * np.matmul(
                self.coords_matrix, self.scaling_diag * noise[i]
            )
            actions[i, :] = action.flatten()
            self._count_eval += 1
        return actions


@pytest.fixture()
def det_cmaes():
    bs = Bounds(low=-10, high=10, shape=(5,))
    return DeterministicCMAES(bounds=bs, sigma=0.3)


class TestESModel:
    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def create_model_states(self, model, batch_size: int = None):
        return StatesModel(batch_size=batch_size, state_dict=model.get_params_dict())

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def create_env_states(self, model, batch_size: int = None):
        return StatesEnv(batch_size=batch_size, state_dict=model.get_params_dict())

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_run_for_1000_predictions(self, model, batch_size):
        for _ in range(100):
            TestModel.test_predict(self, model, batch_size)


init_xmean = np.array([0.66223, 0.44078, 0.76151, 0.12057, 0.38336,]).reshape(-1, 1)
noise_iter_1 = [
    np.array([-0.37583, 1.42318, -0.66132, 0.26609, 1.11349]).reshape(-1, 1),
    np.array([1.3881295, -0.7973751, 0.0091523, 1.3019880, -0.8187276]).reshape(-1, 1),
    np.array([-0.29358, -0.25445, -0.22385, -0.72843, 1.19903]).reshape(-1, 1),
    np.array([0.66587, 0.95149, 0.62977, -0.73383, 0.18110]).reshape(-1, 1),
    np.array([1.11411, 1.22264, -1.98069, 0.18543, -1.23395]).reshape(-1, 1),
    np.array([0.21677, -1.15239, 0.25152, -0.37585, 0.52998]).reshape(-1, 1),
    np.array([2.37239, -1.44102, -0.82046, -0.16501, 0.43072]).reshape(-1, 1),
    np.array([-0.47678, 0.92454, 1.32925, -0.70538, 0.97477]).reshape(-1, 1),
]
fitness_iter_1 = np.array([83.896, 147.895, 120.984, 138.110, 30.584, 166.663, 411.474, 314.332,])
xmean_iter_1 = np.array([0.80284, 0.75808, 0.38818, 0.13250, 0.33627]).reshape(-1, 1)
ps_iter_1 = np.array([0.583924, 1.317643, -1.550294, 0.049556, -0.195543]).reshape(-1, 1)
pc_iter_1 = np.array([0.631321, 1.424597, -1.676132, 0.053579, -0.211416]).reshape(-1, 1)
hsig_iter_1 = 1
actions_iter_1 = np.vstack(
    [
        np.array(
            [
                0.5494822,
                1.0786686,
                0.5741569,
                0.8619898,
                0.9964626,
                0.7272622,
                1.3739453,
                0.5191962,
            ]
        ),
        np.array(
            [
                0.8677339,
                0.2015686,
                0.3644450,
                0.7262285,
                0.8075726,
                0.0950655,
                0.0084765,
                0.7181422,
            ]
        ),
        np.array(
            [
                0.5631104,
                0.7642516,
                0.6943509,
                0.9504361,
                0.1673000,
                0.8369634,
                0.5153690,
                1.1602816,
            ]
        ),
        np.array(
            [
                0.2003983,
                0.5111671,
                -0.0979592,
                -0.0995776,
                0.1762009,
                0.0078159,
                0.0710666,
                -0.0910429,
            ]
        ),
        np.array(
            [
                0.7174045,
                0.1377401,
                0.7430671,
                0.4376869,
                0.0131739,
                0.5423524,
                0.5125742,
                0.6757894,
            ]
        ),
    ]
)
artmp_iter_1 = np.array(
    [
        [1.11411, -0.37583, -0.29358, 0.66587],
        [1.22264, 1.42318, -0.25445, 0.95149],
        [-1.98069, -0.66132, -0.22385, 0.62977],
        [0.18543, 0.26609, -0.72843, -0.73383],
        [-1.23395, 1.11349, 1.19903, 0.18110],
    ]
)
cov_matrix_iter_1 = np.array(
    [
        [0.9612068, 0.0656638, -0.0909456, 0.0050800, -0.0404114],
        [0.0656638, 1.0646287, -0.1709123, 0.0122301, -0.0288684],
        [-0.0909456, -0.1709123, 1.1324262, -0.0134381, 0.0568813],
        [0.0050800, 0.0122301, -0.0134381, 0.9198882, -0.0069058],
        [-0.0404114, -0.0288684, 0.0568813, -0.0069058, 0.9688629],
    ]
)
invsqrt_cov_iter_1 = np.array(
    [
        [1.0246647, -0.0274079, 0.0390921, -0.0020212, 0.0191826],
        [-0.0274079, 0.9794807, 0.0734848, -0.0053737, 0.0103193],
        [0.0390921, 0.0734848, 0.9514068, 0.0056465, -0.0242382],
        [-0.0020212, -0.0053737, 0.0056465, 1.0427739, 0.0033430],
        [0.0191826, 0.0103193, -0.0242382, 0.0033430, 1.0178173],
    ]
)


class TestCMAES:
    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_init_params(self, cmaes: CMAES):
        cmaes._init_algorithm_params(batch_size=8)
        # Constant params
        assert cmaes.n_dims == 5
        assert cmaes.mu_const == 4
        test_weights = np.array([0.529930, 0.285714, 0.142857, 0.041498])
        assert np.allclose(cmaes.weights_const.flatten(), test_weights)
        assert cmaes.weights_const.shape == (4, 1)
        assert np.allclose(cmaes.mu_eff_const, 2.6002)
        assert np.allclose(cmaes.cum_covm_const, 0.45020)
        assert np.allclose(cmaes.cum_sigma_const, 0.36509)
        assert np.allclose(cmaes.lr_covrank1_const, 0.047292)
        assert np.allclose(cmaes.damp_sigma_const, 1.3651)
        assert np.round(cmaes.chi_norm_const, 4) == 2.1285
        # Variable params
        assert (cmaes.invsqrtC == np.eye(cmaes.n_dims)).all()
        assert (cmaes.cov_matrix == np.eye(cmaes.n_dims)).all()
        assert (cmaes.paths_covm == 0).all()
        assert (cmaes.paths_sigma == 0).all()
        assert cmaes.paths_covm.shape == (cmaes.n_dims, 1)
        assert cmaes.paths_sigma.shape == (cmaes.n_dims, 1)
        assert (cmaes.scaling_diag == np.ones((cmaes.n_dims, 1))).all()

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_sample_actions(self, det_cmaes):
        batch_size = 8
        assert det_cmaes._count_eval == 0
        model_states = det_cmaes.reset(
            batch_size=batch_size, init_xmean=init_xmean, noise=noise_iter_1, model_states=None
        )
        assert det_cmaes.pop_size == batch_size
        assert det_cmaes._count_eval == batch_size
        assert np.allclose(det_cmaes.x_mean, init_xmean)
        actions = np.array(model_states.actions.T)
        assert np.allclose(actions, actions_iter_1, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_update_evolution_paths(self, det_cmaes):
        model_states = det_cmaes.reset(
            batch_size=8, init_xmean=init_xmean, noise=noise_iter_1, model_states=None
        )

        assert np.allclose(det_cmaes.x_mean, init_xmean)
        actions = np.array(model_states.actions.T)
        assert np.allclose(actions, actions_iter_1, rtol=1e-5, atol=1e-5)
        sorted_fitness = np.argsort(fitness_iter_1)[: det_cmaes.mu_const]
        selected_actions = model_states.actions[sorted_fitness].T
        det_cmaes._update_evolution_paths(selected_actions)
        assert np.allclose(det_cmaes.old_x_mean, init_xmean)
        assert np.allclose(det_cmaes.x_mean, xmean_iter_1, rtol=1e-5, atol=1e-5), "dif: %s" % (
            det_cmaes.x_mean - xmean_iter_1
        )
        assert np.allclose(det_cmaes.paths_sigma, ps_iter_1, rtol=1e-5, atol=1e-5), "dif: %s" % (
            det_cmaes.paths_sigma - ps_iter_1
        )
        assert det_cmaes.hsig == hsig_iter_1
        assert np.allclose(det_cmaes.paths_covm, pc_iter_1, rtol=1e-5, atol=1e-5), "dif: %s" % (
            det_cmaes.paths_covm - pc_iter_1
        )

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_adapt_covariance_matrix(self, det_cmaes):
        model_states = det_cmaes.reset(
            batch_size=8, init_xmean=init_xmean, noise=noise_iter_1, model_states=None
        )

        assert np.allclose(det_cmaes.x_mean, init_xmean)
        actions = np.array(model_states.actions.T)
        assert np.allclose(actions, actions_iter_1, rtol=1e-5, atol=1e-5)
        sorted_fitness = np.argsort(fitness_iter_1)[: det_cmaes.mu_const]
        selected_actions = model_states.actions[sorted_fitness].T
        det_cmaes._update_evolution_paths(selected_actions)
        det_cmaes._adapt_covariance_matrix(selected_actions)
        assert np.allclose(det_cmaes.artmp, artmp_iter_1), "dif: %s" % (
            det_cmaes.artmp - artmp_iter_1
        )
        assert np.allclose(
            det_cmaes.cov_matrix, cov_matrix_iter_1, rtol=1e-6, atol=1e-6
        ), "dif: %s" % (det_cmaes.cov_matrix - cov_matrix_iter_1)

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_adapt_sigma(self, det_cmaes):
        test_sigma = 0.29992
        model_states = det_cmaes.reset(
            batch_size=8, init_xmean=init_xmean, noise=noise_iter_1, model_states=None
        )

        assert np.allclose(det_cmaes.x_mean, init_xmean)
        actions = np.array(model_states.actions.T)
        assert np.allclose(actions, actions_iter_1, rtol=1e-5, atol=1e-5)
        sorted_fitness = np.argsort(fitness_iter_1)[: det_cmaes.mu_const]
        selected_actions = model_states.actions[sorted_fitness].T
        det_cmaes._update_evolution_paths(selected_actions)
        det_cmaes._adapt_covariance_matrix(selected_actions)
        det_cmaes._adapt_sigma()
        assert np.allclose(det_cmaes.sigma, test_sigma)

    @pytest.mark.skipif(not Backend.is_numpy(), reason="Only for numpy")
    def test_covariance_matrix_diagonalization(self, det_cmaes):
        model_states = det_cmaes.reset(
            batch_size=8, init_xmean=init_xmean, noise=noise_iter_1, model_states=None
        )

        assert np.allclose(det_cmaes.x_mean, init_xmean)
        actions = np.array(model_states.actions.T)
        assert np.allclose(actions, actions_iter_1, rtol=1e-5, atol=1e-5)
        sorted_fitness = np.argsort(fitness_iter_1)[: det_cmaes.mu_const]
        selected_actions = model_states.actions[sorted_fitness].T
        det_cmaes._update_evolution_paths(selected_actions)
        det_cmaes._adapt_covariance_matrix(selected_actions)
        det_cmaes._adapt_sigma()
        det_cmaes._cov_matrix_diagonalization()
        assert np.allclose(
            det_cmaes.invsqrtC, invsqrt_cov_iter_1, rtol=1e-6, atol=1e-6
        ), "dif: %s" % (det_cmaes.invsqrtC - invsqrt_cov_iter_1)
