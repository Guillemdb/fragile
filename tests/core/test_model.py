import judo
from judo import dtype, tensor
import pytest

from fragile.core.dt_samplers import GaussianDt
from fragile.core.models import (
    _DtModel,
    BaseCritic,
    BaseModel,
    BinarySwap,
    Bounds,
    ContinuousModel,
    ContinuousUniform,
    DiscreteModel,
    DiscreteUniform,
    NormalContinuous,
)
from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers


def create_model(name="discrete"):
    if name == "discrete":
        return lambda: DiscreteUniform(n_actions=10)
    elif name == "continuous":
        bs = Bounds(low=-1, high=1, shape=(3,))
        return lambda: ContinuousUniform(bounds=bs)
    elif name == "random_normal":
        bs = Bounds(low=-1, high=1, shape=(3,))
        return lambda: NormalContinuous(loc=0, scale=1, bounds=bs)
    elif name == "discrete_with_critic":
        critic = GaussianDt(max_dt=3)
        return lambda: DiscreteUniform(n_actions=10, critic=critic)
    raise ValueError("Invalid param `name`.")


def create_model_states(model: BaseModel, batch_size: int = 10):
    return StatesModel(batch_size=batch_size, state_dict=model.get_params_dict())


model_fixture_params = ["discrete", "continuous", "random_normal", "discrete_with_critic"]


@pytest.fixture(scope="class", params=model_fixture_params)
def model(request):
    return create_model(request.param)()


@pytest.fixture(scope="class")
def batch_size():
    return 7


class TestModel:
    def create_model_states(self, model: BaseModel, batch_size: int = None):
        return StatesModel(batch_size=batch_size, state_dict=model.get_params_dict())

    def create_env_states(self, model: BaseModel, batch_size: int = None):
        return StatesEnv(batch_size=batch_size, state_dict=model.get_params_dict())

    def test_get_params_dict(self, model):
        params_dict = model.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki, _vi in v.items():
                assert isinstance(ki, str)

    def test_reset(self, model, batch_size):
        states = model.reset(batch_size=batch_size)
        assert isinstance(states, model.STATE_CLASS)
        model_states = self.create_model_states(model=model, batch_size=batch_size)
        env_states = self.create_env_states(model, batch_size=batch_size)
        states = model.reset(
            batch_size=batch_size, model_states=model_states, env_states=env_states
        )
        assert isinstance(states, model.STATE_CLASS), (
            type(states),
            model.STATE_CLASS,
        )
        assert len(model_states.actions) == batch_size

    def test_predict(self, model, batch_size):
        states = self.create_model_states(model=model, batch_size=batch_size)
        env_states = self.create_env_states(model, batch_size=batch_size)
        updated_states = model.predict(model_states=states, env_states=env_states)
        assert isinstance(updated_states, model.STATE_CLASS)
        assert len(updated_states) == batch_size
        updated_states = model.predict(
            model_states=states, env_states=env_states, batch_size=batch_size
        )
        assert isinstance(updated_states, model.STATE_CLASS)
        assert len(updated_states) == batch_size
        if hasattr(model, "bounds"):
            assert model.bounds.points_in_bounds(updated_states.actions).all()
        with pytest.raises(ValueError):
            model.predict()


class DummyCritic(BaseCritic):
    def get_params_dict(self):
        return {"critic_score": {"dtype": float}}

    def calculate(
        self,
        batch_size: int = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
    ) -> States:
        batch_size = batch_size or env_states.n
        return States(batch_size=batch_size, critic_score=5 * judo.ones(batch_size))


class TestDtModel:
    def test_get_params_dict_content(self):
        params = _DtModel().get_params_dict()
        assert "critic_score" in params
        assert "dtype" in params["critic_score"]
        assert params["critic_score"]["dtype"] == dtype.int

    def test_override_get_params_dict(self):
        critic = DummyCritic()
        model = _DtModel(critic=critic)
        params = model.get_params_dict(override_params=False)
        assert "critic_score" in params
        assert "dtype" in params["critic_score"]
        assert params["critic_score"]["dtype"] == dtype.int

        params = model.get_params_dict(override_params=True)
        assert "critic_score" in params
        assert "dtype" in params["critic_score"]
        assert params["critic_score"]["dtype"] == float


class DummyEnv:
    n_actions = 10


class TestDiscreteModel:
    def test_init(self):
        env_1 = DiscreteModel(n_actions=10)
        assert env_1.n_actions == 10
        env_2 = DiscreteModel(env=DummyEnv())
        assert env_2.n_actions == 10

    def test_get_params_dict(self):
        params = DiscreteModel(n_actions=10).get_params_dict()
        assert "actions" in params
        assert "dtype" in params["actions"]
        assert params["actions"]["dtype"] == dtype.int


class TestDiscreteUniform:
    @pytest.mark.parametrize("n_actions", [2, 5, 10, 20])
    def test_sample(self, n_actions):
        model = DiscreteUniform(n_actions=n_actions)
        model_states = model.predict(batch_size=1000)
        actions = model_states.actions
        assert len(actions.shape) == 1
        assert len(judo.unique(actions)) <= n_actions
        assert all(actions >= 0)
        assert all(actions <= n_actions)
        assert "critic_score" in model_states.keys()
        assert dtype.is_tensor(model_states.critic_score)
        assert (model_states.critic_score == 1).all(), model_states.critic_score

        states = create_model_states(batch_size=100, model=model)
        model_states = model.sample(batch_size=states.n, model_states=states)
        actions = model_states.actions
        assert len(actions.shape) == 1
        assert len(judo.unique(actions)) <= n_actions
        assert all(actions >= 0)
        assert all(actions <= n_actions)
        assert judo.allclose(actions, judo.astype(actions, dtype.int))
        assert "critic_score" in model_states.keys()
        assert (model_states.critic_score == 1).all()

    @pytest.mark.parametrize("n_actions", [2, 5, 10, 20])
    def test_sample_with_critic(self, n_actions):
        model = DiscreteUniform(n_actions=n_actions, critic=DummyCritic())
        model_states = model.predict(batch_size=1000)
        actions = model_states.actions
        assert len(actions.shape) == 1
        assert len(judo.unique(actions)) <= n_actions
        assert all(actions >= 0)
        assert all(actions <= n_actions)
        assert "critic_score" in model_states.keys()
        assert (model_states.critic_score == 5).all()

        states = create_model_states(batch_size=100, model=model)
        model_states = model.sample(batch_size=states.n, model_states=states)
        actions = model_states.actions
        assert len(actions.shape) == 1
        assert len(judo.unique(actions)) <= n_actions
        assert all(actions >= 0)
        assert all(actions <= n_actions)
        assert judo.allclose(actions, judo.astype(actions, dtype.int))
        assert "critic_score" in model_states.keys()
        assert (model_states.critic_score == 5).all()


class TestBinarySwap:
    def test_sample(self):
        model = BinarySwap(n_actions=10, n_swaps=3)
        states = model.predict(batch_size=10)
        actions = states.actions
        vectors = actions.sum(axis=1)
        assert actions.min() == 0
        assert actions.max() == 1
        assert (vectors > 0).all(), actions
        assert (vectors <= 3).all(), actions


class TestContinuousModel:
    def test_attributes(self):
        bounds = Bounds(low=-1, high=3, shape=(3,))
        model = ContinuousModel(bounds=bounds)
        assert model.shape == (3,)
        assert model.n_dims == 3

    def test_get_params_dict(self):
        bounds = Bounds(low=-1, high=3, shape=(3,))
        model = ContinuousModel(bounds=bounds)
        params = model.get_params_dict()
        assert params["actions"]["size"] == model.shape


class TestContinuousUniform:
    def test_sample(self):
        bounds = Bounds(low=-1, high=3, shape=(3,))
        model = ContinuousUniform(bounds=bounds)
        actions = model.predict(batch_size=100).actions
        assert actions.min() >= -1
        assert actions.max() <= 3

        bounds = Bounds(low=-1, high=3, shape=(3, 10))
        model = ContinuousUniform(bounds=bounds)
        actions = model.predict(batch_size=100).actions
        assert actions.min() >= -1
        assert actions.max() <= 3


class TestRandomNormal:
    def test_sample(self):
        bounds = Bounds(low=-5, high=5, shape=(3,))
        model = NormalContinuous(bounds=bounds)
        actions = model.predict(batch_size=10000).actions
        assert actions.min() >= -5
        assert actions.max() <= 5
        assert judo.allclose(actions.mean(), tensor(0.0), atol=0.05)
        assert judo.allclose(actions.std(), tensor(1.0), atol=0.05)

        bounds = Bounds(low=-10, high=30, shape=(3, 10))
        model = NormalContinuous(bounds=bounds, loc=5, scale=2)
        actions = model.predict(batch_size=10000).actions
        assert actions.min() >= -10
        assert actions.max() <= 30
        assert judo.allclose(actions.mean(), tensor(5.0), atol=0.05), actions.mean()
        assert judo.allclose(actions.std(), tensor(2.0), atol=0.05), actions.std()
