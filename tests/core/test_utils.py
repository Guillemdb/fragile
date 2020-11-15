import pytest

from fragile.core.utils import get_plangym_env


class TestUtils:
    def test_get_plangym_env(self):
        class dummy_shape:
            shape = (2, 2)

        class DummyEnv:
            def __init__(self):
                class dummy_n:
                    n = 1

                self.action_space = dummy_n
                self.observation_space = dummy_shape

            def get_state(self):
                return dummy_shape

        class DummySwarm:
            @property
            def env(self):
                class dummy_env:
                    _env = DummyEnv

                return dummy_env

        swarm = DummySwarm()
        with pytest.raises(TypeError):
            get_plangym_env(swarm)

        class DummySwarm:
            @property
            def env(self):
                from fragile.core.env import DiscreteEnv

                return DiscreteEnv(DummyEnv())

        swarm = DummySwarm()
        with pytest.raises(TypeError):
            get_plangym_env(swarm)
