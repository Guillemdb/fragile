import judo
from judo import Backend, tensor
from judo.tests.test_states import TestStates
import pytest

from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers

from tests.core.test_swarm import create_atari_swarm

state_classes = [States, StatesEnv, StatesModel, StatesWalkers]


@pytest.fixture(scope="class", params=state_classes)
def states_class(request):
    return request.param


class TestFragileStates:
    def test_clone(self, states_class):
        batch_size = 10
        states = states_class(batch_size=batch_size)
        states.miau = judo.arange(states.n)
        states.miau_2 = judo.arange(states.n)

        will_clone = judo.zeros(states.n, dtype=judo.bool)
        will_clone[3:6] = True
        compas_ix = tensor(list(range(states.n))[::-1])

        states.clone(will_clone=will_clone, compas_ix=compas_ix)
        target_1 = judo.arange(10)

        assert bool(judo.all(target_1 == states.miau)), (target_1 - states.miau, states_class)

    def test_merge_states(self, states_class):
        batch_size = 21
        data = judo.repeat(judo.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        split_states = tuple(new_states.split_states(batch_size))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

        split_states = tuple(new_states.split_states(5))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

    def test_merge_states_with_atari(self):
        swarm = create_atari_swarm()
        for states in (swarm.walkers.states, swarm.walkers.env_states, swarm.walkers.model_states):
            split_states = tuple(states.split_states(states.n))
            merged = states.merge_states(split_states)
            assert len(merged) == states.n
            if (
                Backend.is_numpy() and Backend.use_true_hash()
            ):  # Pytorch hashes are not real hashes
                assert hash(merged) == hash(states)
