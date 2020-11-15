import warnings

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.errors import HypothesisDeprecationWarning
import hypothesis.strategies as st
from judo import Backend, dtype, tensor
from judo.functions.fractalai import (
    calculate_clone,
    calculate_virtual_reward,
    fai_iteration,
)
import numpy


from fragile.core.utils import NUMPY_IGNORE_WARNINGS_PARAMS

warnings.filterwarnings("ignore", category=HypothesisDeprecationWarning)


@given(st.integers(), st.integers())
def test_ints_are_commutative(x, y):
    assert x + y == y + x


class TestFaiNumpy:
    @given(
        arrays(numpy.float32, shape=(10, 3, 10)),
        arrays(numpy.float32, shape=(10,)),
        arrays(numpy.bool, shape=(10, 1)),
    )
    def test_calculate_reward(self, observs, rewards, oobs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            virtual_reward, compas = calculate_virtual_reward(
                observs=tensor(observs),
                rewards=tensor(rewards),
                oobs=tensor(oobs),
                return_compas=True,
            )
            assert dtype.is_tensor(virtual_reward)
            assert len(virtual_reward.shape) == 1
            assert len(virtual_reward) == len(rewards)

    @given(arrays(numpy.float32, shape=(13,)), arrays(numpy.bool, shape=(13,)), st.floats(1e-7, 1))
    def test_calculate_clone(self, virtual_rewards, oobs, eps):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            compas_ix, will_clone = calculate_clone(
                virtual_rewards=tensor(virtual_rewards), oobs=tensor(oobs), eps=tensor(eps)
            )

            assert dtype.is_tensor(compas_ix)
            assert dtype.is_tensor(will_clone)

            assert len(compas_ix.shape) == 1
            assert len(will_clone.shape) == 1

            assert len(compas_ix) == len(virtual_rewards)
            assert len(will_clone) == len(virtual_rewards)
            if Backend.is_numpy():
                assert isinstance(compas_ix[0], dtype.int64), type(compas_ix[0])
                assert isinstance(will_clone[0], dtype.bool), type(will_clone[0])
            elif Backend.is_torch():
                assert compas_ix[0].dtype == dtype.int64, type(compas_ix[0])
                assert will_clone[0].dtype == dtype.bool, type(will_clone[0])

    @given(
        arrays(numpy.float32, shape=(10, 3, 10)),
        arrays(numpy.float32, shape=10),
        arrays(numpy.bool, shape=10),
    )
    def test_fai_iteration(self, observs, rewards, oobs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            compas_ix, will_clone = fai_iteration(
                observs=tensor(observs), rewards=tensor(rewards), oobs=tensor(oobs)
            )
            assert dtype.is_tensor(compas_ix)
            assert dtype.is_tensor(will_clone)

            assert len(compas_ix.shape) == 1
            assert len(will_clone.shape) == 1

            assert len(compas_ix) == len(rewards)
            assert len(will_clone) == len(rewards)
            if Backend.is_numpy():
                assert isinstance(compas_ix[0], dtype.int64), type(compas_ix[0])
                assert isinstance(will_clone[0], dtype.bool), type(will_clone[0])
