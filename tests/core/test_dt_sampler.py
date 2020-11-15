import pytest

from fragile.core.dt_samplers import BaseDtSampler, ConstantDt, GaussianDt, UniformDt
from fragile.core.states import States


def constant_dt():
    return ConstantDt(2)


def gaussian_dt():
    return GaussianDt(min_dt=1, max_dt=10, loc_dt=3, scale_dt=1.0)


def uniform_dt():
    return UniformDt(min_dt=1, max_dt=3)


dt_samplers = [constant_dt, gaussian_dt, uniform_dt]


@pytest.fixture(params=dt_samplers, scope="class")
def dt_sampler(request) -> BaseDtSampler:
    return request.param()


class TestDtSampler:

    batch_size = 13

    def test_get_params_dict(self, dt_sampler):
        param_dict = dt_sampler.get_params_dict()
        assert isinstance(param_dict, dict)
        assert "dt" in param_dict
        assert "critic_score" in param_dict
        assert "dtype" in param_dict["dt"]

    def test_calculate(self, dt_sampler):
        states = dt_sampler.calculate(batch_size=self.batch_size)
        assert isinstance(states, States)
        assert "dt" in states.keys()
        assert "critic_score" in states.keys()
        assert len(states.dt) == self.batch_size
        assert states.dt.dtype == dt_sampler._dtype
        if hasattr(dt_sampler, "min_dt"):
            assert (states.dt >= dt_sampler.min_dt).all()
        if hasattr(dt_sampler, "max_dt"):
            assert (states.dt <= dt_sampler.max_dt).all()
