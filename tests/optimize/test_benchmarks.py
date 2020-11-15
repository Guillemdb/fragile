import itertools

import numpy
import pytest
import judo
from judo import tensor, random_state
from fragile.optimize.benchmarks import (
    EggHolder,
    LennardJones,
    Rastrigin,
    Sphere,
    StyblinskiTang,
)

wiki_bench_classes = [EggHolder, Rastrigin, Sphere, StyblinskiTang]


@pytest.fixture(params=list(itertools.product(wiki_bench_classes, [2,])), scope="class")
def wiki_benchmark(request):
    cls, shape = request.param
    return cls(dims=shape)


class TestBenchmarks:
    def test_optimum(self, wiki_benchmark):
        best = wiki_benchmark.best_state
        new_shape = (1,) + tuple(best.shape)
        val = wiki_benchmark.function(best.reshape(new_shape))
        bench = wiki_benchmark.benchmark
        assert judo.allclose(val[0], bench), wiki_benchmark.__class__.__name__

    @pytest.mark.parametrize("dims", [2, 3, 6])
    def test_get_bounds(self, wiki_benchmark, dims):
        bounds = wiki_benchmark.get_bounds(dims)
        if not isinstance(wiki_benchmark, EggHolder):
            assert len(bounds) == dims
        else:
            assert len(bounds) == 2


class TestLennardJonnes:
    def test_benchmarks(self):
        for k, val in LennardJones.minima.items():
            lennard = LennardJones(n_atoms=int(k))
            lennard.function(tensor(numpy.random.random((1, 3 * int(k)))))
