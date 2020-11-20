import sys
from judo.slogging import setup as setup_logging

from fragile.core import NormalContinuous
from fragile.optimize import FunctionMapper
from fragile.optimize.benchmarks import EggHolder


def main():

    setup_logging(level="INFO", structured=False)

    def gaussian_model(env):
        # Gaussian of mean 0 and std of 10, adapted to the environment bounds
        return NormalContinuous(scale=10, loc=0.0, bounds=env.bounds)

    swarm = FunctionMapper(env=EggHolder, model=gaussian_model, n_walkers=100, max_epochs=5000,)

    swarm.run(report_interval=500, show_pbar=True)


if __name__ == "__main__":
    sys.exit(main())
