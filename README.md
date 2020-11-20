# Fragile
[![Travis build status](https://travis-ci.com/FragileTech/fragile.svg)](https://travis-ci.com/FragileTech/fragile)
[![Documentation Status](https://readthedocs.org/projects/fragile/badge/?version=latest)](https://fragile.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://codecov.io/github/FragileTech/fragile/coverage.svg)](https://codecov.io/github/FragileTech/fragile)
[![PyPI package](https://badgen.net/pypi/v/fragile)](https://pypi.org/project/fragile/)
[![Latest docker image](https://badgen.net/docker/pulls/fragiletech/fragile)](https://hub.docker.com/r/fragiletech/fragile/tags)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![stable](http://badges.github.io/stability-badges/dist/stable.svg)](http://github.com/badges/stability-badges)

Fragile is a framework for developing optimization algorithms inspired by Fractal AI and running them at scale.

## Features

- Provides classes and an API for easily developing planning algorithms
- Provides an classes and an API for function optimization
- Build in visualizations of the sampling process
- Fully documented and tested
- Support for parallelization and distributed search processes


## About FractalAI

FractalAI is based on the framework of [non-equilibrium thermodynamics](https://en.wikipedia.org/wiki/Non-equilibrium_thermodynamics), and can be used to derive new mathematical tools for efficiently exploring state spaces.
 
The principles of our work are accessible online:

- [Arxiv](https://arxiv.org/abs/1803.05049) manuscript describing the fundamental principles of our work.
- [Blog](http://entropicai.blogspot.com) that describes our early research process.
- [Youtube channel](https://www.youtube.com/user/finaysergio/videos) with videos showing how different prototypes work.
- [GitHub repository](https://github.com/FragileTech/FractalAI) containing a prototype that solves most Atari games.

## Getting started 

Check out the [getting started](https://fragile.readthedocs.io/en/latest/resources/examples/01_getting_started.html) 
section of the docs, or the [examples](https://github.com/FragileTech/fragile/tree/master/examples) folder.

## Running in docker
The fragile docker container will execute a Jupyter notebook accessible on port 8080 with password: `fragile`

You can pull a docker image from Docker Hub running:

```bash
    docker pull fragiletech/fragile:version-tag
```

Where version-tag corresponds to the fragile version that will be installed in the pulled image.

## Installation
This framework has been tested in Ubuntu 18.04 and supports Python 3.7 and 3.8. 
If you find any problems running it in a different OS or Python version please open an issue.

It can be installed with `pip install fragile["all"]`.

You can find the pinned versions of the minimum requirements to install the core module in `requirements.txt`,
and the pinned versions of all the optional requirements in `requirements-all.txt`.

Detailed installation instructions can be found in the [docs](https://fragile.readthedocs.io/en/latest/resources/installation.html).

## Documentation

You can access the documentation on [Read The Docs](https://fragile.readthedocs.io/en/latest/).
    
## Roadmap

Upcoming features: _(not necessarily in order)_
- Fix documentation and add examples for the `distributed` module
- Upload Montezuma solver
- Add new algorithms to sample different state spaces.
- Add a benchmarking module
- Add deep learning API

## Contributing

Contribution are welcome. Please take a look at [contributining](docsrc/markdown/CONTRIBUTING.md) 
and respect the [code of conduct](docsrc/markdown/CODE_OF_CONDUCT.md).
    
## Cite us
If you use this framework in your research please cite us as:

    @misc{1803.05049,
        Author = {Sergio Hernández Cerezo and Guillem Duran Ballester},
        Title = {Fractal AI: A fragile theory of intelligence},
        Year = {2018},
        Eprint = {arXiv:1803.05049},
    }
      
## License

This project is MIT licensed. See `LICENSE.md` for the complete text.