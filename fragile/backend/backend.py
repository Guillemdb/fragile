from contextlib import contextmanager
import os

import yaml

try:
    import torch
except ImportError:

    class cuda:
        @staticmethod
        def is_available():
            return False

    class torch_random:
        @staticmethod
        def manual_seed(*args, **kwargs):
            return None

    class torch:
        cuda = cuda
        Tensor = None
        random = torch_random


config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")


def load_backend_config(filepath=config_file):
    with open(filepath, "r") as stream:
        config = yaml.safe_load(stream)
    backend = config["default_backend"]
    device = config["default_device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_grad = config["default_grad"]
    true_hash = config["true_hash"]
    return backend, device, use_grad, true_hash


def update_backend_config(
    backend: str = None,
    device: str = None,
    use_grad: bool = None,
    filepath=config_file,
    true_hash: bool = None,
):
    with open(filepath, "r") as stream:
        config = yaml.safe_load(stream)
        if config is None:
            config = {}
    if backend is not None:
        config["default_backend"] = backend
    if device is not None:
        config["default_device"] = device
    if use_grad is not None:
        config["default_grad"] = use_grad
    if true_hash is not None:
        config["true_hash"] = true_hash
    with open(filepath, "w") as outfile:
        yaml.dump(config, outfile)


@contextmanager
def _use_backend(cls, name, device=None, use_grad=None):
    if name is not None:
        cls._check_valid_backend(name)
    curr_state = cls.get_backend_state()
    cls.set_backend(name=name, device=device, use_grad=use_grad)
    try:
        yield
    finally:
        cls.set_backend(**curr_state)


_backend, _device, _use_grad, _true_hash = load_backend_config()
_backend, _device, _use_grad, _true_hash = (
    str(_backend),
    str(_device),
    bool(_use_grad),
    bool(_true_hash),
)


class Backend:
    AVAILABLE_BACKENDS = ["numpy", "torch"]
    _backend, _device, _use_grad, _true_hash = _backend, _device, _use_grad, _true_hash

    @classmethod
    def _check_valid_backend(cls, name):
        if name not in cls.AVAILABLE_BACKENDS:
            raise ValueError(
                "%s not supported. Available backends: %s" % (name, cls.AVAILABLE_BACKENDS)
            )

    @classmethod
    def get_backend_state(cls):
        state = {
            "name": str(cls._backend),
            "device": str(cls._device),
            "use_grad": bool(cls._use_grad),
        }
        return state

    @classmethod
    def get_current_backend(cls):
        return cls._backend

    @classmethod
    def get_device(cls):
        return str(cls._device)

    @classmethod
    def use_grad(cls):
        return cls._use_grad

    @classmethod
    def use_true_hash(cls):
        return cls._true_hash

    @classmethod
    def set_defaults(
        cls,
        name=None,
        device=None,
        use_grad: bool = None,
        set_backend: bool = True,
        true_hash: bool = None,
    ):
        update_backend_config(backend=name, device=device, use_grad=use_grad, true_hash=true_hash)
        if set_backend:
            cls.set_backend(name, device, use_grad)
            if true_hash is not None:
                cls._true_hash = true_hash

    @classmethod
    def set_backend(cls, name=None, device=None, use_grad: bool = None):
        if name is not None:
            cls._check_valid_backend(name)
            cls._backend = name
        if device is not None:
            cls._device = device
        if use_grad is not None:
            cls._use_grad = use_grad

    @classmethod
    def is_numpy(cls):
        return cls._backend == "numpy"

    @classmethod
    def is_torch(cls):
        return cls._backend == "torch"

    @classmethod
    def execute(cls, value, funcs):
        backend = cls.get_current_backend()
        return funcs[backend](value)

    @classmethod
    def use_backend(cls, name=None, device=None, use_grad=None):
        return _use_backend(cls, name=name, device=device, use_grad=use_grad)


def load_backend_config(filepath=config_file):
    with open(filepath, "r") as stream:
        config = yaml.safe_load(stream)
    backend = config["default_backend"]
    device = config["default_device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_grad = config["default_grad"]
    true_hash = config["true_hash"]
    return backend, device, use_grad, true_hash
