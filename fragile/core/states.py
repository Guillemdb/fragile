import copy
from typing import Optional, Set, Tuple

import judo
from judo import hasher, States as JudoStates, tensor
import numpy

from fragile.core.typing import Scalar, StateDict, Tensor


class States(JudoStates):
    """
    Handles several arrays that will contain the data associated with the \
    walkers of a :class:`Swarm`. Each array will be associated to a class \
    attribute, and it will store the corresponding value of that attribute \
    for all the walkers of a :class:`Swarm`.

    This class behaves as a dictionary of arrays with some extra functionality \
    to make easier the process of cloning the walkers' data. All of its internal \
    arrays will have an extra first dimension equal to the number of walkers.

    In order to define the tensors, a `state_dict` dictionary needs to be \
    specified using the following structure::

        state_dict = {"name_1": {"size": tuple([1]),
                                 "dtype": numpy.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor. The \
    created arrays will accessible the ``name_1`` attribute of the class, or \
    indexing the class with ``states["name_1"]``.

    If ``size`` is not defined the attribute will be considered a vector of \
    length `batch_size`.


    Args:
        batch_size: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: Data can be directly specified as keyword arguments.

    """

    def clone(
        self, will_clone: Tensor, compas_ix: Tensor, ignore: Optional[Set[str]] = None,
    ):
        """
        Clone all the stored data according to the provided arrays.

        Args:
            will_clone: Array of shape (n_walkers,) of booleans indicating the \
                        index of the walkers that will clone to a random companion.
            compas_ix: Array of integers of shape (n_walkers,). Contains the \
                       indexes of the walkers that will be copied.
            ignore: set containing the names of the attributes that will not be \
                    cloned.

        """
        ignore = set() if ignore is None else ignore
        for name in self.keys():
            if judo.is_tensor(self[name]) and name not in ignore:
                self[name][will_clone] = self[name][compas_ix][will_clone]


class StatesEnv(States):
    """
    Keeps track of the data structures used by the :class:`Environment`.

    Attributes:
        states: This data tracks the internal state of the Environment simulation, \
                 and they are only used to save and restore its state.
        observs: This is the data that corresponds to the observations of the \
                 current :class:`Environment` state. The observations are used \
                 for calculating distances.
        rewards: This vector contains the rewards associated with each observation.
        oobs: Stands for **Out Of Bounds**. It is a vector of booleans that \
              represents and arbitrary boundary condition. If a value is ``True`` \
              the corresponding states will be treated as being outside the \
              :class:`Environment` domain. The states considered out of bounds \
              will be avoided by the sampling algorithms.
        terminals: Vector of booleans representing the successful termination \
                   of an environment. A ``True`` value indicates that the \
                   :class:`Environment` has successfully reached a terminal \
                   state that is not out of bounds.

    """

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialise a :class:`StatesEnv`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.observs = None
        self.states = None
        self.rewards = None
        self.oobs = None
        self.terminals = None
        self.times = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesEnv, self).__init__(state_dict=updated_dict, batch_size=batch_size, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`StatesEnv`."""
        params = {
            "states": {"dtype": judo.float64},
            "observs": {"dtype": judo.float32},
            "rewards": {"dtype": judo.float32},
            "times": {"dtype": judo.float32},
            "oobs": {"dtype": judo.bool},
            "terminals": {"dtype": judo.bool},
        }
        state_dict = super(StatesEnv, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesModel(States):
    """
    Keeps track of the data structures used by the :class:`Model`.

    Attributes:
        actions: Represents the actions that will be sampled by a :class:`Model`.

    """

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialise a :class:`StatesModel`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.actions = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesModel, self).__init__(state_dict=updated_dict, batch_size=batch_size, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary with tre attributes common to all Models."""
        params = {
            "actions": {"dtype": judo.float32},
        }
        state_dict = super(StatesModel, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesWalkers(States):
    """
    Keeps track of the data structures used by the :class:`Walkers`.

    Attributes:
        id_walkers: Array of of integers that uniquely identify a given state. \
                    They are obtained by hashing the states.
        compas_clone: Array of integers containing the index of the walkers \
                      selected as companions for cloning.
        processed_rewards: Array of normalized rewards. It contains positive \
                           values with an average of 1. Values greater than one \
                           correspond to rewards above the average, and values \
                           lower than one correspond to rewards below the average.
        virtual_rewards: Array containing the virtual rewards assigned to each walker.
        cum_rewards: Array of rewards used to compute the virtual_reward. This \
                    value can accumulate the rewards provided by the \
                    :class:`Environment` during an algorithm run.
        distances: Array containing the similarity metric of each walker used \
                   to compute the virtual reward.
        clone_probs: Array containing the probability that a walker clones to \
                     its companion during the cloning phase.
        will_clone: Boolean array. A ``True`` value indicates that the \
                    corresponding walker will clone to its companion.
        in_bounds: Boolean array. A `True` value indicates that a walker is \
                   in the domain defined by the :class:`Environment`.

        best_state: State of the walker with the best ``cum_reward`` found \
                   during the algorithm run.
        best_obs: Observation corresponding to the ``best_state``.
        best_reward: Best ``cum_reward`` found during the algorithm run.
        best_id: Integer representing the hash of the ``best_state``.

    """

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialize a :class:`StatesWalkers`.

        Args:
            batch_size: Number of walkers that the class will be tracking.
            state_dict: Dictionary defining the attributes of the tensors.
            kwargs: attributes that will not be set as numpy.ndarrays
        """
        self.will_clone = None
        self.compas_clone = None
        self.processed_rewards = None
        self.cum_rewards = None
        self.virtual_rewards = None
        self.distances = None
        self.clone_probs = None
        self.in_bounds = None
        self.id_walkers = None
        # This is only to allow __repr__. Should be overridden after reset
        self.best_id = None
        self.best_obs = None
        self.best_state = None
        self.best_reward = numpy.NINF
        self.best_epoch = 0
        self.best_time = 0.0

        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesWalkers, self).__init__(
            state_dict=updated_dict, batch_size=batch_size, **kwargs
        )

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of States that can handle all the data generated by the :class:`Walkers`.
        """
        params = {
            "id_walkers": {"dtype": judo.hash_type},
            "compas_clone": {"dtype": judo.int64},
            "processed_rewards": {"dtype": judo.float},
            "virtual_rewards": {"dtype": judo.float},
            "cum_rewards": {"dtype": judo.float},
            "distances": {"dtype": judo.float},
            "clone_probs": {"dtype": judo.float},
            "will_clone": {"dtype": judo.bool},
            "in_bounds": {"dtype": judo.bool},
        }
        state_dict = super(StatesWalkers, self).get_params_dict()
        params.update(state_dict)
        return params

    def clone(self, **kwargs) -> Tuple[Tensor, Tensor]:
        """Perform the clone only on cum_rewards and id_walkers and reset the other arrays."""
        clone, compas = self.will_clone, self.compas_clone
        self.cum_rewards[clone] = self.cum_rewards[compas][clone]
        self.id_walkers[clone] = self.id_walkers[compas][clone]
        self.virtual_rewards[clone] = self.virtual_rewards[compas][clone]
        return clone, compas

    def reset(self):
        """Clear the internal data of the class."""
        params = self.get_params_dict()
        other_attrs = [name for name in self.keys() if name not in params]
        for attr in other_attrs:
            setattr(self, attr, None)
        self.update(
            id_walkers=judo.zeros(self.n, dtype=judo.hash_type),
            compas_dist=judo.arange(self.n),
            compas_clone=judo.arange(self.n),
            processed_rewards=judo.zeros(self.n, dtype=judo.float),
            cum_rewards=judo.zeros(self.n, dtype=judo.float),
            virtual_rewards=judo.ones(self.n, dtype=judo.float),
            distances=judo.zeros(self.n, dtype=judo.float),
            clone_probs=judo.zeros(self.n, dtype=judo.float),
            will_clone=judo.zeros(self.n, dtype=judo.bool),
            in_bounds=judo.ones(self.n, dtype=judo.bool),
        )

    def _ix(self, index: int):
        # TODO(guillemdb): Allow slicing
        data = {
            k: judo.unsqueeze(v[index]) if judo.is_tensor(v) and "best" not in k else v
            for k, v in self.items()
        }
        return self.__class__(batch_size=1, **data)


class OneWalker(States):
    """
    Represent one walker.

    This class is used for initializing a :class:`Swarm` to a given state without having to
    explicitly define the :class:`StatesEnv`, :class:`StatesModel` and :class:`StatesWalkers`.

    """

    def __init__(
        self,
        state: Tensor,
        observ: Tensor,
        reward: Scalar,
        id_walker=None,
        time=0.0,
        state_dict: StateDict = None,
        **kwargs
    ):
        """
        Initialize a :class:`OneWalker`.

        Args:
            state: Non batched numpy array defining the state of the walker.
            observ: Non batched numpy array defining the observation of the walker.
            reward: typing.Scalar value representing the reward of the walker.
            id_walker: Hash of the provided State. If None it will be calculated when the
                       the :class:`OneWalker` is initialized.
            state_dict: External :class:`typing.StateDict` that overrides the default values.
            time: Time step of the current walker. Measures the length of the path followed \
                  by the walker.
            **kwargs: Additional data needed to define the walker. Its structure \
                      needs to be defined in the provided ``state_dict``. These attributes
                      will be assigned to the :class:`EnvStates` of the :class:`Swarm`.

        """
        self.id_walkers = None
        self.rewards = None
        self.observs = None
        self.states = None
        self.times = None
        self._observs_size = observ.shape
        self._observs_dtype = observ.dtype
        self._states_size = state.shape
        self._states_dtype = state.dtype
        self._rewards_dtype = tensor(reward).dtype
        # Accept external definition of param_dict values
        walkers_dict = self.get_params_dict()
        if state_dict is not None:
            for k, v in state_dict.items():
                if k in ["observs", "states"]:  # These two are parsed from the provided opts
                    continue
                if k in walkers_dict:
                    walkers_dict[k] = v
        super(OneWalker, self).__init__(batch_size=1, state_dict=walkers_dict)
        # Keyword arguments must be defined in state_dict
        if state_dict is not None:
            for k in kwargs.keys():
                if k not in state_dict:
                    raise ValueError(
                        "The provided attributes must be defined in state_dict."
                        "param_dict: %s\n kwargs: %s" % (state_dict, kwargs)
                    )
        self.observs[:] = judo.copy(observ)
        self.states[:] = judo.copy(state)
        self.rewards[:] = judo.copy(reward) if judo.is_tensor(reward) else copy.deepcopy(reward)
        self.times[:] = judo.copy(time) if judo.is_tensor(time) else copy.deepcopy(time)
        self.id_walkers[:] = (
            judo.copy(id_walker.squeeze()) if id_walker is not None else hasher.hash_tensor(state)
        )
        self.update(**kwargs)

    def __repr__(self):
        import numpy

        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            string = (
                "reward: %s\n"
                "time: %s\n"
                "observ: %s\n"
                "state: %s\n"
                "id: %s"
                % (
                    self.rewards[0],
                    self.times[0],
                    self.observs[0].flatten(),
                    self.states[0].flatten(),
                    self.id_walkers[0],
                )
            )
            return string

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of States that can handle all the data generated by the :class:`Walkers`.
        """
        params = {
            "id_walkers": {"dtype": judo.hash_type},
            "rewards": {"dtype": self._rewards_dtype},
            "observs": {"dtype": self._observs_dtype, "size": self._observs_size},
            "states": {"dtype": self._states_dtype, "size": self._states_size},
            "times": {"dtype": judo.float32},
        }
        return params
