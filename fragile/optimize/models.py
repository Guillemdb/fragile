from judo import random_state
import numpy

from fragile.core.models import NormalContinuous
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers


class ESModel(NormalContinuous):
    """
    The ESModel implements an evolutionary strategy policy.

    It mutates randomly some of the coordinates of the best solution found by \
    substituting them with a proposal solution. This proposal solution is the \
    difference between two random permutations of the best solution found.

    It applies a gaussian normal perturbation with a probability given by ``mutation``.
    """

    def __init__(
        self,
        mutation: float = 0.5,
        recombination: float = 0.7,
        random_step_prob: float = 0.1,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`ESModel`.

        Args:
            mutation: Probability of mutating a coordinate of the solution vector.
            recombination: Step size of the update applied to the best solution found.
            random_step_prob: Probability of applying a random normal perturbation.
            *args: Passed to the parent :class:`NormalContinuous`.
            **kwargs: Passed to the parent :class:`NormalContinuous`.

        """
        super(ESModel, self).__init__(*args, **kwargs)
        self.mutation = mutation
        self.recombination = recombination
        self.random_step_prob = random_step_prob

    def sample(
        self,
        batch_size: int,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        **kwargs,
    ) -> StatesModel:
        """
        Calculate the corresponding data to interact with the Environment and \
        store it in model states.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.
            kwargs: Passed to the :class:`Critic` if any.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        # There is a chance of performing a gaussian perturbation
        if random_state.random() < self.random_step_prob:
            return super(ESModel, self).sample(
                batch_size=batch_size, env_states=env_states, model_states=model_states, **kwargs,
            )
        observs = (
            env_states.observs
            if env_states is not None
            else numpy.zeros(((batch_size,) + self.shape))
        )
        has_best = walkers_states is not None and walkers_states.best_state is not None
        best = walkers_states.best_state if has_best else observs
        # Choose 2 random indices
        indexes = numpy.arange(observs.shape[0])
        a_rand = self.random_state.permutation(indexes)
        b_rand = self.random_state.permutation(indexes)
        proposal = best + self.recombination * (observs[a_rand] - observs[b_rand])
        # Randomly mutate each coordinate of the original vector
        assert observs.shape == proposal.shape
        rands = random_state.random(observs.shape)
        perturbations = numpy.where(rands < self.mutation, observs, proposal)
        # The Environment will sum the observations to perform the step
        new_states = perturbations - observs
        actions = self.bounds.clip(new_states)
        return self.update_states_with_critic(
            actions=actions,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            **kwargs
        )


class CMAES(NormalContinuous):
    """Implementation of CMAES algorithm from https://en.wikipedia.org/wiki/CMA-ES."""

    def __init__(self, sigma: float, virtual_reward_fitness: bool = False, *args, **kwargs):
        """
        Initialize a :class:`CMAES`.

        Args:
            sigma: Coordinate-wise standard deviation.
            virtual_reward_fitness: Use the virtual reward instead of the environment \
                                    reward to as a fitness measure.
            *args: Passed to :class:`NormalContinuous`.__init__.
            **kwargs: Passed to :class:`NormalContinuous`.__init__.

        """
        super(CMAES, self).__init__(*args, **kwargs)
        # Parameters for selection
        self.virtual_reward_fitness = virtual_reward_fitness
        self.pop_size = None
        self.sigma = sigma
        self._count_eval = 0  # Number of function evaluations performed
        self.x_mean = None  # Objective variables initial point
        self.old_x_mean = None  # x_mean corresponding to the last iteration
        # Parameters for adaptation that are kept constant
        self.mu_const = None  # Number of parents/points for recombination
        self.weights_const = None  # Array for weighted recombination
        self.mu_eff_const = None  # variance-effectiveness of sum w_i x_i
        self.cum_covm_const = None  # Time constant for accumulation for cov_matrix
        self.cum_sigma_const = None  # Time constant for accumulation for sigma control
        self.lr_covrank1_const = None  # Learning rate for rank-one update of cov_matrix
        self.lr_mu_const = None  # Learning rate for rank-mu_const update
        self.damp_sigma_const = None  # Damping for sigma. Usually close to one.
        self.chi_norm_const = None  # expectation of  ||N(0,I)|| == norm(randn(N,1))
        # Dynamic (internal) strategy parameters and constants
        self.paths_covm = None  # Evolution paths for cov_matrix
        self.paths_sigma = None  # Evolution paths for sigma
        self.coords_matrix = None  # Defines the coordinate system
        self.scaling_diag = None  # Diagonal matrix that defines the scaling
        self.cov_matrix = None  # Covariance matrix
        self.invsqrtC = None  # cov_matrix^-1/2
        self.n_eigen_eval = None  # Tracks update of coords_matrix and scaling_diag
        self.artmp = None  # Temporal array that contains terms for updating the covariance matrix
        self.hsig = None

    def sample(
        self,
        batch_size: int,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        **kwargs,
    ) -> StatesModel:
        """
        Calculate the corresponding data to interact with the Environment and \
        store it in model states.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.
            kwargs: Passed to the :class:`Critic` if any.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        if model_states is None or walkers_states is None:
            return super(CMAES, self).sample(
                batch_size=batch_size,
                model_states=model_states,
                env_states=env_states,
                walkers_states=walkers_states,
                **kwargs
            )
        actions = (
            env_states.get("observs")
            if self._count_eval > self.pop_size * 2
            else model_states.get("actions")
        )
        fitness = (
            walkers_states.get("virtual_rewards")
            if self.virtual_reward_fitness
            else walkers_states.get("cum_rewards")
        )
        sorted_fitness = numpy.argsort(fitness)[: self.mu_const]
        selected_actions = actions[sorted_fitness].T
        self._update_evolution_paths(selected_actions)
        self._adapt_covariance_matrix(selected_actions)
        self._adapt_sigma()
        self._cov_matrix_diagonalization()

        actions = self._sample_actions()
        return self.update_states_with_critic(
            actions=actions, batch_size=batch_size, model_states=model_states, **kwargs
        )

    def _update_evolution_paths(self, actions):
        self.old_x_mean = self.x_mean
        self.x_mean = numpy.matmul(actions, self.weights_const)
        # Update evolution paths
        self.paths_sigma = (1 - self.cum_sigma_const) * self.paths_sigma + numpy.sqrt(
            self.cum_sigma_const * (2 - self.cum_sigma_const) * self.mu_eff_const
        ) * numpy.matmul(self.invsqrtC, (self.x_mean - self.old_x_mean)) / self.sigma
        sqrt_term = numpy.sqrt(
            1 - (1 - self.cum_sigma_const) ** (2 * self._count_eval / self.pop_size)
        )
        self.hsig = float(
            numpy.linalg.norm(self.paths_sigma) / sqrt_term / self.chi_norm_const
            < 1.4 + 2 / (self.n_dims + 1)
        )
        self.paths_covm = (1 - self.cum_covm_const) * self.paths_covm + self.hsig * numpy.sqrt(
            self.cum_covm_const * (2 - self.cum_covm_const) * self.mu_eff_const
        ) * (self.x_mean - self.old_x_mean) / self.sigma

    def _adapt_covariance_matrix(self, actions):
        self.artmp = (1 / self.sigma) * (actions - numpy.tile(self.old_x_mean, (1, self.mu_const)))

        self.cov_matrix = (
            (1 - self.lr_covrank1_const - self.lr_mu_const) * self.cov_matrix
            + self.lr_covrank1_const * numpy.matmul(self.paths_covm, self.paths_covm.T)
            + (1 - self.hsig) * self.cum_covm_const * (2 - self.cum_covm_const) * self.cov_matrix
            + self.lr_mu_const
            * numpy.matmul(
                numpy.matmul(self.artmp, numpy.diag(self.weights_const.flatten())), self.artmp.T
            )
        )

    def _adapt_sigma(self):
        self.sigma = self.sigma * numpy.exp(
            (self.cum_sigma_const / self.damp_sigma_const)
            * (numpy.linalg.norm(self.paths_sigma) / self.chi_norm_const - 1)
        )

    def _cov_matrix_diagonalization(self):
        # Decomposition of cov_matrix into coords_matrix*diag(scaling_diag.^2)*coords_matrix'
        # (diagonalization)
        if (
            self._count_eval - self.n_eigen_eval
            > self.pop_size / (self.lr_covrank1_const + self.lr_mu_const) / self.n_dims / 10
        ):
            self.n_eigen_eval = self._count_eval
            self.cov_matrix = numpy.triu(self.cov_matrix) + numpy.triu(self.cov_matrix, 1).T
            eigvals, eigvects = numpy.linalg.eig(self.cov_matrix)
            self.scaling_diag = numpy.diag(eigvals)  # [::-1])
            self.coords_matrix = eigvects  # [:, ::-1]
            self.scaling_diag = numpy.sqrt(numpy.diag(self.scaling_diag)).reshape(-1, 1)
            self.invsqrtC = numpy.matmul(
                numpy.matmul(self.coords_matrix, numpy.diag(self.scaling_diag.flatten() ** -1)),
                self.coords_matrix.T,
            )

    def reset(
        self,
        batch_size: int = 1,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        *args,
        **kwargs
    ) -> StatesModel:
        """
        Return a new blank State for a `DiscreteUniform` instance, and a valid \
        prediction based on that new state.

        Args:
            batch_size: Number of walkers that the new model `State`.
            model_states: :class:`StatesModel` corresponding to the model data.
            env_states: :class:`StatesEnv` containing the environment data.
            *args: Passed to `predict`.
            **kwargs: Passed to `predict`.

        Returns:
            New model states containing sampled data.

        """
        self.pop_size = batch_size
        self._count_eval = 0
        self._init_algorithm_params(batch_size)
        # Take the first sample from a random uniform distribution
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        model_states = model_states or self.create_new_states(batch_size=batch_size)
        init_actions = self.random_state.randn(self.mu_const)
        self.x_mean = numpy.matmul(init_actions.T, self.weights_const)
        actions = self._sample_actions()
        model_states.update(actions=actions)
        return model_states

    def _sample_actions(self) -> numpy.ndarray:

        actions = numpy.empty((self.pop_size, self.n_dims))
        for i in range(self.pop_size):
            normal_noise = self.random_state.randn(self.n_dims).reshape(-1, 1)
            action = self.x_mean + self.sigma * numpy.matmul(
                self.coords_matrix, self.scaling_diag * normal_noise
            )
            actions[i, :] = action.flatten()
            self._count_eval += 1
        return actions

    def _init_algorithm_params(self, batch_size):
        self.pop_size = batch_size
        self.mu_const = self.pop_size / 2  # Number of parents/points for recombination
        self.weights_const = numpy.log(self.mu_const + 0.5) - numpy.log(
            numpy.arange(1, self.mu_const + 1)
        ).reshape((-1, 1))
        self.mu_const = int(numpy.floor(self.mu_const))
        self.weights_const = self.weights_const / numpy.sum(self.weights_const)
        self.mu_eff_const = numpy.sum(self.weights_const) ** 2 / numpy.sum(self.weights_const ** 2)
        # Parameters for adaptation
        self.cum_covm_const = (4 + self.mu_eff_const / self.n_dims) / (
            self.n_dims + 4 + 2 * self.mu_eff_const / self.n_dims
        )
        self.cum_sigma_const = (self.mu_eff_const + 2) / (self.n_dims + self.mu_eff_const + 5)
        self.lr_covrank1_const = 2 / ((self.n_dims + 1.3) ** 2 + self.mu_eff_const)
        self.lr_mu_const = numpy.minimum(
            1 - self.lr_covrank1_const,
            2
            * (self.mu_eff_const - 2 + 1 / self.mu_eff_const)
            / ((self.n_dims + 2) ** 2 + self.mu_eff_const),
        )
        self.damp_sigma_const = (
            1
            + 2 * numpy.maximum(0, numpy.sqrt((self.mu_eff_const - 1) / (self.n_dims + 1)) - 1)
            + self.cum_sigma_const
        )
        # Dynamic (internal) strategy parameters and constants
        self.paths_covm = numpy.zeros((self.n_dims, 1))
        self.paths_sigma = numpy.zeros((self.n_dims, 1))
        self.coords_matrix = numpy.eye(self.n_dims)  # Defines the coordinate system
        self.scaling_diag = numpy.ones(
            (self.n_dims, 1)
        )  # Diagonal matrix that defines the scaling
        self.cov_matrix = (
            self.coords_matrix * numpy.diag(self.scaling_diag ** 2) * self.coords_matrix.T
        )  # Covariance matrix
        self.invsqrtC = (
            self.coords_matrix * numpy.diag(1 / self.scaling_diag.flatten()) * self.coords_matrix.T
        )
        self.n_eigen_eval = 0  # Tracks update of coords_matrix and scaling_diag
        self.chi_norm_const = self.n_dims ** 0.5 * (
            1 - 1 / (4 * self.n_dims) + 1 / (21 * self.n_dims ** 2)
        )
