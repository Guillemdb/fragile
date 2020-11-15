from typing import List, Tuple

import judo
from judo import HistoryTree as JudoTree

from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.typing import NodeId, Tensor


class HistoryTree(JudoTree):
    """
    Tree data structure that keeps track of the visited states.

    It allows to save the :class:`Swarm` data after every iteration and methods to \
    recover the sampled data after the algorithm run.

    The data that will be stored in the graph it's defined in the ``names`` parameter.
    For example:

     - If names is ``["observs", "actions"]``, the observations of every visited \
       state will be stored as node attributes, and actions will be stored as edge attributes.

     - If names if ``["observs", "actions", "next_observs"]`` the same data will be stored,
       but when the data generator methods are called the observation corresponding \
       to the next state will also be returned.

     The attribute ``names`` also defines the order of the data returned by the generator.

     As long as the data is stored in the graph (passing a valid ``names`` list at \
     initialization, the order of the data can be redefined passing the ``names`` \
     parameter to the generator method.

     For example, if the ``names`` passed at initialization is ``["states", "rewards"]``, \
     you can call the generator methods with ``names=["rewards", "states", "next_states"]`` \
     and the returned data will be a tuple containing (rewards, states, next_states).

    """

    def get_states_ids(
        self, walkers_states: StatesWalkers, **kwargs
    ) -> Tuple[Tensor, StatesWalkers]:
        leaf_ids = judo.to_numpy(walkers_states.get("id_walkers"))
        return leaf_ids, walkers_states

    def add_states(
        self,
        parent_ids: List[NodeId],
        env_states: StatesEnv = None,
        model_states: StatesModel = None,
        walkers_states: StatesWalkers = None,
        n_iter: int = None,
    ):
        """
        Update the history of the tree adding the necessary data to recreate a \
        the trajectories sampled by the :class:`Swarm`.

        Args:
            parent_ids: List of states hashes representing the parent nodes of \
                        the current states.
            env_states: :class:`StatesEnv` containing the data that will be \
                        saved as new leaves in the tree.
            model_states: :class:`StatesModel` containing the data that will be \
                        saved as new leaves in the tree.
            walkers_states: :class:`StatesWalkers` containing the data that will be \
                        saved as new leaves in the tree.
            n_iter: Number of iteration of the algorithm when the data was sampled.

        Returns:
            None

        """
        super(HistoryTree, self).add_states(
            parent_ids=parent_ids,
            n_iter=n_iter,
            env_states=env_states,
            model_states=model_states,
            walkers_states=walkers_states,
        )
