from typing import List, Tuple, Union

from judo.data_structures import ReplayMemory

from fragile.core.swarm import Swarm


class SwarmMemory(ReplayMemory):
    """Store replay data extracted from a :class:`HistoryTree`."""

    def __init__(
        self,
        max_size: int,
        names: Union[List[str], Tuple[str]],
        mode: str = "best_leaf",
        min_size: int = None,
    ):
        """
        Initialize a :class:`ReplayMemory`.

        Args:
            max_size: Maximum number of experiences that will be stored.
            names: Names of the replay data attributes that will be stored.
            mode: If ``mode == "best"`` store only data from the best trajectory \
                  of the :class:`Swarm`. Otherwise store data from all the states of \
                  the :class:`HistoryTree`.

            min_size: Minimum number of samples that need to be stored before the \
                     replay memory is considered ready. If ``None`` it will be equal \
                     to max_size.

        """
        super(SwarmMemory, self).__init__(max_size=max_size, min_size=min_size, names=names)
        self.mode = mode

    def append_swarm(self, swarm: Swarm, mode=None):
        """
        Extract the replay data from a :class:`Swarm` and incorporate it to the \
        already saved experiences.
        """
        # extract data from the swarm
        mode = self.mode if mode is None else mode
        if mode == "best_state":
            data = next(swarm.tree.iterate_branch(swarm.best_id, batch_size=-1, names=self.names))
            self.append(**dict(zip(self.names, data)))
        elif mode == "best_leaf":
            best_leaf = swarm.walkers.states.id_walkers[swarm.get("cum_rewards").argmax()]
            data = next(swarm.tree.iterate_branch(best_leaf, batch_size=-1, names=self.names))
            self.append(**dict(zip(self.names, data)))
        elif mode == "branches":
            for node_id in swarm.tree.leafs:
                data = next(swarm.tree.iterate_branch(node_id, batch_size=-1, names=self.names))
                self.append(**dict(zip(self.names, data)))
        else:
            data = next(swarm.tree.iterate_nodes_at_random(batch_size=-1, names=self.names))
            self.append(**dict(zip(self.names, data)))
        # Concatenate the data to the current memory
        self._log.info("Memory now contains %s samples" % len(self))
