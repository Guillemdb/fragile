from typing import Callable, Generator, List, Set, Tuple, Union

from judo.typing import Scalar, StateDict, Tensor  # noqa: F401

DistanceFunction = Callable[[Tensor, Tensor], Tensor]

NodeId = Union[str, int]
NodeData = Union[Tuple[dict, dict], Tuple[dict, dict, dict]]
NodeDataGenerator = Generator[NodeData, None, None]
NamesData = Union[Tuple[str], Set[str], List[str]]
