from typing import Callable

import numpy

from fragile.backend.data_types import dtype, typing
from fragile.backend.fragile_tensor import tensor
from fragile.backend.functions.random import random_state


AVAILABLE_FUNCTIONS = {
    "l2_norm",
    "relativize",
    "get_alive_indexes",
    "calculate_virtual_reward",
    "calculate_clone",
    "calculate_distance",
    "fai_iteration",
    "cross_virtual_reward",
    "cross_clone",
    "cross_fai_iteration",
}


def l2_norm(x: typing.Tensor, y: typing.Tensor) -> typing.Tensor:
    """Euclidean distance between two batches of points stacked across the first dimension."""
    return tensor.norm(x - y, axis=1)


def relativize(x: typing.Tensor) -> typing.Tensor:
    """Normalize the data using a custom smoothing technique."""
    orig = x
    x = tensor.astype(x, dtype.float)
    std = x.std()
    if float(std) == 0:
        return tensor.ones(len(x), dtype=orig.dtype)
    standard = (x - x.mean()) / std
    with numpy.errstate(invalid="ignore", divide="ignore"):
        res = tensor.where(standard > 0.0, tensor.log(1.0 + standard) + 1.0, tensor.exp(standard))
    # standard[standard > 0] = tensor.log(1.0 + standard[standard > 0]) + 1.0
    # standard[standard <= 0] = tensor.exp(standard[standard <= 0])
    return res


def get_alive_indexes(oobs: typing.Tensor):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if tensor.all(oobs):
        return tensor.arange(len(oobs))
    ix = tensor.logical_not(oobs).flatten()
    return random_state.choice(
        tensor.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix)
    )


def calculate_distance(
    observs: typing.Tensor,
    distance_function: Callable = l2_norm,
    return_compas: bool = False,
    oobs: typing.Tensor = None,
    compas: typing.Tensor = None,
):
    """Calculate a distance metric for each walker with respect to a random companion."""
    if compas is None:
        compas = get_alive_indexes(oobs) if oobs is not None else tensor.arange(observs.shape[0])
        compas = random_state.permutation(compas)
    flattened_observs = observs.view(observs.shape[0], -1)
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    return distance_norm if not return_compas else (distance_norm, compas)


def calculate_virtual_reward(
    observs: typing.Tensor,
    rewards: typing.Tensor,
    oobs: typing.Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: typing.Tensor = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards given the required data."""

    compas = get_alive_indexes(oobs) if oobs is not None else tensor.arange(len(rewards))
    compas = random_state.permutation(compas)
    flattened_observs = observs.reshape(len(oobs), -1)
    other_reward = other_reward.flatten() if dtype.is_tensor(other_reward) else other_reward
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards)

    virtual_reward = distance_norm ** dist_coef * rewards_norm ** reward_coef * other_reward
    return virtual_reward.flatten() if not return_compas else (virtual_reward.flatten(), compas)


def calculate_clone(virtual_rewards: typing.Tensor, oobs: typing.Tensor = None, eps=1e-3):
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = (
        get_alive_indexes(oobs) if oobs is not None else tensor.arange(len(virtual_rewards))
    )
    compas_ix = random_state.permutation(compas_ix)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / tensor.where(
        vir_rew > eps, vir_rew, tensor(eps)
    )
    will_clone = clone_probs.flatten() > random_state.random(len(clone_probs))
    return compas_ix, will_clone


def fai_iteration(
    observs: typing.Tensor,
    rewards: typing.Tensor,
    oobs: typing.Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: typing.Tensor = 1.0,
):
    """Perform a FAI iteration."""
    oobs = oobs if oobs is not None else tensor.zeros(rewards.shape, dtype=dtype.bool)
    virtual_reward = calculate_virtual_reward(
        observs,
        rewards,
        oobs,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        other_reward=other_reward,
    )
    compas_ix, will_clone = calculate_clone(virtual_rewards=virtual_reward, oobs=oobs, eps=eps)
    return compas_ix, will_clone


def cross_virtual_reward(
    host_observs: typing.Tensor,
    host_rewards: typing.Tensor,
    ext_observs: typing.Tensor,
    ext_rewards: typing.Tensor,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards between two cloud of points."""
    host_observs = host_observs.reshape(len(host_rewards), -1)
    ext_observs = ext_observs.reshape(len(ext_rewards), -1)
    compas_host = random_state.permutation(tensor.arange(len(host_rewards)))
    compas_ext = random_state.permutation(tensor.arange(len(ext_rewards)))

    # TODO: check if it's better for the distances to be the same for host and ext
    h_dist = distance_function(host_observs, ext_observs[compas_host])
    e_dist = distance_function(ext_observs, host_observs[compas_ext])
    host_distance = relativize(h_dist.flatten())
    ext_distance = relativize(e_dist.flatten())

    host_rewards = relativize(host_rewards)
    ext_rewards = relativize(ext_rewards)

    host_vr = host_distance ** dist_coef * host_rewards ** reward_coef
    ext_vr = ext_distance ** dist_coef * ext_rewards ** reward_coef
    if return_compas:
        return (host_vr, compas_host), (ext_vr, compas_ext)
    return host_vr, ext_vr


def cross_clone(
    host_virtual_rewards: typing.Tensor,
    ext_virtual_rewards: typing.Tensor,
    host_oobs: typing.Tensor = None,
    eps=1e-3,
):
    """Perform a clone operation between two different groups of points."""
    compas_ix = random_state.permutation(tensor.arange(len(ext_virtual_rewards)))
    host_vr = tensor.astype(host_virtual_rewards.flatten(), dtype=dtype.float32)
    ext_vr = tensor.astype(ext_virtual_rewards.flatten(), dtype=dtype.float32)
    clone_probs = (ext_vr[compas_ix] - host_vr) / tensor.where(
        ext_vr > eps, ext_vr, tensor(eps, dtype=dtype.float32)
    )
    will_clone = clone_probs.flatten() > random_state.random(len(clone_probs))
    if host_oobs is not None:
        will_clone[host_oobs] = True
    return compas_ix, will_clone


def cross_fai_iteration(
    host_observs: typing.Tensor,
    host_rewards: typing.Tensor,
    ext_observs: typing.Tensor,
    ext_rewards: typing.Tensor,
    host_oobs: typing.Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    distance_function: Callable = l2_norm,
    eps: float = 1e-8,
):
    """Perform a FractalAI cloning process between two clouds of points."""
    host_vr, ext_vr = cross_virtual_reward(
        host_observs=host_observs,
        host_rewards=host_rewards,
        ext_observs=ext_observs,
        ext_rewards=ext_rewards,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        distance_function=distance_function,
        return_compas=False,
    )

    compas_ix, will_clone = cross_clone(
        host_virtual_rewards=host_vr, ext_virtual_rewards=ext_vr, host_oobs=host_oobs, eps=eps
    )
    return compas_ix, will_clone
