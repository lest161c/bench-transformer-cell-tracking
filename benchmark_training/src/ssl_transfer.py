"""SSL weight-transfer utilities.

Two variants of the transfer are needed because the SSL-pretrained
encoder is a standard ``nn.TransformerEncoder`` whose fused QKV weight
tensor must be split when loading into a
:class:`src.models.SparseEncoder`.

Both helpers mutate ``model`` in place via ``load_state_dict(strict=False)``
and return the number of parameter tensors that were transferred.

Expected caller::

    ssl_state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    if model_uses_sparse:
        init_from_ssl_sparse(model, ssl_state)
    else:
        init_from_ssl_dense(model, ssl_state)
"""

from __future__ import annotations

import logging
from typing import Mapping

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Dense
# ------------------------------------------------------------------ #

def init_from_ssl_dense(
    model: nn.Module,
    ssl_state: Mapping[str, torch.Tensor],
) -> int:
    """Transfer SSL weights to a model with a dense encoder.

    Copies any key whose name and shape match.  The dense encoder
    shares the same ``nn.TransformerEncoder`` key layout as the SSL
    checkpoint, so no transformation is required.

    Parameters
    ----------
    model : nn.Module
        The downstream model.  ``state_dict()`` defines the keys that
        will be considered for transfer.
    ssl_state : Mapping[str, torch.Tensor]
        The ``model_state_dict`` of the SSL checkpoint.

    Returns
    -------
    int
        Number of parameter tensors successfully transferred.
    """
    own = model.state_dict()
    mapped = 0
    for key in own:
        if key in ssl_state and own[key].shape == ssl_state[key].shape:
            own[key] = ssl_state[key].clone()
            mapped += 1
    model.load_state_dict(own, strict=False)
    logger.info("SSL dense transfer: %d/%d keys mapped", mapped, len(own))
    return mapped


# ------------------------------------------------------------------ #
# Sparse
# ------------------------------------------------------------------ #

def init_from_ssl_sparse(
    model: nn.Module,
    ssl_state: Mapping[str, torch.Tensor],
    num_layers: int = 4,
) -> int:
    """Transfer SSL weights to a model with a :class:`SparseEncoder`.

    The SSL checkpoint stores fused QKV tensors as
    ``in_proj_weight`` / ``in_proj_bias``; the sparse encoder expects
    separate ``q_pro``, ``k_pro``, ``v_pro`` tensors.  This helper
    splits the fused tensors and maps them to the corresponding sparse
    parameters.

    The downstream model is expected to use the following key naming
    inside its encoder (``encoder.layers.{i}.attn.{q,k,v}_pro.{weight,bias}``)
    — matching the layout produced by :class:`src.models.SparseEncoder`.

    Parameters
    ----------
    model : nn.Module
        The downstream model.
    ssl_state : Mapping[str, torch.Tensor]
        The ``model_state_dict`` of the SSL checkpoint.
    num_layers : int, default 4
        Number of transformer layers in both encoders.

    Returns
    -------
    int
        Number of parameter tensors successfully transferred.
    """
    own = model.state_dict()
    mapped = 0

    # Feature / coordinate / pair projections and fusion MLP — these
    # are encoder-agnostic and match by name.
    for prefix in ("feat_proj", "coord_proj", "pair_proj", "fusion"):
        for suffix in ("weight", "bias"):
            key = f"{prefix}.{suffix}"
            if key in ssl_state and key in own:
                own[key] = ssl_state[key].clone()
                mapped += 1

    # Per-layer split of the fused QKV + FFN + norm mappings.
    for layer_index in range(num_layers):
        fused_weight = ssl_state.get(
            f"encoder.encoder.layers.{layer_index}.self_attn.in_proj_weight"
        )
        fused_bias = ssl_state.get(
            f"encoder.encoder.layers.{layer_index}.self_attn.in_proj_bias"
        )
        if fused_weight is not None:
            head_dim = fused_weight.shape[0] // 3
            for projection_index, projection_name in enumerate(("q", "k", "v")):
                own[
                    f"encoder.layers.{layer_index}.attn.{projection_name}_pro.weight"
                ] = fused_weight[
                    projection_index * head_dim : (projection_index + 1) * head_dim
                ].clone()
                if fused_bias is not None:
                    own[
                        f"encoder.layers.{layer_index}.attn.{projection_name}_pro.bias"
                    ] = fused_bias[
                        projection_index * head_dim : (projection_index + 1) * head_dim
                    ].clone()
                mapped += 2

        for component in ("linear1", "linear2"):
            for suffix in ("weight", "bias"):
                own_key = f"encoder.layers.{layer_index}.{component}.{suffix}"
                ssl_key = (
                    f"encoder.encoder.layers.{layer_index}.{component}.{suffix}"
                )
                if ssl_key in ssl_state and own_key in own:
                    own[own_key] = ssl_state[ssl_key].clone()
                    mapped += 1

        for norm_name in ("norm1", "norm2"):
            for suffix in ("weight", "bias"):
                own_key = f"encoder.layers.{layer_index}.{norm_name}.{suffix}"
                ssl_key = (
                    f"encoder.encoder.layers.{layer_index}.{norm_name}.{suffix}"
                )
                if ssl_key in ssl_state and own_key in own:
                    own[own_key] = ssl_state[ssl_key].clone()
                    mapped += 1

        for suffix in ("weight", "bias"):
            own_key = f"encoder.layers.{layer_index}.attn.proj.{suffix}"
            ssl_key = (
                f"encoder.encoder.layers.{layer_index}.self_attn.out_proj.{suffix}"
            )
            if ssl_key in ssl_state and own_key in own:
                own[own_key] = ssl_state[ssl_key].clone()
                mapped += 1

    for suffix in ("weight", "bias"):
        own_key = f"encoder.norm.{suffix}"
        if own_key in own and own_key in ssl_state:
            own[own_key] = ssl_state[own_key].clone()
            mapped += 1

    model.load_state_dict(own, strict=False)
    logger.info("SSL sparse transfer: %d keys mapped", mapped)
    return mapped


def init_from_ssl(
    model: nn.Module,
    ssl_state: Mapping[str, torch.Tensor],
    sparse_k: int | None = None,
) -> int:
    """Dispatch to the dense or sparse transfer based on ``sparse_k``.

    Parameters
    ----------
    model : nn.Module
        The downstream model.
    ssl_state : Mapping[str, torch.Tensor]
        SSL checkpoint state dict.
    sparse_k : int | None
        ``None`` or ``0`` selects the dense path; any positive value
        selects the sparse path.

    Returns
    -------
    int
        Number of parameter tensors successfully transferred.
    """
    if sparse_k is None or sparse_k == 0:
        return init_from_ssl_dense(model, ssl_state)
    return init_from_ssl_sparse(model, ssl_state)
