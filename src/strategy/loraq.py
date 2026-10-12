# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Low-rank stage (LoRaQ): a quantized low-rank branch L R added to every layer, W = (W - L R) + L R.

L and R start from the truncated SVD of the (smoothed) weight and are optimized in weight space,
first with Adam so that the residual W - L R quantizes well, then with a rotation r learned on the
Stiefel manifold, (L r, r^T R), so that L and R themselves quantize well (`RotWeightOptimizer`).
"""

import itertools
import os
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import torch
from loguru import logger as log
from torch import Tensor

from ..models.blocks.base import BaseModuleWrapper
from ..models.diffusion import GenericDiTBlock
from ..models.patches.lowrank import LowRankBranch
from ..quantizer.base import BaseQuantizer, InputQuantizer, WeightQuantizer
from ..utils.hook import RemovableHandles
from ..utils.quantizer_utils import low_rank_approximations
from .base import SAVED_RANK, GlobalState, Strategy
from .optim.optimizer import WeightOptimizer


class LoRaQStrategy(Strategy):
    """Low-rank stage. `w_base_quantizer` quantizes the residual and `lr_base_quantizer` the
    factors L and R; with `branch_input_quantizer`, the input of the branch is quantized too.

    The rank-`saved_rank` SVD of each weight is cached in `svd_cache_rank_<saved_rank>.pkl`
    and the optimized factors are stored in `results.pkl`.
    """

    def __init__(
        self,
        w_base_quantizer: BaseQuantizer,
        lr_base_quantizer: BaseQuantizer,
        optimizer: WeightOptimizer,
        rank: int,
        batch_size: int = 8,
        branch_input_quantizer: Optional[BaseQuantizer] = None,
        resume: bool = True,
        saved_rank: int = SAVED_RANK,
    ):
        assert saved_rank >= rank, "The cached SVD rank must be at least the branch rank."
        self.rank = rank
        self.saved_rank = saved_rank
        self.resume = resume
        self.batch_size = batch_size
        self.w_quantizer = WeightQuantizer(w_base_quantizer)
        self.lr_quantizer = WeightQuantizer(lr_base_quantizer)
        self.branch_input_quantizer = InputQuantizer(branch_input_quantizer) if branch_input_quantizer is not None else None
        self.optimizer = optimizer
        self.optimizer.quantizer = self.w_quantizer

        name = f"LoRaQ_{rank}_{self.w_quantizer.dtype.name}_{self.lr_quantizer.dtype.name}"
        if self.branch_input_quantizer is not None:
            name += f"_branch-input-{self.branch_input_quantizer.dtype.name}"
        super().__init__(name)

        self.svd = GlobalState()
        self.results = GlobalState()
        self._update_filenames()

    def _update_filenames(self) -> None:
        self.svd_cache_filename = os.path.join(self.output_dir, f"svd_cache_rank_{self.saved_rank}.pkl")
        self.results_filename = os.path.join(self.output_dir, "results.pkl")

    def _left_cast(self, mat: Tensor) -> Tensor:
        return self.lr_quantizer(mat)

    def _right_cast(self, mat: Tensor) -> Tensor:
        return self.lr_quantizer(mat.T).to(mat).T

    def _compute_svd(self, model: GenericDiTBlock, layer_extractor) -> None:
        if self.resume:
            self.svd = GlobalState(self._load_state(self.svd_cache_filename))
        for block in model.blocks:
            if block.name in self.svd:
                continue
            for _, child in layer_extractor(block):
                for named_layer in child.named_layers:
                    if named_layer.name in self.svd[block.name]:
                        continue
                    l_mat, r_mat = low_rank_approximations(named_layer.module.weight, rank=self.saved_rank)
                    self.svd[block.name][named_layer.name] = (l_mat.detach().cpu(), r_mat.detach().cpu())
            self._save_state(self.svd, self.svd_cache_filename)

    def _optimize(self, model: GenericDiTBlock, layer_extractor) -> None:
        """Optimizes the factors of all layers, by batches of matrices of the same shape."""
        weights, l0s, r0s, layers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        with torch.no_grad():
            for _, child in layer_extractor(model):
                for named_layer in child.named_layers:
                    block_name = named_layer.transformer_block.name
                    weight = named_layer.module.weight.data
                    key = str(tuple(weight.shape))
                    l0, r0 = self.svd[block_name][named_layer.name]
                    l0s[key].append(l0[:, :self.rank].to(weight.device).clone())
                    r0s[key].append(r0[:self.rank, :].to(weight.device).clone())
                    weights[key].append(weight)
                    layers[key].append(named_layer)

        def batches(tensors: List[Tensor]) -> Iterable[Tensor]:
            it = iter(tensors)
            while chunk := list(itertools.islice(it, self.batch_size)):
                yield torch.stack(chunk, dim=0)

        for key in weights:
            log.info(f"Optimizing the low-rank branches of the weights of shape {key}")
            new_l, new_r = [], []
            for w, l0, r0 in zip(batches(weights[key]), batches(l0s[key]), batches(r0s[key])):
                l_opt, r_opt = self.optimizer(w, l0, r0)
                new_l.append(l_opt.detach().cpu())
                new_r.append(r_opt.detach().cpu())
                torch.cuda.empty_cache()
            with torch.no_grad():
                l_all, r_all = torch.cat(new_l, dim=0), torch.cat(new_r, dim=0)
                for i, named_layer in enumerate(layers[key]):
                    self.results[named_layer.transformer_block.name][named_layer.name] = (l_all[i].clone(), r_all[i].clone())
        log.info(f"Saving results to {self.results_filename}")
        self._save_state(self.results, self.results_filename)

    def search(self, models: List[GenericDiTBlock], layer_extractor) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        model = next(iter(models))
        self._compute_svd(model, layer_extractor)
        self._optimize(model, layer_extractor)

    def _low_rank_branch(self, named_layer: BaseModuleWrapper) -> Tuple[Tensor, LowRankBranch]:
        """The quantized branch of a layer and the residual weight W - L R."""
        device, dtype = named_layer.module.weight.device, named_layer.module.weight.dtype
        fp_l, fp_r = self.results[named_layer.transformer_block.name][named_layer.name]
        l_mat, r_mat = self._left_cast(fp_l), self._right_cast(fp_r)
        with torch.no_grad():
            branch = LowRankBranch(l_mat, r_mat, device=device, dtype=dtype)
            residual = named_layer.module.weight - branch.weight
        return residual, branch

    def apply(self, models: List[GenericDiTBlock], layer_extractor) -> RemovableHandles:
        handles = super().apply(models, layer_extractor)
        log.info(f"Loading results from {self.results_filename}")
        self.results = GlobalState(self._load_state(self.results_filename))
        for model in models:
            branch_input_quantizer = self.branch_input_quantizer.clone() if self.branch_input_quantizer is not None else None
            for block in model.blocks:
                for _, child in layer_extractor(block):
                    for named_layer in child.named_layers:
                        residual, branch = self._low_rank_branch(named_layer)
                        named_layer.module.weight.data.copy_(residual)
                        if branch_input_quantizer is not None:
                            handles.append(branch_input_quantizer.hooked_on(branch))
                        handles.append(branch.hooked_on(named_layer.module))
        return handles
