# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

import torch
from loguru import logger as log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ...quantizer.base import BaseQuantizer, WeightQuantizer
from ...utils.hadamard import random_hadamard_matrix
from .stiefel import CayleySGD


class RotationMixin:
    """Refines a low-rank pair (L, R) with a learned rotation r: (L r, r^T R), r on the Stiefel manifold."""

    @staticmethod
    def rotation_param(*params):
        """Add rotation matrix"""
        L0, *_ = params
        try:
            rot = torch.nn.Parameter(
                random_hadamard_matrix(L0.shape[1], L0.device).to(L0.dtype),
                requires_grad=True,
            )
        except ValueError:
            log.warning(f"No Hadamard matrix of size {L0.shape[1]}, the rotation starts from the identity")
            rot = torch.nn.Parameter(
                torch.eye(L0.shape[1], device=L0.device, dtype=L0.dtype),
                requires_grad=True,
            )
        return rot

    @staticmethod
    def rotation_optimizers(rot, lr):
        return CayleySGD([rot], lr=lr)

    @staticmethod
    def apply_rotation(L0, R0, rot):
        """Apply rotation transformation"""
        return torch.mm(L0, rot), torch.mm(rot.T, R0)

    @staticmethod
    def rotation_loss(criterion, quantizer, L0, R0):
        """Compute rotation-specific loss"""
        L0_quantized = quantizer(L0).to(L0)
        R0_quantized = quantizer(R0.T).T.to(R0)
        return criterion(L0_quantized, L0) + criterion(R0_quantized, R0)

    @staticmethod
    def optimize_rotation_phase(
        criterion, quantizer, preprocess_params, epochs, lr, *params
    ):
        """Separate optimization phase for rotation"""
        rot = RotationMixin.rotation_param(*params)
        ref_L, ref_R = preprocess_params(*params)
        ref_L = ref_L.detach()
        ref_R = ref_R.detach()
        optimizer = RotationMixin.rotation_optimizers(rot, lr)
        if __debug__:
            with torch.no_grad():
                L0_rot, R0_rot = RotationMixin.apply_rotation(ref_L, ref_R, rot)
                initial_loss = f"{criterion(L0_rot @ R0_rot, quantizer(L0_rot).to(L0_rot) @ quantizer(R0_rot.T).T.to(R0_rot)).item():.4e}"
                log.debug(
                    f"\tInitial rotation loss (LrrR - Q(Lr)Q(rR)): {initial_loss}"
                )
                log.debug(
                    f"\tReference rotation Loss (LR - Q(L)Q(R)): {criterion(ref_L @ ref_R, quantizer(ref_L) @ quantizer(ref_R.T).T.to(ref_R)).item():.4e}"
                )
                log.debug(
                    f"\tReference rotation Loss (T): {criterion(ref_L @ ref_R, quantizer(ref_L.T).T @ quantizer(ref_R).to(ref_R)).item():.4e}"
                )
        for _ in (
            pbar := tqdm(range(epochs), colour="yellow", leave=False, position=1)
        ):
            optimizer.zero_grad()
            L0_rot, R0_rot = RotationMixin.apply_rotation(ref_L, ref_R, rot)
            loss = RotationMixin.rotation_loss(criterion, quantizer, L0_rot, R0_rot)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f"Rotation phase {quantizer.dtype.name} - Loss: {loss.item():.4e}"
            )
        if __debug__:
            with torch.no_grad():
                L0_rot, R0_rot = RotationMixin.apply_rotation(ref_L, ref_R, rot)
                final_loss = f"{criterion(L0_rot @ R0_rot, quantizer(L0_rot) @ quantizer(R0_rot.T).T.to(R0_rot)).item():.4e}"
                log.debug(f"\nFinal rotation loss (LrrR - Q(Lr)Q(rR)): {final_loss}")
        return L0_rot, R0_rot


class WeightOptimizer:
    """Optimizes a low-rank pair (L, R) so that W - L R quantizes with minimal error."""

    def __init__(
        self,
        w_base_quantizer: BaseQuantizer,
        lr: float,
        epochs: int,
        criterion: torch.nn.Module,
        optimizer: str = "Adam",
    ) -> None:
        self.name: str = self.__class__.__name__
        self.lr: float = lr
        self.criterion: torch.nn.Module = criterion
        self.optimizer_class: torch.optim.Optimizer = getattr(torch.optim, optimizer)
        self.epochs: int = epochs
        self.quantizer = WeightQuantizer(w_base_quantizer)

    def _loss(self, weight, *params):
        L0, R0 = self._preprocess_params(*params)
        lowrank = L0 @ R0
        x = weight - lowrank
        x_quantized = self.quantizer(x).to(x)
        loss = self.criterion(x, x_quantized)
        return loss

    def _preprocess_params(self, *params):
        L0, R0, *_ = params
        return L0, R0

    def _optimize_loop(self, optimizers: list[torch.optim.Optimizer], weight, *params):
        initial_params = [param.clone() for param in params]
        with logging_redirect_tqdm():
            for _ in (
                pbar := tqdm(
                    range(self.epochs), colour="green", position=0, leave=False
                )
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss = self._loss(weight, *params)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                with torch.no_grad():
                    param_diffs = [
                        (i - j).abs().mean() for i, j in zip(initial_params, params)
                    ]
                    norm_str = " ".join(
                        [
                            f"||d{i}||_1: {norm:.2e}"
                            for i, norm in enumerate(param_diffs)
                        ]
                    )
                    pbar.set_description(
                        f"Optimizing {self.name} - Loss: {loss.item():.4e} - {norm_str}"
                    )
        return self._preprocess_params(*params)

    def _get_optimizers(self, *params):
        l, r, *_ = params
        return [self.optimizer_class([l, r], lr=self.lr)]

    def _get_params(self, *args):
        L0, R0, *_ = args
        L0 = torch.nn.Parameter(L0.to(torch.float32))
        R0 = torch.nn.Parameter(R0.to(torch.float32))
        return L0, R0

    def __call__(self, weight: torch.Tensor, *args) -> torch.Tensor:
        x = weight.to(torch.float32)
        args = [arg.to(torch.float32) for arg in args]
        params = self._get_params(*args)
        optimizers = self._get_optimizers(*params)
        if __debug__:
            with torch.no_grad():
                initial_loss = self._loss(x, *params)
                log.debug(
                    f"Initial (batch) loss: {initial_loss.item():.4e} \t Reference Loss : {self.criterion(x, self.quantizer(x).to(x)).item():.4e}"
                )
        L0, R0 = self._optimize_loop(optimizers, x, *params)
        if __debug__:
            with torch.no_grad():
                lowrank = L0 @ R0
                x = weight - lowrank
                x_quantized = self.quantizer(x).to(x)
                final_loss = self.criterion(x, x_quantized)
                log.debug(
                    f"Final (batch) loss (Full Precision): {final_loss.item():.4e}"
                )
        return L0, R0


class RotWeightOptimizer(RotationMixin, WeightOptimizer):
    """`WeightOptimizer` (Adam on L, R), then, per sample, a Stiefel-optimized rotation of (L, R)
    that minimizes the quantization error of the quantized low-rank factors."""

    def __init__(
        self,
        w_base_quantizer: BaseQuantizer,
        lr_base_quantizer: BaseQuantizer,
        lr: float,
        lr_rot: float,
        epochs: int,
        epochs_rot: int,
        criterion: torch.nn.Module,
        optimizer: str = "Adam",
    ) -> None:
        super(RotWeightOptimizer, self).__init__(
            w_base_quantizer, lr, epochs, criterion, optimizer
        )
        self.lr_quantizer = WeightQuantizer(lr_base_quantizer)
        self.lr_rot = lr_rot
        self.epochs_rot = epochs_rot

    def _optimize_loop(self, optimizers, weight, *params):
        super(RotWeightOptimizer, self)._optimize_loop(optimizers, weight, *params)
        L0, *_ = params
        result_L = []
        result_R = []
        with logging_redirect_tqdm():
            for i in tqdm(range(L0.shape[0]), colour="cyan", position=0, leave=False):
                current_params = [param[i] for param in params]
                if __debug__:
                    with torch.no_grad():
                        l, r = self._preprocess_params(*current_params)
                        log.debug(
                            f"Full precision loss for sample {i}: {self.criterion(weight[i], self.quantizer(weight[i] - l @ r).to(weight) + l @ r).item():.4e} : ||W - Q(W - LR) + LR||"
                        )
                        l = self.lr_quantizer(l).to(l)
                        r = self.lr_quantizer(r.T).T.to(r)
                        log.debug(
                            f"Reference Loss for sample {i}: {self.criterion(weight[i], self.quantizer(weight[i])).item():.4e} : ||W - Q(W)||" 
                        )
                        log.debug(
                            f"Initial Loss for sample {i}: {self.criterion(weight[i], self.quantizer(weight[i] - l @ r).to(weight) + l @ r).item():.4e} : ||W - Q(W - Q(L)Q(R)) + Q(L)Q(R)||"
                        )
                l, r = self.optimize_rotation_phase(
                    self.criterion,
                    self.lr_quantizer,
                    self._preprocess_params,
                    self.epochs_rot,
                    self.lr_rot,
                    *current_params,
                )
                if __debug__:
                    with torch.no_grad():
                        l_h = self.lr_quantizer(l).to(l)
                        r_h = self.lr_quantizer(r.T).T.to(r)
                        log.debug(
                            f"Final Loss for sample {i}: {self.criterion(weight[i], self.quantizer(weight[i] - l_h @ r_h).to(weight) + l_h @ r_h).item():.4e} : ||W - Q(W - Q(Lr)Q(rR)) + Q(Lr)Q(rR)||"
                        )
                result_L.append(l)
                result_R.append(r)
        return torch.stack(result_L, dim=0), torch.stack(result_R, dim=0)
