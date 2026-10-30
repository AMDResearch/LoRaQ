# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import os
import typing as tp
from contextlib import AbstractContextManager

from hydra.utils import instantiate
from loguru import logger as log

from src.utils.hook import RemovableHandles
from src.utils.paths import SHIFTED_ACTS_SEGMENT, stack_path

from ..calibrator import DiTCalibrator
from ..models import GenericDiTBlock


class Context(AbstractContextManager):
    """Applies a stack of already-calibrated stages to the models, in order.

    Usage: ``with Context(cfg.context)(models, output_root_dir) as ctx: ...``. Entering
    loads each stage's results and installs its hooks; exiting removes them.
    """

    def __init__(self, context_cfg, shift_activations: bool = False) -> None:
        self.calibrators: tp.List[DiTCalibrator] = [
            instantiate(calibrator_cfg) for calibrator_cfg in context_cfg
        ]
        log.info(f"Context: {len(self.calibrators)} calibrator(s) initialized")
        self.output_dir: str = ""
        self.shift_activations: bool = shift_activations
        if self.shift_activations:
            log.info("Context is set to shift activations in FeedForward blocks")
        self._handles = RemovableHandles()
        self._models: tp.Optional[tp.List[GenericDiTBlock]] = None
        self._root: str = ""

    def __call__(self, models: tp.List[GenericDiTBlock], output_dir: str) -> "Context":
        """Record what to apply. The hooks are installed in `__enter__`."""
        self._models = models
        self._root = output_dir
        self.output_dir = stack_path(output_dir, self.calibrators, self.shift_activations)
        return self

    def _install(self) -> None:
        models, output_dir = self._models, self._root

        if self.shift_activations:
            log.info("Shifting activations in all FeedForward blocks...")
            for model in models:
                for block in model.feed_forwards:
                    block.shift_activations()

        if len(self.calibrators) == 0:
            log.info("No calibrators in the context, the model is not altered")

        current = os.path.join(output_dir, SHIFTED_ACTS_SEGMENT) if self.shift_activations else output_dir
        for calibrator in self.calibrators:
            calibrator.output_dir = current
            self._handles.append(calibrator.load_state_onto(models))
            current = calibrator.output_dir
            log.info(f"Applied stage from {current}")

        assert self.output_dir == current, (
            f"Stack path {self.output_dir} disagrees with the directory threaded through the calibrators ({current})."
        )

    def __enter__(self) -> "Context":
        assert self._models is not None, "Use the context as `with context(models, root):`"
        try:
            self._install()
        except BaseException:
            self._handles.remove()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handles.remove()
        if exc_type is not None:
            log.exception(f"Exception occurred: {exc_value}", exc_info=(exc_type, exc_value, traceback))
            raise exc_value
        return False
