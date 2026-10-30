# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from typing import Any, Dict, List

from loguru import logger as log

from ..models.diffusion import GenericDiTBlock
from ..utils.hook import RemovableHandles

SAVED_RANK = 256


class GlobalState(dict):
    """Nested dict (transformer block -> layer -> result) that creates missing levels on access.

    Results are pickled with this class, so it must stay importable as `src.strategy.base.GlobalState`.
    """

    def __missing__(self, key):
        value = GlobalState()
        self[key] = value
        return value


class Strategy:
    """A calibration stage. `search` fits it on the models, `apply` installs its results.

    Every stage writes into its own directory, named after the stage (`self.name`, built
    from its hyperparameters) and nested in the directory of the previous stage.
    """

    def __init__(self, strategy_name: str):
        self.name: str = strategy_name
        self._output_dir: str = ""

    @property
    def path_segment(self) -> str:
        """The directory component of this stage."""
        return self.name

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str) -> None:
        """Set to the directory of the previous stage; this stage nests its own directory in it."""
        self._output_dir = os.path.join(output_dir, self.path_segment)
        self._update_filenames()

    def _update_filenames(self) -> None:
        """Recompute paths derived from `output_dir` (called on every assignment)."""

    def search(self, models: List[GenericDiTBlock], layer_extractor) -> None:
        raise NotImplementedError

    def apply(self, models: List[GenericDiTBlock], layer_extractor) -> RemovableHandles:
        """Install the stage's results on the models; returns the handles to remove them."""
        log.info(f"Applying strategy {self.name} to the model...")
        return RemovableHandles()

    @staticmethod
    def _save_state(state: Dict[str, Any], filename: str):
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def _load_state(filename: str) -> Dict[str, Any]:
        result = {}
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                result = pickle.load(f)
        else:
            log.warning(f"File {filename} does not exist. Returning empty state.")
        return result
