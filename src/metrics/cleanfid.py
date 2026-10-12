# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

from typing import Tuple

import numpy as np
from cleanfid import fid as fid_clean
from torch import Tensor
from torch.nn import Module
from torchmetrics.metric import Metric


class CleanFrechetInceptionDistance(Metric):
    """FID computed with clean-fid (https://github.com/GaParmar/clean-fid), "clean" mode."""

    _DEFAULT_MODE_CLEANFID : str = 'clean'
    
    _feature_model : Module
    _np_batch_features_real : list[np.ndarray]
    _np_batch_features_fake : list[np.ndarray]

    def __init__(self) -> None:
        super(CleanFrechetInceptionDistance, self).__init__()

        self._feature_model = fid_clean.build_feature_extractor(self._DEFAULT_MODE_CLEANFID)
        self._np_batch_features_real = []
        self._np_batch_features_fake = []
        
    def to(self, device):
        super().to(device)
        
        self._feature_model = fid_clean.build_feature_extractor(self._DEFAULT_MODE_CLEANFID, device)
        
        return self

    def update(self, resized_tensor_batch : Tensor, real : bool) -> None:
        is_batch_valid, error_message = self._validate_tensor_batch(resized_tensor_batch)

        if not is_batch_valid:
            raise ValueError(error_message)
        
        batch_features = fid_clean.get_batch_features(resized_tensor_batch, self._feature_model, self.device)
        
        if real: 
            self._np_batch_features_real.append( batch_features )
        else:
            self._np_batch_features_fake.append( batch_features )


    def compute(self) -> Tensor:
        mu1, sigma1 = self._get_fid_features( self._np_batch_features_real )
        mu2, sigma2 = self._get_fid_features( self._np_batch_features_fake )
        fid_score = fid_clean.frechet_distance(mu1, sigma1, mu2, sigma2)
        fid_score = float(fid_score)
        return Tensor([fid_score])


    def reset(self) -> None:
        self._np_batch_features_real = []
        self._np_batch_features_fake = []


    def _get_fid_features(self, np_batch_features : list[np.ndarray]):
        np_features = np.concatenate(np_batch_features, axis=0)
        mu = np.mean(np_features, axis=0)
        sigma = np.cov(np_features, rowvar=False)
        return mu, sigma
    
    def _validate_tensor_batch(self, tensor_batch : Tensor) -> Tuple[bool, str | None]:

        if tensor_batch.ndim != 4 or tensor_batch.shape[1:] != (3, 299, 299):
            return (False, f"Invalid tensor batch. Expecting (x, 3, 299, 299), but received {tensor_batch.shape}")
        
        return (True, None)
