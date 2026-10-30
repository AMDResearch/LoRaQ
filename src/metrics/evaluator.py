# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
import torch
from cleanfid.resize import build_resizer
from loguru import logger as log
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from ..dataset import get_dataset


class EvaluationDataset(torch.utils.data.Dataset):
    """Pairs each prompt of the dataset with its full-precision (reference) and quantized
    (prediction) images, matched by file name, and with the dataset's image (for FID)."""

    def __init__(self, cfg_dataset, reference_folder: str, prediction_folder: str):
        dataset, num_total, num_batches = get_dataset(**cfg_dataset)
        self.image_names: list[str] = []
        self.prompts: list[str] = []
        self.ground_truth_images = {}
        for item in tqdm(dataset, total=num_batches, desc=f"Loading dataset ({num_batches} batch(es))", colour="yellow"):
            for image, filename, prompt in zip(item["image"], item["filename"], item["prompt"]):
                self.prompts.append(prompt)
                self.image_names.append(filename)
                self.ground_truth_images[filename] = image
        del dataset

        self.reference_folder: str = reference_folder
        self.prediction_folder: str = prediction_folder
        missing = len(set(self.image_names) - set(os.listdir(reference_folder)).intersection(os.listdir(prediction_folder)))
        if missing:
            raise FileNotFoundError(f"{missing} image(s) missing in {reference_folder} or {prediction_folder}: run operation=generation first")

        self.to_tensor = T.ToTensor()
        self.fid_resizer = build_resizer("clean")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        prompt = self.prompts[idx]
        ref_path = os.path.join(self.reference_folder, image_name)
        pred_path = os.path.join(self.prediction_folder, image_name)

        reference_image = Image.open(ref_path).convert("RGB")
        prediction_image = Image.open(pred_path).convert("RGB")
        if reference_image.size != prediction_image.size:
            reference_image = reference_image.resize(prediction_image.size, Image.Resampling.BICUBIC)

        prediction_tensor = self.to_tensor(prediction_image)
        reference_tensor = self.to_tensor(reference_image)

        ground_truth_image = self.ground_truth_images[image_name].convert("RGB")
        ground_truth_tensor = self.to_tensor(ground_truth_image)

        resized_prediction_tensor = self._resize(prediction_image)
        resized_ground_truth_tensor = self._resize(ground_truth_image)

        return reference_tensor, prediction_tensor, ground_truth_tensor, resized_prediction_tensor, resized_ground_truth_tensor, prompt, ref_path, pred_path

    def _resize(self, image: Image.Image):
        """FID input: clean-fid "clean" resize to 299x299, as a float tensor in [0, 255]."""
        resized_image = self.fid_resizer(np.array(image))
        resized_tensor = self.to_tensor(resized_image)
        if resized_image.dtype == "uint8":
            resized_tensor = resized_tensor * 255
        return resized_tensor


class Evaluator:
    def __init__(self,
                 cfg_dataset,
                 metrics: list,
                 reference_folder: str,
                 prediction_folder: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 device: str = "cuda"):
        self.dataset = EvaluationDataset(cfg_dataset, reference_folder=reference_folder, prediction_folder=prediction_folder)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        self.device = device
        self.metrics = metrics
        for metric in self.metrics:
            metric.to(self.device)

    def _evaluate(self):
        for batch in tqdm(self.dataloader, desc="Evaluating", colour="blue"):
            reference_tensor, prediction_tensor, groundtruth_tensor, resized_prediction_tensor, resized_groundtruth_tensor, prompt, ref_path, pred_path = batch
            reference_tensor = reference_tensor.to(self.device)
            prediction_tensor = prediction_tensor.to(self.device)
            groundtruth_tensor = groundtruth_tensor.to(self.device)
            for metric in self.metrics:
                metric.update(reference_tensor, 
                              prediction_tensor, 
                              groundtruth_tensor, 
                              resized_prediction_tensor, 
                              resized_groundtruth_tensor,
                              prompt, 
                              ref_path, 
                              pred_path)
            
    def _get_results(self):
        results = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results
    
    def save_results(self, results: dict, output_filename: str):
        summary_df = pd.DataFrame([results])
        summary_csv_path = os.path.join(f"{output_filename}.csv")
        summary_md_path = os.path.join(f"{output_filename}.md")
        summary_df.to_csv(summary_csv_path, index=False)
        with open(summary_md_path, 'w') as f:
            f.write(summary_df.to_markdown(index=False))
        log.info(f"Results saved to {summary_csv_path} and {summary_md_path}")
        
    def _cleanup(self):
        for metric in self.metrics:
            metric.reset()
        torch.cuda.empty_cache()
    
    def __call__(self):
        self._evaluate()
        results = self._get_results()
        self._cleanup()
        return results
