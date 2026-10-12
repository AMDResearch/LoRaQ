# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""MJHQ-30K prompts (and reference images for FID), used for evaluation."""

import json
import os

import datasets
from PIL import Image

from ._builder_utils import sha256_seed, shuffled_subset

_CITATION = """\
@misc{li2024playground,
      title={Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation},
      author={Daiqing Li and Aleks Kamko and Ehsan Akhgari and Ali Sabet and Linmiao Xu and Suhail Doshi},
      year={2024},
      eprint={2402.17245},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DESCRIPTION = """\
We introduce a new benchmark, MJHQ-30K, for automatic evaluation of a model’s aesthetic quality.
 The benchmark computes FID on a high-quality dataset to gauge aesthetic quality.
"""

_HOMEPAGE = "https://huggingface.co/datasets/playgroundai/MJHQ-30K"

_LICENSE = (
    "Playground v2.5 Community License "
    "(https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md)"
)


class MJHQConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, seed_offset: int = 0, **kwargs):
        super().__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt
        self.seed_offset = seed_offset


class MJHQ(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = MJHQConfig
    BUILDER_CONFIGS = [MJHQConfig(name="mjhq", version=VERSION, description="MJHQ-30K")]
    DEFAULT_CONFIG_NAME = "mjhq"

    IMAGE_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/mjhq30k_imgs.zip"
    META_URL = "https://huggingface.co/datasets/playgroundai/MJHQ-30K/resolve/main/meta_data.json"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "category": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "seed": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        meta_path = dl_manager.download(self.META_URL)
        image_root = dl_manager.download_and_extract(self.IMAGE_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path, "image_root": image_root})
        ]

    def _generate_examples(self, meta_path: str, image_root: str):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        names = shuffled_subset(list(meta.keys()), self.config.max_dataset_size)
        for i, name in enumerate(names):
            category = meta[name]["category"]
            image_path = os.path.join(image_root, category, f"{name}.jpg")
            yield i, {
                "filename": f"{category}_{name}.png",
                "category": category,
                "image": Image.open(image_path) if self.config.return_gt else None,
                "prompt": meta[name]["prompt"],
                "seed": sha256_seed(name) + self.config.seed_offset,
            }
