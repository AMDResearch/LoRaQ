# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""sDCI prompts (summarized Densely Captioned Images), used for evaluation."""

import os

import datasets
import yaml
from PIL import Image

from ._builder_utils import sha256_seed, shuffled_subset

_CITATION = """\
@InProceedings{Urbanek_2024_CVPR,
    author    = {Urbanek, Jack and Bordes, Florian and Astolfi, Pietro and Williamson, Mary and Sharma, Vasu and Romero-Soriano, Adriana},
    title     = {A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26700-26709}
}
"""  # noqa: E501

_DESCRIPTION = """\
The Densely Captioned Images dataset, or DCI, consists of 7805 images from SA-1B,
 each with a complete description aiming to capture the full visual detail of what is present in the image.
 Much of the description is directly aligned to submasks of the image.
"""

_HOMEPAGE = "https://github.com/facebookresearch/DCI"

_LICENSE = "Attribution-NonCommercial 4.0 International (https://github.com/facebookresearch/DCI/blob/main/LICENSE)"

IMAGE_URL = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.gz"
PROMPT_URL = "https://huggingface.co/datasets/mit-han-lab/svdquant-datasets/resolve/main/sDCI.yaml"


class DCIConfig(datasets.BuilderConfig):
    def __init__(self, max_dataset_size: int = -1, return_gt: bool = False, **kwargs):
        super().__init__(
            name=kwargs.get("name", "default"),
            version=kwargs.get("version", "0.0.0"),
            data_dir=kwargs.get("data_dir", None),
            data_files=kwargs.get("data_files", None),
            description=kwargs.get("description", None),
        )
        self.max_dataset_size = max_dataset_size
        self.return_gt = return_gt


class DCI(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = DCIConfig
    BUILDER_CONFIGS = [DCIConfig(name="sdci", version=VERSION, description="sDCI prompts")]
    DEFAULT_CONFIG_NAME = "sdci"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "seed": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        meta_path = dl_manager.download(PROMPT_URL)
        image_root = dl_manager.download_and_extract(IMAGE_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"meta_path": meta_path, "image_root": image_root})
        ]

    def _generate_examples(self, meta_path: str, image_root: str):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        names = shuffled_subset(list(meta.keys()), self.config.max_dataset_size)
        for i, name in enumerate(names):
            yield i, {
                "filename": f"{name}.png",
                "image": Image.open(os.path.join(image_root, f"{name}.jpg")) if self.config.return_gt else None,
                "prompt": meta[name],
                "seed": sha256_seed(name),
            }
