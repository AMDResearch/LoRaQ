# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

"""Calibration prompts: 1024 MS-COCO captions (the calibration set of SVDQuant / DeepCompressor)."""

import datasets
import yaml

from ._builder_utils import rolling_hash_seed, shuffled_subset

_CITATION = """
@inproceedings{li2024svdquant,
  title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
  author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
"""

_DESCRIPTION = """
Captions from the Microsoft COCO Captions dataset, as selected for calibration in DeepCompressor
(https://github.com/nunchaku-tech/deepcompressor/blob/main/examples/diffusion/prompts/qdiff.yaml).
"""

_HOMEPAGE = "https://cocodataset.org"

_LICENSE = "COCO annotations: Creative Commons Attribution 4.0 (CC BY 4.0)"

PROMPTS_FILE = "coco_captains_2024_prompts.yaml"


class CocoCaptains2024Config(datasets.BuilderConfig):
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


class CocoCaptains2024(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIG_CLASS = CocoCaptains2024Config
    BUILDER_CONFIGS = [
        CocoCaptains2024Config(name="coco_captains_2024", version=VERSION, description="COCO calibration captions",
                               max_dataset_size=1000, return_gt=False)
    ]
    DEFAULT_CONFIG_NAME = "coco_captains_2024"

    def _info(self):
        features = datasets.Features(
            {
                "filename": datasets.Value("string"),
                "prompt": datasets.Value("string"),
                "seed": datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.download.DownloadManager):
        prompts_path = dl_manager.download(PROMPTS_FILE)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"prompts_path": prompts_path})]

    def _generate_examples(self, prompts_path: str):
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
        items = [(f"{int(k):04d}", v) for k, v in prompts.items()]
        if self.config.max_dataset_size > 0:
            items = sorted(shuffled_subset(items, self.config.max_dataset_size), key=lambda x: x[0])
        for i, (uid, prompt) in enumerate(items):
            yield i, {"filename": uid, "prompt": prompt, "seed": rolling_hash_seed(prompt)}
