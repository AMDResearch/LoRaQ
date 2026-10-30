# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import queue
import threading

import diffusers
import torch
from hydra.utils import instantiate
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.context.base import Context
from src.quantizer.base import InputQuantizer, WeightQuantizer
from src.utils.gpu_utils import GPUS
from src.utils.hook import RemovableHandles
from src.utils.paths import existing_images, fp16_dir, image_tag, quantized_dir, quantizer_name, stack_segments
from src.utils.quantizer_utils import to_be_quantized


def _load_pipelines(pipeline_config) -> list:
    """One pipeline per visible GPU, without progress bars."""
    pipes = []
    for device in GPUS:
        pipe = instantiate(pipeline_config)
        pipe.to(device)
        if hasattr(pipe, "disable_tqdm"):
            pipe.disable_tqdm = True
        elif hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        pipes.append(pipe)
    return pipes


def _release(pipes: list) -> None:
    pipes.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _generate(pipes: list, iterable, n_batches: int, existing: set, output_dir: str, generate_batch) -> None:
    """Generate the batches with missing images, one worker thread per pipeline (GPU).

    A batch is (re)generated if at least one of its images is missing. If a worker fails
    (e.g. out of memory), the other workers stop and the error is raised here.
    """
    work_queue: queue.Queue = queue.Queue(maxsize=len(pipes) * 10)
    stop_event = threading.Event()
    errors = []

    def worker(pipe):
        while not stop_event.is_set():
            try:
                batch = work_queue.get(timeout=1)
            except queue.Empty:
                continue
            if batch is None:
                return
            try:
                generate_batch(pipe, batch, output_dir)
            except BaseException as e:
                errors.append(e)
                stop_event.set()
                return

    def put(item) -> None:
        while not stop_event.is_set():
            try:
                work_queue.put(item, timeout=1)
                return
            except queue.Full:
                continue

    threads = [threading.Thread(target=worker, args=(pipe,)) for pipe in pipes]
    for t in threads:
        t.start()
    try:
        for batch in tqdm(iterable, total=n_batches, desc="Queueing batches"):
            if stop_event.is_set():
                break
            if not set(batch.get("filename")).issubset(existing):
                put(batch)
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt detected, stopping all threads...")
        stop_event.set()
    finally:
        for _ in threads:
            put(None)
        for t in threads:
            t.join()
    if errors:
        raise errors[0]


@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    """Generate the full-precision reference images, then the images of the quantized model
    with the calibrated stages of `cfg.context` applied."""
    global_seed = cfg.get("seed", 0)

    TAG = image_tag(cfg)
    DATASET = cfg.generation.dataset.name
    OUTPUT_DIR = cfg.output_root_dir

    @torch.inference_mode()
    def generate_batch(pipe, data_items, output_dir):
        common_pipe_args = {
            "prompt": data_items.get("prompt"),
            "negative_prompt": "",
            "num_images_per_prompt": 1,
            "num_inference_steps": cfg.model.num_inference_steps,
            "guidance_scale": cfg.model.guidance_scale,
            "width": cfg.model.width,
            "height": cfg.model.height,
        }
        torch.manual_seed(global_seed)
        diffusers.training_utils.set_seed(global_seed)
        generator = [torch.Generator().manual_seed(int(seed)) for seed in data_items.get("seed")]
        images_pil = pipe(**common_pipe_args, generator=generator).images

        for uid, image in zip(data_items.get("filename"), images_pil):
            filename = os.path.join(output_dir, f"{uid}")
            if not os.path.exists(filename):
                image.save(filename)

    context_config = OmegaConf.to_container(cfg.context, resolve=True)
    pipeline_config = OmegaConf.to_container(cfg.model.wrapper, resolve=True)

    shift_activations = cfg.model.get("shift_activations", False)
    context: Context = Context(context_config, shift_activations=shift_activations)

    log.info(f"Loading {DATASET} dataset...")
    FP16_IMAGES_OUTPUT_DIR = fp16_dir(OUTPUT_DIR, DATASET, TAG)
    os.makedirs(FP16_IMAGES_OUTPUT_DIR, exist_ok=True)
    existing_uids = existing_images(FP16_IMAGES_OUTPUT_DIR)

    iterable, total, n_batches = instantiate(cfg.generation.dataset, return_gt=False)
    log.info(f"Generating {total} images in {n_batches} batches of {cfg.generation.dataset.batch_size}")

    if len(existing_uids) < total:
        log.info(f"{len(existing_uids)} images already exist in {FP16_IMAGES_OUTPUT_DIR}, generating the remaining {total - len(existing_uids)} images...")
        pipes_fp16 = _load_pipelines(cfg.model.pipeline)
        log.info("FP16 pipeline loaded on visible GPUs")
        _generate(pipes_fp16, iterable, n_batches, existing_uids, FP16_IMAGES_OUTPUT_DIR, generate_batch)
        log.info("Releasing FP16 pipeline memory...")
        _release(pipes_fp16)

    iterable, total, n_batches = instantiate(cfg.generation.dataset)

    i_quantizer: InputQuantizer = InputQuantizer(instantiate(cfg.input_quantizer))
    w_quantizer: WeightQuantizer = WeightQuantizer(instantiate(cfg.weight_quantizer))
    pipes_quantized = _load_pipelines(pipeline_config)
    log.info("Quantized pipeline loaded on visible GPUs")

    with context(pipes_quantized, cfg.output_root_dir) as ctx:
        QUANTIZED_IMAGES_OUTPUT_DIR = quantized_dir(
            OUTPUT_DIR,
            stack_segments(ctx.calibrators, shift_activations),
            quantizer_name(i_quantizer, w_quantizer),
            DATASET,
            TAG,
        )
        os.makedirs(QUANTIZED_IMAGES_OUTPUT_DIR, exist_ok=True)
        existing_uids = existing_images(QUANTIZED_IMAGES_OUTPUT_DIR)

        if len(existing_uids) < total:
            log.info(f"{len(existing_uids)} images already exist in {QUANTIZED_IMAGES_OUTPUT_DIR}, generating the remaining {total - len(existing_uids)} images...")
            handles = RemovableHandles()
            for pipe in pipes_quantized:
                gpu_i_quantizer: InputQuantizer = i_quantizer.clone()
                for named_layer in pipe.transformer.named_layers:
                    dtype = named_layer.module.weight.data.dtype
                    if to_be_quantized(named_layer.name, cfg.model.get("input_layers", [])):
                        handles.append(gpu_i_quantizer.hooked_on(named_layer.module))
                    if to_be_quantized(named_layer.name, cfg.model.get("weight_layers", [])):
                        named_layer.module.weight.data.copy_(w_quantizer(named_layer.module.weight.data).to(dtype))
            try:
                _generate(pipes_quantized, iterable, n_batches, existing_uids, QUANTIZED_IMAGES_OUTPUT_DIR, generate_batch)
            finally:
                handles.remove()

        log.info("Releasing quantized pipeline memory...")
        _release(pipes_quantized)
