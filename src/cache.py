# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: MIT

import gc
import math
import os
import random
import sys
import threading
from typing import (
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    TypeAlias,
    Type,
)

import diffusers
import torch
from loguru import logger as log
from torch import Tensor, nn
from tqdm import tqdm

from src.models.diffusion.pipeline import DiTPipelineWrapper
from src.models.transformer import GenericTransformerBlockWrapper
from src.utils.hook import (
    Args,
    Kwargs,
    Outputs,
    RemovableHandles,
)

from .utils.gpu_utils import CPU, GPUS

Kwargs_List: TypeAlias = List[Kwargs]

def kwargs_to(device: torch.device, dtype: torch.dtype, **kwargs) -> Kwargs:
    def move_one(v):
        if isinstance(v, (torch.Tensor, torch.nn.Parameter)):
            t = v.detach()
            if device.type == "cuda":
                t = t.to(device="cpu", dtype=dtype, copy=True).pin_memory()
                return t.to(device=device, non_blocking=True)
            else:
                return t.to(device=device, dtype=dtype, copy=True)
        return v
    return {k: move_one(v) for k, v in kwargs.items()}

def concat_tensor_dicts(
    dicts: Iterable[Dict[Hashable, torch.Tensor | None]],
    dim: int = 0,
    device: Optional[torch.device] = torch.device("cpu"),
    dtype: Optional[torch.dtype] = torch.float16,
    strict_keys: bool = True,
) -> Dict[Hashable, torch.Tensor | None]:
    """
    Concatenate values (tensors) of multiple dicts along `dim`, key-wise.
    Requirements:
    - All dicts must have identical keys when strict_keys=True (default).
    - For each key, all tensors must have the same shape on all dims except `dim`.
    - Tensors must share dtype/device unless `dtype`/`device` are provided to coerce.

    Returns a new dict with the same key order as the first dict.
    """
    dicts = list(dicts)
    if not dicts:
        return {}

    keys0 = list(dicts[0].keys())
    if strict_keys:
        for i, d in enumerate(dicts[1:], start=1):
            if set(d.keys()) != set(keys0):
                missing = set(keys0) - set(d.keys())
                extra = set(d.keys()) - set(keys0)
                raise KeyError(
                    f"Dict {i} key mismatch. Missing: {missing or None}, Extra: {extra or None}"
                )
        keys = keys0
    else:
        keys = sorted(set.intersection(*(set(d.keys()) for d in dicts)))

    out: Dict[Hashable, torch.Tensor | None] = {}
    for k in keys:
        ts: List[torch.Tensor] = []
        for i, d in enumerate(dicts):
            t = d[k]
            if t is not None:
                if device is not None or dtype is not None:
                    t = t.to(
                        device=device if device is not None else t.device,
                        dtype=dtype if dtype is not None else t.dtype,
                    )
                ts.append(t)
        if ts:
            ref = ts[0]
            for j, t in enumerate(ts[1:], start=1):
                if t.ndim != ref.ndim:
                    raise ValueError(
                        f"Key {k}: tensor ndim mismatch ({t.ndim} vs {ref.ndim}) at index {j}"
                    )
                for ax in range(ref.ndim):
                    if ax == dim:
                        continue
                    if t.shape[ax] != ref.shape[ax]:
                        raise ValueError(
                            f"Key {k}: shape mismatch at dim {ax} ({t.shape} vs {ref.shape}) at index {j}"
                        )
                if t.dtype != ref.dtype and dtype is None:
                    raise ValueError(
                        f"Key {k}: dtype mismatch ({t.dtype} vs {ref.dtype}); pass dtype= to coerce"
                    )
                if t.device != ref.device and device is None:
                    raise ValueError(
                        f"Key {k}: device mismatch ({t.device} vs {ref.device}); pass device= to coerce"
                    )
            out[k] = torch.cat(ts, dim=dim)
        else:
            out[k] = None
    return out


def move_blocks_to_gpus(blocks_devices) -> list[GenericTransformerBlockWrapper]:
    streams = [torch.cuda.Stream(device=d) for _, d in blocks_devices]
    for (rep, dev), stream in zip(blocks_devices, streams):
        with torch.cuda.device(dev), torch.cuda.stream(stream):
            rep.module.to(dev, non_blocking=True)
    for (_, dev), stream in zip(blocks_devices, streams):
        with torch.cuda.device(dev):
            stream.synchronize()
    return [rep for rep, _ in blocks_devices]


class TransformerBlockBuffer:
    """Calibration inputs of the transformer blocks.

    `buffer` runs the full-precision pipeline on the calibration prompts, records the inputs
    of the first transformer block at every denoising step, samples `n_samples` of them and
    caches them in `sample_filename` (`samples.pt`). The calibration then propagates these
    samples block by block (`run`), with the previous stages applied.
    """

    def __init__(self):
        self.transformer_block_hidden_states: Kwargs_List = []
        self.previous_transformer_block_outputs: Kwargs_List = []
        self.transformer_block_constant_kwargs: Kwargs_List = []
        self.handle = RemovableHandles()
        self.lock = threading.Lock()
        self.global_seed = 0
        self.current_block_name: str = ""
        if not GPUS:
            raise RuntimeError("No GPU available for caching transformer blocks.")
        self.n_gpus: int = len(GPUS) if GPUS else 1
        self.global_constant_kwargs: Kwargs = {}

    @staticmethod
    def _outputs_to_dict(o: Outputs) -> Kwargs:
        if not isinstance(o, tuple):
            o = (o,)
        return {str(k):v.detach() for k,v in zip(range(len(o)), o)}
 
    def make_hook_cache(self, block: GenericTransformerBlockWrapper):
        def hook_cache(
            module: nn.Module, inputs: Args, kwargs: Kwargs, outputs: Outputs
        ) -> None:
            module_dtype = next(module.parameters()).dtype
            input_kwargs, constant_kwargs = block.extract_state_from_inputs(
                inputs, kwargs
            )
            input_kwargs = kwargs_to(CPU, module_dtype, **input_kwargs)
            constant_kwargs = kwargs_to(CPU, module_dtype, **constant_kwargs)
            prev_out = kwargs_to(CPU, module_dtype, **self._outputs_to_dict(outputs))
            with self.lock:
                self.transformer_block_hidden_states.append(input_kwargs)
                self.transformer_block_constant_kwargs.append(constant_kwargs)
                self.previous_transformer_block_outputs.append(prev_out)
        return block.module.register_forward_hook(hook_cache, with_kwargs=True)

    @staticmethod
    def group_iterable(it: Iterable, group_size: int):
        """Yield lists of up to group_size items from iterable it."""
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == group_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _reset(self):
        self.transformer_block_hidden_states.clear()
        self.previous_transformer_block_outputs.clear()
        self.transformer_block_constant_kwargs.clear()
        self.handle.remove()

    @torch.inference_mode()
    def buffer(
        self,
        sample_filename: str,
        n_samples: int,
        pipelines: List[DiTPipelineWrapper],
        checkpoint: int,
        num_steps: int,
        dataset: Iterable,
        n_batches: int,
        guidance_scale: float,
        seed: int = 0,
    ):
        self._reset()
        all_transformer_blocks = [list(pipe.blocks) for pipe in pipelines]
        first_transformer_blocks = [b[0] for b in all_transformer_blocks]
        self.current_block_name = first_transformer_blocks[0].name
        
        assert all([len(b) for b in all_transformer_blocks]) > 0, (
            f"No transformer blocks found in pipelines after checkpoint {checkpoint}"
        )
        if os.path.exists(sample_filename):
            cache = torch.load(sample_filename, map_location="cpu", weights_only=False)
            self.transformer_block_hidden_states = cache[
                "transformer_block_hidden_states"
            ]
            self.transformer_block_constant_kwargs = cache[
                "transformer_block_constant_kwargs"
            ]
            self.previous_transformer_block_outputs = cache[
                "previous_transformer_block_outputs"
            ]
            self.global_constant_kwargs = cache[
                "global_constant_kwargs"
            ]

            log.info(
                f"Loaded {len(self.transformer_block_hidden_states)} batches from cache file {sample_filename}"
            )
        else:
            last_transformer_blocks = [b[-1] for b in all_transformer_blocks]

            if num_steps <= 1:
                num_steps = max(1, int(num_steps))
                log.info(
                    f"Caching with {num_steps} inference step per sample, the early stopping hook will be set to {first_transformer_blocks[0].name}"
                )
                last_transformer_blocks = first_transformer_blocks

            def make_early_stop(num_steps: int = num_steps):
                i = 0
                def hook_earlystop(*args, **kwargs) -> None:
                    nonlocal i
                    i += 1
                    if i >= num_steps:
                        i = 0
                        raise StopIteration(
                            "Early stopping the diffusion process to skip image generation (GPU VRAM overflow prevention)."
                        )
                return hook_earlystop

            def run_pipeline(pipe, gpu_id: int, data_item):
                common_pipe_args = {
                    "prompt": data_item.get("prompt"),
                    "num_images_per_prompt": 1,
                    "num_inference_steps": num_steps,
                    "guidance_scale": guidance_scale,
                }
                torch.manual_seed(self.global_seed)
                diffusers.training_utils.set_seed(
                    self.global_seed
                )
                generators = [
                    torch.Generator(torch.device(f"cuda:{gpu_id}")).manual_seed(
                        int(current_seed)
                    )
                    for current_seed in data_item.get("seed")
                ]  # type: ignore
                try:
                    with torch.no_grad():
                        pipe(
                            **common_pipe_args,
                            generator=generators,
                            output_type="latent",
                        )
                except StopIteration:
                    log.info(
                        f"Pipeline on GPU {gpu_id} early-stopped to prevent VRAM overflow"
                    )
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(gpu_id)
                        torch.cuda.empty_cache()

            for tb0, tbn in zip(first_transformer_blocks, last_transformer_blocks):
                self.handle.append(self.make_hook_cache(tb0))
                self.handle.append(
                    tbn.module.register_forward_hook(
                        make_early_stop(num_steps), prepend=False
                    )
                )

            with log.contextualize(tqdm=True):
                with tqdm(
                    total=n_batches,
                    file=sys.stdout,
                    desc=f"Caching inputs for transformer block {first_transformer_blocks[0].name}",
                    position=1,
                    leave=False,
                ) as pbar:
                    for batches in self.group_iterable(dataset, len(pipelines)):
                        threads = []
                        for i, batch_item in enumerate(batches):
                            pipe = pipelines[i % len(pipelines)]
                            t = threading.Thread(
                                target=run_pipeline,
                                args=(pipe, i, batch_item),
                                daemon=True,
                            )
                            t.start()
                            threads.append(t)
                        for t in threads:
                            t.join()
                        pbar.update(len(batches))
            self.handle.remove()

            self._sample(n_samples, seed)
            torch.save(
                {
                    "transformer_block_hidden_states": self.transformer_block_hidden_states,
                    "transformer_block_constant_kwargs": self.transformer_block_constant_kwargs,
                    "previous_transformer_block_outputs": self.previous_transformer_block_outputs,
                    "global_constant_kwargs": self.global_constant_kwargs,
                },
                sample_filename,
            )

        skip_transformer_blocks = [
            list_[:checkpoint] for list_ in all_transformer_blocks
        ]
        all_transformer_blocks = [
            list_[checkpoint:] for list_ in all_transformer_blocks
        ]

        for tb in zip(*skip_transformer_blocks):
            self.run(list(tb), cache_next=self.update(tb[0].name, tb[0].__class__))
        if all_transformer_blocks and all_transformer_blocks[0]:
            self.update(all_transformer_blocks[0][0].name, all_transformer_blocks[0][0].__class__)

        for p in pipelines:
            p.to("cpu")
        for _tuple in all_transformer_blocks:
            for b in _tuple:
                b.module.to("cpu")
                b.module.eval()
                for param in b.module.parameters():
                    param.requires_grad = False
                for buffer in b.module.buffers():
                    buffer.requires_grad = False
                b.parent = None

        transformer_blocks = [
            tuple(zip(tb, GPUS)) for tb in zip(*all_transformer_blocks)
        ]

        pipelines.clear()
        del pipelines
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        return transformer_blocks

    def _sample(self, n_samples: int, seed: int) -> None:
        total: int = 0
        for hs in self.transformer_block_hidden_states:
            total += next(iter(hs.values())).shape[0]

        indices = list(range(total))
        random.Random(seed).shuffle(indices)
        indices = indices[:n_samples]

        batch_size = min(math.ceil(n_samples / self.n_gpus), 64)

        def slice_and_concat(
            dicts_list: List[Dict[str, Tensor | None]],
            indices: List[int],
            batch_size: int,
        ):
            n_lists = len(dicts_list)
            new_list = []
            dtype = None
            for d in dicts_list:
                for v in d.values():
                    if isinstance(v, Tensor):
                        dtype = v.dtype
                        break
                if dtype is not None:
                    break
            assert dtype is not None, "No valid dtype found in the tensor dicts."
            for batch_idx in self.group_iterable(indices, batch_size):
                batch_dicts = []
                for idx in batch_idx:
                    bidx = idx % n_lists 
                    batch_dicts.append(
                        {
                            k: (
                                v[(idx % v.shape[0]) : (idx % v.shape[0]) + 1] 
                                if v is not None
                                else v
                            )
                            for k, v in dicts_list[bidx].items()
                        }
                    )

                new_list.append(
                    concat_tensor_dicts(batch_dicts, dim=0, dtype=dtype, device=CPU)
                )
            dicts_list.clear()
            return new_list

        image_rotary_emb = self.transformer_block_constant_kwargs[0].get("image_rotary_emb", None)
        joint_attention_kwargs = self.transformer_block_constant_kwargs[0].get("joint_attention_kwargs", None)
        post_pathch_height = self.transformer_block_constant_kwargs[0].get("height", None)
        post_pathch_width = self.transformer_block_constant_kwargs[0].get("width", None)

        if image_rotary_emb is not None:
            self.global_constant_kwargs.update({"image_rotary_emb": image_rotary_emb})
            for d in self.transformer_block_constant_kwargs:
                if "image_rotary_emb" in d:
                    d.pop("image_rotary_emb")
        if joint_attention_kwargs is not None:
            if isinstance(joint_attention_kwargs, dict) and not joint_attention_kwargs:
                self.global_constant_kwargs.update({"joint_attention_kwargs": joint_attention_kwargs})
                for d in self.transformer_block_constant_kwargs:
                    if "joint_attention_kwargs" in d:
                        d.pop("joint_attention_kwargs")
            else :
                raise ValueError("Non-empty joint_attention_kwargs found in the constant kwargs, unable to sample it.")    
        if post_pathch_height is not None:
            self.global_constant_kwargs.update({"height": post_pathch_height})
            for d in self.transformer_block_constant_kwargs:
                if "height" in d:
                    d.pop("height")
        if post_pathch_width is not None:
            self.global_constant_kwargs.update({"width": post_pathch_width})
            for d in self.transformer_block_constant_kwargs:
                if "width" in d:
                    d.pop("width")
    
        self.transformer_block_hidden_states = slice_and_concat(
            self.transformer_block_hidden_states, indices, batch_size
        )
        
        self.transformer_block_constant_kwargs = slice_and_concat(
            self.transformer_block_constant_kwargs, indices, batch_size
        )
        self.previous_transformer_block_outputs = slice_and_concat(
            self.previous_transformer_block_outputs, indices, batch_size
        )

    def run(
        self,
        current_block: List[GenericTransformerBlockWrapper],
        cache_next: bool = True,
        distribute_batches: bool = False,
    ):
        """Feed the cached inputs to the current block (one replica per GPU), whose hooks collect
        what the strategy needs; if `cache_next`, keep the outputs as the inputs of the next block.

        By default every replica processes every batch (each GPU calibrates different layers).
        With `distribute_batches`, the batches are split across the replicas (each GPU sees a
        share of the batches for all layers).
        """
        module_dtype = current_block[0].module.parameters().__next__().dtype
        assert len(current_block) == self.n_gpus, (
            f"Number of transformer blocks ({len(current_block)}) must match number of GPUs ({self.n_gpus})"
        )

        def run_batch(gpu_id: int, batch_id: int, with_update: bool):
            gpu = GPUS[gpu_id]
            block = current_block[gpu_id]
            arguments = kwargs_to(gpu, module_dtype, **self.transformer_block_hidden_states[batch_id])
            arguments.update(kwargs_to(gpu, module_dtype, **self.transformer_block_constant_kwargs[batch_id]))
            if self.global_constant_kwargs is not None:
                arguments.update(kwargs_to(gpu, module_dtype, **self.global_constant_kwargs))
            with torch.no_grad():
                try:
                    outputs = block.module(**arguments)
                    if with_update:
                        self.previous_transformer_block_outputs[batch_id] = kwargs_to(CPU, module_dtype, **self._outputs_to_dict(outputs))
                except Exception as e:
                    raise RuntimeError(
                        f"Error during transformer block execution on GPU {gpu_id} for batch {batch_id} : {self.current_block_name}"
                    ) from e

        n_batches = len(self.transformer_block_hidden_states)
        with log.contextualize(tqdm=True):
            if distribute_batches:
                with tqdm(total=n_batches, file=sys.stdout, desc=f"Running {current_block[0].name}", position=1) as pbar:
                    for batch_ids in self.group_iterable(range(n_batches), self.n_gpus):
                        threads = [
                            threading.Thread(target=run_batch, args=(i, idx, cache_next), daemon=True)
                            for i, idx in enumerate(batch_ids)
                        ]
                        for t in threads:
                            t.start()
                        for t in threads:
                            t.join()
                        pbar.update(len(batch_ids))
            else:
                def run_replica(gpu_id: int):
                    with tqdm(total=n_batches, file=sys.stdout, desc=f"Running {current_block[0].name}", position=gpu_id + 1) as pbar:
                        for batch_id in range(n_batches):
                            run_batch(gpu_id, batch_id, cache_next and gpu_id == 0)
                            pbar.update(1)

                threads = [threading.Thread(target=run_replica, args=(i,), daemon=True) for i in range(self.n_gpus)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def update(self, block_name: str, next_block_cls: Type[GenericTransformerBlockWrapper]) -> bool:
        if self.current_block_name == block_name:
            log.info(
                f"Transformer block {block_name} is already loaded on GPUs, skipping update."
            )
            return False
        else:
            log.info(f"Updating cache to transformer block {block_name}...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.current_block_name = block_name
        new_hidden_states: Kwargs_List = []
        for prev_out in self.previous_transformer_block_outputs:
            output_batch = tuple(prev_out.values())
            if len(prev_out) == 1:
                extracted = next_block_cls.extract_inputs_from_previous_block_outputs(output_batch[0])
            else:
                extracted = next_block_cls.extract_inputs_from_previous_block_outputs(output_batch)
            cpu_extracted = {
                k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v)
                for k, v in extracted.items()
            }
            new_hidden_states.append(cpu_extracted)
                
        self.transformer_block_hidden_states.clear()
        self.transformer_block_hidden_states = new_hidden_states
        return True
