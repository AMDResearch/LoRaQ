# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileContributor: Yann Bouquet
# SPDX-License-Identifier: Apache-2.0

"""Smoothing stage: per-layer search of the smoothing scale s = |x|^alpha / |W|^beta.

For every group of linear layers sharing the same input (e.g. q/k/v), the (alpha, beta) pairs of
a grid are evaluated on calibration inputs: the weights are scaled by s and the activations
divided by s, both are quantized (the weights with a low-rank branch kept in high precision, as
in SVDQuant), and the pair minimizing the output error of the enclosing module is kept.
"""

import functools
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Tuple, TypeVar

import torch
from loguru import logger as log
from torch import Tensor, nn
from tqdm import tqdm

try:
    from typing import Self, override
except ImportError:
    from typing_extensions import Self, override  # type: ignore

from ..cache import TransformerBlockBuffer, move_blocks_to_gpus
from ..models.blocks import BaseModuleWrapper
from ..models.diffusion import GenericDiTBlock
from ..models.patches.smoothscale import SmoothScale
from ..models.transformer import GenericTransformerBlockWrapper
from ..quantizer.base import BaseQuantizer, InputQuantizer, WeightQuantizer
from ..utils.hook import Args, Kwargs, Outputs, RemovableHandles
from .base import GlobalState, Strategy

_value = TypeVar("_value")
_accumulated = TypeVar("_accumulated")

WEIGHTS = "weights"
INPUTS = "inputs"


class Reduce(ABC, Generic[_value, _accumulated]):
    """A running reduction; the initial value counts as the first value."""

    def __init__(self, initial: _accumulated):
        self.acc: _accumulated = initial
        self.cnt: float = 0.0

    @abstractmethod
    def _map(self, value: _value) -> _accumulated: ...

    @abstractmethod
    def _aggregate(self, value: _accumulated) -> None: ...

    def update(self, value: _value) -> Self:
        self._aggregate(self._map(value))
        return self

    def result(self) -> _accumulated:
        return self.acc


class ReduceError(Reduce[float, float]):
    """Sum of errors."""

    def __init__(self, initial: float = 0.0):
        super().__init__(initial)

    @override
    def _map(self, value: float) -> float:
        return value

    @override
    def _aggregate(self, value: float) -> None:
        self.acc += value
        self.cnt += 1.0


class AbsMax(Reduce[Tensor, Tensor]):
    """Per-channel maximum of |x| over all but the last dimension."""

    def __init__(self, initial: Tensor = torch.tensor([0.0])):  # type: ignore
        super().__init__(initial)

    @override
    def _map(self, value: Tensor) -> Tensor:
        value = value.abs()
        if value.ndim == 3:
            out: Tensor = value.amax(dim=(0, 1), keepdim=True)
        elif value.ndim == 2:
            out = value.amax(dim=0, keepdim=True)
        elif value.ndim == 1:
            out = value
        else:
            raise ValueError(f"Unsupported tensor shape for {self.__class__.__name__}: {value.shape}")
        return out

    @override
    def _aggregate(self, value: Tensor) -> None:
        assert self.acc.shape == (1,) or value.ndim == self.acc.ndim, (
            "Aggregated tensor must have the same number of dimensions as accumulator."
        )
        self.acc = torch.maximum(self.acc.to(device=value.device, dtype=value.dtype), value).to("cpu")  # type: ignore


class Accumulator(GlobalState):
    """Nested dict of reductions (block -> layer -> key -> reduction)."""

    @abstractmethod
    def _default_value(self) -> Any: ...

    @abstractmethod
    def _aggregate(self, current: Any, new: Tensor | float) -> Any: ...

    def accumulate(self, new_value: Tensor | float, *keys) -> None:
        current = self
        *head, tail = keys
        for key in head:
            current = current[key]
        current_value = current.get(tail, self._default_value())
        current[tail] = self._aggregate(current_value, new_value)

    def update(self, other: "Accumulator") -> Self:
        """Merge the reductions of `other` into this accumulator."""
        def walk(node: Any, path: Tuple[str, ...] = ()):
            if isinstance(node, Reduce):
                yield path, node
            elif isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(k, str):
                        yield from walk(v, path + (k,))

        for path, leaf in walk(other):
            current = self
            for key in path[:-1]:
                current = current[key]
            if path[-1] not in current:
                current[path[-1]] = leaf
            else:
                current[path[-1]].update(leaf.result())
        return self


class ErrorAccumulator(Accumulator):
    @override
    def _default_value(self) -> ReduceError:
        return ReduceError()

    @override
    def _aggregate(self, current: ReduceError, new: float) -> ReduceError:
        return current.update(new)


class SpanAccumulator(Accumulator):
    """Per-channel AbsMax of the inputs and weights of each layer group."""

    @override
    def _default_value(self) -> Dict[str, AbsMax]:
        return {"AbsMax": AbsMax()}

    @override
    def _aggregate(self, current: Dict[str, AbsMax], new: Tensor) -> Dict[str, AbsMax]:
        for span in current.values():
            span.update(new)
        return current

    def span(self, layer: BaseModuleWrapper, key: str) -> Tensor:
        assert layer.transformer_block, f"Layer {layer.name} must be part of a transformer block."
        return self[layer.transformer_block.name][layer.name][key]["AbsMax"].result()


@dataclass
class SmoothingGrid:
    """(alpha, beta) candidates on a regular grid of step 1/num_candidates, always with (0, 0):
    `one_minus` adds beta = 1 - alpha, `alpha_zero` adds beta = 0 (the search space of SVDQuant)."""

    num_candidates: int
    one_minus: bool = False
    alpha_zero: bool = False

    def build(self) -> List[Tuple[float, float]]:
        axis = [i / self.num_candidates for i in range(1, self.num_candidates)]
        pairs = {(0.0, 0.0)}
        if self.one_minus:
            pairs.update((float(a), 1.0 - float(a)) for a in axis)
        if self.alpha_zero:
            pairs.update((float(a), 0.0) for a in axis)
        return sorted({(float(a), float(b)) for a, b in pairs})


class SmoothingCandidates:
    """The smoothing candidates of one layer group, and the original weights to restore."""

    @torch.no_grad()
    def __init__(self, layers: BaseModuleWrapper, a_span: Tensor, w_span: Tensor, grid: List[Tuple[float, float]]):
        assert all(isinstance(layer.module.weight, Tensor | torch.nn.Parameter) for layer in layers.named_layers), (
            "Layer module weight must be a tensor."
        )
        self._layers = layers
        self._initial_weights: List[Tensor] = [
            layer.module.weight.clone() for layer in layers.named_layers if isinstance(layer.module, nn.Linear)
        ]
        self.grid = grid
        self.a_span = a_span
        self.w_span = w_span

    def __iter__(self) -> Iterator[SmoothScale]:
        for alpha, beta in self.grid:
            yield SmoothScale(self.a_span, self.w_span, alpha, beta)

    def __len__(self) -> int:
        return len(self.grid)

    @property
    def name(self) -> str:
        return self._layers.name

    @property
    def modules(self) -> List[nn.Linear]:
        return [layer.module for layer in self._layers.named_layers if isinstance(layer.module, nn.Linear)]

    @torch.no_grad()
    def reset_modules(self) -> None:
        assert len(self.modules) == len(self._initial_weights)
        for module, w0 in zip(self.modules, self._initial_weights):
            module.weight.copy_(w0.to(device=module.weight.device, dtype=module.weight.dtype))


class OutputEvaluator:
    """Sum of squared errors (float64) between a module's outputs and reference outputs."""

    dev_dtype = torch.float64

    @torch.inference_mode()
    def _compute_error(self, outputs: Outputs, targets: Outputs) -> float:
        if isinstance(outputs, Tensor) and isinstance(targets, Tensor):
            return (outputs.to(self.dev_dtype) - targets.to(self.dev_dtype)).pow(2).sum().item()
        if isinstance(outputs, tuple) and isinstance(targets, tuple):
            loss = 0.0
            for output, target in zip(outputs, targets):
                loss += (output.to(self.dev_dtype) - target.to(self.dev_dtype)).pow(2).sum().item()
            return loss
        raise ValueError(f"Outputs and targets must be both Tensor or tuple of Tensor. Got {type(outputs)} and {type(targets)}")

    @torch.inference_mode()
    def evaluate(self, module: nn.Module, *, ipts_args: Args, ipts_kwargs: Kwargs, targets: Outputs) -> float:
        if len(ipts_args) > 0 and len(ipts_kwargs) > 0:
            result = module(*ipts_args, **ipts_kwargs)
        elif len(ipts_args) > 0:
            result = module(*ipts_args)
        elif len(ipts_kwargs) > 0:
            result = module(**ipts_kwargs)
        else:
            raise ValueError("Inputs for evaluation block are empty. Please check the cache or the inputs provided.")
        return self._compute_error(result, targets)


class InputCollector(RemovableHandles):
    """Collects the inputs of a module and processes them by batches."""

    def __init__(self, process: Callable[[Tensor], None], batch_size: int = 2):
        super().__init__()
        self.stack: List[Tensor] = []
        self.batch_size = batch_size
        self.process = process

    def process_stack(self) -> None:
        while self.stack:
            self.process(self.stack.pop(0))

    def hooked_on(self, module: nn.Module) -> Self:
        def hook(module: nn.Module, inputs: Args) -> None:
            self.stack.append(inputs[0])
            if len(self.stack) > self.batch_size:
                self.process_stack()

        self.append(module.register_forward_pre_hook(hook, with_kwargs=False))
        return self

    @override
    def remove(self) -> None:
        self.process_stack()
        super().remove()


class HookOverseer:
    """Suspends the evaluation hooks of a block while one of them re-runs its module."""

    def __init__(self):
        self.hooks: List["CandidateEvalHook"] = []

    def register(self, hook: "CandidateEvalHook") -> None:
        self.hooks.append(hook)

    @contextmanager
    def suspend_all(self):
        for hook in self.hooks:
            hook.unhook()
        try:
            yield
        finally:
            for hook in self.hooks:
                hook.hook()


class CandidateEvalHook(RemovableHandles):
    """Captures the inputs and output of a module (e.g. an attention block) and evaluates every
    smoothing candidate of one of its layer groups on them.

    For a candidate, the layers are smoothed, their weights quantized (low-rank branch + quantized
    residual) and their inputs quantized. By default the low-rank branch reads the unquantized
    (smoothed) input; with `quantize_branch_input` it reads the quantized one.
    """

    def __init__(
        self,
        overseer: HookOverseer,
        evalblock: BaseModuleWrapper,
        candidates: SmoothingCandidates,
        process: Callable[[nn.Module, str, int, Args, Kwargs, Outputs], None],
        i_quantizer: InputQuantizer,
        w_quantizer: WeightQuantizer,
        quantize_branch_input: bool = False,
        batch_size: int = 1,
    ):
        super().__init__()
        self.module_name = evalblock.name
        self.module = evalblock.module
        self.batch_size = batch_size
        self.process = process
        self.candidates = candidates
        self.i_quantizer = i_quantizer
        self.w_quantizer = w_quantizer
        self.quantize_branch_input = quantize_branch_input
        self.stack: List[Tuple[Args, Kwargs, Outputs]] = []
        self.overseer = overseer
        self.overseer.register(self)

    def _quantize_inputs(self, handles: RemovableHandles) -> None:
        for layer_module in self.candidates.modules:
            handles.append(self.i_quantizer.hooked_on(layer_module))

    def process_stack(self) -> None:
        if self.stack:
            with self.overseer.suspend_all():
                for cand_id, candidate in tqdm(
                    enumerate(self.candidates), total=len(self.candidates),
                    desc=f"Candidates for {self.module_name}", position=0,
                ):
                    handles = RemovableHandles()
                    for layer_module in self.candidates.modules:
                        handles.append(candidate.hooked_on(layer_module))
                    if self.quantize_branch_input:
                        self._quantize_inputs(handles)
                        handles.append(self.w_quantizer.quantize_modules(*self.candidates.modules))
                    else:
                        handles.append(self.w_quantizer.quantize_modules(*self.candidates.modules))
                        self._quantize_inputs(handles)
                    for inputs, kwargs, target in self.stack:
                        self.process(self.module, self.candidates.name, cand_id, inputs, kwargs, target)
                    handles.remove()
                    self.candidates.reset_modules()
        self.stack.clear()

    def _pre_hook(self, module: nn.Module, inputs: Args, kwargs: Kwargs) -> None:
        assert not self.stack, "Stack should be empty before pre-hook."
        self.stack.append((inputs, kwargs, torch.empty(0)))

    def _post_hook(self, module: nn.Module, inputs: Args, output: Outputs) -> None:
        assert self.stack
        ipt, kwargs, _ = self.stack[-1]
        self.stack[-1] = ipt, kwargs, output
        if len(self.stack) >= self.batch_size:
            self.process_stack()

    def hook(self) -> Self:
        self.append(self.module.register_forward_pre_hook(self._pre_hook, with_kwargs=True, prepend=True))
        self.append(self.module.register_forward_hook(self._post_hook, prepend=False))
        return self

    def unhook(self) -> None:
        super().remove()

    @override
    def remove(self) -> None:
        if self.stack:
            self.process_stack()
        self.unhook()


def _is_empty(layers: BaseModuleWrapper) -> bool:
    return next(iter(layers.named_layers), None) is None


class SmoothingSearchStrategy(Strategy):
    """Smoothing stage: searches the best (alpha, beta) of each layer group, block by block.

    Results (`result/best_candidates.pkl`, the candidate index per layer group) are stored with
    the spans (`spans.pkl`) and the accumulated errors (`checkpoint/errors.pkl`).
    """

    def __init__(
        self,
        grid: SmoothingGrid,
        i_base_quantizer: BaseQuantizer,
        w_base_quantizer: BaseQuantizer,
        dataset: Tuple[Iterable, int, int],
        quantize_branch_input: bool = False,
        num_samples: int = 128,
        num_steps: int = 28,
        guidance_scale: float = 7.0,
        seed: int = 0,
        resume: bool = False,
        dataset_path: str = "data/samples.pt",
    ):
        self.grid: List[Tuple[float, float]] = grid.build()
        self.dataset = dataset
        self.io_cache = TransformerBlockBuffer()
        self.w_quantizer: WeightQuantizer = WeightQuantizer(w_base_quantizer)
        self.i_quantizer: InputQuantizer = InputQuantizer(i_base_quantizer)
        self.quantize_branch_input = quantize_branch_input
        self.resume = resume
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.num_samples = num_samples
        self.sample_filenames = dataset_path

        name = f"Smoothing_{self.i_quantizer.name}_{self.w_quantizer.name}"
        if quantize_branch_input:
            name += "_quantized-branch-input"
        super().__init__(name)
        self._update_filenames()

        self.best_candidates = GlobalState()
        self.errors = ErrorAccumulator()
        self.span_accumulator = SpanAccumulator()

    @override
    def _update_filenames(self) -> None:
        self.result_filename = os.path.join(self.output_dir, "result", "best_candidates.pkl")
        self.checkpoint_filename = os.path.join(self.output_dir, "checkpoint", "errors.pkl")
        self.spans_filename = os.path.join(self.output_dir, "spans.pkl")

    def _save(self, state: Dict, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self._save_state(state, filename)

    def _candidates(self, layers: BaseModuleWrapper) -> SmoothingCandidates:
        a_span = self.span_accumulator.span(layers, INPUTS).squeeze()
        w_span = self.span_accumulator.span(layers, WEIGHTS).squeeze()
        assert a_span.ndim == 1 and w_span.ndim == 1, f"Spans of {layers.name} must be 1D tensors."
        return SmoothingCandidates(layers, a_span, w_span, self.grid)

    def _accumulate_spans(self, blocks: List[GenericTransformerBlockWrapper], layer_extractor) -> None:
        """Weight spans from the weights, input spans from the cached calibration inputs."""
        block_name = blocks[0].name
        for _, child in layer_extractor(blocks[0]):
            for layer in child.named_layers:
                assert isinstance(layer.module.weight.data, Tensor)
                self.span_accumulator.accumulate(layer.module.weight.data, block_name, child.name, WEIGHTS)

        handles = RemovableHandles()
        accumulators = []
        for tb in blocks:
            accumulator = SpanAccumulator()
            accumulators.append(accumulator)
            for _, child in layer_extractor(tb):
                first_layer = next(iter(child.named_layers), None)
                if first_layer is None:
                    continue
                keys = (tb.name, child.name, INPUTS)
                collect = lambda x, acc=accumulator, keys=keys: acc.accumulate(x, *keys)  # noqa: E731
                handles.append(InputCollector(process=collect, batch_size=2).hooked_on(first_layer.module))

        self.io_cache.run(blocks, cache_next=self.io_cache.update(blocks[0].name, blocks[0].__class__), distribute_batches=True)
        handles.remove()

        for accumulator in accumulators:
            self.span_accumulator.update(accumulator)
        self._save(self.span_accumulator, self.spans_filename)

    def _accumulate_errors(self, blocks: List[GenericTransformerBlockWrapper], layer_extractor) -> None:
        """Evaluates the candidates of every layer group on the cached calibration inputs."""
        handles = RemovableHandles()
        accumulators = []
        evaluator = OutputEvaluator()
        for tb in blocks:
            errors = ErrorAccumulator()
            accumulators.append(errors)
            i_quantizer = self.i_quantizer.clone()
            w_quantizer = self.w_quantizer.clone()
            overseer = HookOverseer()
            for eval_layer, child in layer_extractor(tb):
                if _is_empty(child) or child.name in self.errors[tb.name]:
                    continue
                evalhook = CandidateEvalHook(
                    overseer=overseer,
                    evalblock=eval_layer,
                    candidates=self._candidates(child),
                    process=functools.partial(self._compute_error, errors=errors, evaluator=evaluator, block_name=tb.name),
                    i_quantizer=i_quantizer,
                    w_quantizer=w_quantizer,
                    quantize_branch_input=self.quantize_branch_input,
                )
                handles.append(evalhook.hook())

        self.io_cache.run(blocks, cache_next=self.io_cache.update(blocks[0].name, blocks[0].__class__), distribute_batches=True)
        handles.remove()

        for errors in accumulators:
            self.errors.update(errors)
        self._save(self.errors, self.checkpoint_filename)

    @staticmethod
    def _compute_error(module: nn.Module, layer_name: str, candidate_id: int, inputs: Args, kwargs: Kwargs,
                       output: Outputs, *, errors: ErrorAccumulator, evaluator: OutputEvaluator, block_name: str) -> None:
        error = evaluator.evaluate(module, ipts_args=inputs, ipts_kwargs=kwargs, targets=output)
        errors.accumulate(error, block_name, layer_name, str(candidate_id))

    def _select_best_candidates(self, block_name: str) -> None:
        for layer_name, candidates in self.errors[block_name].items():
            best_id, _ = min(candidates.items(), key=lambda x: x[1].result(), default=(None, float("inf")))
            assert best_id is not None, f"No candidates found for layer: {layer_name}"
            self.best_candidates[block_name][layer_name] = best_id

    @override
    def search(self, pipelines: List, layer_extractor: Callable) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.resume:
            self.span_accumulator = SpanAccumulator(self._load_state(self.spans_filename))
            self.best_candidates = GlobalState(self._load_state(self.result_filename))
            self.errors = ErrorAccumulator(self._load_state(self.checkpoint_filename))
        else:
            self.span_accumulator = SpanAccumulator()

        dataset, _, n_batches = self.dataset
        transformer_blocks_gpus = self.io_cache.buffer(
            sample_filename=self.sample_filenames,
            n_samples=self.num_samples,
            pipelines=pipelines,
            checkpoint=len(self.best_candidates),
            num_steps=self.num_steps,
            dataset=dataset,
            n_batches=n_batches,
            guidance_scale=self.guidance_scale,
            seed=self.seed,
        )
        with log.contextualize(tqdm=True):
            with tqdm(transformer_blocks_gpus, desc="Transformer Blocks", position=0) as pbar:
                while transformer_blocks_gpus:
                    transformer_blocks = transformer_blocks_gpus.pop(0)
                    blocks = move_blocks_to_gpus(transformer_blocks)
                    self._accumulate_spans(blocks, layer_extractor)
                    self._accumulate_errors(blocks, layer_extractor)
                    self._select_best_candidates(transformer_blocks[0][0].name)
                    self._save(self.best_candidates, self.result_filename)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    pbar.update()
        log.info("Smoothing search completed for all transformer blocks.")
        self._save(self.best_candidates, self.result_filename)

    @override
    def apply(self, models: List[GenericDiTBlock], layer_extractor) -> RemovableHandles:
        """Smooth the weights in place and divide the activations by the scale (pre-hook)."""
        handles = super().apply(models, layer_extractor)
        if not self.best_candidates:
            self.best_candidates = GlobalState(self._load_state(self.result_filename))
        if not self.span_accumulator:
            self.span_accumulator = SpanAccumulator(self._load_state(self.spans_filename))
        assert self.best_candidates, "Best candidates must be available to apply the smoothing."

        for model in models:
            for block in model.blocks:
                if block.name not in self.best_candidates:
                    log.warning(f"Block {block.name} not found in best candidates, skipping.")
                    continue
                for _, child in layer_extractor(block):
                    if child.name not in self.best_candidates[block.name]:
                        log.warning(f"Layers {child.name} of block {block.name} not found in best candidates, skipping.")
                        continue
                    alpha, beta = self.grid[int(self.best_candidates[block.name][child.name])]
                    scale = SmoothScale(
                        self.span_accumulator.span(child, INPUTS).squeeze(),
                        self.span_accumulator.span(child, WEIGHTS).squeeze(),
                        alpha, beta,
                    )
                    for named_layer in child.named_layers:
                        handles.append(scale.hooked_on(named_layer.module))
        return handles
