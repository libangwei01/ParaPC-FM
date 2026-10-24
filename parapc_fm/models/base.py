"""Common single-prompt inference interface for model adapters."""

import math
import time
import torch
from parapc_fm.distributed import Runtime
from parapc_fm.sampling import SamplingOutput, sample_flow
from parapc_fm.prediction import validate_orders


class FlowPipeline:
    model_id = None
    default_dtype = torch.bfloat16
    default_tolerance = 0.5

    def __init__(self, pipeline, runtime):
        self.pipeline = pipeline
        self.runtime = runtime

    @classmethod
    def from_pretrained(cls, model_id=None, *, runtime=None, torch_dtype=None, **kwargs):
        runtime = runtime or Runtime()
        pipeline = cls.pipeline_type().from_pretrained(
            model_id or cls.model_id, torch_dtype=torch_dtype or cls.default_dtype, **kwargs)
        pipeline.to(runtime.device)
        pipeline.set_progress_bar_config(disable=True)
        return cls(pipeline, runtime)

    @staticmethod
    def validate(prompt, num_inference_steps, height, width):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide one nonempty text prompt")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if any(type(v) is not int or v <= 0 or v % 16 for v in (height, width)):
            raise ValueError("height and width must be positive multiples of 16")

    @torch.no_grad()
    def __call__(self, prompt, *, negative_prompt="", num_inference_steps=50,
                 height=512, width=512, seed=0, guidance_scale=3.5,
                 update_order=0, init_order=1, window_size=8, tolerance=None,
                 output_type="pil"):
        validate_orders(update_order, init_order)
        self.validate(prompt, num_inference_steps, height, width)
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if output_type not in ("pil", "latent"):
            raise ValueError("output_type must be pil or latent")
        if tolerance is None:
            tolerance = self.default_tolerance
        runtime = self.runtime
        runtime.synchronize()
        started = time.perf_counter()
        generator = torch.Generator("cpu").manual_seed(seed)
        initial, times, velocity = self.prepare(
            prompt, negative_prompt, num_inference_steps, height, width,
            generator, guidance_scale)
        runtime.synchronize()
        prepared = time.perf_counter()
        latent, iterations, evaluations = sample_flow(
            initial, times, velocity, runtime, update_order=update_order,
            init_order=init_order, window_size=window_size, tolerance=tolerance)
        runtime.synchronize()
        sampled = time.perf_counter()
        images = None
        if runtime.is_main:
            images = latent if output_type == "latent" else self.decode(latent, height, width)
        runtime.synchronize()
        finished = time.perf_counter()
        timings = {"preparation_seconds": runtime.maximum(prepared - started),
                   "sampling_seconds": runtime.maximum(sampled - prepared),
                   "decode_seconds": runtime.maximum(finished - sampled),
                   "generation_seconds": runtime.maximum(finished - started)}
        return SamplingOutput(images, iterations, evaluations, timings)

    def decode(self, latent, height, width):
        pipe = self.pipeline
        dtype = next(pipe.vae.parameters()).dtype
        latent = latent.to(dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latent, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")
