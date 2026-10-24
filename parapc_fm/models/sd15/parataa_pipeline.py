"""Distributed SD1.5 ParaTAA DDIM with centrally defined q initialization."""

import math
import time
import torch
from parapc_fm.models.base import FlowPipeline
from parapc_fm.prediction import initialize_ddim_states, validate_orders
from parapc_fm.sampling import SamplingOutput
from .parataa_sampler import ParallelDDIM


class ParallelPipeline(FlowPipeline):
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    default_dtype = torch.float32

    @staticmethod
    def pipeline_type():
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from diffusers import DDIMScheduler
        instance = super().from_pretrained(*args, **kwargs)
        instance.pipeline.scheduler = DDIMScheduler.from_config(instance.pipeline.scheduler.config)
        return instance

    @torch.no_grad()
    def __call__(self, prompt, *, negative_prompt="", num_inference_steps=50,
                 height=512, width=512, seed=0, guidance_scale=4.0,
                 init_order=1, window_size=8, taa_order=8, memory_size=2,
                 tol1=0.001, tol2=0.001, max_iterations=None, output_type="pil"):
        validate_orders(0, init_order)
        self.validate(prompt, num_inference_steps, height, width)
        if (height, width) != (512, 512):
            raise ValueError("The original ParaTAA core currently supports only 512x512 images")
        if type(window_size) is not int or type(taa_order) is not int or not 1 <= taa_order <= window_size:
            raise ValueError("Require 1 <= taa_order <= window_size")
        if type(memory_size) is not int or memory_size < 1:
            raise ValueError("memory_size must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (tol1, tol2, guidance_scale)):
            raise ValueError("Guidance and tolerances must be finite and nonnegative")
        if output_type not in ("pil", "latent"):
            raise ValueError("output_type must be pil or latent")
        if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
            raise ValueError("max_iterations must be a positive integer")
        runtime, pipe = self.runtime, self.pipeline
        runtime.synchronize()
        started = time.perf_counter()
        pipe.scheduler.set_timesteps(num_inference_steps, device=runtime.device)
        positive, negative = pipe.encode_prompt(
            prompt, runtime.device, 1, True, negative_prompt=negative_prompt)
        dtype = next(pipe.unet.parameters()).dtype
        sampler = ParallelDDIM(
            timesteps=num_inference_steps, scheduler=pipe.scheduler,
            window_size=window_size, eta=0.0, cfg_scale=guidance_scale,
            order=taa_order, memory_size=memory_size, tol1=tol1, tol2=tol2,
            device=runtime.device, use_fp16=dtype == torch.float16)
        noise = torch.randn((num_inference_steps + 1, 4, 64, 64),
                            generator=torch.Generator("cpu").manual_seed(seed),
                            dtype=torch.float64).to(runtime.device)
        runtime.broadcast(noise)
        sampler.initialize(noise)
        runtime.synchronize()
        prepared = time.perf_counter()
        iterations = evaluations = 0
        limit = max_iterations if max_iterations is not None else 10 * num_inference_steps
        while not sampler.is_finished():
            if iterations >= limit:
                raise RuntimeError(f"ParaTAA did not converge in {limit} iterations; no image was saved")
            initialize_ddim_states(sampler, init_order)
            end = int(sampler.finished_index) + 1
            begin = max(end - window_size, 1)
            indices = list(range(begin - 1, end - 1))

            def predict(index):
                state = sampler._samples[index + 1:index + 2].to(dtype)
                model_input = torch.cat([state, state])
                timestep = sampler.scheduler_timesteps[index].expand(2)
                model_input = pipe.scheduler.scale_model_input(model_input, timestep)
                prediction = pipe.unet(
                    model_input, timestep, encoder_hidden_states=torch.cat([negative, positive]),
                    return_dict=False)[0]
                uncond, cond = prediction.chunk(2)
                return (uncond + guidance_scale * (cond - uncond))[0].double()

            directions = runtime.evaluate(indices, sampler._samples[0], predict)
            if not torch.isfinite(directions).all():
                raise RuntimeError("Non-finite UNet prediction")
            ts = sampler.scheduler_timesteps[indices]
            sampler.step(directions, {"timestep": torch.cat([ts, ts])})
            runtime.broadcast(sampler._samples)
            iterations += 1
            evaluations += len(indices)
        runtime.synchronize()
        sampled = time.perf_counter()
        images = None
        if runtime.is_main:
            latent = sampler.get_last_sample().unsqueeze(0).to(next(pipe.vae.parameters()).dtype)
            if output_type == "latent":
                images = latent
            else:
                decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
                images = pipe.image_processor.postprocess(decoded, output_type="pil")
        runtime.synchronize()
        finished = time.perf_counter()
        timings = {"preparation_seconds": runtime.maximum(prepared - started),
                   "sampling_seconds": runtime.maximum(sampled - prepared),
                   "decode_seconds": runtime.maximum(finished - sampled),
                   "generation_seconds": runtime.maximum(finished - started)}
        return SamplingOutput(images, iterations, evaluations, timings)
