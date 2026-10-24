"""Stable Diffusion 3 adapter; classifier-free guidance stays local to a timestep."""

import torch
from parapc_fm.models.base import FlowPipeline


class ParallelPipeline(FlowPipeline):
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    default_dtype = torch.float16

    @staticmethod
    def pipeline_type():
        from diffusers import StableDiffusion3Pipeline
        return StableDiffusion3Pipeline

    def prepare(self, prompt, negative_prompt, steps, height, width, generator, guidance_scale):
        pipe, device = self.pipeline, self.runtime.device
        use_cfg = guidance_scale > 1
        embeds, negative, pooled, negative_pooled = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt,
            device=device, do_classifier_free_guidance=use_cfg,
            num_images_per_prompt=1, max_sequence_length=256)
        if use_cfg:
            embeds = torch.cat([negative, embeds])
            pooled = torch.cat([negative_pooled, pooled])
        latent = pipe.prepare_latents(1, pipe.transformer.config.in_channels,
                                      height, width, embeds.dtype, device, generator)
        kwargs = {}
        if pipe.scheduler.config.get("use_dynamic_shifting", False):
            size = (height // pipe.vae_scale_factor // pipe.transformer.config.patch_size) * (
                width // pipe.vae_scale_factor // pipe.transformer.config.patch_size)
            cfg = pipe.scheduler.config
            low, high = cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096)
            kwargs["mu"] = cfg.get("base_shift", 0.5) + (size - low) * (
                cfg.get("max_shift", 1.16) - cfg.get("base_shift", 0.5)) / (high - low)
        pipe.scheduler.set_timesteps(steps, device=device, **kwargs)
        times = torch.cat([pipe.scheduler.timesteps.double() / 1000,
                           torch.zeros(1, device=device, dtype=torch.float64)])

        def velocity(state, index):
            model_input = torch.cat([state, state]) if use_cfg else state
            prediction = pipe.transformer(
                hidden_states=model_input,
                timestep=(times[index] * 1000).to(state.dtype).expand(model_input.shape[0]),
                encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
            if use_cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + guidance_scale * (cond - uncond)
            return prediction

        return latent, times, velocity
