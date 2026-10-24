"""FLUX.1-dev adapter for distributed ParaPC-FM sampling."""

import torch
from parapc_fm.models.base import FlowPipeline


class ParallelPipeline(FlowPipeline):
    model_id = "black-forest-labs/FLUX.1-dev"

    @staticmethod
    def pipeline_type():
        from diffusers import FluxPipeline
        return FluxPipeline

    def prepare(self, prompt, negative_prompt, steps, height, width, generator, guidance_scale):
        if negative_prompt:
            raise ValueError("This FLUX adapter uses distilled guidance; negative_prompt is unsupported")
        pipe, device = self.pipeline, self.runtime.device
        embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1, max_sequence_length=512)
        latent, image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, height, width,
            embeds.dtype, device, generator)
        config = pipe.scheduler.config
        low, high = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096)
        slope = (config.get("max_shift", 1.15) - config.get("base_shift", 0.5)) / (high - low)
        mu = slope * (latent.shape[1] - low) + config.get("base_shift", 0.5)
        sigmas = torch.linspace(1, 1 / steps, steps).tolist()
        pipe.scheduler.set_timesteps(steps, device=device, sigmas=sigmas, mu=mu)
        times = torch.cat([pipe.scheduler.timesteps.double() / 1000,
                           torch.zeros(1, device=device, dtype=torch.float64)])
        guidance = (torch.full((1,), guidance_scale, device=device, dtype=torch.float32)
                    if pipe.transformer.config.guidance_embeds else None)

        def velocity(state, index):
            return pipe.transformer(
                hidden_states=state, timestep=(times[index] * 1000).to(state.dtype).expand(1) / 1000,
                guidance=guidance, pooled_projections=pooled, encoder_hidden_states=embeds,
                txt_ids=text_ids, img_ids=image_ids, return_dict=False)[0]

        return latent, times, velocity

    def decode(self, latent, height, width):
        pipe = self.pipeline
        unpacked = pipe._unpack_latents(latent, height, width, pipe.vae_scale_factor)
        return super().decode(unpacked, height, width)
