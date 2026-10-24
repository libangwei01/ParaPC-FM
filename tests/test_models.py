"""Weight-free adapter tests, including real DDIM scheduler/ParaTAA updates."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from diffusers import DDIMScheduler
from parapc_fm.distributed import Runtime
from parapc_fm.models.flux.paradigms_pipeline import ParallelPipeline as Flux
from parapc_fm.models.sd3.paradigms_pipeline import ParallelPipeline as SD3
from parapc_fm.models.sd35.paradigms_pipeline import ParallelPipeline as SD35
from parapc_fm.models.sd15.parataa_pipeline import ParallelPipeline as SD15


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=True, patch_size=2)

    def forward(self, hidden_states=None, sample=None, encoder_hidden_states=None, **kwargs):
        value = hidden_states if hidden_states is not None else sample
        if encoder_hidden_states is not None and value.shape[0] == 2:
            return (torch.cat([torch.zeros_like(value[:1]), torch.ones_like(value[:1])]),)
        return (torch.ones_like(value),)


class Stub:
    def __init__(self, flux=False, ddim=False):
        self.flux = flux
        self.ddim = ddim
        self.transformer = Model()
        self.unet = Model()
        self.vae = Model()
        self.vae_scale_factor = 8
        self.scheduler = DDIMScheduler() if ddim else SimpleNamespace(config={}, set_timesteps=self.set_timesteps)

    def set_timesteps(self, steps, device=None, **kwargs):
        self.scheduler.timesteps = torch.linspace(1000, 1000/steps, steps, device=device)

    def encode_prompt(self, *args, **kwargs):
        value = torch.ones(1, 2, 3)
        if self.flux:
            return value, value, torch.zeros(2, 3)
        if self.ddim:
            return value, torch.zeros_like(value)
        return value, torch.zeros_like(value), value, torch.zeros_like(value)

    def prepare_latents(self, *args):
        if self.flux:
            return torch.zeros(1, 2, 4), torch.zeros(2, 3)
        return torch.zeros(1, 4, 2, 2)


class ModelTests(unittest.TestCase):
    def test_three_flow_adapters_all_orders(self):
        for adapter in (Flux, SD3, SD35):
            pipeline = adapter(Stub(flux=adapter is Flux), Runtime("cpu"))
            for p in range(3):
                for q in range(3):
                    result = pipeline("test", num_inference_steps=4, update_order=p,
                                      init_order=q, window_size=3, tolerance=0, output_type="latent")
                    expected = -1. if adapter is Flux else -3.5
                    torch.testing.assert_close(result.images, torch.full_like(result.images, expected))

    def test_parataa_all_q_and_repeat_call(self):
        torch.set_num_threads(1)
        pipeline = SD15(Stub(ddim=True), Runtime("cpu"))
        for q in (0, 1, 2, 0):
            result = pipeline("test", num_inference_steps=8, init_order=q,
                              window_size=3, taa_order=3, output_type="latent")
            self.assertEqual(tuple(result.images.shape), (1, 4, 64, 64))
            self.assertTrue(torch.isfinite(result.images).all())
            self.assertGreater(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
