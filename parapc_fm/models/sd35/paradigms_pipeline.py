"""SD3.5 shares the SD3 algorithm and overrides only model defaults."""

import torch
from parapc_fm.models.sd3.paradigms_pipeline import ParallelPipeline as SD3Pipeline


class ParallelPipeline(SD3Pipeline):
    model_id = "stabilityai/stable-diffusion-3.5-large"
    default_dtype = torch.bfloat16
    default_tolerance = 1.0
