"""ParaPC-FM. Model dependencies are imported only when a model is selected."""

__version__ = "0.1.0"


def load_pipeline(model, **kwargs):
    """Load one of flux, sd3, sd35 or sd15 from Hugging Face or a local path."""
    from importlib import import_module

    if model not in {"flux", "sd3", "sd35", "sd15"}:
        raise ValueError(f"Unknown model: {model}")
    module = import_module(f"parapc_fm.models.{model}")
    return module.ParallelPipeline.from_pretrained(**kwargs)
