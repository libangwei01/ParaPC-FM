# ParaPC-FM

Code for **ParaPC-FM: Accelerating Parallel Sampling via Principled Initialization**.
[中文文档](README_zh.md)

ParaPC-FM reuses past predictions to initialize future states in parallel
sampling. Multiple GPUs evaluate different timesteps of the same image.

## Supported models

| Model | Model ID | Demo |
|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | `demos/demo_flux_paradigms.py` |
| Stable Diffusion 3 | `stabilityai/stable-diffusion-3-medium-diffusers` | `demos/demo_sd3_paradigms.py` |
| Stable Diffusion 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | `demos/demo_sd35_paradigms.py` |
| Stable Diffusion 1.5 + ParaTAA | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `demos/demo_sd15_parataa.py` |

## Installation

Use Python 3.10+ and install PyTorch for your CUDA environment. From the
repository root:

```bash
python -m pip install -e .
```

The pipelines use Diffusers 0.36.0.

## Quick start

Generate an image with eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 demos/demo_flux_paradigms.py \
  --prompt "A cat holding a sign that says hello world" \
  --update_order 1 --init_order 1 --output_dir outputs/flux
```

Run the other models with their configuration files:

```bash
torchrun --standalone --nproc_per_node=8 demos/demo_sd3_paradigms.py --config configs/sd3.yaml
torchrun --standalone --nproc_per_node=8 demos/demo_sd35_paradigms.py --config configs/sd35.yaml
torchrun --standalone --nproc_per_node=8 demos/demo_sd15_parataa.py --config configs/sd15.yaml
```

For a single GPU:

```bash
python demos/demo_flux_paradigms.py --prompt "A watercolor landscape"
```

Set `--nproc_per_node` to the number of GPUs. Each GPU holds one complete
model replica, including the text encoders and VAE. Each invocation generates
one image. SD1.5 uses 512×512 resolution and deterministic DDIM (`eta=0`);
FLUX uses distilled guidance.

The FLUX, SD3 and SD3.5 pipelines use the ParaDiGMS-style framework.
SD1.5 uses ParaTAA. Framework-specific Python filenames contain
`paradigms` or `parataa`.

## Parameters

For FLUX, SD3 and SD3.5, `update_order` selects the update rule p and
`init_order` selects the initialization order q:

| Argument | Value | Meaning |
|---|---|---|
| `--update_order` | 0 | Euler |
| `--update_order` | 1 | Heun corrector |
| `--update_order` | 2 | UniPC-2 corrector |
| `--init_order` | 0 | Constant state initialization |
| `--init_order` | 1 | Linear extrapolation |
| `--init_order` | 2 | Second-order extrapolation using a historical velocity difference |

SD1.5 uses `--init_order` with ParaTAA parameters `--taa_order`,
`--memory_size`, `--tol1` and `--tol2`.

Common options include `--model_id`, `--local_files_only`, `--prompt`,
`--seed`, `--num_inference_steps`, `--height`, `--width`, `--dtype`,
`--guidance_scale`, `--window_size` and `--output_dir`.
`--model_id` accepts a Hugging Face model ID or a local weight directory.
SD3, SD3.5 and SD1.5 also accept `--negative_prompt`.
Command-line arguments override YAML configuration values.
Use `--help` to see the options for each demo.

## Outputs

The main process saves `image.png` and `metadata.json` to `--output_dir`.
Metadata contains generation parameters, software versions, GPU count,
correction iterations, guided timestep evaluations and timings. Each
classifier-free guidance pair counts as one guided timestep evaluation.
Model loading, preparation, sampling and decoding times are recorded
separately. Reusing an output directory overwrites its demo results.

## Python API

Run the following program with `torchrun` for multiple GPUs or `python` for
one GPU:

```python
from parapc_fm import load_pipeline
from parapc_fm.distributed import Runtime

runtime = Runtime()
try:
    pipe = load_pipeline("flux", runtime=runtime)
    result = pipe("A watercolor landscape", update_order=1, init_order=2)
    if runtime.is_main:
        result.images[0].save("landscape.png")
finally:
    runtime.close()
```

## Experiments

Create `prompts.txt` with one prompt per nonempty line, then compare p and q:

```bash
torchrun --standalone --nproc_per_node=8 scripts/benchmark.py \
  --model flux --prompts prompts.txt --count 1000 --all_update_orders \
  --num_inference_steps 50 --tolerance 0.5 --output_dir outputs/benchmark
```

The script compares q=0,1,2 with the same prompts and seed. Use
`--all_update_orders` to also compare p=0,1,2 for Flow models.
`--warmup` sets the number of warmup generations per configuration (default: 1).
See [experiment settings](docs/reproduction.md).

## Project structure

```text
parapc_fm/
  prediction/     Prediction, initialization and numerical coefficients
  sampling/       Sliding-window updates and convergence
  distributed/    Timestep distribution and communication
  models/         FLUX, SD3, SD3.5 and SD1.5 pipelines
demos/            Per-model examples
configs/          Model configurations
scripts/          Benchmark scripts
tests/            Numerical, pipeline and distributed tests
docs/             Algorithm and experiment settings
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests run without model weights. Distributed tests use two local CPU/Gloo
processes. See [algorithm details](docs/algorithm.md) for update and
initialization formulas.

## License

Original ParaPC-FM contributions are licensed under the [MIT License](LICENSE).
Third-party code is covered separately; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation and acknowledgments

Paper: **ParaPC-FM: Accelerating Parallel Sampling via Principled Initialization**.

The Flow sampling framework builds on
[ParaDiGMS](https://github.com/AndyShih12/paradigms). Please also cite Shih et al.,
*Parallel Sampling of Diffusion Models*, NeurIPS 2023, when using this framework.

The SD1.5 sampler builds on
[ParaTAA-Diffusion](https://github.com/TZW1998/ParaTAA-Diffusion).
Please also cite Tang et al., *Accelerating Parallel Sampling of Diffusion
Models*, ICML 2024, when using the ParaTAA sampler.
[Third-party attribution](THIRD_PARTY_NOTICES.md).
