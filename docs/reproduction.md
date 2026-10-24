# Experiment configurations

The implementation uses NVIDIA GPUs with NCCL. The paper's main Flow
experiments used eight Ascend 910B3 NPUs; SD1.5 experiments used eight RTX 4090
GPUs. Record hardware and software versions alongside timing measurements.

## Flow main tables

Use 512×512, guidance 3.5, window 8, the same fixed prompt file and seed 0.
The paper reports 1,000 prompts. Select the prompt range with `--start` and
`--count`, and use identical prompts and ordering across compared settings.

| Model | Steps | p | Tolerance |
|---|---|---|---|
| FLUX | 25 | 0,1,2 | 1.0 |
| FLUX | 50 | 0,1,2 | 0.5 |
| FLUX | 100 | 0,1 | 0.4 |
| FLUX | 100 | 2 | 0.5 |
| SD3 | 50 | 0,1,2 | 0.5 |
| SD3.5 | 50 | 0,1,2 | 1.0 |

For each row compare q=0,1,2. Example:

```bash
torchrun --standalone --nproc_per_node=8 scripts/benchmark.py \
  --model flux --prompts prompts.txt --start 1000 --count 1000 \
  --all_update_orders --num_inference_steps 50 --tolerance 0.5 \
  --output_dir outputs/flux50
```

For FLUX 100 steps, run p=0/1 with tolerance 0.4 and p=2 with tolerance 0.5
separately; `--all_update_orders` does not silently change tolerances.
Default demo YAML files represent the 50-step setting, not every paper row.

## SD1.5 ParaTAA

Paper settings: 50 steps, guidance 4.0, eta=0, seed=0, one image per prompt,
window=8, taa_order=8, memory_size=2, tol1=tol2=0.001, float32 UNet,
q=0/1/2; 10,000 prompts. The paper hardware was eight RTX 4090 GPUs.

```bash
torchrun --standalone --nproc_per_node=8 scripts/benchmark.py \
  --model sd15 --prompts prompts.txt --count 10000 \
  --config configs/sd15.yaml --output_dir outputs/sd15
```

ParaTAA states and coefficients use float64 independently of UNet precision.
See `algorithm.md` for initialization and guidance conventions.

## Metrics and timing

The benchmark writes images and per-image metadata. ImageReward, PE-Core/CLIP and
GenEval require their own model weights and evaluation setup. The paper's
normalized RGB RMSE is sqrt(mean((image/255-reference/255)^2)), measured against
the corresponding same-order sequential reference. Raw latent RMSE is not the
paper metric.

Warmup runs are not recorded. Loading is separate from preparation, sampling
and VAE decode. Every phase is synchronized and wall-clock times use the
maximum across processes. Cold-start and warm-run numbers should be reported
separately. Initialization speedup compares q=0 to q>0 at fixed p, steps,
hardware, tolerance and prompts; it is distinct from speedup over a sequential
solver.
