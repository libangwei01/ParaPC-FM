"""Shared demo interface and reproducible metadata output."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import torch
import yaml
from parapc_fm import load_pipeline, __version__
from parapc_fm.distributed import Runtime


def parser_for(model):
    parser = argparse.ArgumentParser(description=f"ParaPC-FM {model} single-image demo")
    parser.add_argument("--config", type=Path, help="Optional YAML defaults; command line overrides YAML")
    parser.add_argument("--model_id", help="Hugging Face model ID or local model directory")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / model)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default={"flux": "bfloat16", "sd3": "float16",
                                 "sd35": "bfloat16", "sd15": "float32"}[model])
    parser.add_argument("--guidance_scale", type=float, default=4.0 if model == "sd15" else 3.5)
    parser.add_argument("--init_order", type=int, choices=[0, 1, 2], default=1, help="Paper q")
    parser.add_argument("--window_size", type=int, default=8)
    if model == "sd15":
        parser.add_argument("--taa_order", type=int, default=8)
        parser.add_argument("--memory_size", type=int, default=2)
        parser.add_argument("--tol1", type=float, default=0.001)
        parser.add_argument("--tol2", type=float, default=0.001)
        parser.add_argument("--max_iterations", type=int)
    else:
        parser.add_argument("--update_order", type=int, choices=[0, 1, 2], default=0, help="Paper p")
        parser.add_argument("--tolerance", type=float, default=1.0 if model == "sd35" else 0.5)
    return parser


def parse_args(model, argv=None):
    parser = parser_for(model)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        config = yaml.safe_load(preliminary.config.read_text())
        if not isinstance(config, dict):
            parser.error("Configuration must be a YAML mapping")
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        unknown = set(config) - allowed
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        # Route YAML through argparse so types and choices are checked as for CLI.
        values = []
        for key, value in config.items():
            if key == "local_files_only":
                if type(value) is not bool:
                    parser.error("local_files_only must be a boolean")
                if value:
                    values.append("--local_files_only")
            else:
                values.extend([f"--{key}", str(value)])
        import sys
        return parser.parse_args(values + (list(argv) if argv is not None else sys.argv[1:]))
    return parser.parse_args(argv)


def generation_kwargs(args):
    excluded = {"config", "model_id", "local_files_only", "output_dir", "dtype"}
    return {k: v for k, v in vars(args).items() if k not in excluded}


def save_result(pipe, args, result, model, load_seconds):
    if not pipe.runtime.is_main:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output_dir / "image.png")
    metadata = {
        "model": model, "model_id": args.model_id or pipe.model_id,
        "parameters": generation_kwargs(args), "dtype": args.dtype,
        "world_size": pipe.runtime.world_size, "iterations": result.iterations,
        "model_evaluations": result.model_evaluations,
        "model_evaluations_definition": "Total guided timestep evaluations across all ranks; CFG pair counts once",
        "timings": {"model_load_seconds": load_seconds, **result.timings},
        "versions": {"parapc_fm": __version__, "torch": torch.__version__,
                     "diffusers": version("diffusers"), "transformers": version("transformers")},
        "hardware": torch.cuda.get_device_name(pipe.runtime.device),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output_dir / 'image.png'}; iterations={result.iterations}")


def main(model, argv=None):
    args = parse_args(model, argv)
    runtime = Runtime()
    try:
        runtime.synchronize()
        start = time.perf_counter()
        pipe = load_pipeline(model, model_id=args.model_id, runtime=runtime,
                             torch_dtype=getattr(torch, args.dtype),
                             local_files_only=args.local_files_only)
        runtime.synchronize()
        load_seconds = runtime.maximum(time.perf_counter() - start)
        result = pipe(**generation_kwargs(args))
        save_result(pipe, args, result, model, load_seconds)
    finally:
        runtime.close()
