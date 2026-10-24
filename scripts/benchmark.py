"""Sweep q (and optionally p), reuse loaded weights, save per-image metadata."""

import argparse
from pathlib import Path
import time
import torch
from parapc_fm import load_pipeline
from parapc_fm.cli import parse_args, generation_kwargs, save_result
from parapc_fm.distributed import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["flux", "sd3", "sd35", "sd15"], required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="One prompt per nonempty line")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--all_update_orders", action="store_true")
    parser.add_argument("--warmup", type=int, default=1, help="Unrecorded generations per configuration")
    options, remaining = parser.parse_known_args()
    if options.start < 0 or options.count < 1 or options.warmup < 0:
        parser.error("Require start >= 0, count >= 1, warmup >= 0")
    if options.model == "sd15" and options.all_update_orders:
        parser.error("SD1.5 uses ParaTAA's taa_order, not Flow update_order")
    args = parse_args(options.model, remaining)
    prompts = [s.strip() for s in options.prompts.read_text().splitlines() if s.strip()]
    selected = prompts[options.start:options.start + options.count]
    if len(selected) != options.count:
        parser.error("Requested prompt range exceeds the prompt file")
    runtime = Runtime()
    try:
        runtime.synchronize()
        start = time.perf_counter()
        pipe = load_pipeline(options.model, model_id=args.model_id, runtime=runtime,
                             torch_dtype=getattr(torch, args.dtype),
                             local_files_only=args.local_files_only)
        runtime.synchronize()
        load_seconds = runtime.maximum(time.perf_counter() - start)
        root = args.output_dir
        orders = range(3) if options.all_update_orders else [getattr(args, "update_order", None)]
        for p in orders:
            if p is not None:
                args.update_order = p
            for q in range(3):
                args.init_order = q
                args.prompt = selected[0]
                for _ in range(options.warmup):
                    pipe(**generation_kwargs(args))
                for index, prompt in enumerate(selected, options.start):
                    args.prompt = prompt
                    args.output_dir = root / (f"p{p}_q{q}" if p is not None else f"q{q}") / f"{index:05d}"
                    result = pipe(**generation_kwargs(args))
                    save_result(pipe, args, result, options.model, load_seconds)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
