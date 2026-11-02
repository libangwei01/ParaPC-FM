"""Adaptive ParaPC-FM sliding window based on ParaDiGMS.

ParaDiGMS: Copyright (c) 2023 Andy Shih, MIT License.
See LICENSE for the MIT License text.
"""

import math
import torch
from parapc_fm.prediction import validate_orders, initialize_future_states, update_window
from parapc_fm.prediction.paradigms_predictor import repair_window_tail


@torch.no_grad()
def sample_flow(initial, times, velocity, runtime, *, update_order=0,
                init_order=1, window_size=8, tolerance=0.5):
    validate_orders(update_order, init_order)
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    times = times.to(device=initial.device, dtype=torch.float64)
    differences = times.diff()
    if times.ndim != 1 or len(times) < 2 or not torch.isfinite(times).all() or not (
        (differences > 0).all() or (differences < 0).all()
    ):
        raise ValueError("times must be finite and strictly monotone")
    steps = len(times) - 1
    states = initial.unsqueeze(0).repeat(steps + 1, *([1] * initial.ndim))
    velocities = torch.zeros_like(states)
    runtime.broadcast(states)
    begin, end = 0, min(window_size, steps)
    iterations = evaluations = 0
    while begin < steps:
        indices = list(range(begin, end))
        velocities[begin:end] = runtime.evaluate(
            indices, initial, lambda i: velocity(states[i], i))
        if not torch.isfinite(velocities[begin:end]).all():
            raise RuntimeError("Non-finite model velocity; window cannot be accepted")
        evaluations += len(indices)
        anchor, updated = update_window(states, velocities, times, begin, end, update_order)
        if not torch.isfinite(updated).all():
            raise RuntimeError("Non-finite state update; window cannot be accepted")
        # Each image must meet the criterion; normalize over latent elements.
        error = (updated.float() - states[begin + 1:end + 1].float()).square()
        error = error.reshape(end - begin, initial.shape[0], -1).mean(dim=-1)
        error = error / differences[begin:end].square().unsqueeze(-1)
        failed = torch.nonzero((error > tolerance**2).any(dim=1)).flatten()
        advance = int(failed[0]) + 1 if failed.numel() else end - begin
        new_begin = begin + advance
        new_end = min(new_begin + window_size, steps)
        states[begin] = anchor
        states[begin + 1:end + 1] = updated
        if new_begin <= end:
            repair_window_tail(states, velocities, times, new_begin, end, update_order)
        initialize_future_states(states, velocities, times, end, new_end, init_order)
        # Root is authoritative even on devices with slightly differing kernels.
        runtime.broadcast(states)
        begin, end = new_begin, new_end
        iterations += 1
    return states[-1], iterations, evaluations
