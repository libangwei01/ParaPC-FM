"""Moment-matching coefficients; all public orders use the paper's p and q."""

import torch


def validate_orders(update_order=0, init_order=1):
    for name, value in (("update_order", update_order), ("init_order", init_order)):
        if type(value) is not int or value not in (0, 1, 2):
            raise ValueError(f"{name} must be an integer in {{0, 1, 2}}")


def correction_weights(offsets, step):
    """Solve R a = b for velocity differences at normalized time offsets.

    offsets[m] = (t_m - t_reference) / step; b[k] = step/(k+2).
    Solving in float64 avoids unsupported/unstable low-precision linear solves.
    Signed steps support both increasing and decreasing time grids.
    """
    offsets = torch.as_tensor(offsets)
    if offsets.numel() == 0:
        return offsets.to(torch.float64)
    offsets = offsets.to(torch.float64)
    step = torch.as_tensor(step, device=offsets.device, dtype=torch.float64)
    if not torch.isfinite(step) or step == 0 or not torch.isfinite(offsets).all():
        raise ValueError("Coefficient time offsets must be finite with a nonzero step")
    powers = torch.arange(1, offsets.numel() + 1, device=offsets.device)
    matrix = offsets.unsqueeze(0) ** powers.unsqueeze(1)
    return torch.linalg.solve(matrix, step / (powers + 1))
