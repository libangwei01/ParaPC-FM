"""Flow update and truncated initialization, independent of model and transport."""

import torch
from .coefficients import correction_weights, validate_orders


def _increment(velocities, times, index, update_order, corrector=True):
    step = times[index + 1] - times[index]
    result = step.to(velocities.dtype) * velocities[index]
    if update_order == 0:
        return result
    # p corrector terms: p-1 backward terms and one forward term.
    # A predictor without a forward velocity has only p-1 backward terms.
    backward = min(update_order - 1, index)
    indices = list(range(index - 1, index - backward - 1, -1))
    if corrector:
        indices.append(index + 1)
    if indices:
        offsets = (times[indices] - times[index]) / step
        weights = correction_weights(offsets, step).to(velocities.dtype)
        for weight, j in zip(weights, indices):
            result = result + weight * (velocities[j] - velocities[index])
    return result


def update_window(states, velocities, times, begin, end, update_order):
    """Return updated states [begin+1, end], without changing old iterates.

    This preserves the research implementation's corrected anchor and predictor
    at the window's last step, where v[end] has not been evaluated yet.
    """
    validate_orders(update_order, 0)
    anchor = states[begin].clone()
    if update_order and begin > 0:
        anchor = states[begin - 1] + _increment(
            velocities, times, begin - 1, update_order, corrector=True)
    increments = torch.stack([
        _increment(velocities, times, i, update_order, corrector=i + 1 < end)
        for i in range(begin, end)
    ])
    return anchor, anchor.unsqueeze(0) + increments.cumsum(dim=0)


def repair_window_tail(states, velocities, times, begin, end, update_order):
    """Predict remaining states after acceptance, as in the original solver."""
    if update_order < 2:
        return
    for j in range(begin, end + 1):
        states[j] = states[j - 1] + _increment(
            velocities, times, j - 1, update_order, corrector=False)


def initialize_future_states(states, velocities, times, old_end, new_end, init_order):
    """Initialize [old_end, new_end], retaining the original overlap convention.

    q=0 copies x[old_end]. q=1/2 extrapolate from the last evaluated velocity
    at old_end-1 and refresh x[old_end] as well as newly exposed states.
    """
    validate_orders(0, init_order)
    if init_order == 0:
        states[old_end:new_end + 1] = states[old_end].clone()
        return
    reference = old_end - 1
    anchor = states[reference].clone()
    for j in range(old_end, new_end + 1):
        step = times[j] - times[reference]
        value = anchor + step.to(velocities.dtype) * velocities[reference]
        if init_order == 2 and reference > 0:
            offsets = ((times[reference - 1] - times[reference]) / step).reshape(1)
            weight = correction_weights(offsets, step)[0].to(velocities.dtype)
            value = value + weight * (velocities[reference - 1] - velocities[reference])
        states[j] = value
