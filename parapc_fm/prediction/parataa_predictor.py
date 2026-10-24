"""ParaTAA initialization on its reversed DDIM timestep grid.

    Slopes are finite differences of latent states, not flow-model velocities.
"""

import torch
from .coefficients import correction_weights, validate_orders


def initialize_ddim_states(sampler, init_order):
    """Initialize newly exposed ParaTAA states with q=0,1,2.

    Uses signed displacements consistently. Missing history falls back to a
    constant state (then linear prediction once one interval is available).
    """
    validate_orders(0, init_order)
    if sampler.uninitialized_index < 0:
        return
    if sampler.finished_index - sampler.uninitialized_index >= sampler.window_size:
        return
    end = max(-1, int(sampler.finished_index) - sampler.window_size)
    reference = sampler.uninitialized_index + 1
    states, times = sampler._samples, sampler.scheduler_timesteps
    targets = range(end + 1, sampler.uninitialized_index + 1)
    # The initial noise has no preceding interval on the reversed grid.
    if init_order == 0 or sampler.new_image or reference + 1 >= len(times):
        for j in targets:
            states[j] = states[reference]
    else:
        velocity = (states[reference] - states[reference + 1]) / (
            times[reference] - times[reference + 1])
        previous = None
        if init_order == 2 and reference + 2 < len(times):
            previous = (states[reference + 1] - states[reference + 2]) / (
                times[reference + 1] - times[reference + 2])
        for j in targets:
            step = times[j] - times[reference]
            value = states[reference] + step * velocity
            if previous is not None:
                offset = ((times[reference + 1] - times[reference]) / step).reshape(1)
                weight = correction_weights(offset, step)[0].to(states.dtype)
                value = value + weight * (previous - velocity)
            states[j] = value
    sampler.new_image = False
    sampler.uninitialized_index = end


def build_ddim_grid(scheduler, num_steps, eta, device):
    scheduler_timesteps = scheduler.timesteps.tolist()
    scheduler_prev_timesteps = scheduler_timesteps[1:]
    scheduler_prev_timesteps.append(0)
    scheduler_timesteps = scheduler_timesteps[::-1]
    scheduler_prev_timesteps = scheduler_prev_timesteps[::-1]
    alphas_cumprod = [1 - scheduler.alphas_cumprod[t] for t in scheduler_timesteps]
    alphas_cumprod_prev = [1 - scheduler.alphas_cumprod[t] for t in scheduler_prev_timesteps]
    now_coeff = torch.tensor(alphas_cumprod, dtype=torch.float64)
    next_coeff = torch.tensor(alphas_cumprod_prev, dtype=torch.float64)
    now_coeff = torch.clamp(now_coeff, min=0)
    next_coeff = torch.clamp(next_coeff, min=0)
    m_now_coeff = torch.clamp(1 - now_coeff, min=0)
    m_next_coeff = torch.clamp(1 - next_coeff, min=0)
    noise_thr = torch.sqrt(next_coeff / now_coeff) * torch.sqrt(1 - (1 - now_coeff) / (1 - next_coeff))
    nl = noise_thr * eta
    nl[0] = 0.0
    m_nl_next_coeff = torch.clamp(next_coeff - nl ** 2, min=0)
    coeff_x = torch.sqrt(m_next_coeff) / torch.sqrt(m_now_coeff)
    coeff_d = torch.sqrt(m_nl_next_coeff) - torch.sqrt(now_coeff) * coeff_x
    noise_thr = noise_thr.to(device)
    nl = nl.to(device)
    coeff_x = coeff_x.to(device)
    coeff_d = coeff_d.to(device)
    coeff_x_mat = torch.zeros(num_steps, num_steps, device=device, dtype=torch.float64)
    coeff_d_mat = torch.zeros(num_steps, num_steps, device=device, dtype=torch.float64)
    nl_mat = torch.zeros(num_steps, num_steps, device=device, dtype=torch.float64)
    for t in range(num_steps):
        coeff_x_mat[t, t:] = torch.cumprod(coeff_x[t:], dim=0)
        coeff_d_mat[t, t] = coeff_d[t]
        coeff_d_mat[t, t + 1:] = coeff_x_mat[t, t:-1] * coeff_d[t + 1:]
        nl_mat[t, t] = nl[t]
        nl_mat[t, t + 1:] = coeff_x_mat[t, t:-1] * nl[t + 1:]
    scheduler_timesteps_list = scheduler_timesteps.copy()
    scheduler_timesteps = torch.tensor(scheduler_timesteps, device=device)
    return {'scheduler_timesteps': scheduler_timesteps, 'scheduler_timesteps_list': scheduler_timesteps_list, 'noise_thr': noise_thr, 'nl': nl, 'coeff_x': coeff_x, 'coeff_d': coeff_d, 'coeff_x_mat': coeff_x_mat, 'coeff_d_mat': coeff_d_mat, 'nl_mat': nl_mat}
