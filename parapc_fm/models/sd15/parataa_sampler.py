"""ParaTAA DDIM core adapted from the local ParaTAA-Diffusion project.

Upstream: https://github.com/TZW1998/ParaTAA-Diffusion
Authors: Zhiwei Tang, Jiasheng Tang, Hao Luo, Fan Wang, Tsung-Hui Chang.
Source: parallel_sd_unic_2.py.
Prediction and initialization have been moved to parapc_fm.prediction.
"""
import torch
from parapc_fm.prediction.parataa_predictor import build_ddim_grid

class ParallelDDIM:

    def __init__(self, timesteps=50, scheduler=None, window_size=8, eta=0.0, cfg_scale=4.0, order=8, memory_size=2, lmd=0.001, tol1=0.001, tol2=0.001, device='cuda', use_fp16=False):
        self.eta = eta
        self.timesteps = timesteps
        self.num_steps = timesteps
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale = cfg_scale
        self.tol1 = tol1
        self.tol2 = tol2
        self.lmd = lmd
        self.memory_size = memory_size
        self.use_fp16 = use_fp16
        self.window_size = window_size
        self.order = order
        assert order <= window_size
        self.__dict__.update(build_ddim_grid(self.scheduler, self.num_steps, self.eta, self.device))

    def is_finished(self):
        return self._is_finished

    def get_last_sample(self):
        return self._samples[0].clone()

    def step(self, direction, model_kwargs):
        direction = direction.to(dtype=torch.float64)
        timestep, _ = model_kwargs['timestep'].chunk(2, dim=0)
        t = torch.tensor([self.scheduler_timesteps_list.index(ts) for ts in timestep], device=self.device)
        new_samples = torch.einsum('i,ijkl->ijkl', self.coeff_x[t], self._samples[t + 1]) + torch.einsum('i,ijkl->ijkl', self.coeff_d[t], direction) + torch.einsum('i,ijkl->ijkl', self.nl[t], self.noise_vectors[t])
        residual = new_samples - self._samples[t]
        residual_norm = torch.sqrt(torch.mean(residual ** 2, dim=(1, 2, 3)))
        unfinished_index = torch.where((residual_norm > self.tol1) & (residual_norm > self.tol2 * self.noise_thr[t]))[0]
        self.taa_step(t, direction, residual, unfinished_index)
        self.uninitialized_index = min([self.uninitialized_index, t[0].item()])
        if self.finished_index == 0:
            self._is_finished = True

    def compute_ho_residual(self, t, direction):
        update_times = min([self.order, len(t)])
        order_index = torch.clamp(t + update_times, max=t[-1] + 1)
        coeff_d_sub_mat = self.coeff_d_mat[t[0]:t[-1] + 1, t[0]:t[-1] + 1]
        coeff_nl_sub_mat = self.nl_mat[t[0]:t[-1] + 1, t[0]:t[-1] + 1]
        coeff_d_sub_mat = torch.tril(coeff_d_sub_mat, diagonal=update_times - 1)
        coeff_nl_sub_mat = torch.tril(coeff_nl_sub_mat, diagonal=update_times - 1)
        new_samples = torch.einsum('i,ijkl->ijkl', self.coeff_x_mat[t, order_index - 1], self._samples[order_index]) + torch.einsum('ip,pjkl->ijkl', coeff_d_sub_mat, direction) + torch.einsum('ip,pjkl->ijkl', coeff_nl_sub_mat, self.noise_vectors[t])
        ho_residual = new_samples - self._samples[t]
        return ho_residual

    def taa_step(self, t, direction, residual, unfinished_index):
        if len(unfinished_index) == 0:
            self.finished_index = t[0]
        else:
            self.finished_index = t[unfinished_index.max()].item()
        ho_residual = self.compute_ho_residual(t, direction)
        Gf = torch.zeros_like(ho_residual)
        if self.residual_memory is None:
            self.residual_memory = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
            self.samples_memory = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
            self.residual_memory[0, t] = ho_residual
            self.samples_memory[0, t] = self._samples[t]
            self.memory_indexes[t] = torch.clamp(self.memory_indexes[t] + 1, max=self.memory_size)
        else:
            padded_residual = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
            padded_samples = torch.zeros(1, self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
            padded_residual[0, t] = ho_residual
            padded_samples[0, t] = self._samples[t]
            self.residual_memory = torch.cat([self.residual_memory, padded_residual], dim=0)
            self.samples_memory = torch.cat([self.samples_memory, padded_samples], dim=0)
            self.residual_memory = self.residual_memory[-self.memory_size:]
            self.samples_memory = self.samples_memory[-self.memory_size:]
            self.memory_indexes[t] = torch.clamp(self.memory_indexes[t] + 1, max=self.memory_size)
            residual_diff = self.residual_memory[1:] - self.residual_memory[:-1]
            sample_diff = self.samples_memory[1:] - self.samples_memory[:-1]
            residual_diff_t = residual_diff[:, t, :, :, :]
            sample_diff_t = sample_diff[:, t, :, :, :]
            update_indexes = unfinished_index.max() if len(unfinished_index) > 0 else 0
            update_indexes = torch.arange(update_indexes, device=self.device)
            update_indexes = update_indexes[self.memory_indexes[t[update_indexes]] > 1]
            if len(update_indexes) > 0:
                use_memory = self.memory_indexes[t[update_indexes]] - 1
                use_memory_max = use_memory.max()
                sample_diff_mat = sample_diff_t[:use_memory_max, update_indexes[0]:, :, :, :]
                res_diff_mat = residual_diff_t[:use_memory_max, update_indexes[0]:, :, :, :]
                flip_res_diff_mat = torch.flip(res_diff_mat, dims=[1])
                B = torch.einsum('ijklm,pjklm->ipj', flip_res_diff_mat, flip_res_diff_mat)
                B = torch.flip(torch.cumsum(B, dim=2), dims=[2])[:, :, :len(update_indexes)]
                B = B.permute(2, 0, 1)
                flip_ho_residual = torch.flip(ho_residual[update_indexes[0]:, :, :, :], dims=[0])
                d = torch.einsum('ijklm,jklm->ji', flip_res_diff_mat, flip_ho_residual)
                d = torch.flip(torch.cumsum(d, dim=0), dims=[0])[:len(update_indexes)]
                indices = torch.arange(d.shape[1], device=d.device).unsqueeze(0).expand(d.shape)
                mask_d = indices + 1 > use_memory.unsqueeze(1)
                d[mask_d] = 0
                mask_B = mask_d.unsqueeze(2) | mask_d.unsqueeze(1)
                B[mask_B] = 0
                B = B + self.lmd * torch.eye(use_memory_max, device=B.device, dtype=torch.float64).unsqueeze(0)
                if use_memory_max == 1:
                    solve_d = d / B.squeeze(-1)
                else:
                    solve_d = torch.linalg.solve(B, d)
                A = sample_diff_mat[:, :len(update_indexes), :, :, :] + res_diff_mat[:, :len(update_indexes), :, :, :]
                Gf_flat = torch.einsum('ijklm,ji->jklm', A, solve_d)
                Gf[update_indexes] = Gf_flat
        self._samples[t] = self._samples[t] + ho_residual - Gf

    def initialize(self, noise=None):
        self._is_finished = False
        self.finished_index = self.num_steps
        self.uninitialized_index = self.num_steps - 1
        if noise is not None:
            self.noise_vectors = noise.to(self.device, dtype=torch.float64)
        else:
            self.noise_vectors = torch.randn(self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
        self._samples = torch.zeros(self.num_steps + 1, 4, 64, 64, device=self.device, dtype=torch.float64)
        self._samples[-1] = self.noise_vectors[-1]
        self.samples_memory = None
        self.residual_memory = None
        self.memory_indexes = torch.zeros(self.num_steps + 1, dtype=torch.long, device=self.device)
        self.new_image = True
